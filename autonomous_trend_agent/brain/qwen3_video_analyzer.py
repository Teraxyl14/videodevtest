"""
Qwen3.5-0.8B Video Analyzer (Phase 3, D3.1-D3.3)

Natively multimodal (Early Fusion) visual analysis via vLLM sidecar.
Replaces the old Qwen3-4B-Instruct-AWQ direct vLLM Python approach
with an OpenAI-compatible HTTP client for the vLLM server.

vLLM sidecar command:
    vllm serve Qwen/Qwen3.5-0.8B --port 8002 --tensor-parallel-size 1 \
        --max-model-len 4096 --gpu-memory-utilization 0.15 \
        --enable-chunked-prefill

VRAM footprint: ~1.5-2GB (BF16 weights + 4K context KV cache + vLLM overhead)

Objectives: D3.1 (visual analysis), D3.2 (scene detection), D3.3 (<2GB VRAM)
"""

import json
import base64
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ViralMoment:
    """A potential viral clip identified in the video."""
    start_time: float
    end_time: float
    hook: str
    reason: str
    viral_score: float              # 0-1
    editing_notes: Optional[dict] = None
    segment_groups: Optional[List[Tuple[float, float]]] = None


@dataclass
class SceneChange:
    """A detected scene transition."""
    timestamp: float
    description: str
    visual_elements: List[str] = field(default_factory=list)
    scene_type: str = ""            # "talking_head", "action", "b-roll", etc.


@dataclass
class VideoAnalysis:
    """Complete analysis of a video."""
    video_path: str
    duration: float
    viral_moments: List[ViralMoment]
    scene_changes: List[SceneChange]
    overall_theme: str
    target_audience: str


class Qwen35VideoAnalyzer:
    """
    Visual analysis using Qwen3.5-0.8B via vLLM OpenAI-compatible API.

    Always resident in VRAM as part of Parallel Residency (~1.5-2GB).
    Communicates with vLLM sidecar at localhost:8002.

    Usage:
        analyzer = Qwen35VideoAnalyzer()
        analysis = analyzer.analyze_video("/path/to/video.mp4")
    """

    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    VLLM_BASE_URL = "http://localhost:8002/v1"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.model_id = model_id or self.MODEL_ID
        self.base_url = base_url or self.VLLM_BASE_URL
        self._client = OpenAI(base_url=self.base_url, api_key="none")
        logger.info(
            f"[Qwen3.5-VL] Initialized (model={self.model_id}, "
            f"endpoint={self.base_url})"
        )

    def analyze_frame(self, image_b64: str, prompt: str) -> str:
        """
        Analyze a single frame with a custom prompt.

        Args:
            image_b64: Base64-encoded JPEG image
            prompt: Analysis question/instruction

        Returns:
            Model response text
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=512,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[Qwen3.5-VL] Frame analysis failed: {e}")
            return ""

    def analyze_video(
        self,
        video_path: str,
        num_frames: int = 16,
        transcript_text: Optional[str] = None,
    ) -> VideoAnalysis:
        """
        Full video analysis: extract frames, detect scenes, identify viral moments.

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            transcript_text: Optional transcript for context

        Returns:
            VideoAnalysis with viral moments and scene changes
        """
        logger.info(f"[Qwen3.5-VL] Analyzing video: {video_path}")

        # Extract frames
        frames_b64, duration, timestamps = self._extract_frames(
            video_path, num_frames
        )

        if not frames_b64:
            logger.error("[Qwen3.5-VL] No frames extracted.")
            return VideoAnalysis(
                video_path=video_path, duration=0,
                viral_moments=[], scene_changes=[],
                overall_theme="unknown", target_audience="unknown",
            )

        # Analyze frames for scene understanding
        scene_changes = self._detect_scenes(frames_b64, timestamps)

        # Identify viral moments
        context = f"\nTranscript: {transcript_text[:2000]}" if transcript_text else ""
        viral_moments = self._identify_viral_moments(
            frames_b64, timestamps, duration, context
        )

        # Get overall theme
        theme_response = self._analyze_overall(frames_b64[0], duration, context)

        return VideoAnalysis(
            video_path=video_path,
            duration=duration,
            viral_moments=viral_moments,
            scene_changes=scene_changes,
            overall_theme=theme_response.get("theme", "Unknown"),
            target_audience=theme_response.get("audience", "General"),
        )

    def _extract_frames(
        self,
        video_path: str,
        num_frames: int = 16,
        resize: Tuple[int, int] = (448, 448),
    ) -> Tuple[List[str], float, List[float]]:
        """
        Extract evenly-spaced frames as base64 JPEG.
        Uses FFmpeg fast-seek (-ss before -i) for instant keyframe access.

        Returns:
            (list of base64 strings, duration in seconds, timestamps)
        """
        video_path = str(video_path)

        # Get duration via ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        try:
            probe_result = subprocess.run(
                probe_cmd, capture_output=True, text=True, check=True
            )
            probe_data = json.loads(probe_result.stdout)
        except Exception as e:
            logger.error(f"[Qwen3.5-VL] ffprobe failed: {e}")
            return [], 0.0, []

        duration = float(probe_data.get('format', {}).get('duration', 0))
        if duration <= 0:
            return [], 0.0, []

        timestamps = np.linspace(0, duration - 0.1, num_frames).tolist()
        temp_dir = tempfile.mkdtemp(prefix="qwen35_frames_")
        frames_b64: List[str] = []

        try:
            for i, ts in enumerate(timestamps):
                frame_path = Path(temp_dir) / f"frame_{i:03d}.jpg"

                # FFmpeg fast-seek: -ss BEFORE -i
                cmd = [
                    'ffmpeg', '-y', '-ss', str(ts), '-i', video_path,
                    '-frames:v', '1', '-q:v', '2',
                    '-vf', f'scale={resize[0]}:{resize[1]}',
                    str(frame_path)
                ]
                subprocess.run(cmd, capture_output=True, check=False)

                if frame_path.exists():
                    with open(frame_path, "rb") as f:
                        frames_b64.append(base64.b64encode(f.read()).decode())

            logger.info(f"[Qwen3.5-VL] Extracted {len(frames_b64)}/{num_frames} frames")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        return frames_b64, duration, timestamps[:len(frames_b64)]

    def _detect_scenes(
        self, frames_b64: List[str], timestamps: List[float]
    ) -> List[SceneChange]:
        """Detect scene transitions by comparing consecutive frames."""
        scenes: List[SceneChange] = []

        # Analyze first frame as the opening scene
        if frames_b64:
            desc = self.analyze_frame(
                frames_b64[0],
                "Describe this video frame in one sentence. "
                "What type of scene is this? (talking_head, action, b-roll, text_overlay, other)"
            )
            scenes.append(SceneChange(
                timestamp=timestamps[0],
                description=desc,
                scene_type="opening",
            ))

        # Sample every 4th frame for scene change detection
        for i in range(4, len(frames_b64), 4):
            prompt = (
                "Compare this frame to the previous content. "
                "Has the scene changed significantly? If yes, describe the new scene. "
                "If no, say 'SAME'. Be brief."
            )
            response = self.analyze_frame(frames_b64[i], prompt)

            if response and "SAME" not in response.upper():
                scenes.append(SceneChange(
                    timestamp=timestamps[i],
                    description=response,
                ))

        return scenes

    def _identify_viral_moments(
        self,
        frames_b64: List[str],
        timestamps: List[float],
        duration: float,
        context: str,
    ) -> List[ViralMoment]:
        """Identify 3-4 viral-worthy segments using visual + context analysis."""
        # Use representative frames (every 4th) for the viral analysis
        sample_indices = list(range(0, len(frames_b64), max(1, len(frames_b64) // 8)))
        sample_frames = [frames_b64[i] for i in sample_indices]
        sample_times = [timestamps[i] for i in sample_indices]

        prompt = f"""Analyze these video frames sampled at timestamps {sample_times}.
Video duration: {duration:.1f} seconds.{context}

Identify 3-4 segments that would make viral short-form videos (30-90 seconds each).
For each segment, provide:
- start_time (seconds)
- end_time (seconds)  
- hook (attention-grabbing first sentence)
- reason (why this would go viral)
- viral_score (0.0 to 1.0)

Return ONLY valid JSON array:
[{{"start_time": X, "end_time": Y, "hook": "...", "reason": "...", "viral_score": Z}}]"""

        # Send first frame with the prompt (0.8B may not handle multi-image well)
        if sample_frames:
            response = self.analyze_frame(sample_frames[0], prompt)
        else:
            return []

        # Parse response
        moments: List[ViralMoment] = []
        try:
            # Try to extract JSON from the response
            text = response.strip()
            # Find JSON array in response
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                data = json.loads(text[start_idx:end_idx])
                for item in data:
                    moments.append(ViralMoment(
                        start_time=float(item.get("start_time", 0)),
                        end_time=float(item.get("end_time", 30)),
                        hook=item.get("hook", ""),
                        reason=item.get("reason", ""),
                        viral_score=float(item.get("viral_score", 0.5)),
                    ))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[Qwen3.5-VL] Failed to parse viral moments: {e}")
            # Fallback: create evenly-spaced segments
            seg_duration = min(60, duration / 3)
            for i in range(3):
                start = i * seg_duration
                moments.append(ViralMoment(
                    start_time=start,
                    end_time=min(start + seg_duration, duration),
                    hook=f"Segment {i+1}",
                    reason="Auto-generated segment",
                    viral_score=0.3,
                ))

        return moments

    def _analyze_overall(
        self, first_frame_b64: str, duration: float, context: str
    ) -> Dict:
        """Get overall video theme and target audience."""
        prompt = f"""Based on this opening frame of a {duration:.0f}-second video{context},
determine:
1. The overall theme/topic
2. The target audience

Return JSON: {{"theme": "...", "audience": "..."}}"""

        response = self.analyze_frame(first_frame_b64, prompt)

        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(response[start_idx:end_idx])
        except (json.JSONDecodeError, ValueError):
            pass

        return {"theme": "Unknown", "audience": "General"}
