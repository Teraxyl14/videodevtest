"""
Qwen3-VL Video Analyzer - Local GPU-Accelerated Video Understanding
Uses Qwen3-VL-8B-Instruct with Int4 quantization (~5.5GB VRAM)
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ViralMoment:
    """A potential viral clip identified in the video"""
    start_time: float
    end_time: float
    hook: str
    reason: str
    viral_score: float  # 0-1
    # Optional fields for Gemini-enhanced analysis
    editing_notes: Optional[object] = None  # EditingNotes from Gemini
    segment_groups: Optional[List[Tuple[float, float]]] = None  # For non-contiguous grouping


@dataclass
class VideoAnalysis:
    """Complete analysis of a video"""
    video_path: str
    duration: float
    viral_moments: List[ViralMoment]
    overall_theme: str
    target_audience: str


class Qwen3VideoAnalyzer:
    """
    GPU-accelerated video analyzer using Qwen3-VL-8B-Instruct.
    Runs locally in Docker with Int4 quantization for memory efficiency.
    """
    
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    
    def __init__(self, device: str = "cuda", load_in_4bit: bool = True):
        """
        Initialize Qwen3-VL model.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            load_in_4bit: Use Int4 quantization (recommended for 16GB VRAM)
        """
        self.device = device
        self.model = None
        self.processor = None
        self.load_in_4bit = load_in_4bit
        
        print(f"[Qwen3-VL] Initializing (4-bit: {load_in_4bit}, device: {device})")
    
    def _ensure_loaded(self):
        """Lazy load model on first use to save VRAM"""
        if self.model is not None:
            return
        
        
        # Check for configured path or default
        configured_path = os.environ.get("QWEN_MODEL_PATH")
        # CRITICAL: Force Qwen3-VL-8B-Instruct to avoid legacy shape errors
        if "Qwen2.5" in str(configured_path) or "2B" in str(configured_path):
             print(f"[Qwen3-VL] Warning: QWEN_MODEL_PATH points to legacy '{configured_path}'. Overriding to force Qwen3-VL.")
             configured_path = None # Force fallback to self.MODEL_ID
        if configured_path:
            local_path = Path(configured_path)
            # Handle relative path in Docker
            if not local_path.exists() and not str(local_path).startswith("/"):
                 local_path = Path("/workspace") / configured_path
        else:
            # Try Qwen3 first
            paths_to_try = [
                Path("models/Qwen3-VL-8B-Instruct"),
                Path("/app/models/Qwen3-VL-8B-Instruct"),
                Path("/workspace/models/Qwen3-VL-8B-Instruct"),
            ]
            local_path = Path("models/Qwen3-VL-8B-Instruct") # Default
            for p in paths_to_try:
                if p.exists():
                    local_path = p
                    break
        
        model_id = str(local_path) if local_path.exists() else (configured_path or self.MODEL_ID)
        
        print(f"[Qwen3-VL] Loading {model_id}...")
        
        # Auto-disable 4-bit for 2B model (it's small enough and avoids bnb errors)
        is_small_model = "2B" in model_id or "2b" in model_id
        if is_small_model and self.load_in_4bit:
            print("[Qwen3-VL] Detected 2B model - Disabling 4-bit quantization (running in FP16)")
            self.load_in_4bit = False

        print(f"[Debug] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Debug] Current Device: {torch.cuda.get_device_name(0)}")
            print(f"[Debug] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        try:
            # Import AutoModel because Qwen3 uses custom code
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            # Load with 4-bit quantization
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="cuda" if torch.cuda.is_available() else "auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    device_map="cuda" if torch.cuda.is_available() else "auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            self._process_vision_info = process_vision_info
            
            print(f"[Qwen3-VL] Model loaded successfully")
            
        except Exception as e:
            print(f"[Qwen3-VL] Failed to load model: {e}")
            raise
    
    def _extract_frames(
        self, 
        video_path: str, 
        num_frames: int = 16,
        resize: Tuple[int, int] = (448, 448)
    ) -> Tuple[List, float]:
        """
        Extract evenly-spaced frames from video using FFmpeg.
        FFmpeg works reliably in Docker while OpenCV often fails.
        
        Returns:
            Tuple of (frame_images, duration, timestamps)
        """
        import subprocess
        import tempfile
        import shutil
        from PIL import Image
        
        video_path = str(video_path)
        
        # Get video metadata using ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            import json as json_module
            probe_data = json_module.loads(probe_result.stdout)
        except Exception as e:
            raise RuntimeError(f"[Qwen3-VL] Failed to probe video: {e}")
        
        # Extract duration and fps from probe data
        duration = float(probe_data.get('format', {}).get('duration', 0))
        fps = 30.0  # Default
        for stream in probe_data.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Parse frame rate (e.g., "30000/1001" or "30/1")
                r_frame_rate = stream.get('r_frame_rate', '30/1')
                if '/' in r_frame_rate:
                    num, den = map(float, r_frame_rate.split('/'))
                    fps = num / den if den > 0 else 30.0
                else:
                    fps = float(r_frame_rate)
                break
        
        if duration <= 0:
            raise RuntimeError(f"[Qwen3-VL] Video has invalid duration: {duration}")
        
        print(f"[Qwen3-VL] Video: {duration:.1f}s, {fps:.2f} fps")
        
        # Calculate timestamps for evenly-spaced frames
        timestamps = np.linspace(0, duration - 0.1, num_frames).tolist()
        
        # Create temp directory for extracted frames
        temp_dir = tempfile.mkdtemp(prefix="qwen_frames_")
        images = []
        
        try:
            for i, ts in enumerate(timestamps):
                frame_path = Path(temp_dir) / f"frame_{i:03d}.png"
                
                # Extract single frame at timestamp using FFmpeg
                extract_cmd = [
                    'ffmpeg', '-y', '-ss', str(ts), '-i', video_path,
                    '-frames:v', '1', '-q:v', '2',
                    '-vf', f'scale={resize[0]}:{resize[1]}',
                    str(frame_path)
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, check=False)
                
                if frame_path.exists():
                    pil_image = Image.open(frame_path).convert('RGB')
                    images.append(pil_image.copy())  # Copy to detach from file
                    pil_image.close()
                else:
                    print(f"[Qwen3-VL] Warning: Failed to extract frame at {ts:.1f}s")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not images:
            raise RuntimeError(f"[Qwen3-VL] Failed to extract any frames from {video_path}")
             
        print(f"[Qwen3-VL] Extracted {len(images)} frames from {duration:.1f}s video")
        return images, duration, timestamps
    
    def analyze_video(
        self,
        video_path: str,
        transcript_json: Optional[str] = None,
        num_clips: int = 4,
        num_frames: int = 16
    ) -> VideoAnalysis:
        """
        Analyze video to find viral moments.
        
        Args:
            video_path: Path to video file
            transcript_json: Optional path to transcript JSON
            num_clips: Number of viral clips to identify (will be adjusted for short videos)
            num_frames: Number of frames to sample
            
        Returns:
            VideoAnalysis with identified moments
        """
        self._ensure_loaded()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"[Qwen3-VL] Analyzing: {video_path.name}")
        
        # Extract frames
        images, duration, timestamps = self._extract_frames(
            str(video_path), 
            num_frames=num_frames
        )
        
        # Adjust num_clips for short videos
        # If video is short, asking for 4 clips forces hallucinations of tiny segments
        min_clip_duration = 45.0
        max_possible_clips = max(1, int(duration / min_clip_duration))
        if num_clips > max_possible_clips:
            print(f"[Qwen3-VL] Duration {duration:.1f}s too short for {num_clips} clips. Adjusting to {max_possible_clips}.")
            num_clips = max_possible_clips
        
        # Load transcript if available
        transcript_text = ""
        if transcript_json:
            try:
                with open(transcript_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    parts = [
                        f"[{s['start']:.1f}s] {s['text']}" 
                        for s in data.get('segments', [])[:30]
                    ]
                    transcript_text = "\n".join(parts)
            except Exception as e:
                print(f"[Qwen3-VL] Warning: Could not load transcript: {e}")
        
        # Build frame descriptions for prompt
        frame_descriptions = ", ".join([f"frame at {t:.1f}s" for t in timestamps])
        
        # Chain-of-Thought prompt for better analysis (based on research)
        prompt = f"""You are a professional TikTok/Reels video editor. Analyze these frames and transcript to identify the {num_clips} most viral segments.

CRITICAL INSTRUCTIONS:
1. THINK FIRST: Briefly list the key events and their timestamps.
2. SELECT SEGMENTS: Choose exactly {num_clips} non-overlapping segments.
3. STRICT FORMAT: Output the result in a valid JSON block inside ```json ... ``` tags.

CONSTRAINTS:
- Each segment MUST be 45-90 seconds (Target: 60s).
- Viral Score: 0.0 to 1.0 (float).
- The "hook" must be specific to the start of the clip.

Output format:
```json
{{
    "analysis_summary": "1-sentence summary",
    "viral_moments": [
        {{
            "start_time": 10.5,
            "end_time": 65.0,
            "hook": "Specific hook text...",
            "reason": "Why this is viral...",
            "viral_score": 0.95
        }}
    ],
    "overall_theme": "Theme...",
    "target_audience": "Audience..."
}}
```"""

        # Build messages for Qwen VL
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Process with Qwen
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = self._process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        print("[Qwen3-VL] Running inference...")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[Qwen3-VL] Response received ({len(response)} chars)")
        
        # Robust JSON extraction (handles CoT reasoning text before JSON)
        result = self._extract_json_from_response(response)
        
        if result is None:
            print(f"[Qwen3-VL] JSON parse failed. Response preview: {response[:500]}...")
            # Fallback
            fallback_duration = min(60.0, duration)
            result = {
                "viral_moments": [{
                    "start_time": 0.0,
                    "end_time": fallback_duration,
                    "hook": "Opening segment",
                    "reason": "Analysis parsing failed - using fallback",
                    "viral_score": 0.5
                }],
                "overall_theme": "Unknown",
                "target_audience": "General"
            }
        else:
            print(f"[Qwen3-VL] Successfully parsed {len(result.get('viral_moments', []))} moments")
        
        # Build analysis object
        moments = []
        for m in result.get("viral_moments", []):
            moments.append(ViralMoment(
                start_time=float(m.get("start_time", 0)),
                end_time=float(m.get("end_time", 0)),
                hook=str(m.get("hook", "")),
                reason=str(m.get("reason", "")),
                viral_score=float(m.get("viral_score", 0))
            ))
        
        return VideoAnalysis(
            video_path=str(video_path),
            duration=duration,
            viral_moments=moments,
            overall_theme=result.get("overall_theme", "Unknown"),
            target_audience=result.get("target_audience", "General")
        )
    
    @staticmethod
    def _extract_json_from_response(text: str) -> Optional[dict]:
        """Extract JSON from VLM response using multiple strategies."""
        import re
        
        # Strategy 1: Find JSON in code blocks
        json_block_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        ]
        
        for pattern in json_block_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Find JSON object with "viral_moments" key
        json_pattern = r'\{[^{}]*"viral_moments"\s*:\s*\[[\s\S]*?\]\s*[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Find any complete JSON object (outermost braces)
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        start_idx = -1
                        continue
        
        return None
    
    def unload(self):
        """Unload model to free VRAM"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            print("[Qwen3-VL] Model unloaded, VRAM freed")
    
    def save_analysis(self, analysis: VideoAnalysis, output_path: str):
        """Save analysis to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)
        
        print(f"[Qwen3-VL] Saved analysis to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python qwen3_video_analyzer.py <video_path> [transcript.json]")
        sys.exit(1)
    
    video = sys.argv[1]
    transcript = sys.argv[2] if len(sys.argv) > 2 else None
    output = f"{Path(video).stem}_qwen3_analysis.json"
    
    analyzer = Qwen3VideoAnalyzer()
    
    try:
        analysis = analyzer.analyze_video(video, transcript)
        analyzer.save_analysis(analysis, output)
        
        print(f"\n✅ Analysis complete!")
        print(f"Theme: {analysis.overall_theme}")
        print(f"Target Audience: {analysis.target_audience}")
        print(f"Viral Moments Found: {len(analysis.viral_moments)}")
        
        for i, moment in enumerate(analysis.viral_moments, 1):
            print(f"  {i}. [{moment.start_time:.1f}s - {moment.end_time:.1f}s] Score: {moment.viral_score:.2f}")
            print(f"     Hook: {moment.hook}")
    finally:
        analyzer.unload()
