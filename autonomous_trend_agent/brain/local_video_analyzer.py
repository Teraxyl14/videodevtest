# brain/local_video_analyzer.py
"""
Local Video Analysis using Ollama (Qwen2-VL / Qwen3-VL)
Runs locally via Ollama API
"""
import json
import cv2
import ollama
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import io

@dataclass
class ViralMoment:
    """A potential viral clip identified in the video"""
    start_time: float
    end_time: float
    hook: str
    reason: str
    viral_score: float  # 0-1

@dataclass
class VideoAnalysis:
    """Complete analysis of a video"""
    video_path: str
    duration: float
    viral_moments: List[ViralMoment]
    overall_theme: str
    target_audience: str

class LocalVideoAnalyzer:
    """Local video analysis using Ollama Vision Models"""
    
    def __init__(self, model_name: str = "llama3.2-vision"):
        """
        Initialize local model client
        Args:
            model_name: Ollama model tag (e.g., 'qwen2-vl', 'llava', 'qwen3vl-local')
        """
        self.model_name = model_name
        print(f"Initialized LocalVideoAnalyzer with model: {model_name}")
        self._check_ollama()

    def _check_ollama(self):
        print(f"Checking status for model: {self.model_name}...")
        try:
            models = ollama.list()
            # Normalize model names to handle tags
            available_models = [m['name'] for m in models.get('models', [])]
            print(f"Available models: {available_models}")
            
            # Check if our model exists in the list (fuzzy match)
            found = False
            for m in available_models:
                if self.model_name in m:
                    found = True
                    break
            
            if not found:
                print(f"⚠️ Warning: {self.model_name} not found in Ollama library. Pulling...")
                ollama.pull(self.model_name)
                print("✅ Model pulled successfully")
            else:
                print(f"✅ {self.model_name} is ready")
                
        except Exception as e:
            print(f"❌ Ollama connection check failed: {e}")
            print("Ensure Ollama is running (Start Menu -> Ollama)")

    def _extract_frames(self, video_path: str, num_frames: int = 16) -> List[bytes]:
        """Extract frames and convert to JPEG bytes for Ollama"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames_bytes = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Encode to JPEG bytes
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frames_bytes.append(buffer.tobytes())
        
        cap.release()
        return frames_bytes, duration

    def analyze_video(
        self,
        video_path: str,
        transcript_json: Optional[str] = None,
        num_clips: int = 4
    ) -> VideoAnalysis:
        """
        Analyze video to find viral moments using local vision model
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"Analyzing video locally: {video_path.name}")
        
        # Extract frames
        frames_bytes, duration = self._extract_frames(str(video_path), num_frames=16)
        
        # Load transcript
        transcript_text = ""
        if transcript_json:
            try:
                with open(transcript_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract first 50 segments for context
                    parts = [f"[{s['start']:.1f}s] {s['text']}" for s in data.get('segments', [])[:50]]
                    transcript_text = "\n".join(parts)
            except Exception as e:
                print(f"Warning: Could not load transcript: {e}")

        # Build Prompt
        prompt = f"""
        Analyze this video frames and identify {num_clips} VIRAL MOMENTS (15-60s).
        
        Context (Transcript):
        {transcript_text[:1000]}...

        Return ONLY valid JSON format:
        {{
            "viral_moments": [
                {{
                    "start_time": 0.0,
                    "end_time": 15.0,
                    "hook": "Description of hook",
                    "reason": "Why it's viral",
                    "viral_score": 0.9
                }}
            ],
            "overall_theme": "Theme description",
            "target_audience": "Audience"
        }}
        """

        print(f"Sending to Ollama ({self.model_name})...")
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': frames_bytes
                }],
                format='json',  # Force JSON mode if model supports it
                options={'temperature': 0.7}
            )
            
            content = response['message']['content']
            print("Response received")
            
            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON block
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    raise Exception("No JSON found in response")

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            # Fallback
            result = {
                "viral_moments": [{
                    "start_time": 0.0, "end_time": min(30.0, duration),
                    "hook": "Fallback clip", "reason": "Analysis failed", "viral_score": 0.5
                }],
                "overall_theme": "Unknown",
                "target_audience": "General"
            }

        # Validate and build object
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

    def save_analysis(self, analysis: VideoAnalysis, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python local_video_analyzer.py <video> [transcript]")
        sys.exit(1)
    
    analyzer = LocalVideoAnalyzer(model_name="llama3.2-vision") # Default to Llama 3.2 Vision
    res = analyzer.analyze_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    analyzer.save_analysis(res, f"{Path(sys.argv[1]).stem}_analysis.json")
