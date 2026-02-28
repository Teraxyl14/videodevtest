# brain/video_analyzer.py
"""
Gemini Video Analysis Engine (Modern Stack)
Identifies viral moments in long-form content using Gemini 3 Flash Multimodal.
Uses google-genai SDK (v1.0+)
"""
import json
import time
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict

# Modern Stack
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ViralMoment(BaseModel):
    """A potential viral clip identified in the video"""
    hook: str = Field(description="One punchy sentence that grabs attention")
    reason: str = Field(description="Why this moment is viral-worthy")
    viral_score: float = Field(ge=0.0, le=1.0, description="Viral potential score (0-1)")
    segments: List[dict] = Field(
        description="List of {'start': float, 'end': float} defining the cuts to stitch together.",
        default_factory=list
    )
    
    # Backward compatibility properties for single-clip logic fallback
    @property
    def start_time(self) -> float:
        return self.segments[0].get('start', 0.0) if self.segments else 0.0
        
    @property
    def end_time(self) -> float:
        return self.segments[-1].get('end', 0.0) if self.segments else 0.0

class VideoAnalysis(BaseModel):
    """Complete analysis of a video"""
    video_path: str = Field(description="Path to local video file")
    duration: float = Field(description="Video duration in seconds")
    viral_moments: List[ViralMoment] = Field(description="List of identified viral moments")
    overall_theme: str = Field(description="Overall theme of the video")
    target_audience: str = Field(description="Target audience demographics")

# =============================================================================
# ANALYZER
# =============================================================================

class GeminiVideoAnalyzer:
    """Analyzes videos using Gemini 3 Flash Multimodal capabilities via google-genai SDK."""
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL_NAME", model_name)
        print(f"[GeminiVideoAnalyzer] Initialized with {self.model_name}")
        
    def analyze_video(
        self, 
        video_path: str,
        transcript_json: Optional[str] = None,
        num_clips: int = 4
    ) -> VideoAnalysis:
        """
        Analyze video to find viral moments
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"[GeminiVideoAnalyzer] Uploading video: {video_path.name}")
        
        # 1. Upload Video (Modern SDK)
        # Note: 'file' argument is used in new SDK
        video_file = self.client.files.upload(file=str(video_path))
        
        # 2. Wait for processing
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
            
        print(f"\n[GeminiVideoAnalyzer] Video ready. analyzing...")
        
        # 3. Prepare Context
        transcript_text = ""
        if transcript_json:
            try:
                with open(transcript_json, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    # Support various formats
                    segments = tdata.get("segments", [])
                    transcript_text = "\n".join([f"[{s.get('start',0):.1f}s] {s.get('text','')}" for s in segments])
            except Exception as e:
                print(f"[GeminiVideoAnalyzer] Warning: Could not load transcript: {e}")

        # 4. Build Prompt
        prompt = f"""Analyze this video and identify {num_clips} VIRAL MOMENTS perfect for short-form content (45-90s).

TRANSCRIPT CONTEXT:
{transcript_text[:15000]}... (truncated)

REQUIREMENTS:
1. Find moments with strong visual or emotional hooks.
2. Return exactly {num_clips} moments.
3. Timestamps must be accurate.

Provide structured analysis."""

        try:
            # 5. Generate Content with Video + Text
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VideoAnalysis,
                    temperature=0.1
                )
            )
            
            # 6. Parse
            analysis = VideoAnalysis.model_validate_json(response.text)
            
            # Contextualize with local path (model doesn't know local path)
            analysis.video_path = str(video_path)
            
            return analysis
            
        except Exception as e:
            print(f"[GeminiVideoAnalyzer] Analysis failed: {e}")
            return VideoAnalysis(
                video_path=str(video_path),
                duration=0,
                viral_moments=[],
                overall_theme="Error",
                target_audience="Error"
            )
        finally:
            # Cleanup
            try:
                self.client.files.delete(name=video_file.name)
                print("[GeminiVideoAnalyzer] Remote file cleaned up.")
            except:
                pass

    def save_analysis(self, analysis: VideoAnalysis, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"[GeminiVideoAnalyzer] Saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <video_path>")
        sys.exit(1)
    
    analyzer = GeminiVideoAnalyzer()
    res = analyzer.analyze_video(sys.argv[1])
    analyzer.save_analysis(res, f"{Path(sys.argv[1]).stem}_analysis.json")
