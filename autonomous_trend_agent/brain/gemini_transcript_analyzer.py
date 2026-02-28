# brain/gemini_transcript_analyzer.py
"""
Gemini Flash Transcript Analyzer (Modern Stack)
Uses google-genai SDK (v1.0+) for Gemini 3 Flash compatibility.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

# Modern SDK & Typing
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Models for Structured Output (The Golden Way) ---

class SegmentRange(BaseModel):
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")

class EditingNotes(BaseModel):
    pacing: str = Field(description="fast, normal, or slow", default="normal")
    emphasis_timestamps: List[float] = Field(description="Timestamps for visual emphasis", default_factory=list)
    suggested_caption_style: str = Field(description="tiktok, hormozi, or minimal", default="tiktok")
    notes: str = Field(description="Specific editing instructions", default="")

class EditableSegment(BaseModel):
    title: str = Field(description="Viral hook title")
    segments: List[SegmentRange] = Field(description="Time ranges to stitch together")
    hook: str = Field(description="The opening hook script")
    reason: str = Field(description="Why this is viral")
    viral_score: float = Field(description="0.0 to 1.0 score")
    editing_notes: EditingNotes = Field(default_factory=EditingNotes)

    @property
    def total_duration(self) -> float:
        return sum(seg.end - seg.start for seg in self.segments)
    
    @property
    def start_time(self) -> float:
        return self.segments[0].start if self.segments else 0

    @property
    def end_time(self) -> float:
        return self.segments[-1].end if self.segments else 0

class TranscriptAnalysis(BaseModel):
    shorts: List[EditableSegment] = Field(description="List of identified viral shorts")
    overall_theme: str = Field(description="Main topic of the video")
    target_audience: str = Field(description="Who this content is for")
    content_warnings: List[str] = Field(description="List of potential issues", default_factory=list)


class GeminiTranscriptAnalyzer:
    """
    Analyzes transcripts using Gemini 3 Flash via google-genai SDK.
    """
    
    def __init__(self, model_name: str = None):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        # Modern Client Initialization
        self.client = genai.Client(api_key=api_key)
        
        # Default to Gemini 1.5 Flash if 3 is unstable/404, but User wants 3.
        # We respect env var first.
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        print(f"[GeminiTranscriptAnalyzer] Initialized with model: {self.model_name}")

    @staticmethod
    def verify_api_key(model_name: str = None) -> bool:
        """Verify API connectivity using Modern SDK."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Gemini Check] ❌ No API key found")
            return False
            
        model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents="Say OK."
            )
            print(f"[Gemini Check] ✅ Connected to {model_name}: {response.text.strip()}")
            return True
        except Exception as e:
            print(f"[Gemini Check] ❌ Failed: {e}")
            return False

    def analyze_transcript(
        self,
        transcript: Dict,
        video_duration: float = None,
        num_shorts: int = 4,
        target_duration: tuple = (30, 60)
    ) -> TranscriptAnalysis:
        """
        Analyze transcript and return structured TranscriptAnalysis object.
        """
        transcript_text = self._format_transcript(transcript)
        prompt = self._build_prompt(duration=video_duration or 0, num_shorts=num_shorts, target_duration=target_duration)
        
        print(f"[GeminiTranscriptAnalyzer] Analyzing with {self.model_name}...")
        
        try:
            # Modern SDK Call with Structured Output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, transcript_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TranscriptAnalysis,
                    temperature=0.1
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Parse result
            # The SDK guarantees JSON matching the schema, so we can validate directly
            analysis = TranscriptAnalysis.model_validate_json(response.text)
            print(f"[GeminiTranscriptAnalyzer] Found {len(analysis.shorts)} segments")
            return analysis
            
        except Exception as e:
            print(f"[GeminiTranscriptAnalyzer] Error: {e}")
            # Return empty structure on failure
            return TranscriptAnalysis(shorts=[], overall_theme="Error", target_audience="Error")

    def _format_transcript(self, transcript: Dict) -> str:
        """Format transcript dict into readable text with timestamps"""
        lines = []
        if 'segments' in transcript:
            for seg in transcript['segments']:
                start = seg.get('start', 0)
                text = seg.get('text', '').strip()
                lines.append(f"[{start:.1f}s] {text}")
        elif 'words' in transcript:
            # Simple word consolidation
            words = transcript['words']
            chunk = []
            chunk_start = 0
            for w in words:
                chunk.append(w.get('word', w.get('text', '')))
                if w.get('end', 0) - chunk_start > 10:
                    lines.append(f"[{chunk_start:.1f}s] {' '.join(chunk)}")
                    chunk = []
                    chunk_start = w.get('end', 0)
            if chunk:
                lines.append(f"[{chunk_start:.1f}s] {' '.join(chunk)}")
        return "\n".join(lines)

    def _build_prompt(self, duration: float, num_shorts: int, target_duration: tuple) -> str:
        min_dur, max_dur = target_duration
        return f"""
        You are an expert viral content editor. Analyze the transcript.
        Identify exactly {num_shorts} highly engaging, viral segments.
        
        CRITICAL MULTI-CUT REQUIREMENT:
        Do NOT just provide one long continuous time block. Instead, you MUST piece together each short from **multiple non-contiguous cuts**. 
        For example, if the speaker rambles, provide a segment array like:
        [
          {{"start": 10.5, "end": 15.0}},  // The Hook
          {{"start": 25.0, "end": 37.0}},  // The Core Point
          {{"start": 80.0, "end": 85.0}}   // The Punchline/Conclusion
        ]
        
        STRICT PACING AND CUTTING RULES:
        1. **NO MID-SENTENCE CUTS**: EVERY single cut `end` time MUST fall naturally after a completed sentence or natural pause (period, comma, question mark). NEVER cut someone off while they are still speaking a sentence.
        2. **CONTEXTUAL COMPLETENESS**: Ensure the thought in each clip is fully resolved before moving to the next cut.
        3. **PADDING**: Add ~0.5s to 1.0s of padding to your mental timestamps to ensure the speaker's last word is fully captured before the cut.
        
        Guidelines:
        1. Total stitched duration for each short must be between {min_dur} and {max_dur} seconds. (Aim for >55s).
        2. Identify hooks, pacing, and editing needs for each stitched short.
        3. Output must strictly follow the Schema provided.
        """

    def save_analysis(self, analysis: TranscriptAnalysis, output_path: str):
        """Save analysis to JSON file"""
        # Convert Pydantic model to dict
        data = analysis.model_dump()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[GeminiTranscriptAnalyzer] Saved to {output_path}")

if __name__ == "__main__":
    # Test stub
    pass
