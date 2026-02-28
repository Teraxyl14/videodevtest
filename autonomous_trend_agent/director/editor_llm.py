# director/editor_llm.py
"""
LLM Director / Edit Decision Engine (Modern Stack)
Uses google-genai SDK (v1.0+) to generate intelligent edit decisions.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Modern Stack
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PYDANTIC MODELS (EDL Schema)
# =============================================================================

class Cut(BaseModel):
    """A single cut/edit point"""
    start_time: float = Field(description="Start time of the cut")
    end_time: float = Field(description="End time of the cut")
    cut_type: str = Field(default="hard", description="hard, fade, j-cut, l-cut")
    reason: str = Field(default="", description="Why this cut was made")

class EditDecision(BaseModel):
    """Complete edit decision for a segment"""
    optimal_start: float = Field(description="Best start time for the hook")
    optimal_end: float = Field(description="Best end time for the loop/resolution")
    hook_score: float = Field(ge=0.0, le=1.0, description="Strength of the hook (0-1)")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Does it work standalone (0-1)")
    pacing: str = Field(default="normal", description="fast, normal, slow")
    suggested_cuts: List[Cut] = Field(default_factory=list, description="Internal cuts (silence removal)")
    edit_reason: str = Field(default="", description="Explanation of the edit")

class EDLInternal(BaseModel):
    """Wrapper for internal EDL logic if needed"""
    decisions: List[EditDecision]

# =============================================================================
# LEGACY DATACLASSES (For backward compatibility with pipeline)
# =============================================================================
from dataclasses import dataclass, field, asdict

@dataclass
class LegacyCut:
    start_time: float
    end_time: float
    cut_type: str = "hard"
    reason: str = ""
    confidence: float = 1.0

@dataclass
class LegacyEditDecision:
    source_start: float
    source_end: float
    output_start: float
    output_duration: float
    cuts: List[LegacyCut] = field(default_factory=list)
    pacing: str = "normal"
    hook_score: float = 0.0
    coherence_score: float = 0.0
    reason: str = ""

@dataclass
class EDL:
    video_path: str
    created_at: str
    decisions: List[LegacyEditDecision]
    total_output_duration: float
    segments_count: int
    metadata: Dict = field(default_factory=dict)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

# =============================================================================
# LLM DIRECTOR
# =============================================================================

class LLMDirector:
    """
    AI-powered video editor using Gemini 3 Flash.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("[LLMDirector] Warning: No API key. LLM features disabled.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
                print(f"[LLMDirector] Initialized with {self.model_name}")
            except Exception as e:
                print(f"[LLMDirector] Failed to init Gemini: {e}")
                self.client = None

    def generate_edl(
        self,
        video_path: str,
        transcript: Dict,
        viral_moments: List[Dict],
        num_shorts: int = 4,
        target_duration: Tuple[int, int] = (30, 90)
    ) -> EDL:
        """Generate an Edit Decision List."""
        decisions = []
        output_offset = 0
        
        for i, moment in enumerate(viral_moments[:num_shorts]):
            # Extract text
            segment_transcript = self._extract_segment_text(
                transcript, moment.get("start_time", 0), moment.get("end_time", 0)
            )
            
            # Analyze
            if self.client:
                decision = self._llm_analyze_segment(moment, segment_transcript, target_duration)
            else:
                decision = self._rule_based_analysis(moment, segment_transcript, target_duration)
            
            # Set timeline position
            decision.output_start = output_offset
            output_offset += decision.output_duration
            decisions.append(decision)
        
        return EDL(
            video_path=video_path,
            created_at=datetime.now().isoformat(),
            decisions=decisions,
            total_output_duration=output_offset,
            segments_count=len(decisions),
            metadata={"llm_used": self.client is not None}
        )
    
    def _llm_analyze_segment(
        self,
        moment: Dict,
        transcript_text: str,
        target_duration: Tuple[int, int]
    ) -> LegacyEditDecision:
        """Analyze segment using Gemini 3 Flash with Pydantic Schema."""
        prompt = f"""Analyze this video segment for VIRAL short-form content.

SEGMENT INFO:
Start: {moment.get('start_time', 0):.1f}s
End: {moment.get('end_time', 0):.1f}s
Hook: {moment.get('hook', 'N/A')}

TRANSCRIPT:
{transcript_text[:3000]}

CONSTRAINTS:
1. Result duration: {target_duration[0]}s - {target_duration[1]}s
2. Viral Loop: Ensure end connects to start contextually.
3. Hook: First 3 seconds must be strong.

Provide structured edit decision."""

        try:
            # Modern Structured Output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EditDecision,
                    temperature=0.1
                )
            )
            
            # Parse
            data = EditDecision.model_validate_json(response.text)
            
            # Convert Internal Pydantic -> Legacy Dataclass for Pipeline Compatibility
            legacy_cuts = [
                LegacyCut(start_time=c.start_time, end_time=c.end_time, cut_type=c.cut_type, reason=c.reason)
                for c in data.suggested_cuts
            ]
            
            duration = data.optimal_end - data.optimal_start
            
            return LegacyEditDecision(
                source_start=data.optimal_start,
                source_end=data.optimal_end,
                output_start=0,
                output_duration=duration,
                cuts=legacy_cuts,
                pacing=data.pacing,
                hook_score=data.hook_score,
                coherence_score=data.coherence_score,
                reason=data.edit_reason
            )

        except Exception as e:
            print(f"[LLMDirector] Analysis failed: {e}")
            return self._rule_based_analysis(moment, transcript_text, target_duration)

    def _rule_based_analysis(
        self, moment: Dict, transcript_text: str, target_duration: Tuple[int, int]
    ) -> LegacyEditDecision:
        """Fallback logic."""
        start = moment.get('start_time', 0)
        end = moment.get('end_time', 0)
        duration = end - start
        
        # Clamp
        if duration > target_duration[1]:
            end = start + target_duration[1]
            duration = target_duration[1]
        
        return LegacyEditDecision(
            source_start=start,
            source_end=end,
            output_start=0,
            output_duration=duration,
            cuts=[],
            pacing="normal",
            hook_score=0.5,
            coherence_score=0.5,
            reason="Rule-based fallback"
        )

    def _extract_segment_text(self, transcript: Dict, start: float, end: float) -> str:
        """Extract text from transcript range."""
        words = []
        source_words = []
        
        if "segments" in transcript:
            for seg in transcript["segments"]:
                 # Handling different potential structures
                 if "words" in seg:
                     source_words.extend(seg["words"])
                 else:
                     # If no word level, use segment level roughly
                     if seg.get("start", 0) >= start and seg.get("end", 0) <= end:
                         words.append(seg.get("text", ""))
        elif "words" in transcript:
             source_words = transcript["words"]
             
        for w in source_words:
            t = w.get("start", w.get("start_time", 0))
            if start <= t <= end:
                words.append(w.get("word", w.get("text", "")))
                
        return " ".join(words)

# CLI Stub
if __name__ == "__main__":
    pass
