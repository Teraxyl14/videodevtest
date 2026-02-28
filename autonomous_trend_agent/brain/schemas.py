# brain/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class VideoClip(BaseModel):
    start_time: float
    end_time: float
    transcript_text: str
    relevance_score: float

class EditDecision(BaseModel):
    clips: List[VideoClip]
    hook_text: str = Field(description="Viral hook text to display in first 3 seconds")
    visual_style: str = Field(description="Style of simple/minimalist/hype")
    music_vibe: str = Field(description="Type of background music")

class AgentState(BaseModel):
    video_path: str
    transcript: str
    current_plan: Optional[EditDecision] = None
    critique_feedback: Optional[str] = None
    iteration_count: int = 0
    is_approved: bool = False
