"""
PydanticAI Agents (v2.1 — Orchestration Layer)

Typed AI agents using PydanticAI to enforce structured outputs from
Gemini 3 Pro. Replaces the old gemini_trend_ranker.py, gemini_services.py,
gemini_transcript_analyzer.py, and gemini_context_cache.py.

Each agent returns a Pydantic BaseModel — no more JSON parsing/validation.

Objectives: B2.1-B2.3 (trend scoring), D4.1 (segment identification),
            F2.1-F2.4 (metadata generation)
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ─── Pydantic Output Models ─────────────────────────────────────────────────

class TrendScore(BaseModel):
    """NPU/API scoring result for a single trend."""
    score: int = Field(ge=0, le=100, description="Viral potential 0-100")
    reason: str = Field(description="Brief explanation of the score")


class ContentBrief(BaseModel):
    """
    Final output of the Trend Discovery phase (B2.3).
    Replaces the old dataclass-based ContentBrief.
    """
    topic: str
    keywords: List[str] = Field(max_length=8)
    content_angle: str
    hook: str                         # First 3-second hook suggestion
    target_audience: str
    search_query: str                 # YouTube search query for Phase 2
    viral_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    is_breakout: bool = False
    platform_sources: List[str] = Field(default_factory=list)
    notes: str = ""                   # Free-form notes from the AI
    raw_candidates: List[dict] = Field(default_factory=list)


class VideoManifest(BaseModel):
    """
    Editing plan for a single short — output of the Director node (H1.1).
    """
    start_time: float = Field(description="Start timestamp in seconds")
    end_time: float = Field(description="End timestamp in seconds")
    hook_text: str = Field(description="Caption for the hook (first 3s)")
    segment_title: str = Field(description="Short descriptive title")
    viral_score: float = Field(ge=0, le=100)
    reasoning: str = Field(description="Why this segment was selected")


class DirectorPlan(BaseModel):
    """
    Complete editing plan output by the Director node (D4.1).
    Contains 3-4 segments to extract from the source video.
    """
    source_summary: str
    segments: List[VideoManifest] = Field(min_length=1, max_length=6)
    total_segments: int


class VideoMetadata(BaseModel):
    """
    Metadata for a finished short — output of the Export phase (F2.1-F2.4).
    """
    title: str = Field(max_length=100)
    description: str = Field(max_length=5000)
    tags: List[str] = Field(max_length=30)
    hashtags: List[str] = Field(max_length=10)
    thumbnail_prompt: str = Field(description="Prompt for generating thumbnail")


class QAResult(BaseModel):
    """Quality assurance check result — output of the Compliance node (G2)."""
    passes: bool
    hook_present: bool
    coherent: bool
    natural_ending: bool
    captions_accurate: bool
    issues: List[str] = Field(default_factory=list)
    overall_score: float = Field(ge=0, le=100)


# ─── Agent Definitions (lazy-loaded to avoid requiring API keys at import) ────

_agents = {}


def _get_model_name() -> str:
    """Get the Gemini model name from environment."""
    return os.getenv("GEMINI_MODEL_NAME", "google-gla:gemini-3-pro")


def _get_agent(name: str, result_type, system_prompt: str) -> "Agent":
    """Lazy-load an agent — only instantiated on first use."""
    if name not in _agents:
        _agents[name] = Agent(
            _get_model_name(),
            result_type=result_type,
            system_prompt=system_prompt,
        )
    return _agents[name]


def get_trend_ranker():
    return _get_agent(
        "trend_ranker",
        ContentBrief,
        """You are an expert viral content strategist and social media analyst.
You are given a list of trending topics with their virality metrics.
Your job is to:
1. Rank them by their potential for a SHORT-FORM VERTICAL VIDEO (TikTok/YouTube Shorts/Reels)
2. Identify the single best topic to create content about RIGHT NOW
3. Suggest a specific content angle that would make this video go viral
4. Generate a YouTube search query to find a high-quality long-form source video

Key criteria:
- Emotional resonance (surprising, controversial, heartwarming, educational)
- Broad audience appeal (not too niche)
- Exists long-form source content on YouTube
- Visual/storytelling potential
""",
    )


def get_director():
    return _get_agent(
        "director",
        DirectorPlan,
        """You are an expert video editor and content director.
Given a transcript with timestamps, identify 3-4 segments that would make
viral short-form videos (30-90 seconds each).

Each segment must:
1. Have a strong hook in the first 3 seconds
2. Tell a complete, self-contained story
3. End naturally (not mid-sentence)
4. Have high emotional or informational impact

Return precise start/end timestamps from the transcript.
""",
    )


def get_metadata_generator():
    return _get_agent(
        "metadata_generator",
        VideoMetadata,
        """You are an expert social media content optimizer.
Generate SEO-optimized metadata for a short-form video.
- Title: catchy, under 100 chars, uses trending keywords
- Description: includes relevant keywords naturally
- Tags: 10-30 relevant tags for discoverability
- Hashtags: 5-10 trending/relevant hashtags
- Thumbnail prompt: describe an eye-catching thumbnail image
""",
    )


def get_compliance_checker():
    return _get_agent(
        "compliance_checker",
        QAResult,
        """You are a content quality reviewer.
Review the video transcript and metadata for:
1. Hook present in first 3 seconds (attention-grabbing opening)
2. Content coherence (makes sense as a standalone clip)
3. Natural ending (no abrupt cutoff mid-sentence)
4. Caption accuracy (captions match the speech)
Provide an overall quality score 0-100.
""",
    )


# ─── Convenience Functions ───────────────────────────────────────────────────

async def rank_trends(candidates_text: str) -> ContentBrief:
    """Run the trend ranker agent on formatted candidate text."""
    result = await get_trend_ranker().run(candidates_text)
    return result.data


async def plan_cuts(transcript_text: str) -> DirectorPlan:
    """Run the director agent to plan video cuts."""
    result = await get_director().run(
        f"Plan the cuts for this transcript:\n\n{transcript_text}"
    )
    return result.data


async def generate_metadata(segment_info: str) -> VideoMetadata:
    """Generate SEO metadata for a finished short."""
    result = await get_metadata_generator().run(
        f"Generate metadata for this video segment:\n\n{segment_info}"
    )
    return result.data


async def check_quality(transcript_and_meta: str) -> QAResult:
    """Run QA checks on a finished segment."""
    result = await get_compliance_checker().run(
        f"Review this content for quality:\n\n{transcript_and_meta}"
    )
    return result.data

