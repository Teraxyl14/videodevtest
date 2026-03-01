"""
Gemini Flash Trend Ranker (Phase 1, B2.1-B2.4)

Uses the Gemini Flash API (via google-genai SDK) to:
  1. Receive a list of ViralCandidates from the Z-Score engine
  2. Rank them by content creation potential (not just raw virality)
  3. Recommend a specific "content angle" for the winning trend
  4. Output structured JSON: topic, keywords, sources, confidence, angle

This is the "brain" that converts raw signal data into an actionable
content brief ready for the Video Sourcing phase.

Objectives: B2.1 (Gemini 3 Flash integration), B2.2 (ranks by viral potential),
            B2.3 (recommends content angle), B2.4 (structured JSON output)
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

from .z_score_virality import ViralCandidate

logger = logging.getLogger(__name__)


@dataclass
class ContentBrief:
    """
    Final output of the Trend Discovery phase.
    Passed directly to Phase 2 (Video Sourcing).
    """
    # B2.4: structured fields
    topic: str
    keywords: List[str]
    content_angle: str          # e.g. "Debunking the myth that X causes Y"
    hook: str                   # First 3-second hook suggestion
    target_audience: str        # e.g. "Tech enthusiasts aged 18-34"
    platform_sources: List[str] # Platforms that detected the trend
    viral_score: float          # 0-100 composite score
    confidence: float           # 0.0-1.0
    is_breakout: bool
    search_query: str           # YouTube search query to find source video
    gemini_notes: str           # Free-form notes from Gemini
    raw_candidates: List[dict] = field(default_factory=list)  # Top-N competitors


class GeminiTrendRanker:
    """
    Uses Gemini Flash to evaluate and rank ViralCandidates and generate
    a ContentBrief for the winning trend.

    B2.1: Uses google-genai SDK (not the REST API directly)
    """

    SYSTEM_PROMPT = """You are an expert viral content strategist and social media analyst.
You are given a list of trending topics with their virality metrics.
Your job is to:
1. Rank them by their potential for a SHORT-FORM VERTICAL VIDEO (TikTok/YouTube Shorts/Reels)
2. Identify the single best topic to create content about RIGHT NOW
3. Suggest a specific content angle that would make this video go viral
4. Generate a YouTube search query to find a high-quality long-form source video

Key criteria for your ranking:
- Emotional resonance (surprising, controversial, heartwarming, educational)
- Broad audience appeal (not too niche)
- Evergreen potential vs pure news
- Exists long-form source content (is someone discussing this in depth on YouTube?)
- Visual/storytelling potential (can be shown, not just narrated)

Output ONLY valid JSON in this exact format:
{
  "winner_topic": "<topic>",
  "winner_keywords": ["<kw1>", "<kw2>", "<kw3>"],
  "content_angle": "<specific angle/hook for the short>",
  "hook": "<first 3-second hook sentence>",
  "target_audience": "<audience description>",
  "search_query": "<YouTube search query to find source video>",
  "confidence": <0.0-1.0>,
  "notes": "<brief explanation of your choice>"
}"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Set it in your .env file or pass api_key= directly."
            )

        self._client = genai.Client(api_key=self.api_key)
        logger.info(f"[GeminiTrendRanker] Initialized with model: {self.model_name}")

    def rank_and_brief(
        self,
        candidates: List[ViralCandidate],
        top_n: int = 10,
    ) -> ContentBrief:
        """
        Rank top_n candidates using Gemini Flash and return a ContentBrief.

        Args:
            candidates:  Sorted list of ViralCandidates from Z-Score engine
            top_n:       How many to send to Gemini (avoid exceeding context)

        Returns:
            ContentBrief with the winning topic and a content strategy
        """
        if not candidates:
            raise ValueError("No candidates to rank — run trend discovery first.")

        top = candidates[:top_n]
        prompt = self._build_prompt(top)

        logger.info(f"[GeminiTrendRanker] Sending {len(top)} candidates to Gemini...")

        raw_text = ""
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text.strip()
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"[GeminiTrendRanker] JSON parse failed: {e}\nRaw: {raw_text[:500]}")
            return self._fallback_brief(candidates[0])
        except Exception as e:
            logger.error(f"[GeminiTrendRanker] Gemini API error: {e}")
            return self._fallback_brief(candidates[0])

        # Find the winning candidate by topic name
        winner_topic = data.get("winner_topic", candidates[0].topic)
        winner = next(
            (c for c in top if c.topic.lower() in winner_topic.lower()
             or winner_topic.lower() in c.topic.lower()),
            candidates[0]
        )

        brief = ContentBrief(
            topic=winner_topic,
            keywords=data.get("winner_keywords", winner.keywords[:5]),
            content_angle=data.get("content_angle", ""),
            hook=data.get("hook", ""),
            target_audience=data.get("target_audience", "General audience"),
            platform_sources=[s.platform for s in winner.signals],
            viral_score=winner.viral_score,
            confidence=data.get("confidence", winner.confidence),
            is_breakout=winner.is_breakout,
            search_query=data.get("search_query", winner_topic + " explained"),
            gemini_notes=data.get("notes", ""),
            raw_candidates=[{
                "topic": c.topic,
                "viral_score": c.viral_score,
                "platforms": c.platform_count,
            } for c in top],
        )

        logger.info(
            f"[GeminiTrendRanker] Winner: '{brief.topic}' "
            f"(score={brief.viral_score}, confidence={brief.confidence})"
        )
        return brief

    def _build_prompt(self, candidates: List[ViralCandidate]) -> str:
        """Format candidates as a readable prompt for Gemini."""
        lines = [
            "Here are the current trending topics detected across YouTube, Twitter/X, "
            "TikTok, and Google Trends. Each has been scored using a Z-Score virality "
            "algorithm (0-100, higher = more viral):\n"
        ]
        for i, c in enumerate(candidates, 1):
            platforms = ", ".join({s.platform for s in c.signals})
            lines.append(
                f"{i}. Topic: '{c.topic}'\n"
                f"   Viral Score: {c.viral_score}/100 | "
                f"Platforms: {platforms} | "
                f"Breakout: {'YES' if c.is_breakout else 'no'} | "
                f"Estimated duration: {c.estimated_duration}\n"
                f"   Keywords: {', '.join(c.keywords[:5])}\n"
            )
        lines.append(
            "\nBased on this data, select the BEST topic for a viral short-form video "
            "and provide your strategic recommendation in the specified JSON format."
        )
        return "\n".join(lines)

    @staticmethod
    def _fallback_brief(winner: ViralCandidate) -> ContentBrief:
        """Return a minimal brief when Gemini fails, using the top Z-Score candidate."""
        return ContentBrief(
            topic=winner.topic,
            keywords=winner.keywords[:5],
            content_angle=f"Breaking down why '{winner.topic}' is trending right now",
            hook=f"Has anyone noticed what's happening with {winner.topic}?",
            target_audience="General social media audience",
            platform_sources=[s.platform for s in winner.signals],
            viral_score=winner.viral_score,
            confidence=winner.confidence * 0.7,   # Lower confidence since Gemini failed
            is_breakout=winner.is_breakout,
            search_query=f"{winner.topic} explained full video",
            gemini_notes="Gemini ranking unavailable — using top Z-Score candidate.",
        )
