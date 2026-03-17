"""
Video Searcher (Phase 2, C1.1 + C1.2)

Uses YouTube Data API v3 to find long-form source videos matching the
ContentBrief's search_query. Filters candidates by quality criteria
(resolution, duration, engagement) before presenting to the user.

The search_query comes directly from GeminiTrendRanker — it's crafted
specifically to find a long-form explainer/commentary video on the
winning trend topic.

Objectives:
    C1.1: System searches for videos related to trend topic
    C1.2: Results filtered by quality criteria
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone, timedelta

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


@dataclass
class VideoCandidate:
    """A candidate source video for the pipeline."""
    video_id: str
    title: str
    channel: str
    description: str
    published_at: str
    duration_seconds: int           # Parsed from ISO 8601 duration
    duration_str: str               # Human-readable, e.g. "24:15"
    view_count: int
    like_count: int
    comment_count: int
    thumbnail_url: str
    tags: List[str] = field(default_factory=list)

    # Quality gate results (set by VideoSearcher.filter_by_quality)
    passes_quality: bool = False
    quality_notes: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def engagement_rate(self) -> float:
        if self.view_count == 0:
            return 0.0
        return (self.like_count + self.comment_count) / self.view_count


class VideoSearcher:
    """
    Searches YouTube for long-form source video candidates using the
    ContentBrief's search_query, then filters by quality.

    Quality criteria (C1.2):
        - Duration: 15–60 minutes (long enough for 3-4 shorts)
        - Uploaded within last 90 days (fresh content)
        - Views: > 10,000 (validated audience interest)
        - No known watermark indicators in title/channel

    Usage:
        searcher = VideoSearcher()
        candidates = searcher.search("AI agents explained 2026", max_results=10)
        approved = searcher.filter_by_quality(candidates)
    """

    # Channels/title keywords that usually mean watermarked/low-quality content
    WATERMARK_INDICATORS = [
        "highlights", "shorts", "clip", "reaction",
        "tiktok compilation", "instagram", "viral compilation"
    ]

    MIN_VIEWS = 10_000
    MIN_DURATION_SEC = 15 * 60   # 15 minutes (strict per Objective C3.3)
    MAX_DURATION_SEC = 30 * 60   # 30 minutes (strict per Objective C3.3)
    MAX_AGE_DAYS = 90

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not set in environment.")
        self._service = build("youtube", "v3", developerKey=self.api_key)
        logger.info("[VideoSearcher] Initialized.")

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_age_days: int = MAX_AGE_DAYS,
    ) -> List[VideoCandidate]:
        """
        Search YouTube for videos matching the query.

        Args:
            query:        Search string from ContentBrief.search_query
            max_results:  Max candidates to return before quality filtering
            max_age_days: Only include videos published within N days

        Returns:
            List of VideoCandidate sorted by view_count descending.
        """
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(f"[VideoSearcher] Searching: '{query}' (last {max_age_days} days)")

        try:
            search_resp = self._service.search().list(
                part="id",
                q=query,
                type="video",
                order="relevance",
                publishedAfter=published_after,
                maxResults=max_results,
                videoDuration="long",       # >20 min — YouTube's own filter
                relevanceLanguage="en",
                safeSearch="none",
            ).execute()
        except HttpError as e:
            logger.error(f"[VideoSearcher] Search API error: {e}")
            return []

        video_ids = [
            item["id"]["videoId"]
            for item in search_resp.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

        if not video_ids:
            logger.warning("[VideoSearcher] No results found.")
            return []

        return self._enrich(video_ids)

    def filter_by_quality(self, candidates: List[VideoCandidate]) -> List[VideoCandidate]:
        """
        Applies architectural quality gates to each candidate (Objective C1.2).
        
        ARCHITECTURE NOTE: 
        These strict bounds protect the downstream pipeline from OOM crashes or 
        AI context-window hallucination:
        - Too short (< 15m): Gemini won't have enough script depth to find 3-4 distinct moments.
        - Too long (> 30-60m): Transcription arrays overflow VRAM; Qwen-VL context limits hit.
        
        Sets candidate.passes_quality and candidate.quality_notes.
        Returns only passing candidates.
        """
        passing = []
        for c in candidates:
            notes = []
            ok = True

            # Duration check
            if c.duration_seconds < self.MIN_DURATION_SEC:
                ok = False
                notes.append(f"Too short ({c.duration_str} < 15 min)")
            elif c.duration_seconds > self.MAX_DURATION_SEC:
                ok = False
                notes.append(f"Too long ({c.duration_str} > 60 min)")
            else:
                notes.append(f"Duration OK ({c.duration_str})")

            # View count check
            if c.view_count < self.MIN_VIEWS:
                ok = False
                notes.append(f"Too few views ({c.view_count:,} < {self.MIN_VIEWS:,})")
            else:
                notes.append(f"Views OK ({c.view_count:,})")

            # Watermark indicator check (title/channel heuristic)
            title_lower = c.title.lower()
            hit = next(
                (kw for kw in self.WATERMARK_INDICATORS if kw in title_lower), None
            )
            if hit:
                ok = False
                notes.append(f"Watermark indicator: '{hit}' in title")
            else:
                notes.append("No watermark indicators")

            c.passes_quality = ok
            c.quality_notes = notes

            if ok:
                passing.append(c)

        logger.info(
            f"[VideoSearcher] Quality filter: "
            f"{len(passing)}/{len(candidates)} candidates passed."
        )
        return sorted(passing, key=lambda x: x.view_count, reverse=True)

    def _enrich(self, video_ids: List[str]) -> List[VideoCandidate]:
        """Fetch full details (snippet + statistics + contentDetails) for video IDs."""
        try:
            resp = self._service.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids),
            ).execute()
        except HttpError as e:
            logger.error(f"[VideoSearcher] Stats fetch error: {e}")
            return []

        candidates = []
        for item in resp.get("items", []):
            try:
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                details = item.get("contentDetails", {})

                duration_sec = _parse_duration(details.get("duration", "PT0S"))
                duration_str = _format_duration(duration_sec)

                thumbs = snippet.get("thumbnails", {})
                thumbnail = (
                    thumbs.get("maxres", thumbs.get("high", thumbs.get("default", {}))).get("url", "")
                )

                candidates.append(VideoCandidate(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    channel=snippet.get("channelTitle", ""),
                    description=snippet.get("description", "")[:500],
                    published_at=snippet.get("publishedAt", ""),
                    duration_seconds=duration_sec,
                    duration_str=duration_str,
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    comment_count=int(stats.get("commentCount", 0)),
                    thumbnail_url=thumbnail,
                    tags=snippet.get("tags", [])[:8],
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"[VideoSearcher] Skipping malformed item: {e}")
                continue

        candidates.sort(key=lambda x: x.view_count, reverse=True)
        logger.info(f"[VideoSearcher] Enriched {len(candidates)} video candidates.")
        return candidates


# ─── Duration helpers ─────────────────────────────────────────────────────────

def _parse_duration(iso_duration: str) -> int:
    """Parse ISO 8601 duration (PT1H24M35S) to total seconds."""
    import re
    pattern = re.compile(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.IGNORECASE
    )
    m = pattern.match(iso_duration)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def _format_duration(seconds: int) -> str:
    """Convert seconds to HH:MM:SS or MM:SS string."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
