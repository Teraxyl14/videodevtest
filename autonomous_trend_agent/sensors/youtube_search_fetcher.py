"""
YouTube Search Fetcher (Phase 1 — Supplementary)

Complements the YouTubeFetcher (trending chart) by querying YouTube's
Search API for videos uploaded in the last 48 hours, sorted by view count.
This surfaces EMERGING topics — content that is gaining velocity before
it appears on the official trending chart.

Uses the same YOUTUBE_API_KEY as youtube_fetcher.py.

Strategy:
    - Queries a set of "signal probe" search terms
    - Filters to recently-published videos (last 24–48 hrs)
    - Computes velocity (views/hr since upload)
    - Deduplicates against the trending chart results
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


# Probe terms that indicate fast-rising, broad-audience content
DEFAULT_PROBE_TERMS = [
    "viral 2026",
    "trending today",
    "breaking news",
    "shocking moment",
    "you won't believe",
    "everyone's talking about",
    "went viral",
    "most watched",
]


@dataclass
class YouTubeSearchTrend:
    """An emerging topic found via YouTube Search API."""
    video_id: str
    title: str
    channel: str
    topic: str                  # Keyword / inferred topic
    view_count: int
    like_count: int
    comment_count: int
    published_at: str
    search_query: str           # Which probe term found this video
    tags: List[str] = field(default_factory=list)

    # Computed
    views_per_hour: float = 0.0
    is_emerging: bool = True    # Always True for Search results (< 48hrs old)


class YouTubeSearchFetcher:
    """
    Uses YouTube Data API v3 Search endpoint to find recently-published
    videos that are already gaining massive view velocity.

    This is fundamentally different from YouTubeFetcher (trending chart):
    - Trending chart = already peaked / well-known
    - Search fetcher = catching the wave as it starts

    Usage:
        fetcher = YouTubeSearchFetcher()
        trends = fetcher.fetch_emerging(hours_back=24)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "YOUTUBE_API_KEY not found. "
                "Set it in your .env file or pass api_key= directly."
            )
        self._service = build("youtube", "v3", developerKey=self.api_key)
        logger.info("[YouTubeSearchFetcher] Initialized.")

    def fetch_emerging(
        self,
        probe_terms: Optional[List[str]] = None,
        hours_back: int = 24,
        max_per_query: int = 10,
        region_code: str = "US",
        min_views: int = 10_000,
    ) -> List[YouTubeSearchTrend]:
        """
        Fetch recent videos gaining view velocity across probe search terms.

        Args:
            probe_terms:    Search queries to probe (defaults to DEFAULT_PROBE_TERMS)
            hours_back:     Only consider videos uploaded in the last N hours
            max_per_query:  Max results per probe term (YouTube Search max: 50)
            region_code:    ISO 3166-1 region (only affects search ranking bias)
            min_views:      Minimum view count to include (filters noise)

        Returns:
            Deduplicated list of YouTubeSearchTrend sorted by views_per_hour desc.
        """
        probe_terms = probe_terms or DEFAULT_PROBE_TERMS
        published_after = (
            datetime.now(timezone.utc) - timedelta(hours=hours_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        seen_ids: Dict[str, YouTubeSearchTrend] = {}

        for term in probe_terms:
            try:
                video_ids = self._search(
                    query=term,
                    published_after=published_after,
                    max_results=max_per_query,
                    region_code=region_code,
                )
                if not video_ids:
                    continue

                # Enrich with statistics
                enriched = self._get_stats(video_ids, search_query=term)
                for t in enriched:
                    if t.video_id not in seen_ids and t.view_count >= min_views:
                        self._compute_velocity(t)
                        seen_ids[t.video_id] = t

                time.sleep(0.5)  # Light pacing between queries

            except HttpError as e:
                logger.error(f"[YouTubeSearchFetcher] API error on '{term}': {e}")
                continue
            except Exception as e:
                logger.error(f"[YouTubeSearchFetcher] Unexpected error: {e}")
                continue

        results = list(seen_ids.values())
        results.sort(key=lambda x: x.views_per_hour, reverse=True)

        logger.info(
            f"[YouTubeSearchFetcher] Found {len(results)} emerging videos "
            f"across {len(probe_terms)} probe terms."
        )
        return results

    def _search(
        self,
        query: str,
        published_after: str,
        max_results: int,
        region_code: str,
    ) -> List[str]:
        """
        Run a YouTube Search and return list of video IDs.
        Uses order=viewCount so highest-velocity videos surface first.
        """
        resp = self._service.search().list(
            part="id",
            q=query,
            type="video",
            order="viewCount",
            publishedAfter=published_after,
            maxResults=max_results,
            regionCode=region_code,
            relevanceLanguage="en",
            safeSearch="none",
            videoDuration="medium",   # 4-20 minutes — good source material
        ).execute()

        return [
            item["id"]["videoId"]
            for item in resp.get("items", [])
            if item.get("id", {}).get("videoId")
        ]

    def _get_stats(self, video_ids: List[str], search_query: str) -> List[YouTubeSearchTrend]:
        """Batch-fetch statistics for a list of video IDs."""
        if not video_ids:
            return []

        try:
            resp = self._service.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids),
            ).execute()
        except HttpError as e:
            logger.error(f"[YouTubeSearchFetcher] Stats fetch error: {e}")
            return []

        trends = []
        for item in resp.get("items", []):
            try:
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                tags = snippet.get("tags", [])

                # Topic: first 3 tags + first 4 title words
                title_words = snippet.get("title", "").split()[:4]
                topic_parts = (tags[:3] + title_words)[:5]
                topic = " ".join(topic_parts)

                trends.append(YouTubeSearchTrend(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    channel=snippet.get("channelTitle", ""),
                    topic=topic,
                    view_count=int(stats.get("viewCount", 0)),
                    like_count=int(stats.get("likeCount", 0)),
                    comment_count=int(stats.get("commentCount", 0)),
                    published_at=snippet.get("publishedAt", ""),
                    search_query=search_query,
                    tags=tags[:8],
                ))
            except (KeyError, ValueError):
                continue

        return trends

    def _compute_velocity(self, trend: YouTubeSearchTrend):
        """Compute views per hour since upload."""
        try:
            pub = datetime.fromisoformat(trend.published_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours = max(0.1, (now - pub).total_seconds() / 3600)
            trend.views_per_hour = trend.view_count / hours
        except Exception:
            trend.views_per_hour = 0.0
