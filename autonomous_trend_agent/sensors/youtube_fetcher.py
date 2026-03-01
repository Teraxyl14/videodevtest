"""
YouTube Data API v3 — Trend Fetcher (Phase 1, B1.1)

Fetches trending videos and computes Velocity and Acceleration
(rate of change in views/likes) to surface rising topics.

Requires:
    YOUTUBE_API_KEY in environment

Objectives: B1.1 (YouTube velocity/acceleration tracking)
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


@dataclass
class YouTubeTrend:
    """A trending topic detected from YouTube."""
    video_id: str
    title: str
    channel: str
    topic: str                      # Inferred topic / keywords
    view_count: int
    like_count: int
    comment_count: int
    published_at: str               # ISO 8601
    region_code: str
    category_id: str
    tags: List[str] = field(default_factory=list)
    thumbnail_url: str = ""

    # Velocity / Acceleration (computed)
    views_per_hour: float = 0.0     # B1.1: Velocity
    acceleration: float = 0.0      # B1.1: Acceleration (delta velocity)
    virality_score: float = 0.0     # Raw score before Z-normalization


class YouTubeFetcher:
    """
    Fetches trending YouTube videos and computes velocity metrics.

    Usage:
        fetcher = YouTubeFetcher()
        trends = fetcher.fetch_trending(region_codes=["US", "IN"], max_results=50)
    """

    # YouTube categories most relevant for viral shorts
    RELEVANT_CATEGORIES = {
        "1":  "Film & Animation",
        "2":  "Autos & Vehicles",
        "10": "Music",
        "15": "Pets & Animals",
        "17": "Sports",
        "20": "Gaming",
        "22": "People & Blogs",
        "23": "Comedy",
        "24": "Entertainment",
        "25": "News & Politics",
        "26": "Howto & Style",
        "27": "Education",
        "28": "Science & Technology",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "YOUTUBE_API_KEY not found. "
                "Set it in your .env file or pass api_key= directly."
            )
        self._service = build("youtube", "v3", developerKey=self.api_key)
        self._prev_stats: Dict[str, Dict] = {}   # For acceleration calculation
        logger.info("[YouTubeFetcher] Initialized.")

    def fetch_trending(
        self,
        region_codes: List[str] = None,
        category_ids: List[str] = None,
        max_results: int = 50,
    ) -> List[YouTubeTrend]:
        """
        Fetches currently trending videos across regions and categories.
        Computes Velocity (views/hr since upload) and Acceleration
        (delta from last poll, if available).

        Args:
            region_codes: ISO 3166-1 codes, e.g. ["US", "IN", "GB"]
            category_ids: YouTube category IDs to filter by
            max_results: Max videos to fetch per region (YouTube max: 50)

        Returns:
            Deduplicated list of YouTubeTrend objects sorted by virality.
        """
        region_codes = region_codes or ["US"]
        all_trends: Dict[str, YouTubeTrend] = {}   # video_id → trend

        for region in region_codes:
            try:
                page_token = None
                fetched = 0

                while fetched < max_results:
                    batch_size = min(50, max_results - fetched)
                    params = {
                        "part": "snippet,statistics,contentDetails",
                        "chart": "mostPopular",
                        "regionCode": region,
                        "maxResults": batch_size,
                        "hl": "en",
                    }
                    if category_ids:
                        params["videoCategoryId"] = category_ids[0]  # API only allows 1
                    if page_token:
                        params["pageToken"] = page_token

                    response = self._service.videos().list(**params).execute()

                    for item in response.get("items", []):
                        trend = self._parse_video(item, region)
                        if trend and trend.video_id not in all_trends:
                            all_trends[trend.video_id] = trend

                    page_token = response.get("nextPageToken")
                    fetched += len(response.get("items", []))
                    if not page_token:
                        break

            except HttpError as e:
                logger.error(f"[YouTubeFetcher] API error for region {region}: {e}")
                continue
            except Exception as e:
                logger.error(f"[YouTubeFetcher] Unexpected error: {e}")
                continue

        # Compute velocity / acceleration, then virality raw score
        results = list(all_trends.values())
        for t in results:
            self._compute_velocity(t)

        # Sort by virality (highest first)
        results.sort(key=lambda x: x.virality_score, reverse=True)

        # Save stats snapshot for acceleration calc next poll
        self._save_snapshot(results)

        logger.info(f"[YouTubeFetcher] Found {len(results)} trending videos.")
        return results

    def _parse_video(self, item: dict, region_code: str) -> Optional[YouTubeTrend]:
        """Parse a raw API video item into a YouTubeTrend."""
        try:
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            video_id = item["id"]

            published_at = snippet.get("publishedAt", "")
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            tags = snippet.get("tags", [])

            # Thumbnail — prefer maxres, fallback to high
            thumbs = snippet.get("thumbnails", {})
            thumbnail = (
                thumbs.get("maxres", thumbs.get("high", thumbs.get("default", {}))).get("url", "")
            )

            # Infer topic from title + tags (first 3 tags + first 5 title words)
            title_words = snippet.get("title", "").split()[:5]
            topic_keywords = (tags[:3] + title_words)[:6]
            topic = " ".join(topic_keywords)

            return YouTubeTrend(
                video_id=video_id,
                title=snippet.get("title", ""),
                channel=snippet.get("channelTitle", ""),
                topic=topic,
                view_count=views,
                like_count=likes,
                comment_count=comments,
                published_at=published_at,
                region_code=region_code,
                category_id=snippet.get("categoryId", ""),
                tags=tags[:10],
                thumbnail_url=thumbnail,
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"[YouTubeFetcher] Skipping malformed video item: {e}")
            return None

    def _compute_velocity(self, trend: YouTubeTrend):
        """
        Compute Velocity (views/hr since upload) and Acceleration
        (delta velocity vs last snapshot).

        B1.1: velocity = views / hours_since_publish
        B1.1: acceleration = (current_velocity - prev_velocity)
        """
        try:
            pub = datetime.fromisoformat(trend.published_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_alive = max(0.1, (now - pub).total_seconds() / 3600)
            trend.views_per_hour = trend.view_count / hours_alive
        except Exception:
            trend.views_per_hour = 0.0

        # Acceleration from previous snapshot
        prev = self._prev_stats.get(trend.video_id)
        if prev:
            prev_vph = prev.get("views_per_hour", 0.0)
            trend.acceleration = trend.views_per_hour - prev_vph
        else:
            trend.acceleration = 0.0

        # Raw virality: weighted velocity + acceleration + engagement
        engagement = (trend.like_count + trend.comment_count) / max(1, trend.view_count)
        trend.virality_score = (
            trend.views_per_hour * 0.6
            + trend.acceleration * 0.3
            + engagement * 100000 * 0.1
        )

    def _save_snapshot(self, trends: List[YouTubeTrend]):
        """Save current stats for acceleration calculation next poll."""
        for t in trends:
            self._prev_stats[t.video_id] = {
                "view_count": t.view_count,
                "views_per_hour": t.views_per_hour,
                "snapshot_time": time.time(),
            }
