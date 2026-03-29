"""
Trend Discovery Orchestrator v2.1 (Phase 1 — Section B)

Coordinates Crawl4AI discovery, YouTube Data API validation, Google Trends
confirmation, and PydanticAI-based ranking via Gemini 3 Pro.

Replaces: old trend_discovery.py (custom scrapers + Z-Score + Gemini Flash)

Single method: TrendDiscovery.run() → ContentBrief

Objectives: B1.1–B1.4, B2.1–B2.3
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .crawl4ai_discovery import Crawl4AIDiscovery, CrawledTrend
from .youtube_fetcher import YouTubeFetcher, YouTubeTrend
from .youtube_search_fetcher import YouTubeSearchFetcher, YouTubeSearchTrend
from .pytrends_validator import PyTrendsValidator

from autonomous_trend_agent.brain.pydantic_agents import (
    rank_trends, ContentBrief
)

logger = logging.getLogger(__name__)


@dataclass
class TrendDiscoveryConfig:
    """Configuration for the Trend Discovery phase."""
    # Platform toggles
    use_crawl4ai: bool = True          # Crawl4AI for TikTok/Twitter/YouTube web
    use_youtube_api: bool = True       # YouTube Data API v3 (structured)
    use_youtube_search: bool = True    # YouTube search (emerging topics)
    use_google_trends: bool = True     # pytrends validation

    # Crawl4AI settings
    crawl4ai_platforms: List[str] = None  # ["tiktok", "twitter", "youtube"]

    # YouTube settings
    youtube_regions: List[str] = None
    youtube_max_results: int = 50
    youtube_search_hours_back: int = 24

    # Ranking settings
    gemini_top_n: int = 10

    # Output
    output_dir: str = "/app/output/trends"


class TrendDiscovery:
    """
    Phase 1 orchestrator — discovers trending topics and produces
    a ContentBrief ready for Phase 2 (Video Sourcing).

    v2.1: Uses Crawl4AI + PydanticAI instead of custom scrapers + Z-Score.

    Usage:
        td = TrendDiscovery()
        brief = await td.run()
        print(brief.search_query)
    """

    def __init__(self, config: Optional[TrendDiscoveryConfig] = None):
        self.config = config or TrendDiscoveryConfig()

        # Set defaults for mutable fields
        if self.config.crawl4ai_platforms is None:
            self.config.crawl4ai_platforms = ["tiktok", "twitter", "youtube"]
        if self.config.youtube_regions is None:
            self.config.youtube_regions = ["US", "IN", "GB"]

        # Initialize structured API sensors (these can fail independently)
        self._youtube = None
        self._youtube_search = None
        self._pytrends = None

        try:
            if self.config.use_youtube_api:
                self._youtube = YouTubeFetcher()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] YouTube API init failed (non-fatal): {e}")

        try:
            if self.config.use_youtube_search:
                self._youtube_search = YouTubeSearchFetcher()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] YouTube Search init failed (non-fatal): {e}")

        try:
            if self.config.use_google_trends:
                self._pytrends = PyTrendsValidator()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] pytrends init failed (non-fatal): {e}")

        logger.info("[TrendDiscovery v2.1] Initialized.")

    async def run(self) -> ContentBrief:
        """
        Execute the full Phase 1 trend discovery pipeline.

        Steps:
            1. Crawl web signals via Crawl4AI (async, concurrent)
            2. Fetch structured signals from YouTube Data API
            3. Merge and deduplicate all signals
            4. Validate with Google Trends (pytrends)
            5. Send to PydanticAI trend_ranker → ContentBrief
            6. Save brief to disk

        Returns:
            ContentBrief with the winning trend and content strategy.
        """
        logger.info("=" * 60)
        logger.info("[TrendDiscovery] Phase 1 starting...")
        logger.info("=" * 60)

        all_topics: List[dict] = []

        # ── Step 1: Crawl4AI discovery ─────────────────────────────────
        if self.config.use_crawl4ai:
            crawled = await self._crawl_web()
            for trend in crawled:
                all_topics.append({
                    "platform": trend.platform,
                    "title": trend.title,
                    "engagement": trend.estimated_engagement,
                    "url": trend.url,
                    "source": "crawl4ai",
                })

        # ── Step 2: YouTube Data API (structured) ──────────────────────
        yt_trends = self._fetch_youtube()
        for t in yt_trends:
            all_topics.append({
                "platform": "youtube_api",
                "title": t.title,
                "engagement": t.view_count,
                "velocity": t.views_per_hour,
                "acceleration": t.acceleration,
                "url": f"https://youtube.com/watch?v={t.video_id}",
                "tags": t.tags[:5],
                "source": "youtube_api",
            })

        yts_trends = self._fetch_youtube_search()
        for t in yts_trends:
            all_topics.append({
                "platform": "youtube_search",
                "title": t.title,
                "engagement": t.view_count,
                "velocity": t.views_per_hour,
                "url": f"https://youtube.com/watch?v={t.video_id}",
                "tags": t.tags[:5],
                "source": "youtube_search",
            })

        if not all_topics:
            logger.error("[TrendDiscovery] No signals from any source.")
            return self._fallback_brief()

        logger.info(f"[TrendDiscovery] Collected {len(all_topics)} raw signals.")

        # ── Step 3: Google Trends validation ───────────────────────────
        google_validated = {}
        if self._pytrends:
            try:
                top_titles = list({t["title"][:50] for t in all_topics})[:20]
                google_validated = self._pytrends.validate(top_titles)
                for title, score in google_validated.items():
                    if score and hasattr(score, 'breakout') and score.breakout:
                        # Mark breakout trends
                        for t in all_topics:
                            if title.lower() in t["title"].lower():
                                t["google_breakout"] = True
                logger.info("[TrendDiscovery] Google Trends validation complete.")
            except Exception as e:
                logger.warning(f"[TrendDiscovery] Google Trends failed (non-fatal): {e}")

        # ── Step 4: PydanticAI ranking → ContentBrief ──────────────────
        prompt = self._format_candidates_for_ranking(all_topics)

        try:
            brief = await rank_trends(prompt)
            # Enrich with platform sources
            brief.platform_sources = list({t["platform"] for t in all_topics})
        except Exception as e:
            logger.error(f"[TrendDiscovery] PydanticAI ranking failed: {e}")
            brief = self._fallback_brief()

        # ── Step 5: Save brief to disk ─────────────────────────────────
        self._save_brief(brief)

        logger.info(
            f"[TrendDiscovery] DONE. Winner: '{brief.topic}' "
            f"| Score: {brief.viral_score} "
            f"| Confidence: {brief.confidence}"
        )
        return brief

    # ── Crawl4AI ──────────────────────────────────────────────────────────

    async def _crawl_web(self) -> List[CrawledTrend]:
        """Run Crawl4AI discovery across configured platforms."""
        try:
            async with Crawl4AIDiscovery(
                platforms=self.config.crawl4ai_platforms
            ) as discovery:
                return await discovery.discover_all()
        except Exception as e:
            logger.error(f"[TrendDiscovery] Crawl4AI failed: {e}")
            return []

    # ── YouTube Data API ──────────────────────────────────────────────────

    def _fetch_youtube(self) -> List[YouTubeTrend]:
        if not self._youtube:
            return []
        try:
            trends = self._youtube.fetch_trending(
                region_codes=self.config.youtube_regions,
                max_results=self.config.youtube_max_results,
            )
            logger.info(f"[TrendDiscovery] YouTube API: {len(trends)} trends fetched.")
            return trends
        except Exception as e:
            logger.error(f"[TrendDiscovery] YouTube API fetch failed: {e}")
            return []

    def _fetch_youtube_search(self) -> List[YouTubeSearchTrend]:
        if not self._youtube_search:
            return []
        try:
            trends = self._youtube_search.fetch_emerging(
                hours_back=self.config.youtube_search_hours_back
            )
            logger.info(f"[TrendDiscovery] YouTube Search: {len(trends)} emerging videos.")
            return trends
        except Exception as e:
            logger.error(f"[TrendDiscovery] YouTube Search failed: {e}")
            return []

    # ── Formatters ────────────────────────────────────────────────────────

    def _format_candidates_for_ranking(self, topics: List[dict]) -> str:
        """Format collected signals into a prompt for the PydanticAI ranker."""
        lines = [
            "Here are the current trending topics detected across TikTok, Twitter/X, "
            "YouTube, and Google Trends via Crawl4AI and YouTube Data API v3.\n"
        ]

        # Deduplicate by similar title
        seen = set()
        unique = []
        for t in topics:
            key = t["title"].lower()[:30]
            if key not in seen:
                seen.add(key)
                unique.append(t)

        for i, t in enumerate(unique[:30], 1):
            vel = t.get("velocity", "N/A")
            eng = t.get("engagement", 0)
            breakout = " [BREAKOUT]" if t.get("google_breakout") else ""
            tags = ", ".join(t.get("tags", []))
            lines.append(
                f"{i}. '{t['title']}'{breakout}\n"
                f"   Platform: {t['platform']} | Engagement: {eng:,.0f} | "
                f"Velocity: {vel}\n"
                f"   Tags: {tags or 'N/A'}\n"
            )

        lines.append(
            "\nBased on this data, select the BEST topic for a viral short-form video "
            "and provide your strategic recommendation."
        )
        return "\n".join(lines)

    # ── Fallback & Output ─────────────────────────────────────────────────

    def _fallback_brief(self) -> ContentBrief:
        """Return a minimal brief when everything fails."""
        return ContentBrief(
            topic="AI technology",
            keywords=["AI", "technology", "2026"],
            content_angle="Latest developments in AI technology",
            search_query="AI technology latest developments explained 2026",
            hook="Here's what's happening in AI right now...",
            target_audience="General tech audience",
            viral_score=20.0,
            confidence=0.1,
            is_breakout=False,
            platform_sources=["fallback"],
            notes="Fallback brief — no platform signals available.",
        )

    def _save_brief(self, brief: ContentBrief):
        """Save the ContentBrief to disk as JSON for Phase 2 to consume."""
        try:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "content_brief.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(brief.model_dump(), f, indent=2)
            logger.info(f"[TrendDiscovery] Brief saved to {out_path}")
        except Exception as e:
            logger.warning(f"[TrendDiscovery] Failed to save brief (non-fatal): {e}")
