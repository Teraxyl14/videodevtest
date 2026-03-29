"""
Crawl4AI Discovery Module (Phase 1, B1.1-B1.2)

Replaces the custom TikTok/Twitter/YouTube scrapers with Crawl4AI's
AsyncWebCrawler + JsonCssExtractionStrategy for async, headless scraping.

Keeps youtube_fetcher.py and pytrends_validator.py as supplementary signals
since they use structured APIs (YouTube Data API v3, Google Trends).

Objectives: B1.1 (Crawl4AI integration), B1.2 (JSON schemas per platform)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

logger = logging.getLogger(__name__)


# ─── Platform-specific extraction schemas (B1.2) ────────────────────────────

TIKTOK_SCHEMA = {
    "name": "TikTok Trending",
    "baseSelector": "[data-e2e='trending-item'], .trending-item, .tiktok-trending-card",
    "fields": [
        {"name": "title", "selector": "h2, .title, [data-e2e='trending-title']", "type": "text"},
        {"name": "engagement", "selector": ".count, .engagement, [data-e2e='trending-count']", "type": "text"},
        {"name": "hashtag", "selector": "a[href*='tag'], .hashtag", "type": "text"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"},
    ]
}

TWITTER_SCHEMA = {
    "name": "Twitter/X Trending",
    "baseSelector": "[data-testid='trend'], .trend-item, .css-1dbjc4n",
    "fields": [
        {"name": "title", "selector": "span, .trend-name", "type": "text"},
        {"name": "tweet_count", "selector": ".tweet-count, span:last-child", "type": "text"},
        {"name": "category", "selector": ".trend-category, span:first-child", "type": "text"},
    ]
}

YOUTUBE_TRENDING_SCHEMA = {
    "name": "YouTube Trending",
    "baseSelector": "ytd-video-renderer, .ytd-rich-item-renderer",
    "fields": [
        {"name": "title", "selector": "#video-title, .title", "type": "text"},
        {"name": "channel", "selector": "#channel-name, .channel-name", "type": "text"},
        {"name": "views", "selector": "#metadata-line span, .view-count", "type": "text"},
        {"name": "url", "selector": "a#video-title-link, a#thumbnail", "type": "attribute", "attribute": "href"},
    ]
}

PLATFORM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tiktok": {
        "url": "https://www.tiktok.com/trending",
        "schema": TIKTOK_SCHEMA,
        "fallback_urls": [
            "https://www.tiktok.com/discover",
        ],
    },
    "twitter": {
        "url": "https://x.com/explore/tabs/trending",
        "schema": TWITTER_SCHEMA,
        "fallback_urls": [
            "https://trends24.in/",
            "https://getdaytrends.com/",
        ],
    },
    "youtube": {
        "url": "https://www.youtube.com/feed/trending",
        "schema": YOUTUBE_TRENDING_SCHEMA,
        "fallback_urls": [],
    },
}


@dataclass
class CrawledTrend:
    """A single trend item extracted by Crawl4AI."""
    platform: str
    title: str
    engagement_text: str = ""         # Raw text like "1.2M views"
    url: str = ""
    category: str = ""
    hashtags: List[str] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)

    @property
    def estimated_engagement(self) -> float:
        """Parse engagement text into a numeric value."""
        return _parse_engagement(self.engagement_text)


class Crawl4AIDiscovery:
    """
    Async web crawler for discovering trending topics across platforms.

    Usage:
        async with Crawl4AIDiscovery() as discovery:
            trends = await discovery.discover_all()
    """

    def __init__(self, platforms: Optional[List[str]] = None):
        """
        Args:
            platforms: Which platforms to crawl. Defaults to all.
        """
        self.platforms = platforms or list(PLATFORM_CONFIGS.keys())
        self._crawler: Optional[AsyncWebCrawler] = None

    async def __aenter__(self):
        self._crawler = AsyncWebCrawler()
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._crawler:
            await self._crawler.__aexit__(*args)

    async def discover_all(self) -> List[CrawledTrend]:
        """
        Crawl all configured platforms concurrently.

        Returns:
            Combined list of CrawledTrend from all platforms.
        """
        if not self._crawler:
            raise RuntimeError("Use 'async with Crawl4AIDiscovery() as d:' context manager.")

        tasks = [self._crawl_platform(p) for p in self.platforms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_trends: List[CrawledTrend] = []
        for platform, result in zip(self.platforms, results):
            if isinstance(result, Exception):
                logger.error(f"[Crawl4AI] {platform} crawl failed: {result}")
                continue
            all_trends.extend(result)
            logger.info(f"[Crawl4AI] {platform}: {len(result)} trends extracted.")

        logger.info(f"[Crawl4AI] Total: {len(all_trends)} trends across {len(self.platforms)} platforms.")
        return all_trends

    async def discover_platform(self, platform: str) -> List[CrawledTrend]:
        """Crawl a single platform."""
        if not self._crawler:
            raise RuntimeError("Use context manager.")
        return await self._crawl_platform(platform)

    async def _crawl_platform(self, platform: str) -> List[CrawledTrend]:
        """Attempt to crawl a platform, falling back to alternate URLs."""
        config = PLATFORM_CONFIGS.get(platform)
        if not config:
            logger.warning(f"[Crawl4AI] Unknown platform: {platform}")
            return []

        urls_to_try = [config["url"]] + config.get("fallback_urls", [])
        schema = config["schema"]

        for url in urls_to_try:
            try:
                trends = await self._extract_from_url(url, schema, platform)
                if trends:
                    return trends
                logger.info(f"[Crawl4AI] {platform}: No results from {url}, trying fallback...")
            except Exception as e:
                logger.warning(f"[Crawl4AI] {platform}: Error on {url}: {e}")
                continue

        logger.warning(f"[Crawl4AI] {platform}: All URLs exhausted, returning empty.")
        return []

    async def _extract_from_url(
        self, url: str, schema: Dict, platform: str
    ) -> List[CrawledTrend]:
        """Run Crawl4AI extraction on a single URL."""
        strategy = JsonCssExtractionStrategy(schema)

        result = await self._crawler.arun(
            url=url,
            extraction_strategy=strategy,
            bypass_cache=True,
        )

        if not result.extracted_content:
            return []

        # Parse the JSON content extracted by Crawl4AI
        import json
        try:
            items = json.loads(result.extracted_content)
        except (json.JSONDecodeError, TypeError):
            items = []

        if not isinstance(items, list):
            items = [items] if items else []

        trends: List[CrawledTrend] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            title = (
                item.get("title", "")
                or item.get("name", "")
                or item.get("text", "")
            ).strip()

            if not title:
                continue

            engagement = (
                item.get("engagement", "")
                or item.get("tweet_count", "")
                or item.get("views", "")
                or ""
            )

            trend = CrawledTrend(
                platform=platform,
                title=title,
                engagement_text=str(engagement).strip(),
                url=item.get("url", ""),
                category=item.get("category", ""),
                hashtags=[h.strip() for h in str(item.get("hashtag", "")).split("#") if h.strip()],
                raw_data=item,
            )
            trends.append(trend)

        return trends


# ─── Utilities ───────────────────────────────────────────────────────────────

def _parse_engagement(text: str) -> float:
    """
    Parse human-readable engagement strings like '1.2M', '500K', '3.4B'.
    Returns a float value.
    """
    if not text:
        return 0.0

    import re
    text = text.strip().upper().replace(",", "")

    # Match patterns like "1.2M views", "500K", "3400"
    match = re.search(r"([\d.]+)\s*([KMBT])?", text)
    if not match:
        return 0.0

    value = float(match.group(1))
    suffix = match.group(2) or ""

    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    return value * multipliers.get(suffix, 1)
