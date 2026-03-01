"""
Trend Discovery Orchestrator (Phase 1 — Section B)

The top-level entry point for Phase 1. Coordinates all sensors
(YouTube, Twitter, TikTok, Google Trends), runs the Z-Score algorithm,
validates with pytrends, and sends results to Gemini Flash for final
ranking and content brief generation.

Single method: TrendDiscovery.run() → ContentBrief

Objectives: B1.1–B1.5, B2.1–B2.4
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

from .youtube_fetcher import YouTubeFetcher, YouTubeTrend
from .youtube_search_fetcher import YouTubeSearchFetcher, YouTubeSearchTrend
from .twitter_scraper import TwitterScraper, TwitterTrend
from .pytrends_validator import PyTrendsValidator

from autonomous_trend_agent.brain.z_score_virality import ZScoreVirality, PlatformSignal, ViralCandidate
from autonomous_trend_agent.brain.gemini_trend_ranker import GeminiTrendRanker, ContentBrief

logger = logging.getLogger(__name__)


@dataclass
class TrendDiscoveryConfig:
    """Configuration for the Trend Discovery phase."""
    # Platform toggles
    use_youtube: bool = True           # YouTube trending chart
    use_youtube_search: bool = True    # YouTube search (emerging topics)
    use_twitter: bool = True
    use_google_trends: bool = True

    # YouTube settings
    youtube_regions: List[str] = None      # e.g. ["US", "IN", "GB"]
    youtube_max_results: int = 50
    youtube_search_hours_back: int = 24   # How far back to search

    # Twitter settings
    twitter_proxy_url: Optional[str] = None  # Residential proxy URL (optional)
    twitter_woeid: int = 1                   # Where On Earth ID (1 = worldwide)

    # Ranking settings
    gemini_top_n: int = 10                 # Candidates to send to Gemini
    min_google_interest: float = 15.0      # Minimum pytrends score to pass

    # Output
    output_dir: str = "/app/output/trends"


class TrendDiscovery:
    """
    Phase 1 orchestrator — discovers trending topics and produces
    a ContentBrief ready for Phase 2 (Video Sourcing).

    Usage:
        td = TrendDiscovery()
        brief = td.run()
        print(brief.search_query)   # YouTube query for Phase 2
    """

    def __init__(self, config: Optional[TrendDiscoveryConfig] = None):
        self.config = config or TrendDiscoveryConfig()

        # Set defaults for mutable fields
        if self.config.youtube_regions is None:
            self.config.youtube_regions = ["US", "IN", "GB"]

        # Initialize sensors (only what's available — each can fail independently)
        self._youtube = None
        self._youtube_search = None
        self._twitter = None
        self._pytrends = None

        try:
            if self.config.use_youtube:
                self._youtube = YouTubeFetcher()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] YouTube init failed (non-fatal): {e}")

        try:
            if self.config.use_youtube_search:
                self._youtube_search = YouTubeSearchFetcher()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] YouTube Search init failed (non-fatal): {e}")

        try:
            if self.config.use_twitter:
                self._twitter = TwitterScraper(
                    proxy_url=self.config.twitter_proxy_url
                )
        except Exception as e:
            logger.warning(f"[TrendDiscovery] Twitter init failed (non-fatal): {e}")

        try:
            if self.config.use_google_trends:
                self._pytrends = PyTrendsValidator()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] pytrends init failed (non-fatal): {e}")

        # Scoring pipeline
        self._z_score = ZScoreVirality()
        try:
            self._ranker = GeminiTrendRanker()
        except Exception as e:
            logger.warning(f"[TrendDiscovery] Gemini ranker init failed: {e}")
            self._ranker = None

        logger.info("[TrendDiscovery] Initialized.")

    def run(self) -> ContentBrief:
        """
        Execute the full Phase 1 trend discovery pipeline.

        Steps:
            1. Fetch raw signals from all platforms
            2. Convert to PlatformSignals for Z-Score engine
            3. Run Z-Score algorithm → ranked ViralCandidates
            4. Validate top candidates via Google Trends (pytrends)
            5. Send to Gemini Flash → ContentBrief
            6. Save brief to disk

        Returns:
            ContentBrief with the winning trend and content strategy.
            On total failure, returns a minimal fallback brief rather than crashing.
        """
        logger.info("=" * 60)
        logger.info("[TrendDiscovery] Phase 1 starting...")
        logger.info("=" * 60)

        all_signals: List[PlatformSignal] = []

        # ── Step 1: Fetch raw signals ──────────────────────────────────
        yt_trends = self._fetch_youtube()
        yts_trends = self._fetch_youtube_search()
        tw_trends = self._fetch_twitter()

        # ── Step 2: Convert to PlatformSignals ────────────────────────
        all_signals.extend(self._youtube_to_signals(yt_trends))
        all_signals.extend(self._youtube_search_to_signals(yts_trends))
        all_signals.extend(self._twitter_to_signals(tw_trends))

        if not all_signals:
            logger.error(
                "[TrendDiscovery] No signals from any platform. "
                "Check API keys and network connectivity."
            )
            # Return a minimal fallback instead of crashing
            fallback = ContentBrief(
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
                gemini_notes="Fallback brief — no platform signals available.",
                raw_candidates=[],
            )
            self._save_brief(fallback)
            return fallback

        logger.info(f"[TrendDiscovery] Collected {len(all_signals)} raw signals.")

        # ── Step 3: Z-Score algorithm ──────────────────────────────────
        candidates: List[ViralCandidate] = self._z_score.compute(all_signals)
        logger.info(f"[TrendDiscovery] Scored {len(candidates)} unique topic candidates.")

        if not candidates:
            logger.warning("[TrendDiscovery] Z-Score produced 0 candidates. Using raw signal topics.")
            # Create synthetic candidates from raw signals
            seen = set()
            for sig in all_signals[:5]:
                if sig.topic not in seen:
                    candidates.append(ViralCandidate(
                        topic=sig.topic, viral_score=sig.raw_velocity,
                        platforms=[sig.platform], signal_count=1,
                    ))
                    seen.add(sig.topic)

        # ── Step 4: Google Trends validation ──────────────────────────
        if self._pytrends and candidates:
            candidates = self._validate_with_google(candidates)

        # ── Step 5: Gemini Flash ranking → ContentBrief ───────────────
        if self._ranker:
            try:
                brief = self._ranker.rank_and_brief(candidates, top_n=self.config.gemini_top_n)
            except Exception as e:
                logger.error(f"[TrendDiscovery] Gemini ranking failed: {e}. Using top Z-Score candidate.")
                top = candidates[0]
                brief = ContentBrief(
                    topic=top.topic,
                    keywords=top.keywords[:5] if top.keywords else [top.topic],
                    content_angle=f"Deep analysis of {top.topic}",
                    search_query=f"{top.topic} explained analysis deep dive",
                    hook=f"Everyone is talking about {top.topic}...",
                    target_audience="General social media audience",
                    viral_score=top.viral_score,
                    confidence=0.4,
                    is_breakout=top.is_breakout,
                    platform_sources=[s.platform for s in top.signals],
                    gemini_notes="Gemini ranking failed — using top Z-Score candidate.",
                    raw_candidates=[],
                )
        else:
            # No Gemini ranker available — use raw Z-Score results
            top = candidates[0]
            brief = ContentBrief(
                topic=top.topic,
                keywords=top.keywords[:5] if top.keywords else [top.topic],
                content_angle=f"Deep analysis of {top.topic}",
                search_query=f"{top.topic} explained analysis deep dive",
                hook=f"Everyone is talking about {top.topic}...",
                target_audience="General social media audience",
                viral_score=top.viral_score,
                confidence=0.3,
                is_breakout=top.is_breakout,
                platform_sources=[s.platform for s in top.signals],
                gemini_notes="Gemini ranker unavailable — using top Z-Score candidate.",
                raw_candidates=[],
            )

        # ── Step 6: Save brief to disk ────────────────────────────────
        self._save_brief(brief)

        logger.info(
            f"[TrendDiscovery] DONE. Winner: '{brief.topic}' "
            f"| Score: {brief.viral_score} "
            f"| Confidence: {brief.confidence}"
        )
        return brief

    # ── Platform fetchers ──────────────────────────────────────────────

    def _fetch_youtube(self) -> List[YouTubeTrend]:
        if not self._youtube:
            return []
        try:
            trends = self._youtube.fetch_trending(
                region_codes=self.config.youtube_regions,
                max_results=self.config.youtube_max_results,
            )
            logger.info(f"[TrendDiscovery] YouTube: {len(trends)} trends fetched.")
            return trends
        except Exception as e:
            logger.error(f"[TrendDiscovery] YouTube fetch failed: {e}")
            return []

    def _fetch_twitter(self) -> List[TwitterTrend]:
        if not self._twitter:
            return []
        try:
            trends = self._twitter.fetch_trending(woeid=self.config.twitter_woeid)
            logger.info(f"[TrendDiscovery] Twitter: {len(trends)} trends fetched.")
            return trends
        except Exception as e:
            logger.error(f"[TrendDiscovery] Twitter fetch failed (non-fatal): {e}")
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
            logger.error(f"[TrendDiscovery] YouTube Search fetch failed (non-fatal): {e}")
            return []

    # ── Signal converters ──────────────────────────────────────────────

    def _youtube_to_signals(self, trends: List[YouTubeTrend]) -> List[PlatformSignal]:
        signals = []
        for t in trends:
            eng = (t.like_count + t.comment_count) / max(1, t.view_count)
            signals.append(PlatformSignal(
                platform="youtube",
                topic=t.topic,
                raw_velocity=t.views_per_hour,
                raw_engagement=eng,
                raw_acceleration=t.acceleration,
                source_url=f"https://youtube.com/watch?v={t.video_id}",
                extra={"title": t.title, "tags": t.tags, "channel": t.channel},
            ))
        return signals

    def _twitter_to_signals(self, trends: List[TwitterTrend]) -> List[PlatformSignal]:
        signals = []
        for t in trends:
            eng = min(1.0, t.impressions / max(1, t.tweet_volume * 100))
            signals.append(PlatformSignal(
                platform="twitter",
                topic=t.topic,
                raw_velocity=t.velocity,
                raw_engagement=eng,
                raw_acceleration=t.acceleration,
                extra={"sentiment": t.sentiment_score},
            ))
        return signals

    def _youtube_search_to_signals(self, trends: List[YouTubeSearchTrend]) -> List[PlatformSignal]:
        signals = []
        for t in trends:
            eng = (t.like_count + t.comment_count) / max(1, t.view_count)
            signals.append(PlatformSignal(
                platform="youtube_search",
                topic=t.topic,
                raw_velocity=t.views_per_hour,
                raw_engagement=eng,
                raw_acceleration=0.0,   # No prior snapshot for emerging videos
                source_url=f"https://youtube.com/watch?v={t.video_id}",
                extra={"title": t.title, "tags": t.tags, "query": t.search_query},
            ))
        return signals

    # ── Google Trends validation ───────────────────────────────────────

    def _validate_with_google(self, candidates: List[ViralCandidate]) -> List[ViralCandidate]:
        """
        Validate + re-rank top candidates with Google Trends.
        Lowers viral_score for topics below min_google_interest.
        Never removes a candidate entirely — just de-prioritizes.
        """
        top_keywords = [c.topic for c in candidates[:20]]
        try:
            scores = self._pytrends.validate(top_keywords)
            for c in candidates:
                gs = scores.get(c.topic)
                if gs:
                    # Boost if breakout on Google
                    if gs.breakout:
                        c.viral_score = min(100, c.viral_score * 1.2)
                        c.is_breakout = True
                    # Penalize if low Google interest
                    elif gs.interest_7d < self.config.min_google_interest:
                        c.viral_score *= 0.7
            # Re-sort after adjustment
            candidates.sort(key=lambda x: x.viral_score, reverse=True)
            logger.info("[TrendDiscovery] Google Trends validation complete.")
        except Exception as e:
            logger.warning(f"[TrendDiscovery] Google Trends validation failed (non-fatal): {e}")
        return candidates

    # ── Output ─────────────────────────────────────────────────────────

    def _save_brief(self, brief: ContentBrief):
        """Save the ContentBrief to disk as JSON for Phase 2 to consume."""
        try:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "content_brief.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(asdict(brief), f, indent=2)
            logger.info(f"[TrendDiscovery] Brief saved to {out_path}")
        except Exception as e:
            logger.warning(f"[TrendDiscovery] Failed to save brief (non-fatal): {e}")
