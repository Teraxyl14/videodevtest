"""
Google Trends Validator — pytrends integration (Phase 1, B1.4)

Cross-validates trend candidates against Google Search interest
to confirm broad public awareness before committing to a topic.

Requires:
    pip install pytrends

Objectives: B1.4 (pytrends Google Trends validation)
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError

logger = logging.getLogger(__name__)


@dataclass
class GoogleTrendScore:
    """Result of a Google Trends validation for a single keyword."""
    keyword: str
    interest_7d: float          # Average search interest over last 7 days (0-100)
    peak_interest: float        # Peak value in last 7 days
    rising: bool                # True if trend is still rising
    related_topics: List[str]   # Top related topics from Google
    breakout: bool              # True if Google marked it as "Breakout" (>5000% rise)


class PyTrendsValidator:
    """
    Validates keywords/topics using Google Trends via pytrends.

    Uses conservative request pacing to avoid 429 rate limits.
    Fails gracefully — a validation error returns a neutral score
    so it never blocks the pipeline.
    """

    # Max keywords per pytrends batch request
    BATCH_SIZE = 5

    # Seconds between batches (avoids 429)
    RATE_LIMIT_DELAY = 2.0

    def __init__(self, hl: str = "en-US", tz: int = 330):
        """
        Args:
            hl:  Language code (default: en-US)
            tz:  Timezone offset in minutes (330 = India, 0 = UTC, -300 = EST)
        """
        self._pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 25), retries=2, backoff_factor=0.5)
        logger.info("[PyTrendsValidator] Initialized.")

    def validate(self, keywords: List[str], timeframe: str = "now 7-d") -> Dict[str, GoogleTrendScore]:
        """
        Validate a list of trend keywords against Google Trends.

        Args:
            keywords:  Up to ~20 keywords (will be batched automatically)
            timeframe: pytrends timeframe string, default "now 7-d"

        Returns:
            Dict mapping keyword → GoogleTrendScore
        """
        results: Dict[str, GoogleTrendScore] = {}

        if not keywords:
            return results

        # Deduplicate and batch
        unique_kws = list(dict.fromkeys(keywords))  # preserve order
        batches = [
            unique_kws[i:i + self.BATCH_SIZE]
            for i in range(0, len(unique_kws), self.BATCH_SIZE)
        ]

        for batch in batches:
            try:
                batch_results = self._fetch_batch(batch, timeframe)
                results.update(batch_results)
            except Exception as e:
                logger.warning(f"[PyTrendsValidator] Batch {batch} failed: {e}")
                # Neutral scores so the pipeline doesn't stall
                for kw in batch:
                    results[kw] = self._neutral_score(kw)

            time.sleep(self.RATE_LIMIT_DELAY)

        return results

    def _fetch_batch(self, keywords: List[str], timeframe: str) -> Dict[str, GoogleTrendScore]:
        """Fetch interest over time + related topics for one batch."""
        self._pytrends.build_payload(
            kw_list=keywords,
            cat=0,
            timeframe=timeframe,
            geo="",     # Worldwide
            gprop="",
        )

        # Interest over time (DataFrame: date index, keyword columns, isPartial col)
        iot = self._pytrends.interest_over_time()
        results = {}

        for kw in keywords:
            if iot.empty or kw not in iot.columns:
                results[kw] = self._neutral_score(kw)
                continue

            series = iot[kw]
            interest_avg = float(series.mean())
            interest_peak = float(series.max())

            # Rising: last value > 80% of the period's mean (still climbing)
            last_val = float(series.iloc[-1]) if len(series) > 0 else 0.0
            rising = last_val >= (interest_avg * 0.8)

            results[kw] = GoogleTrendScore(
                keyword=kw,
                interest_7d=round(interest_avg, 1),
                peak_interest=round(interest_peak, 1),
                rising=rising,
                related_topics=[],
                breakout=False,
            )

        # Related topics (separate API call) — best effort
        try:
            related = self._pytrends.related_topics()
            for kw in keywords:
                if kw in related and related[kw]["rising"] is not None:
                    rising_df = related[kw]["rising"]
                    if not rising_df.empty and "topic_title" in rising_df.columns:
                        topics = rising_df["topic_title"].tolist()[:5]
                        results[kw].related_topics = topics
                        # Breakout: pytrends marks it as ">5000%" in the value column
                        if "value" in rising_df.columns:
                            val = str(rising_df["value"].iloc[0]) if len(rising_df) > 0 else ""
                            results[kw].breakout = "Breakout" in val
        except Exception as e:
            logger.debug(f"[PyTrendsValidator] Related topics fetch failed (non-fatal): {e}")

        return results

    @staticmethod
    def _neutral_score(keyword: str) -> GoogleTrendScore:
        """Return a neutral score when API fails, so pipeline doesn't stall."""
        return GoogleTrendScore(
            keyword=keyword,
            interest_7d=50.0,
            peak_interest=50.0,
            rising=False,
            related_topics=[],
            breakout=False,
        )

    def rank_by_interest(
        self,
        candidates: List[str],
        min_interest: float = 20.0,
    ) -> List[str]:
        """
        Validate and rank candidates by Google interest.
        Filters out topics below min_interest threshold.

        Returns list of keywords sorted by interest_7d descending.
        """
        scores = self.validate(candidates)
        ranked = [
            kw for kw, s in sorted(scores.items(), key=lambda x: x[1].interest_7d, reverse=True)
            if s.interest_7d >= min_interest
        ]
        logger.info(
            f"[PyTrendsValidator] {len(ranked)}/{len(candidates)} passed "
            f"min_interest={min_interest}"
        )
        return ranked
