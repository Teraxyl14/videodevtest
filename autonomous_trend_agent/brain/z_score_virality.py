"""
Z-Score Virality Algorithm (Phase 1, B1.5)

Normalizes raw scores from different platforms (YouTube, Twitter/X, TikTok,
Google Trends) into a single comparable Z-Score virality metric. This lets
us rank trends fairly across platforms that have wildly different scales
(YouTube views in millions vs. Twitter impressions in thousands).

Algorithm:
    1. Collect raw signals per platform (views/hr, engagement rate, etc.)
    2. Z-normalize each metric within its platform population
    3. Weight platforms by signal freshness and reliability
    4. Output a single 0-100 ViralScore per topic

Objectives: B1.5 (Z-Score Virality Algorithm, cross-platform normalization)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlatformSignal:
    """A single platform's contribution to a trend candidate."""
    platform: str                   # "youtube", "twitter", "tiktok", "google_trends"
    topic: str                      # Normalized topic/keyword
    raw_velocity: float             # Views/engagements per hour
    raw_engagement: float           # Engagement rate (0.0–1.0)
    raw_acceleration: float         # Delta velocity from last period
    source_url: str = ""            # URL of the trending item (if available)
    extra: Dict = field(default_factory=dict)  # Platform-specific fields


@dataclass
class ViralCandidate:
    """A deduplicated trend topic with cross-platform viral scoring."""
    topic: str
    keywords: List[str]             # Synonym/related keywords
    signals: List[PlatformSignal]   # One per platform that detected it

    # Computed scores
    platform_count: int = 0         # Number of platforms detecting it
    z_score: float = 0.0            # Cross-platform Z-Score (raw)
    viral_score: float = 0.0        # Final 0-100 score
    confidence: float = 0.0         # 0.0-1.0 confidence in the signal

    # Top-level metadata
    peak_platform: str = ""         # Which platform shows the strongest signal
    is_breakout: bool = False       # True if rising extremely fast
    estimated_duration: str = ""    # "hours", "days", "weeks"


class ZScoreVirality:
    """
    Computes cross-platform Z-Score virality scores.

    Platform weights (importance to our pipeline):
        YouTube: 0.40  — Long-form content source; velocity is most actionable
        Twitter: 0.30  — Real-time sentiment and breakout detection
        TikTok:  0.20  — Fastest-moving consumer trends
        Google:  0.10  — Broad confirmation signal (lags behind by hours/days)
    """

    PLATFORM_WEIGHTS = {
        "youtube":        0.35,   # Trending chart — validated, high reach
        "youtube_search": 0.25,   # Search emerging — early signal, same API
        "twitter":        0.30,   # Real-time sentiment and breakout detection
        "google_trends":  0.10,   # Broad confirmation (lags by hours/days)
    }

    def __init__(self):
        # Stores running population stats per platform for Z-scoring
        self._platform_stats: Dict[str, Dict] = {}

    def compute(self, all_signals: List[PlatformSignal]) -> List[ViralCandidate]:
        """
        Group signals by topic, compute Z-Scores, return sorted candidates.

        Args:
            all_signals: Raw signals from all platform fetchers.

        Returns:
            List of ViralCandidate sorted by viral_score descending.
        """
        if not all_signals:
            return []

        # 1. Compute per-platform population stats (mean, std)
        platform_groups: Dict[str, List[PlatformSignal]] = {}
        for sig in all_signals:
            platform_groups.setdefault(sig.platform, []).append(sig)

        platform_stats: Dict[str, Dict] = {}
        for platform, sigs in platform_groups.items():
            vels = [s.raw_velocity for s in sigs]
            engs = [s.raw_engagement for s in sigs]
            accs = [s.raw_acceleration for s in sigs]
            platform_stats[platform] = {
                "vel_mean": _mean(vels), "vel_std": _std(vels),
                "eng_mean": _mean(engs), "eng_std": _std(engs),
                "acc_mean": _mean(accs), "acc_std": _std(accs),
            }

        # 2. Z-normalize each signal
        normalized: List[Dict] = []
        for sig in all_signals:
            st = platform_stats[sig.platform]
            z_vel = _zscore(sig.raw_velocity, st["vel_mean"], st["vel_std"])
            z_eng = _zscore(sig.raw_engagement, st["eng_mean"], st["eng_std"])
            z_acc = _zscore(sig.raw_acceleration, st["acc_mean"], st["acc_std"])
            # Composite Z: velocity dominates (60%), engagement (25%), acceleration (15%)
            composite_z = z_vel * 0.60 + z_eng * 0.25 + z_acc * 0.15
            normalized.append({
                "signal": sig,
                "composite_z": composite_z,
            })

        # 3. Group by normalized topic keyword
        topic_groups: Dict[str, List[Dict]] = {}
        for item in normalized:
            key = _normalize_topic(item["signal"].topic)
            topic_groups.setdefault(key, []).append(item)

        # 4. Build ViralCandidates
        candidates: List[ViralCandidate] = []
        for topic, items in topic_groups.items():
            candidate = self._build_candidate(topic, items)
            candidates.append(candidate)

        # 5. Sort by viral_score
        candidates.sort(key=lambda c: c.viral_score, reverse=True)

        logger.info(
            f"[ZScoreVirality] Scored {len(candidates)} unique topics "
            f"from {len(all_signals)} signals across "
            f"{len(platform_groups)} platform(s)."
        )
        return candidates

    def _build_candidate(self, topic: str, items: List[Dict]) -> ViralCandidate:
        """Aggregate normalized signals for a single topic into a ViralCandidate."""
        signals = [i["signal"] for i in items]
        platform_zscores: Dict[str, float] = {}

        for item in items:
            plat = item["signal"].platform
            # Average Z if multiple signals from same platform for same topic
            platform_zscores[plat] = max(
                platform_zscores.get(plat, -999),
                item["composite_z"]
            )

        # Weighted cross-platform Z
        total_weight = 0.0
        weighted_z = 0.0
        for plat, z in platform_zscores.items():
            w = self.PLATFORM_WEIGHTS.get(plat, 0.1)
            weighted_z += z * w
            total_weight += w

        cross_platform_z = weighted_z / max(total_weight, 1e-9)

        # Clamp and scale to 0-100
        viral_score = _clamp(_sigmoid(cross_platform_z) * 100, 0, 100)

        # Confidence: more platforms = higher confidence
        num_platforms = len(platform_zscores)
        confidence = min(1.0, num_platforms / len(self.PLATFORM_WEIGHTS))

        # Breakout: any signal has extremely high Z (>2.5 sigma above mean)
        is_breakout = any(i["composite_z"] > 2.5 for i in items)

        # Best platform (highest weighted Z)
        peak_platform = max(platform_zscores, key=platform_zscores.get, default="")

        # Duration estimate based on acceleration
        avg_acc = _mean([s.raw_acceleration for s in signals])
        if avg_acc > 1000:
            duration = "hours"
        elif avg_acc > 100:
            duration = "days"
        else:
            duration = "weeks"

        # Keywords: collect all topic strings and tags
        kws = list({s.topic for s in signals})
        for s in signals:
            kws.extend(s.extra.get("tags", []))
        kws = list(dict.fromkeys(kws))[:8]  # Deduplicated, max 8

        return ViralCandidate(
            topic=topic,
            keywords=kws,
            signals=signals,
            platform_count=num_platforms,
            z_score=round(cross_platform_z, 3),
            viral_score=round(viral_score, 1),
            confidence=round(confidence, 2),
            peak_platform=peak_platform,
            is_breakout=is_breakout,
            estimated_duration=duration,
        )


# ─── Math helpers ────────────────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0  # Avoid division by zero; treat as no spread
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return max(math.sqrt(variance), 1e-9)


def _zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def _sigmoid(z: float) -> float:
    """Smooth mapping of Z → (0, 1) to get viral score."""
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_topic(topic: str) -> str:
    """Lowercase, strip punctuation, collapse spaces for deduplication."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", topic.lower())).strip()
