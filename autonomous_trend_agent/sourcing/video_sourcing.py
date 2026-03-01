"""
Video Sourcing Orchestrator (Phase 2)

Fully automated pipeline:
    1. VideoSearcher.search(ContentBrief.search_query)  →  List[VideoCandidate]
    2. VideoSearcher.filter_by_quality(candidates)      →  filtered + ranked list
    3. Auto-select top candidate by view_count           →  VideoCandidate
    4. VideoDownloader.download(candidate.url)          →  DownloadResult (C2.1/C2.2)
    5. VideoValidator.validate(download.video_path)     →  ValidationResult (C2.3, C3.1–C3.4)
    6. Return SourcedVideo (path + metadata) for Phase 3

If the top candidate fails download or validation, falls back to the next
best candidate automatically (up to max_retries attempts).

Objectives: C1.1–C1.3, C2.1–C2.3, C3.1–C3.4
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

from .video_searcher import VideoSearcher, VideoCandidate
from .video_downloader import VideoDownloader, DownloadResult
from .video_validator import VideoValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class SourcedVideo:
    """
    Output of Phase 2, consumed by Phase 3 (Analysis).
    Contains the validated local path and full metadata.
    """
    video_path: str             # Absolute path to validated video.mp4
    metadata_path: str          # Absolute path to metadata.json
    title: str
    channel: str
    url: str
    duration_sec: int
    file_size_bytes: int
    resolution: str             # e.g. "1920x1080"
    fps: float
    topic: str                  # Trend topic this video addresses
    search_query: str           # Query used to find it


# (No user-facing types needed — pipeline is fully automated)


class VideoSourcing:
    """
    Phase 2 orchestrator — fully automated.

    Searches YouTube for source videos matching the ContentBrief query,
    ranks them by quality + view count, and downloads the best one.
    Falls back to the next candidate if download or validation fails.

    Usage:
        sourcing = VideoSourcing()
        brief = ...  # ContentBrief from Phase 1
        video = sourcing.run(
            search_query=brief.search_query,
            topic=brief.topic,
        )
    """

    def __init__(self, download_dir: Optional[str] = None):
        self._searcher = VideoSearcher()
        self._downloader = VideoDownloader(download_dir=download_dir)
        self._validator = VideoValidator()
        logger.info("[VideoSourcing] Initialized.")

    def run(
        self,
        search_query: str,
        topic: str = "",
        max_search_results: int = 10,
        max_retries: int = 3,
    ) -> SourcedVideo:
        """
        Execute Phase 2 end-to-end, fully automated.

        Args:
            search_query:    From ContentBrief.search_query
            topic:           Trend topic label (for metadata)
            max_search_results: Max candidates to fetch from YouTube
            max_retries:     How many candidates to try before giving up

        Returns:
            SourcedVideo ready for Phase 3.

        Raises:
            RuntimeError: If no valid video found after all retries.
        """
        logger.info("=" * 60)
        logger.info("[VideoSourcing] Phase 2 starting...")
        logger.info(f"[VideoSourcing] Query: '{search_query}'")
        logger.info("=" * 60)

        # ── Step 1 & 2: Search + Quality filter ─────────────────────
        candidates = self._searcher.search(search_query, max_results=max_search_results)

        # If zero results, try broadening the query
        if not candidates:
            broader_query = " ".join(search_query.split()[:3])  # First 3 words only
            logger.warning(
                f"[VideoSourcing] No results for '{search_query}'. "
                f"Trying broader query: '{broader_query}'"
            )
            candidates = self._searcher.search(broader_query, max_results=max_search_results)

        if not candidates:
            raise RuntimeError(
                f"[VideoSourcing] No videos found for query: '{search_query}' "
                f"(also tried broader query). YouTube API may be rate-limited."
            )

        quality_candidates = self._searcher.filter_by_quality(candidates)
        if not quality_candidates:
            logger.warning(
                "[VideoSourcing] No candidates passed quality filter. "
                "Relaxing: using all search results."
            )
            quality_candidates = candidates[:5]

        # ── Step 3: Auto-select top candidate (sorted by view_count desc) ──
        # Already sorted by VideoSearcher.filter_by_quality — pick the first.
        candidate_queue = quality_candidates[:max_retries]
        logger.info(
            f"[VideoSourcing] Auto-selected top candidate: "
            f"'{candidate_queue[0].title}' ({candidate_queue[0].view_count:,} views)"
        )

        for attempt, candidate in enumerate(candidate_queue[:max_retries], 1):
            logger.info(
                f"[VideoSourcing] Download attempt {attempt}/{max_retries}: "
                f"'{candidate.title}'"
            )

            # Download (C2.1 + C2.2)
            dl: DownloadResult = self._downloader.download(
                video_url=candidate.url,
                title_hint=candidate.title,
            )

            if not dl.success:
                logger.warning(
                    f"[VideoSourcing] Download failed: {dl.error}. "
                    f"{'Trying next candidate.' if attempt < max_retries else 'No more candidates.'}"
                )
                continue

            # Validate (C2.3, C3.1-C3.4)
            vr: ValidationResult = self._validator.validate(dl.video_path)

            if vr.valid:
                sourced = SourcedVideo(
                    video_path=dl.video_path,
                    metadata_path=dl.metadata_path,
                    title=dl.title,
                    channel=dl.channel,
                    url=candidate.url,
                    duration_sec=dl.duration_sec,
                    file_size_bytes=dl.file_size_bytes,
                    resolution=f"{vr.width}x{vr.height}",
                    fps=vr.fps,
                    topic=topic,
                    search_query=search_query,
                )
                self._save_phase2_record(sourced)
                logger.info(
                    f"[VideoSourcing] ✅ Phase 2 complete: '{dl.title}' "
                    f"({dl.file_size_bytes / 1e6:.0f} MB)"
                )
                return sourced
            else:
                logger.warning(
                    f"[VideoSourcing] Validation failed: "
                    + " | ".join(n for n in vr.notes if n.startswith("❌"))
                )
                if attempt < max_retries and candidate_queue: # Check if there are more candidates in the queue
                    # The loop iterates over candidate_queue[:max_retries], so `candidate` is already from this slice.
                    # If validation fails, we just continue to the next iteration, which will pick the next candidate
                    # from the `candidate_queue` slice, up to `max_retries`.
                    # No need to manually pop or manage `remaining` here as the loop handles it.
                    pass # The loop will naturally move to the next candidate if available.

        raise RuntimeError(
            f"[VideoSourcing] All {max_retries} download/validation attempts failed. "
            "Check network, YouTube availability, or widen the search query."
        )

    def _save_phase2_record(self, video: SourcedVideo):
        """Append a record of the sourced video to the metadata.json file."""
        try:
            meta_path = Path(video.metadata_path)
            if meta_path.exists():
                with open(meta_path, "r+", encoding="utf-8") as f:
                    meta = json.load(f)
                    meta["phase2"] = {
                        "topic": video.topic,
                        "search_query": video.search_query,
                        "resolution": video.resolution,
                        "fps": video.fps,
                    }
                    f.seek(0)
                    json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"[VideoSourcing] Could not update metadata: {e}")
