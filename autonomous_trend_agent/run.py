"""
Autonomous Trend Agent — End-to-End Pipeline Runner (H1.1)
===========================================================
Single command triggers the entire pipeline:

    Phase 1: Trend Discovery   → ContentBrief (topic + search_query)
    Phase 2: Video Sourcing    → SourcedVideo (validated local .mp4)
    Phase 3: Analysis          → Viral segments identified
    Phase 4: Editing           → Reframed, captioned, polished shorts
    Phase 5: Output + QA       → Organized directories with metadata

Usage:
    # Full autonomous run (discovers trend, finds video, creates shorts)
    python -m autonomous_trend_agent.run

    # Skip Phase 1+2, process a local video directly
    python -m autonomous_trend_agent.run --video /path/to/video.mp4

    # Custom topic (skip Phase 1, run Phase 2+ with custom query)
    python -m autonomous_trend_agent.run --topic "AI agents 2026"

Objectives: H1.1 (single command), H1.3 (resume), H1.4 (progress logged),
            H3.1 (edge cases), H3.2 (errors logged), H3.3 (no corruption)
"""

import os
import gc
import sys
import json
import time
import signal
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional

from dotenv import load_dotenv

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("AutoTrendAgent")

# ─── Fallback topics when Phase 1 fails ───────────────────────────────────────

FALLBACK_TOPICS = [
    {"topic": "AI agents",       "query": "AI agents explained deep dive 2026"},
    {"topic": "Claude AI",       "query": "Claude AI latest features review 2026"},
    {"topic": "GPT-5",           "query": "GPT-5 capabilities analysis explained"},
    {"topic": "Sora video AI",   "query": "Sora AI video generation deep dive"},
    {"topic": "Robot humanoid",  "query": "humanoid robots latest breakthroughs 2026"},
]


# ─── VRAM Safety ──────────────────────────────────────────────────────────────

def _force_vram_cleanup():
    """Force-release all GPU memory between pipeline phases."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            logger.info(f"[VRAM] Cleanup done. Allocated: {allocated:.2f} GB")
    except ImportError:
        pass  # No torch on Windows host — expected
    except Exception as e:
        logger.warning(f"[VRAM] Cleanup error (non-fatal): {e}")


# ─── Graceful shutdown ────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C — set flag so current phase finishes, then exit cleanly."""
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("[Signal] Second interrupt — FORCE EXIT")
        sys.exit(1)
    _shutdown_requested = True
    logger.warning("[Signal] Shutdown requested (Ctrl+C). Finishing current phase...")


signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _signal_handler)


def _check_shutdown():
    """Raise if shutdown requested — checked between phases."""
    if _shutdown_requested:
        raise KeyboardInterrupt("Graceful shutdown requested between phases.")


# ─── Resume detection (H1.3) ─────────────────────────────────────────────────

def _find_resume_state(run_dir: Path) -> dict:
    """
    Check if a previous run has partial output we can resume from.

    Returns dict with resume info:
        {"phase": int, "content_brief": dict|None, "video_path": str|None}
    """
    state = {"phase": 0, "content_brief": None, "video_path": None}

    # Check for Phase 1 output
    brief_path = run_dir / "content_brief.json"
    if brief_path.exists():
        try:
            with open(brief_path, "r", encoding="utf-8") as f:
                state["content_brief"] = json.load(f)
            state["phase"] = 1
            logger.info(f"[Resume] Found Phase 1 output: {brief_path}")
        except Exception:
            pass

    # Check for Phase 2 output (video already downloaded)
    video_dirs = list(run_dir.glob("**/video.mp4"))
    if video_dirs:
        state["video_path"] = str(video_dirs[0])
        state["phase"] = 2
        logger.info(f"[Resume] Found Phase 2 output: {state['video_path']}")

    # Check for completed run
    run_meta = run_dir / "pipeline_run.json"
    if run_meta.exists():
        try:
            with open(run_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "output_qa" in meta.get("phases_completed", []):
                state["phase"] = 5
                logger.info("[Resume] Previous run was COMPLETE. Starting fresh.")
                state = {"phase": 0, "content_brief": None, "video_path": None}
        except Exception:
            pass

    return state


# ─── Disk space check ─────────────────────────────────────────────────────────

def _check_disk_space(path: str, min_gb: float = 5.0) -> bool:
    """Verify enough disk space for downloads + rendering."""
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_gb:
            logger.error(
                f"[Disk] Only {free_gb:.1f} GB free in {path}. "
                f"Need at least {min_gb} GB. Aborting."
            )
            return False
        logger.info(f"[Disk] {free_gb:.1f} GB free — OK")
        return True
    except Exception as e:
        logger.warning(f"[Disk] Could not check space: {e}. Continuing anyway.")
        return True


# ─── Phase 1: Trend Discovery ────────────────────────────────────────────────

def run_phase1(output_dir: Path) -> dict:
    """
    Phase 1 — Discover a trending topic and produce a ContentBrief.

    On failure, returns a hardcoded fallback topic instead of crashing.

    Returns:
        dict with keys: topic, search_query, keywords, content_angle, hook,
                        viral_score, confidence, platform_sources
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 — TREND DISCOVERY")
    logger.info("=" * 60)

    try:
        import asyncio
        from autonomous_trend_agent.sensors.trend_discovery import (
            TrendDiscovery, TrendDiscoveryConfig,
        )

        config = TrendDiscoveryConfig(output_dir=str(output_dir / "phase1"))
        td = TrendDiscovery(config)
        brief = asyncio.run(td.run())
        brief_dict = brief.model_dump()

    except Exception as e:
        logger.error(f"[Phase 1] FAILED: {e}")
        logger.warning("[Phase 1] Using fallback trending topic.")

        # Pick first fallback topic
        fb = FALLBACK_TOPICS[0]
        brief_dict = {
            "topic": fb["topic"],
            "search_query": fb["query"],
            "keywords": [fb["topic"]],
            "content_angle": f"Deep dive into {fb['topic']}",
            "hook": f"Here's what everyone needs to know about {fb['topic']}...",
            "target_audience": "General tech audience",
            "viral_score": 50.0,
            "confidence": 0.3,
            "is_breakout": False,
            "platform_sources": ["fallback"],
            "gemini_notes": "Phase 1 failed — using hardcoded fallback topic.",
            "raw_candidates": [],
        }

    # Save brief to disk for resume capability (H1.3)
    brief_path = output_dir / "content_brief.json"
    try:
        with open(brief_path, "w", encoding="utf-8") as f:
            json.dump(brief_dict, f, indent=2)
        logger.info(f"Phase 1 complete → Topic: '{brief_dict['topic']}'")
        logger.info(f"  Search query: '{brief_dict.get('search_query', 'N/A')}'")
        logger.info(f"  Brief saved: {brief_path}")
    except Exception as e:
        logger.warning(f"[Phase 1] Could not save brief (non-fatal): {e}")

    return brief_dict


# ─── Phase 2: Video Sourcing ─────────────────────────────────────────────────

def run_phase2(
    search_query: str,
    topic: str,
    download_dir: str,
) -> dict:
    """
    Phase 2 — Find, download, and validate a source video.

    On failure, raises RuntimeError (caller handles retry/fallback).

    Returns:
        dict with keys: video_path, metadata_path, title, channel, url, etc.
    """
    from autonomous_trend_agent.sourcing.video_sourcing import VideoSourcing

    logger.info("=" * 60)
    logger.info("PHASE 2 — VIDEO SOURCING")
    logger.info("=" * 60)
    logger.info(f"  Query: '{search_query}'")
    logger.info(f"  Topic: '{topic}'")

    # Pre-check: disk space
    if not _check_disk_space(download_dir, min_gb=3.0):
        raise RuntimeError("Insufficient disk space for video download.")

    sourcing = VideoSourcing(download_dir=download_dir)
    sourced = sourcing.run(search_query=search_query, topic=topic)

    sourced_dict = asdict(sourced)
    logger.info(f"Phase 2 complete → '{sourced.title}'")
    logger.info(f"  Path: {sourced.video_path}")
    logger.info(f"  Resolution: {sourced.resolution} @ {sourced.fps}fps")
    logger.info(f"  Duration: {sourced.duration_sec}s")

    return sourced_dict


# ─── Phase 3+4+5: Analysis → Editing → Output + QA ───────────────────────────

def run_phase345(
    video_path: str,
    output_dir: str,
    topic: str = "",
    num_shorts: int = 4,
    caption_style: str = "tiktok",
) -> dict:
    """
    Phases 3-5 — Analyze, edit, and export shorts.

    Uses the v2.1 LangGraph pipeline (langgraph_pipeline.py) which handles:
        - Transcription (easytranscriber)
        - Visual Analysis (Qwen3.5-0.8B via vLLM sidecar)
        - Director (PydanticAI + Gemini 3 Pro)
        - Subject Tracking (YOLO26s-Pose)
        - Editing (ZeroCopy pipeline)
        - QA + Export

    Handles:
        - GPU OOM → reduces batch size and retries
        - Missing dependencies → graceful degradation

    Returns:
        dict with pipeline result (success, shorts_created, duration, errors)
    """
    import asyncio
    from autonomous_trend_agent.pipeline.langgraph_pipeline import (
        run_pipeline, PipelineConfig,
    )

    logger.info("=" * 60)
    logger.info("PHASES 3-5 — ANALYSIS → EDITING → OUTPUT (LangGraph v2.1)")
    logger.info("=" * 60)
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Output: {output_dir}")

    # Verify video exists + is non-zero
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if vp.stat().st_size < 1_000_000:  # < 1MB = corrupt
        raise ValueError(f"Video file too small ({vp.stat().st_size} bytes) — likely corrupt: {video_path}")

    # Pre-check disk space for rendering
    if not _check_disk_space(output_dir, min_gb=5.0):
        raise RuntimeError("Insufficient disk space for video rendering.")

    batch_size = 50
    max_oom_retries = 3

    for oom_attempt in range(max_oom_retries):
        try:
            config = PipelineConfig(
                video_path=video_path,
                output_dir=output_dir,
                trend_topic=topic,
                num_shorts=num_shorts,
                caption_style=caption_style,
                batch_size=batch_size,
                use_gpu_reframing=True,
                use_yolo_tracking=True,
                use_qwen_analysis=True,
                use_captions=True,
                use_boredom_detector=True,
            )

            # Run the async LangGraph pipeline
            result = asyncio.run(
                run_pipeline(
                    video_path=video_path,
                    config=config,
                    use_redis=False,  # Use memory saver for now
                )
            )
            _force_vram_cleanup()

            # Extract results from the LangGraph state dict
            shorts_created = result.get("shorts", [])
            errors = result.get("errors", [])
            elapsed = result.get("elapsed_time", 0)

            result_dict = {
                "success": len(shorts_created) > 0 and not errors,
                "shorts_created": shorts_created,
                "output_dir": output_dir,
                "duration_seconds": elapsed,
                "errors": errors,
            }

            if result_dict["success"]:
                logger.info(f"Phases 3-5 complete → {len(shorts_created)} shorts generated")
                logger.info(f"  Output dir: {output_dir}")
                logger.info(f"  Duration: {elapsed:.1f}s")
            else:
                logger.warning(f"Pipeline completed with errors: {errors}")

            return result_dict

        except (RuntimeError, Exception) as e:
            err_str = str(e).lower()
            is_oom = any(k in err_str for k in [
                "out of memory", "cuda oom", "cublas", "alloc",
                "cudnn", "not enough memory"
            ])

            if is_oom and oom_attempt < max_oom_retries - 1:
                batch_size = max(5, batch_size // 2)
                logger.warning(
                    f"[OOM] GPU memory exhausted. Retrying with batch_size={batch_size} "
                    f"(attempt {oom_attempt + 2}/{max_oom_retries})"
                )
                _force_vram_cleanup()
                continue
            else:
                raise  # Non-OOM error or final OOM attempt — re-raise

    # Should never reach here, but safety
    raise RuntimeError("Pipeline failed after all OOM retries.")


# ─── Full End-to-End Runner ───────────────────────────────────────────────────

def run_full_pipeline(
    video_path: Optional[str] = None,
    topic: Optional[str] = None,
    search_query: Optional[str] = None,
    output_dir: str = "./output",
    download_dir: str = "./downloaded_videos",
    num_shorts: int = 4,
    caption_style: str = "tiktok",
    resume_dir: Optional[str] = None,
) -> dict:
    """
    Run the complete autonomous pipeline end-to-end.

    Entry modes:
        1. No args       → Phase 1+2+3+4+5 (fully autonomous)
        2. --topic        → Phase 2+3+4+5  (skip discovery, use given topic)
        3. --video        → Phase 3+4+5    (skip discovery + sourcing)

    Error handling:
        - Phase 1 fails → uses fallback trending topics
        - Phase 2 fails → retries with broadened query, then simpler fallback
        - Phase 3-5 OOM → halves batch size and retries up to 3x
        - Ctrl+C        → finishes current phase then exits cleanly

    Returns:
        dict with full pipeline results and metadata
    """
    start = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(resume_dir) if resume_dir else Path(output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Ensure download dir exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  AUTONOMOUS TREND AGENT — FULL PIPELINE                  ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {run_dir}")

    pipeline_meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "phases_completed": [],
        "errors": [],
        "warnings": [],
    }

    # ── Check for resumable state (H1.3) ──────────────────────────────
    resume = _find_resume_state(run_dir) if resume_dir else {"phase": 0}

    if resume["phase"] >= 1 and resume.get("content_brief") and not topic:
        logger.info("[Resume] Using Phase 1 output from previous run.")
        topic = resume["content_brief"].get("topic")
        search_query = resume["content_brief"].get("search_query")
        pipeline_meta["phase1"] = resume["content_brief"]
        pipeline_meta["phases_completed"].append("trend_discovery")

    if resume["phase"] >= 2 and resume.get("video_path") and not video_path:
        logger.info("[Resume] Using Phase 2 video from previous run.")
        video_path = resume["video_path"]
        pipeline_meta["phases_completed"].append("video_sourcing")

    try:
        # ── Phase 1: Trend Discovery ──────────────────────────────────
        _check_shutdown()

        if video_path is None and topic is None:
            brief = run_phase1(run_dir)
            topic = brief["topic"]
            search_query = brief["search_query"]
            pipeline_meta["phase1"] = brief
            pipeline_meta["phases_completed"].append("trend_discovery")
            _force_vram_cleanup()

        elif topic and not search_query:
            search_query = f"{topic} analysis deep dive 2026"
            logger.info(f"Using provided topic: '{topic}'")
            logger.info(f"Auto-generated search query: '{search_query}'")

        # ── Phase 2: Video Sourcing ───────────────────────────────────
        _check_shutdown()

        if video_path is None:
            if not search_query:
                raise RuntimeError(
                    "No search_query available. Provide --video, --topic, or let Phase 1 run."
                )

            # Try primary query; if fails, try broader/fallback queries
            phase2_queries = [
                search_query,
                f"{topic} explained" if topic else None,
                f"{topic} full video" if topic else None,
            ]
            phase2_queries = [q for q in phase2_queries if q]

            for qi, query in enumerate(phase2_queries):
                try:
                    logger.info(
                        f"[Phase 2] Attempt {qi + 1}/{len(phase2_queries)}: '{query}'"
                    )
                    sourced = run_phase2(
                        search_query=query,
                        topic=topic or "trending",
                        download_dir=download_dir,
                    )
                    video_path = sourced["video_path"]
                    pipeline_meta["phase2"] = sourced
                    pipeline_meta["phases_completed"].append("video_sourcing")
                    break  # Success!

                except Exception as e:
                    logger.warning(f"[Phase 2] Query '{query}' failed: {e}")
                    if qi == len(phase2_queries) - 1:
                        # All queries exhausted
                        raise RuntimeError(
                            f"Phase 2 failed with all query variants. "
                            f"Last error: {e}"
                        ) from e
                    _force_vram_cleanup()

        else:
            logger.info(f"Skipping Phases 1-2: using provided video: {video_path}")
            # Validate provided video exists
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Provided video not found: {video_path}")

        # ── Phases 3-5: Analysis → Editing → Output ──────────────────
        _check_shutdown()
        _force_vram_cleanup()  # Clean slate before heavy GPU work

        result = run_phase345(
            video_path=video_path,
            output_dir=str(run_dir),
            topic=topic or "",
            num_shorts=num_shorts,
            caption_style=caption_style,
        )
        pipeline_meta["phase345"] = result
        pipeline_meta["phases_completed"].extend(["analysis", "editing", "output_qa"])

    except KeyboardInterrupt:
        logger.warning("[Signal] Pipeline interrupted by user.")
        pipeline_meta["errors"].append("Pipeline interrupted by user (Ctrl+C)")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        pipeline_meta["errors"].append(str(e))

    # ── Save run metadata ─────────────────────────────────────────────
    elapsed = time.time() - start
    pipeline_meta["completed_at"] = datetime.now().isoformat()
    pipeline_meta["total_duration_seconds"] = round(elapsed, 1)

    meta_path = run_dir / "pipeline_run.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_meta, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Could not save run metadata: {e}")

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    if not pipeline_meta["errors"]:
        shorts_count = 0
        if "phase345" in pipeline_meta:
            shorts_count = len(pipeline_meta["phase345"].get("shorts_created", []))
        logger.info(f"║  ✅ PIPELINE COMPLETE — {shorts_count} shorts generated            ║")
    else:
        logger.info("║  ❌ PIPELINE FINISHED WITH ERRORS                         ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"  Phases: {' → '.join(pipeline_meta['phases_completed'])}")
    logger.info(f"  Run metadata: {meta_path}")
    if pipeline_meta["errors"]:
        for err in pipeline_meta["errors"]:
            logger.error(f"  Error: {err}")

    return pipeline_meta


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Trend Agent — End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully autonomous — discover trend, find video, create shorts
  python -m autonomous_trend_agent.run

  # Process a specific video
  python -m autonomous_trend_agent.run --video /path/to/video.mp4

  # Discover video for a specific topic
  python -m autonomous_trend_agent.run --topic "AI agents explained"

  # Resume a failed run
  python -m autonomous_trend_agent.run --resume ./output/run_20260228_191244

  # Custom output directory
  python -m autonomous_trend_agent.run --output ./my_shorts
        """,
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a local video file (skips Phase 1+2)",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Trend topic to search for (skips Phase 1, runs Phase 2+)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Specific YouTube search query (skips Phase 1)",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--download-dir", type=str, default="./downloaded_videos",
        help="Directory for downloaded videos (default: ./downloaded_videos)",
    )
    parser.add_argument(
        "--shorts", type=int, default=4,
        help="Number of shorts to generate (default: 4)",
    )
    parser.add_argument(
        "--caption-style", type=str, default="tiktok",
        choices=["tiktok", "hormozi", "mrbeast", "minimal", "neon"],
        help="Caption animation style (default: tiktok)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume a previous run from its output directory",
    )

    args = parser.parse_args()

    # Load environment — resolve .env relative to this file's directory
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=True)
    logger.info(f"Loaded .env from: {env_path} (exists={env_path.exists()})")

    result = run_full_pipeline(
        video_path=args.video,
        topic=args.topic,
        search_query=args.query,
        output_dir=args.output,
        download_dir=args.download_dir,
        num_shorts=args.shorts,
        caption_style=args.caption_style,
        resume_dir=args.resume,
    )

    sys.exit(0 if not result.get("errors") else 1)


if __name__ == "__main__":
    main()
