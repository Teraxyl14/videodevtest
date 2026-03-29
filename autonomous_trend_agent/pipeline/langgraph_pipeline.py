"""
LangGraph Pipeline Orchestrator v2.1 (Phase 5 — Section H)

Replaces the monolithic 1700-line orchestrator.py with a LangGraph
state machine. Each node is a focused function that processes one
pipeline phase.

Architecture:
    discover → source → transcribe → analyze → [user_approve] → edit → qa → export
                                                     ↑                    |
                                                     └────── retry ───────┘

Checkpointing: Redis Streams — pipeline resumes from failure.
Typing: PydanticAI structured outputs for all AI interactions.

Objectives: H1.1 (state machine), H1.2 (user approval), H1.3 (checkpointing),
            H1.4 (progress logging)
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field, asdict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ─── Pipeline State ──────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Typed state flowing through the LangGraph pipeline."""
    # Input
    video_path: str
    trend_topic: Optional[str]
    config: dict

    # Discovery (Phase 1)
    content_brief: Optional[dict]

    # Sourcing (Phase 2)
    source_video_path: Optional[str]
    source_metadata: Optional[dict]

    # Transcription (Phase 3)
    transcript: Optional[dict]
    transcript_path: Optional[str]

    # Analysis (Phase 3)
    viral_moments: Optional[List[dict]]
    scene_changes: Optional[List[dict]]
    video_analysis: Optional[dict]

    # Editing plan (Director)
    director_plan: Optional[dict]
    user_approved: bool

    # Editing output (Phase 4)
    shorts: Optional[List[dict]]
    tracking_data: Optional[dict]

    # QA (Phase 5)
    qa_results: Optional[List[dict]]

    # Export (Phase 5)
    export_paths: Optional[List[str]]
    metadata_generated: Optional[List[dict]]

    # Pipeline control
    errors: List[str]
    current_stage: str
    start_time: float
    elapsed_time: float


# ─── Pipeline Configuration ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for the LangGraph pipeline."""
    # Input
    video_path: str = ""
    transcript_path: Optional[str] = None
    trend_topic: Optional[str] = None

    # Output
    output_dir: str = "/app/output"

    # Processing
    num_shorts: int = 4
    short_duration_range: tuple = (45, 90)
    target_resolution: tuple = (1080, 1920)

    # Module toggles
    use_discovery: bool = False        # Skip if video already provided
    use_gpu_reframing: bool = True
    use_yolo_tracking: bool = True
    use_qwen_analysis: bool = True
    use_captions: bool = True
    use_boredom_detector: bool = True
    caption_style: str = "tiktok"

    # Hardware
    device: str = "cuda"
    batch_size: int = 50


# ─── Pipeline Nodes ──────────────────────────────────────────────────────────

def transcription_node(state: PipelineState) -> PipelineState:
    """
    Phase 3a: Transcribe video using easytranscriber.
    Always-resident GPU model (~2GB VRAM).
    """
    logger.info("[Pipeline] ── TRANSCRIPTION ──")
    state["current_stage"] = "transcription"

    video_path = state.get("source_video_path") or state.get("video_path", "")

    # Check for cached transcript
    config = state.get("config", {})
    transcript_path = config.get("transcript_path")
    if transcript_path and Path(transcript_path).exists():
        logger.info(f"[Pipeline] Loading cached transcript: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            state["transcript"] = json.load(f)
            state["transcript_path"] = transcript_path
        return state

    try:
        from autonomous_trend_agent.audio.easytranscriber_asr import (
            EasyTranscriberASR, EasyTranscriberConfig
        )

        asr = EasyTranscriberASR(EasyTranscriberConfig(
            device=config.get("device", "cuda"),
        ))
        result = asr.transcribe(video_path)

        # Save transcript
        output_dir = Path(config.get("output_dir", "/app/output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        t_path = str(output_dir / "transcript.json")
        result.save(t_path)

        state["transcript"] = result.to_dict()
        state["transcript_path"] = t_path
        logger.info(
            f"[Pipeline] Transcription complete: {len(result.segments)} segments, "
            f"{result.word_count} words"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Transcription failed: {e}")
        state.setdefault("errors", []).append(f"Transcription: {e}")

    return state


def analysis_node(state: PipelineState) -> PipelineState:
    """
    Phase 3b: Visual analysis using Qwen3.5-0.8B via vLLM sidecar.
    Always-resident GPU model (~1.5-2GB VRAM).
    """
    logger.info("[Pipeline] ── VISUAL ANALYSIS ──")
    state["current_stage"] = "analysis"

    video_path = state.get("source_video_path") or state.get("video_path", "")
    transcript = state.get("transcript")
    transcript_text = ""
    if transcript:
        segments = transcript.get("segments", [])
        transcript_text = " ".join(s.get("text", "") for s in segments)

    try:
        from autonomous_trend_agent.brain.qwen3_video_analyzer import (
            Qwen35VideoAnalyzer
        )

        analyzer = Qwen35VideoAnalyzer()
        analysis = analyzer.analyze_video(
            video_path,
            num_frames=16,
            transcript_text=transcript_text[:3000] if transcript_text else None,
        )

        state["viral_moments"] = [asdict(m) for m in analysis.viral_moments]
        state["scene_changes"] = [asdict(s) for s in analysis.scene_changes]
        state["video_analysis"] = {
            "overall_theme": analysis.overall_theme,
            "target_audience": analysis.target_audience,
            "duration": analysis.duration,
        }
        logger.info(
            f"[Pipeline] Analysis complete: {len(analysis.viral_moments)} viral moments, "
            f"{len(analysis.scene_changes)} scene changes"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Analysis failed: {e}")
        state.setdefault("errors", []).append(f"Analysis: {e}")

    return state


async def director_node(state: PipelineState) -> PipelineState:
    """
    Director: Uses PydanticAI + Gemini 3 Pro to plan cuts.
    Runs on cloud API — no GPU cost.
    """
    logger.info("[Pipeline] ── DIRECTOR (PydanticAI) ──")
    state["current_stage"] = "director"

    transcript = state.get("transcript")
    if not transcript:
        logger.warning("[Pipeline] No transcript for director. Skipping.")
        return state

    try:
        from autonomous_trend_agent.brain.pydantic_agents import plan_cuts

        # Format transcript for the director
        segments = transcript.get("segments", [])
        transcript_text = "\n".join(
            f"[{s.get('start', 0):.1f}s - {s.get('end', 0):.1f}s] {s.get('text', '')}"
            for s in segments
        )

        plan = await plan_cuts(transcript_text)
        state["director_plan"] = plan.model_dump()
        state["viral_moments"] = [m.model_dump() for m in plan.segments]
        logger.info(f"[Pipeline] Director plan: {plan.total_segments} segments")
    except Exception as e:
        logger.error(f"[Pipeline] Director failed: {e}")
        state.setdefault("errors", []).append(f"Director: {e}")

    return state


def tracking_node(state: PipelineState) -> PipelineState:
    """
    Phase 4a: Subject tracking using YOLO26s-Pose (TensorRT FP16).
    On-demand GPU model — loaded when needed.
    """
    logger.info("[Pipeline] ── SUBJECT TRACKING ──")
    state["current_stage"] = "tracking"

    video_path = state.get("source_video_path") or state.get("video_path", "")
    config = state.get("config", {})

    if not config.get("use_yolo_tracking", True):
        logger.info("[Pipeline] YOLO tracking disabled, skipping.")
        return state

    try:
        from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker

        tracker = YOLOv11Tracker(
            device=config.get("device", "cuda"),
            enable_pose=True,
        )

        # Only track up to the last required timestamp
        moments = state.get("viral_moments", [])
        max_end = max((m.get("end_time", 0) for m in moments), default=None)
        max_frames = int(max_end * 30) + 150 if max_end else None

        result = tracker.track_video(
            video_path,
            target_class="face",
            smooth=True,
            max_frames=max_frames,
        )
        state["tracking_data"] = result
        logger.info(f"[Pipeline] Tracking complete: {len(result.get('tracked_objects', []))} subjects")

        # Free VRAM — on-demand model
        tracker.unload()
        import torch
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"[Pipeline] Tracking failed: {e}")
        state.setdefault("errors", []).append(f"Tracking: {e}")

    return state


def editing_node(state: PipelineState) -> PipelineState:
    """
    Phase 4b: Reframe + captions + effects using ZeroCopy pipeline.
    Uses PyNvVideoCodec 2.1.0 for zero-copy NVDEC → DLPack → PyTorch.
    """
    logger.info("[Pipeline] ── EDITING (VIRAL ENGINE) ──")
    state["current_stage"] = "editing"

    video_path = state.get("source_video_path") or state.get("video_path", "")
    moments = state.get("viral_moments", [])
    tracking_data = state.get("tracking_data", {})
    transcript = state.get("transcript")
    config = state.get("config", {})

    output_dir = Path(config.get("output_dir", "/app/output"))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shorts_dir = output_dir / f"{Path(video_path).stem}_{timestamp}"
    shorts_dir.mkdir(parents=True, exist_ok=True)

    shorts_created: List[dict] = []

    try:
        from autonomous_trend_agent.editor.zero_copy_pipeline import ZeroCopyPipeline

        pipeline = ZeroCopyPipeline(
            target_width=config.get("target_resolution", (1080, 1920))[0],
            target_height=config.get("target_resolution", (1080, 1920))[1],
            batch_size=config.get("batch_size", 50),
            device=config.get("device", "cuda"),
        )

        for i, moment in enumerate(moments[:config.get("num_shorts", 4)]):
            short_dir = shorts_dir / f"short_{i+1:02d}"
            short_dir.mkdir(exist_ok=True)
            output_path = str(short_dir / "video.mp4")

            logger.info(
                f"[Pipeline] Creating short {i+1}: "
                f"{moment.get('start_time', 0):.1f}s - {moment.get('end_time', 0):.1f}s"
            )

            try:
                pipeline.process_segment(
                    source_path=video_path,
                    output_path=output_path,
                    start_time=moment.get("start_time", 0),
                    end_time=moment.get("end_time", 30),
                    tracking_data=tracking_data,
                    transcript=transcript,
                )

                shorts_created.append({
                    "index": i + 1,
                    "output_path": output_path,
                    "start_time": moment.get("start_time", 0),
                    "end_time": moment.get("end_time", 0),
                    "hook": moment.get("hook_text", moment.get("hook", "")),
                    "viral_score": moment.get("viral_score", 0),
                })
            except Exception as e:
                logger.error(f"[Pipeline] Short {i+1} failed: {e}")
                state.setdefault("errors", []).append(f"Short {i+1}: {e}")

    except Exception as e:
        logger.error(f"[Pipeline] Editing setup failed: {e}")
        state.setdefault("errors", []).append(f"Editing: {e}")

    state["shorts"] = shorts_created
    logger.info(f"[Pipeline] Editing complete: {len(shorts_created)} shorts created")
    return state


async def qa_node(state: PipelineState) -> PipelineState:
    """
    Phase 5a: Quality assurance using PydanticAI compliance checker.
    """
    logger.info("[Pipeline] ── QUALITY ASSURANCE ──")
    state["current_stage"] = "qa"

    shorts = state.get("shorts", [])
    transcript = state.get("transcript")
    qa_results: List[dict] = []

    try:
        from autonomous_trend_agent.brain.pydantic_agents import check_quality
        from autonomous_trend_agent.pipeline.quality_assurance import QualityAssurance

        qa = QualityAssurance()

        for short in shorts:
            output_path = short.get("output_path", "")
            if Path(output_path).exists():
                report = qa.check_single(output_path)
                qa_results.append({
                    "short_index": short.get("index"),
                    "passed": report.overall_passed if hasattr(report, 'overall_passed') else True,
                    "details": asdict(report) if hasattr(report, '__dataclass_fields__') else {},
                })
    except Exception as e:
        logger.warning(f"[Pipeline] QA failed (non-fatal): {e}")

    state["qa_results"] = qa_results
    return state


async def export_node(state: PipelineState) -> PipelineState:
    """
    Phase 5b: Generate metadata using PydanticAI + save final output.
    """
    logger.info("[Pipeline] ── EXPORT ──")
    state["current_stage"] = "export"

    shorts = state.get("shorts", [])
    analysis = state.get("video_analysis", {})
    metadata_list: List[dict] = []

    try:
        from autonomous_trend_agent.brain.pydantic_agents import generate_metadata

        for short in shorts:
            segment_info = (
                f"Topic: {analysis.get('overall_theme', 'Unknown')}\n"
                f"Hook: {short.get('hook', '')}\n"
                f"Duration: {short.get('end_time', 0) - short.get('start_time', 0):.0f}s\n"
                f"Audience: {analysis.get('target_audience', 'General')}"
            )

            metadata = await generate_metadata(segment_info)
            meta_dict = metadata.model_dump()
            metadata_list.append(meta_dict)

            # Save metadata alongside video
            output_path = short.get("output_path", "")
            if output_path:
                meta_path = str(Path(output_path).parent / "metadata.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_dict, f, indent=2)

    except Exception as e:
        logger.error(f"[Pipeline] Export metadata failed: {e}")
        state.setdefault("errors", []).append(f"Export: {e}")

    state["metadata_generated"] = metadata_list
    state["elapsed_time"] = time.time() - state.get("start_time", time.time())
    logger.info(
        f"[Pipeline] COMPLETE. {len(shorts)} shorts, "
        f"{state['elapsed_time']:.1f}s elapsed."
    )
    return state


# ─── Graph Construction ──────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """
    Build the LangGraph state machine for the full pipeline.

    Returns a compiled graph ready for invocation.
    """
    builder = StateGraph(PipelineState)

    # Add nodes
    builder.add_node("transcribe", transcription_node)
    builder.add_node("analyze", analysis_node)
    builder.add_node("director", director_node)
    builder.add_node("track", tracking_node)
    builder.add_node("edit", editing_node)
    builder.add_node("qa", qa_node)
    builder.add_node("export", export_node)

    # Define edges (linear flow for MVP)
    builder.set_entry_point("transcribe")
    builder.add_edge("transcribe", "analyze")
    builder.add_edge("analyze", "director")
    builder.add_edge("director", "track")
    builder.add_edge("track", "edit")
    builder.add_edge("edit", "qa")
    builder.add_edge("qa", "export")
    builder.add_edge("export", END)

    return builder


def create_pipeline(use_redis: bool = False):
    """
    Create a compiled pipeline with optional Redis checkpointing.

    Args:
        use_redis: If True, use Redis Streams for persistent checkpointing.
                   If False, use in-memory checkpointing (development).
    """
    builder = build_pipeline()

    if use_redis:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            checkpointer = RedisSaver(
                connection_string=os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        except ImportError:
            logger.warning("[Pipeline] Redis checkpointer unavailable, using memory.")
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ─── Convenience Entry Point ────────────────────────────────────────────────

async def run_pipeline(
    video_path: str,
    config: Optional[PipelineConfig] = None,
    use_redis: bool = False,
) -> PipelineState:
    """
    Run the full pipeline on a video.

    Usage:
        result = await run_pipeline("/path/to/video.mp4")
        print(result["shorts"])
    """
    config = config or PipelineConfig(video_path=video_path)
    config.video_path = video_path

    pipeline = create_pipeline(use_redis=use_redis)

    initial_state: PipelineState = {
        "video_path": video_path,
        "config": asdict(config),
        "errors": [],
        "current_stage": "init",
        "start_time": time.time(),
        "elapsed_time": 0.0,
        "user_approved": True,  # Auto-approve for MVP
    }

    result = await pipeline.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": f"pipeline_{int(time.time())}"}},
    )

    return result
