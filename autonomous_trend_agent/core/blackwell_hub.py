"""
BlackwellHub — CPU-Only Orchestration FSM
==========================================
The central controller for the Blackwell-Triad Hub & Spoke pattern.
Manages sequential model execution on a single 16GB RTX 5080.

This process NEVER imports torch.cuda or initializes any GPU resources.
It orchestrates ephemeral Spoke processes via multiprocessing.

FSM States: IDLE → ASR → HANDOFF → YOLO → HANDOFF → QWEN → DONE

References:
    - GPU VRAM Orchestration and IPC.txt, Section 4 & 8
    - Objective A3.1-A3.3 (VRAM Management)
"""

import multiprocessing as mp
import time
import logging
import json
from enum import Enum, auto
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger("BlackwellHub")


class PipelineState(Enum):
    """Finite State Machine states for the pipeline."""
    IDLE = auto()
    ASR_RUNNING = auto()
    ASR_HANDOFF = auto()
    YOLO_RUNNING = auto()
    YOLO_HANDOFF = auto()
    QWEN_RUNNING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class SpokeResult:
    """Result from an ephemeral spoke process."""
    spoke_name: str
    success: bool
    exit_code: int
    duration_s: float
    data: Optional[Any] = None
    error: Optional[str] = None


class BlackwellHub:
    """
    CPU-only orchestrator implementing the Blackwell-Triad Hub pattern.

    Architecture:
        - Hub (this): Lightweight CPU process. Zero VRAM footprint.
        - Locker: Persistent background process holding shared CUDA buffer (~33MB).
        - Spokes: Ephemeral GPU processes (ASR, YOLO, Qwen). One at a time.

    The Hub guarantees:
        1. Only ONE spoke runs at any time (sequential VRAM usage).
        2. Spokes die after completing work (OS-level VRAM reclamation).
        3. Data persists between spokes via the Locker's IPC buffer.
        4. OOM failures trigger automatic retry with reduced parameters.
    """

    # Maximum retries for a spoke that crashes (OOM or other GPU error)
    MAX_RETRIES = 2

    # Timeout per spoke (seconds)
    SPOKE_TIMEOUTS = {
        "asr": 300,    # 5 min for long audio
        "yolo": 120,   # 2 min for detection pass
        "qwen": 180,   # 3 min for VLM reasoning
    }

    def __init__(self, models_dir: str = "./models", output_dir: str = "./output"):
        """
        Initialize the Hub. Does NOT touch the GPU.

        Args:
            models_dir: Path to model weights directory
            output_dir: Path to output directory
        """
        # CRITICAL: Use 'spawn' context for clean CUDA context per process
        self.ctx = mp.get_context('spawn')
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.state = PipelineState.IDLE

        # Locker state
        self.locker_process: Optional[mp.Process] = None
        self.locker_ipc_handle = None

        # Results from each stage
        self.results: Dict[str, SpokeResult] = {}

        logger.info("BlackwellHub initialized (CPU-only, zero VRAM)")

    # =========================================================================
    # Locker Management
    # =========================================================================

    def start_locker(self):
        """
        Start the persistent VRAM Locker process.
        Allocates a 4K RGBA buffer (~33MB) and shares the IPC handle.
        """
        from autonomous_trend_agent.core.vram_locker import VRAMLocker

        handle_queue = self.ctx.Queue()
        metadata_queue = self.ctx.Queue()

        locker = VRAMLocker(
            handle_queue=handle_queue,
            metadata_queue=metadata_queue,
            buffer_shape=(2160, 3840, 4)  # 4K RGBA
        )
        self.locker_process = self.ctx.Process(
            target=locker.run,
            name="VRAM-Locker",
            daemon=True
        )
        self.locker_process.start()

        # Wait for IPC handle (timeout 10s)
        try:
            self.locker_ipc_handle = handle_queue.get(timeout=10)
            if self.locker_ipc_handle is None:
                raise RuntimeError("Locker failed to allocate CUDA buffer")
            logger.info("VRAM Locker started. IPC handle received.")
        except Exception as e:
            logger.error(f"Failed to start VRAM Locker: {e}")
            self.shutdown_locker()
            raise

    def shutdown_locker(self):
        """Terminate the Locker process, freeing VRAM."""
        if self.locker_process and self.locker_process.is_alive():
            self.locker_process.terminate()
            self.locker_process.join(timeout=5)
            logger.info("VRAM Locker terminated.")
        self.locker_process = None
        self.locker_ipc_handle = None

    # =========================================================================
    # Spoke Execution
    # =========================================================================

    def _run_spoke(
        self,
        spoke_name: str,
        target_fn,
        args: tuple,
        timeout: Optional[int] = None,
        retry_args: Optional[tuple] = None
    ) -> SpokeResult:
        """
        Spawn an ephemeral spoke process and wait for completion.

        Implements OOM Watchdog: if the spoke crashes, retry with fallback args.

        Args:
            spoke_name: Human-readable name (e.g., "asr", "yolo", "qwen")
            target_fn: The function to run in the spoke process
            args: Arguments to pass to target_fn
            timeout: Max seconds to wait (None = use default)
            retry_args: Fallback args for OOM retry (e.g., reduced context)

        Returns:
            SpokeResult with success status and data
        """
        if timeout is None:
            timeout = self.SPOKE_TIMEOUTS.get(spoke_name, 120)

        result_queue = self.ctx.Queue()
        attempts = 0

        while attempts <= self.MAX_RETRIES:
            current_args = retry_args if (attempts > 0 and retry_args) else args
            full_args = current_args + (result_queue,)

            logger.info(
                f"[{spoke_name.upper()}] Spawning spoke "
                f"(attempt {attempts + 1}/{self.MAX_RETRIES + 1})"
            )
            start_time = time.time()

            process = self.ctx.Process(
                target=target_fn,
                args=full_args,
                name=f"Spoke-{spoke_name}"
            )
            process.start()
            process.join(timeout=timeout)

            duration = time.time() - start_time

            # Check if process is still alive (timeout)
            if process.is_alive():
                logger.warning(f"[{spoke_name.upper()}] Timed out after {timeout}s. Killing.")
                process.kill()
                process.join(timeout=5)
                return SpokeResult(
                    spoke_name=spoke_name,
                    success=False,
                    exit_code=-9,
                    duration_s=duration,
                    error=f"Timeout after {timeout}s"
                )

            exit_code = process.exitcode

            if exit_code == 0:
                # Success: retrieve data from queue
                data = None
                if not result_queue.empty():
                    try:
                        data = result_queue.get_nowait()
                    except Exception:
                        pass

                result = SpokeResult(
                    spoke_name=spoke_name,
                    success=True,
                    exit_code=0,
                    duration_s=duration,
                    data=data
                )
                logger.info(
                    f"[{spoke_name.upper()}] Completed in {duration:.1f}s. "
                    f"VRAM reclaimed via process termination."
                )
                return result

            # Non-zero exit: likely OOM or CUDA error
            logger.warning(
                f"[{spoke_name.upper()}] Failed with exit code {exit_code} "
                f"(attempt {attempts + 1})"
            )
            attempts += 1

            if attempts <= self.MAX_RETRIES and retry_args:
                logger.info(f"[{spoke_name.upper()}] Retrying with reduced parameters...")
                time.sleep(1)  # Brief pause for VRAM to clear

        return SpokeResult(
            spoke_name=spoke_name,
            success=False,
            exit_code=exit_code,
            duration_s=duration,
            error=f"Failed after {attempts} attempts (exit code: {exit_code})"
        )

    # =========================================================================
    # Pipeline Execution
    # =========================================================================

    def process_pipeline(
        self,
        audio_path: str,
        video_path: str,
        prompt: str = "Describe what you see in this video frame.",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Run the full sequential pipeline: ASR → YOLO → Qwen.

        Args:
            audio_path: Path to audio file (WAV/MP3)
            video_path: Path to video file (MP4/MKV)
            prompt: Text prompt for Qwen VLM reasoning
            progress_callback: Optional fn(state, message) for progress updates

        Returns:
            Dict with transcript, detections, and reasoning results
        """
        from autonomous_trend_agent.spokes.audio_spoke import run_asr_spoke
        from autonomous_trend_agent.spokes.video_spoke import (
            run_yolo_spoke, run_qwen_spoke
        )

        pipeline_start = time.time()

        def _report(state, msg):
            self.state = state
            logger.info(f"[FSM → {state.name}] {msg}")
            if progress_callback:
                progress_callback(state.name, msg)

        try:
            # ---- Phase 1: ASR (Ears) ----
            _report(PipelineState.ASR_RUNNING, f"Transcribing: {audio_path}")

            asr_result = self._run_spoke(
                spoke_name="asr",
                target_fn=run_asr_spoke,
                args=(audio_path, str(self.models_dir))
            )
            self.results["asr"] = asr_result

            if not asr_result.success:
                _report(PipelineState.ERROR, f"ASR failed: {asr_result.error}")
                return {"error": asr_result.error}

            transcript = asr_result.data
            _report(PipelineState.ASR_HANDOFF, "ASR complete. VRAM freed.")

            # ---- Phase 2: YOLO (Eyes) ----
            _report(PipelineState.YOLO_RUNNING, f"Detecting objects: {video_path}")

            yolo_result = self._run_spoke(
                spoke_name="yolo",
                target_fn=run_yolo_spoke,
                args=(video_path, self.locker_ipc_handle)
            )
            self.results["yolo"] = yolo_result

            if not yolo_result.success:
                _report(PipelineState.ERROR, f"YOLO failed: {yolo_result.error}")
                return {"error": yolo_result.error, "transcript": transcript}

            detections = yolo_result.data
            _report(PipelineState.YOLO_HANDOFF, "YOLO complete. Frame in Locker.")

            # ---- Phase 3: Qwen (Brain) ----
            _report(PipelineState.QWEN_RUNNING, "Running VLM reasoning...")

            # Build prompt with transcript context
            full_prompt = (
                f"Transcript: {transcript}\n\n"
                f"Task: {prompt}"
            )

            qwen_result = self._run_spoke(
                spoke_name="qwen",
                target_fn=run_qwen_spoke,
                args=(full_prompt, self.locker_ipc_handle),
                # OOM retry: reduce max model length
                retry_args=(full_prompt, self.locker_ipc_handle, 2048)
            )
            self.results["qwen"] = qwen_result

            if not qwen_result.success:
                _report(PipelineState.ERROR, f"Qwen failed: {qwen_result.error}")
                return {
                    "error": qwen_result.error,
                    "transcript": transcript,
                    "detections": detections
                }

            reasoning = qwen_result.data

            # ---- Done ----
            total_time = time.time() - pipeline_start
            _report(PipelineState.DONE, f"Pipeline complete in {total_time:.1f}s")

            return {
                "transcript": transcript,
                "detections": detections,
                "reasoning": reasoning,
                "timing": {
                    "asr_s": asr_result.duration_s,
                    "yolo_s": yolo_result.duration_s,
                    "qwen_s": qwen_result.duration_s,
                    "total_s": total_time,
                }
            }

        except Exception as e:
            _report(PipelineState.ERROR, f"Pipeline crashed: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def shutdown(self):
        """Clean shutdown of all resources."""
        self.shutdown_locker()
        self.state = PipelineState.IDLE
        logger.info("BlackwellHub shut down.")

    def __enter__(self):
        self.start_locker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
