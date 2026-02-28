"""
WebSocket Progress Callback (Hardened)
Bridges the PipelineOrchestrator's callback system to WebSocket clients.
Every on_stage_start/on_step/on_stage_complete/on_error call is forwarded
as a JSON event to all connected WebSocket clients in real-time.

Crash protections:
  - Thread-safe broadcasting with Lock
  - Dead client auto-removal on send failure
  - Exception-safe event loop interaction
  - Graceful degradation if no clients connected
  - Capped log buffer to prevent memory leak
"""

import json
import time
import asyncio
import threading
from typing import Any, Set
from .orchestrator import ProgressCallback


class WebSocketCallback(ProgressCallback):
    """
    ProgressCallback that broadcasts events to WebSocket clients.
    Thread-safe: pipeline runs in a thread, WebSocket is async.
    """
    
    # Pipeline phases with their weight toward overall progress
    # Names must match the actual stage names from the orchestrator
    # Duplicates are aliases — only one will match per run, so total stays ~100
    PHASE_WEIGHTS = {
        "Transcription": 10,
        "Audio Transcription": 10,
        "Video Analysis": 20,
        "Subject Tracking": 15,
        "Tracking": 15,
        "Short Generation": 30,
        "Editing": 30,
        "Metadata Generation": 5,
        "Export": 5,
        "Thumbnail Extraction": 5,
        "Quality Assurance": 5,
    }
    
    MAX_LOG_LINES = 1000  # Cap log buffer to prevent memory leak
    
    def __init__(self):
        self._clients: Set = set()
        self._clients_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop = None
        self._current_stage = None
        self._stage_progress = {}  # stage -> {current, total, pct}
        self._completed_stages = set()
        self._errors = []
        self._log_lines = []
        self._pipeline_state = "idle"  # idle, running, paused, complete, error, cancelled
        self._start_time = None
        self._overall_pct = 0.0
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for thread-safe broadcasting."""
        self._loop = loop
    
    def add_client(self, ws):
        with self._clients_lock:
            self._clients.add(ws)
    
    def remove_client(self, ws):
        with self._clients_lock:
            self._clients.discard(ws)
    
    def reset(self):
        """Reset state for a new pipeline run."""
        self._current_stage = None
        self._stage_progress = {}
        self._completed_stages = set()
        self._errors = []
        self._log_lines = []
        self._start_time = time.time()
        self._overall_pct = 0.0
        # Use the setter so it broadcasts the state_change event
        self.state = "running"
    
    @property
    def state(self):
        return self._pipeline_state
    
    @state.setter
    def state(self, val):
        self._pipeline_state = val
        self._broadcast({
            "type": "state_change",
            "state": val,
            "timestamp": time.time(),
        })
    
    def get_snapshot(self) -> dict:
        """Get full state snapshot for newly connected clients."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "type": "snapshot",
            "state": self._pipeline_state,
            "current_stage": self._current_stage,
            "stage_progress": dict(self._stage_progress),  # Copy for thread safety
            "completed_stages": list(self._completed_stages),
            "errors": list(self._errors),
            "overall_pct": self._overall_pct,
            "elapsed_seconds": elapsed,
            "log_lines": self._log_lines[-100:],  # Last 100 log lines
        }
    
    def _calc_overall(self):
        """Calculate overall 0-100% from completed phases and current phase progress."""
        total = 0.0
        seen_aliases = set()
        
        for stage, weight in self.PHASE_WEIGHTS.items():
            # Avoid double-counting aliases (e.g., "Transcription" and "Audio Transcription")
            if stage in seen_aliases:
                continue
            
            if stage in self._completed_stages:
                total += weight
                seen_aliases.add(stage)
            elif stage == self._current_stage and stage in self._stage_progress:
                pct = self._stage_progress[stage].get("pct", 0)
                total += weight * (pct / 100.0)
                seen_aliases.add(stage)
        
        self._overall_pct = min(100.0, total)
    
    def _add_log(self, line: str):
        """Add a log line with buffer cap."""
        self._log_lines.append(line)
        if len(self._log_lines) > self.MAX_LOG_LINES:
            self._log_lines = self._log_lines[-self.MAX_LOG_LINES:]
    
    def _broadcast(self, event: dict):
        """Thread-safe broadcast to all WebSocket clients."""
        with self._clients_lock:
            if not self._clients:
                return
            clients_copy = self._clients.copy()
        
        if not self._loop:
            return
        
        msg = json.dumps(event)
        
        async def _send_all():
            dead = set()
            for ws in clients_copy:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            if dead:
                with self._clients_lock:
                    self._clients -= dead
        
        try:
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, _send_all()
            )
        except RuntimeError:
            pass  # Loop closed or not running
        except Exception:
            pass  # Never crash the pipeline due to broadcast failure
    
    # === ProgressCallback overrides ===
    
    def on_stage_start(self, stage: str, total_steps: int = 0):
        try:
            super().on_stage_start(stage, total_steps)
        except Exception:
            pass
        
        self._current_stage = stage
        self._stage_progress[stage] = {"current": 0, "total": total_steps, "pct": 0}
        self._calc_overall()
        
        self._add_log(f"[STAGE] {stage}")
        
        self._broadcast({
            "type": "stage_start",
            "stage": stage,
            "total_steps": total_steps,
            "overall_pct": self._overall_pct,
            "timestamp": time.time(),
        })
    
    def on_step(self, step: str, current: int = 0, total: int = 0):
        try:
            super().on_step(step, current, total)
        except Exception:
            pass
        
        stage = self._current_stage or "Unknown"
        pct = (current / total * 100) if total > 0 else 0
        
        self._stage_progress[stage] = {
            "current": current,
            "total": total,
            "pct": pct,
            "step": step,
        }
        self._calc_overall()
        
        if total > 0:
            self._add_log(f"  [{current}/{total}] ({pct:.0f}%) {step}")
        else:
            self._add_log(f"  → {step}")
        
        self._broadcast({
            "type": "step",
            "stage": stage,
            "step": step,
            "current": current,
            "total": total,
            "stage_pct": pct,
            "overall_pct": self._overall_pct,
            "timestamp": time.time(),
        })
    
    def on_stage_complete(self, stage: str, result: Any = None):
        try:
            super().on_stage_complete(stage, result)
        except Exception:
            pass
        
        self._completed_stages.add(stage)
        self._stage_progress[stage] = {"current": 1, "total": 1, "pct": 100.0}
        self._calc_overall()
        
        self._add_log(f"  ✅ {stage} complete")
        
        self._broadcast({
            "type": "stage_complete",
            "stage": stage,
            "overall_pct": self._overall_pct,
            "timestamp": time.time(),
        })
    
    def on_error(self, stage: str, error: str):
        try:
            super().on_error(stage, error)
        except Exception:
            pass
        
        self._errors.append({"stage": stage, "error": error, "timestamp": time.time()})
        
        # Cap errors list too
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]
        
        self._add_log(f"  ❌ Error in {stage}: {error}")
        
        self._broadcast({
            "type": "error",
            "stage": stage,
            "error": error,
            "overall_pct": self._overall_pct,
            "timestamp": time.time(),
        })
