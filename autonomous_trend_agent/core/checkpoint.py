"""
Checkpoint Manager - V2 Pipeline Reliability
=============================================
Atomic state persistence for crash recovery.
Handles SIGTERM gracefully to save progress before shutdown.
"""

import os
import json
import signal
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PipelineCheckpoint:
    """State that can be saved and restored."""
    video_path: str
    current_phase: str  # "transcription", "analysis", "editing", "export"
    phase_progress: float  # 0.0 to 1.0
    processed_frames: int
    total_frames: int
    shorts_completed: list
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        return cls(**data)


class CheckpointManager:
    """
    Manages atomic checkpoint saves and graceful shutdown.
    
    Usage:
        ckpt = CheckpointManager("/app/checkpoints")
        state = ckpt.load("video_abc")
        
        # In processing loop:
        ckpt.save("video_abc", current_state)
        
        # On shutdown, GracefulKiller triggers final save
    """
    
    def __init__(self, checkpoint_dir: str = "/app/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, job_id: str) -> Path:
        return self.checkpoint_dir / f"{job_id}.checkpoint.json"
    
    def save(self, job_id: str, checkpoint: PipelineCheckpoint) -> None:
        """
        Atomic save: write to temp file, then rename.
        Prevents corruption if process dies mid-write.
        """
        filepath = self._get_path(job_id)
        temp_path = filepath.with_suffix(".tmp")
        
        checkpoint.timestamp = datetime.now().isoformat()
        
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, filepath)
            logger.info(f"Checkpoint saved: {job_id} @ {checkpoint.current_phase} ({checkpoint.phase_progress:.1%})")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def load(self, job_id: str) -> Optional[PipelineCheckpoint]:
        """Load existing checkpoint if available."""
        filepath = self._get_path(job_id)
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            checkpoint = PipelineCheckpoint.from_dict(data)
            logger.info(f"Checkpoint loaded: {job_id} @ {checkpoint.current_phase}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear(self, job_id: str) -> None:
        """Remove checkpoint after successful completion."""
        filepath = self._get_path(job_id)
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Checkpoint cleared: {job_id}")


class GracefulKiller:
    """
    Handles SIGTERM/SIGINT for graceful shutdown.
    
    Integrates with CheckpointManager to save state before exit.
    Supervisor sends SIGTERM, waits stopwaitsecs, then SIGKILL.
    
    Usage:
        killer = GracefulKiller(checkpoint_mgr, job_id)
        
        for frame in frames:
            if killer.should_exit:
                break
            process_frame(frame)
            killer.update_state(new_checkpoint)
    """
    
    def __init__(
        self, 
        checkpoint_mgr: CheckpointManager, 
        job_id: str,
        initial_state: Optional[PipelineCheckpoint] = None
    ):
        self.checkpoint_mgr = checkpoint_mgr
        self.job_id = job_id
        self.current_state = initial_state
        self.should_exit = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received {sig_name}. Initiating graceful shutdown...")
        
        # Save current state before exiting
        if self.current_state:
            try:
                self.checkpoint_mgr.save(self.job_id, self.current_state)
                logger.info("Final checkpoint saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save final checkpoint: {e}")
        
        self.should_exit = True
    
    def update_state(self, checkpoint: PipelineCheckpoint) -> None:
        """Update current state (will be saved on shutdown)."""
        self.current_state = checkpoint


def create_checkpoint(
    video_path: str,
    phase: str,
    progress: float,
    processed: int = 0,
    total: int = 0,
    shorts: list = None
) -> PipelineCheckpoint:
    """Helper to create checkpoint objects."""
    return PipelineCheckpoint(
        video_path=video_path,
        current_phase=phase,
        phase_progress=progress,
        processed_frames=processed,
        total_frames=total,
        shorts_completed=shorts or [],
        timestamp=datetime.now().isoformat()
    )
