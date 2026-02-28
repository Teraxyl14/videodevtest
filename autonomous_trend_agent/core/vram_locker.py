"""
VRAM Locker — Persistent CUDA IPC Buffer
=========================================
A minimal background process that holds a shared CUDA tensor buffer.
Allows zero-copy data transfer between ephemeral spokes via IPC handles.

The Locker acts as a VRAM-resident "clipboard":
    - YOLO spoke WRITES a frame to the buffer
    - Qwen spoke READS the frame from the buffer (zero-copy)
    - The buffer persists while spokes are born and die

VRAM Footprint: ~33MB (4K RGBA: 3840 x 2160 x 4 bytes)

Docker Requirement: --ipc=host (non-negotiable for cudaIpcMemHandle)

References:
    - GPU VRAM Orchestration and IPC.txt, Section 4.3 & 8.3.1
    - Objective A3.2 (Hub-and-Spoke + VRAM Locker)
"""

import multiprocessing as mp
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger("VRAMLocker")


class VRAMLocker:
    """
    Persistent process that holds a shared CUDA tensor buffer.
    Allows zero-copy data transfer between ephemeral spokes via IPC handles.

    The buffer is shaped as (H, W, C) to match 4K RGBA video frames.
    This is more structured than a flat byte array and allows spokes
    to directly index pixels without reshaping.

    Args:
        handle_queue: Queue to send the IPC handle to the Hub
        metadata_queue: Queue to send/receive buffer metadata
        buffer_shape: Shape of the VRAM buffer (H, W, C)
    """

    def __init__(
        self,
        handle_queue: mp.Queue,
        metadata_queue: Optional[mp.Queue] = None,
        buffer_shape: Tuple[int, ...] = (2160, 3840, 4)
    ):
        self.handle_queue = handle_queue
        self.metadata_queue = metadata_queue
        self.buffer_shape = buffer_shape
        self._calculate_size()

    def _calculate_size(self):
        """Calculate buffer size in MB for logging."""
        numel = 1
        for s in self.buffer_shape:
            numel *= s
        self.size_mb = numel / (1024 * 1024)

    def run(self):
        """
        Main loop for the Locker process.

        Allocates the CUDA buffer, shares the IPC handle, and stays alive.
        The OS releases the memory when this process terminates.
        """
        import torch

        try:
            # ---- 1. Initialize Minimal CUDA Context ----
            if not torch.cuda.is_available():
                logger.error("CUDA not available! Locker cannot start.")
                self.handle_queue.put(None)
                return

            device = torch.device('cuda:0')
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(
                f"Locker starting on {gpu_name}. "
                f"Allocating {self.size_mb:.1f}MB buffer "
                f"(shape: {self.buffer_shape})"
            )

            # ---- 2. Allocate Persistent Shared Buffer ----
            # We use uint8 as the generic container for video frame data.
            # Shape: (H, W, C) = (2160, 3840, 4) for 4K RGBA = ~33MB
            buffer = torch.zeros(
                self.buffer_shape,
                dtype=torch.uint8,
                device=device
            )

            # ---- 3. Get IPC Handle ----
            # _share_cuda_() returns a tuple of handle details that can
            # be passed to другому process to reconstruct the tensor.
            # This is the core of the zero-copy IPC mechanism.
            storage = buffer.untyped_storage()
            handle_info = storage._share_cuda_()

            logger.info("Buffer allocated. Sharing IPC handle with Hub.")
            self.handle_queue.put(handle_info)

            # Send metadata if queue is provided
            if self.metadata_queue:
                self.metadata_queue.put({
                    "shape": self.buffer_shape,
                    "dtype": "uint8",
                    "size_mb": self.size_mb,
                    "gpu": gpu_name
                })

            # ---- 4. Keep Process Alive ----
            # The OS releases the memory when this process dies.
            # We log VRAM usage periodically as a health heartbeat.
            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()

            while True:
                time.sleep(1)

                # Periodic health heartbeat
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    vram_allocated = torch.cuda.memory_allocated(0) / (1024**2)
                    vram_reserved = torch.cuda.memory_reserved(0) / (1024**2)
                    logger.debug(
                        f"Locker heartbeat: "
                        f"allocated={vram_allocated:.0f}MB, "
                        f"reserved={vram_reserved:.0f}MB"
                    )
                    last_heartbeat = now

                # Check for shutdown signal on metadata queue
                if self.metadata_queue and not self.metadata_queue.empty():
                    try:
                        msg = self.metadata_queue.get_nowait()
                        if msg == "SHUTDOWN":
                            logger.info("Shutdown signal received.")
                            break
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Locker crashed: {e}")
            import traceback
            traceback.print_exc()
            self.handle_queue.put(None)


def start_locker(
    buffer_shape: Tuple[int, ...] = (2160, 3840, 4)
) -> tuple:
    """
    Helper to spawn the Locker process and retrieve the IPC handle.

    Args:
        buffer_shape: Shape of the VRAM buffer

    Returns:
        (process, ipc_handle, metadata_queue) tuple

    Raises:
        RuntimeError: If the Locker fails to initialize
    """
    ctx = mp.get_context('spawn')  # Must use 'spawn' for CUDA
    handle_queue = ctx.Queue()
    metadata_queue = ctx.Queue()

    locker = VRAMLocker(
        handle_queue=handle_queue,
        metadata_queue=metadata_queue,
        buffer_shape=buffer_shape
    )

    process = ctx.Process(
        target=locker.run,
        name="VRAM-Locker",
        daemon=True
    )
    process.start()

    # Wait for handle (timeout 10s)
    handle = handle_queue.get(timeout=10)
    if handle is None:
        process.terminate()
        raise RuntimeError(
            "Failed to initialize VRAM Locker. "
            "Ensure CUDA is available and --ipc=host is set in Docker."
        )

    # Get metadata
    metadata = metadata_queue.get(timeout=5)
    logger.info(
        f"VRAM Locker started (PID={process.pid}, "
        f"{metadata.get('size_mb', 0):.1f}MB on {metadata.get('gpu', 'unknown')})"
    )

    return process, handle, metadata_queue
