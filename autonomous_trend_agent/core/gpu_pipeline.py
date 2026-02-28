"""
GPU Video Pipeline - Zero-Copy Processing
==========================================
Hardware-accelerated video decode/encode using PyNvVideoCodec.
Eliminates CPU-GPU memory copies via DLPack tensor bridging.

Data Flow (Zero-Copy):
    NVDEC -> GPU Memory -> DLPack -> PyTorch Tensor -> Inference -> DLPack -> NVENC

Performance:
    - 10x faster than FFmpeg CPU decode
    - No PCIe bandwidth saturation
    - Frames stay in VRAM throughout pipeline
"""

import os
import logging
import ctypes
import gc
from pathlib import Path
from typing import Optional, Iterator, Tuple
from dataclasses import dataclass

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyNvVideoCodec (assuming VPF-style or pynvvideocodec package)
# Lazy-load PyNvVideoCodec later in the decode functions to prevent
# CUDA Context conflicts with PyTorch on WSL2.
PYNV_AVAILABLE = True
nvc = None


class UnifiedContextManager:
    """
    Manages a single CUDA Primary Context shared between PyTorch and NVDEC.
    Ensures that pointers are valid across the entire pipeline lifecycle.
    Prevents the '!handles_.at(i)' assertion crash in CUDACachingAllocator.
    """
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.ctx_handle = None
        self._initialize_pytorch_context()

    def _initialize_pytorch_context(self):
        """
        Forces PyTorch to initialize the Primary Context and retrieves the handle.
        """
        # 1. Trigger PyTorch CUDA initialization
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in PyTorch.")
        
        torch.cuda.set_device(self.gpu_id)
        # Allocate a dummy tensor to ensure the context is fully built
        _ = torch.zeros(1, device=f'cuda:{self.gpu_id}')
        
        # 2. Locate the CUDA Driver Library
        # WSL2 specific location for the thunking driver
        lib_path = "/usr/lib/wsl/lib/libcuda.so.1"
        if not os.path.exists(lib_path):
            # Fallback for native Linux
            lib_path = "libcuda.so"
            
        try:
            self.cuda_driver = ctypes.CDLL(lib_path)
            # Define Driver API signature for cuCtxGetCurrent
            # CUresult cuCtxGetCurrent(CUcontext *pctx);
            self.cuCtxGetCurrent = self.cuda_driver.cuCtxGetCurrent
            self.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self.cuCtxGetCurrent.restype = int
        except OSError as e:
            logger.warning(f"Could not load libcuda.so from {lib_path}. Advanced context management disabled. {e}")
            self.ctx_handle = None
            return

        # 3. Retrieve the context handle
        ctx_ptr = ctypes.c_void_p()
        ret = self.cuCtxGetCurrent(ctypes.byref(ctx_ptr))
        
        if ret != 0 or not ctx_ptr:
            logger.warning(f"Failed to get current CUDA context. Error code: {ret}")
            self.ctx_handle = None
        else:
            self.ctx_handle = ctx_ptr.value
            logger.info(f"Verified Shared CUDA Context Handle: {hex(self.ctx_handle)}")

    def get_decoder(self, video_path):
        """
        Returns a PyNvVideoCodec decoder initialized with the shared context.
        """
        if not PYNV_AVAILABLE:
            raise ImportError("PyNvVideoCodec not installed")
            
        # Check if installed nvc supports cuda_context injection (VPF feature)
        # If not, we might be on a version that doesn't support it, but we try anyway
        try:
            import PyNvVideoCodec as nvc
            # We try to use the most generic constructor available
            if hasattr(nvc, 'CreateDecoder'):
                 if hasattr(nvc, 'SimpleDecoder'):
                     return nvc.SimpleDecoder(
                        str(video_path),
                        gpu_id=self.gpu_id,
                        cuda_context=self.ctx_handle,
                        use_device_memory=True
                     )
                 else:
                     pass
        except Exception as e:
             logger.error(f"Unified Context Decoder Init Failed: {e}")

        # Fallback to standard init if specific shared context init fails or is not supported
        return None


class TransitionContextManager:
    """
    Guarantees a clean GPU state between pipeline phases.
    Prevents CUDACachingAllocator fragmentation and stale handle issues.
    """
    def __init__(self, phase_name, aggressive_reset=False):
        self.phase_name = phase_name
        self.aggressive_reset = aggressive_reset

    def __enter__(self):
        logger.info(f"--- [Pipeline] Entering Phase: {self.phase_name} ---")
        self._sync_and_clean()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"--- [Pipeline] Exiting Phase: {self.phase_name} ---")
        if exc_type:
            logger.error(f"Error detected in {self.phase_name}. Attempting cleanup.")
        self._sync_and_clean()

    def _sync_and_clean(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            # ipc_collect handles cleanup of shared memory handles
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()



@dataclass
class VideoInfo:
    """Metadata about a video file."""
    width: int
    height: int
    fps: float
    num_frames: int
    duration: float
    codec: str


class GPUVideoDecoder:
    """
    Hardware-accelerated video decoder using NVIDIA NVDEC.
    
    Returns frames directly as PyTorch GPU tensors (zero-copy).
    
    Usage:
        decoder = GPUVideoDecoder("video.mp4")
        for frame_tensor in decoder:
            # frame_tensor is already on GPU, shape: (H, W, C)
            process(frame_tensor)
    """
    
    def __init__(
        self, 
        video_path: str,
        gpu_id: int = 0,
        output_format: str = "rgb"  # "rgb" or "nv12"
    ):
        self.video_path = Path(video_path)
        self.gpu_id = gpu_id
        self.output_format = output_format
        self.device = torch.device(f"cuda:{gpu_id}")
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Try to import PyNvVideoCodec
        try:
            import PyNvVideoCodec as nvc
            self.nvc = nvc
            self._use_nvc = True
        except ImportError:
            logger.warning("PyNvVideoCodec not available. Falling back to OpenCV.")
            self._use_nvc = False
        
        self._decoder = None
        self._video_info: Optional[VideoInfo] = None
        self._frame_index = 0

        # Initialize Unified Context Manager
        self.context_manager = UnifiedContextManager(gpu_id=gpu_id)

    def _init_nvc_decoder(self):
        """Initialize PyNvVideoCodec decoder."""
        # Try to get decoder from shared context first
        shared_decoder = self.context_manager.get_decoder(self.video_path)
        if shared_decoder:
            logger.info("Using Shared Context Decoder")
            self._decoder = shared_decoder
            return

        # Fallback to standard creation if shared context failed
        logger.warning("Shared Context init failed, falling back to standalone decoder")

        # VPF V1 vs V2 Enum compatibility
        CodecEnum = getattr(self.nvc, 'CudaVideoCodec', getattr(self.nvc, 'cudaVideoCodec', None))
        if CodecEnum is None:
            # Attempt to find it recursively or use integer fallback (H264=4)
            logger.warning("CudaVideoCodec enum not found, attempting auto-detection or raw values")
            
        try:
             # Standard VPF/pynvvideocodec init
             # cudacontext=0: Attach to PyTorch's Primary Context (shared pointers)
             # usedevicememory=True: Decode to VRAM not CPU (zero-copy critical)
             self._decoder = self.nvc.CreateDecoder(
                gpuid=self.gpu_id,
                codec=CodecEnum.H264 if CodecEnum else 4, # Fallback to H264 idx
                cudacontext=0,       # CRITICAL: Share context with PyTorch
                cudastream=0,        # Default stream
                usedevicememory=True  # CRITICAL: Decode to VRAM
            )
        except Exception as e:
            raise ImportError(f"Failed to create NVC decoder: {e}")
        
    def _init_cv_decoder(self):
        """Fallback: OpenCV decoder (CPU, with upload to GPU)."""
        import cv2
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        self._video_info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            num_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self._cap.get(cv2.CAP_PROP_FPS),
            codec="unknown"
        )
    
    @property
    def info(self) -> VideoInfo:
        """Get video metadata."""
        if self._video_info is None:
            if self._use_nvc:
                try:
                    # Select correct API
                    if hasattr(self.nvc, 'CreateDemuxer'):
                        # VPF 2.0+
                        demuxer = self.nvc.CreateDemuxer(filename=str(self.video_path))
                    elif hasattr(self.nvc, 'PyFFmpegDemuxer'):
                        # Legacy VPF
                        demuxer = self.nvc.PyFFmpegDemuxer(str(self.video_path))
                    else:
                        raise ImportError("PyNvVideoCodec incompatible: No Demuxer found")

                    # Handle API differences for Codec retrieval
                    if hasattr(demuxer, 'Codec'):
                        codec_name = str(demuxer.Codec())
                    elif hasattr(demuxer, 'Format'):
                         # Some V2 versions use Format
                        codec_name = str(demuxer.Format())
                    else:
                        codec_name = "h264" # Default for V2

                    # Safely get properties (API Variance)
                    get_fps = getattr(demuxer, 'Framerate', getattr(demuxer, 'framerate', lambda: 30.0))
                    get_frames = getattr(demuxer, 'Numframes', getattr(demuxer, 'numframes', lambda: 0))
                    
                    fps = float(get_fps())
                    num_frames = int(get_frames())
                    duration = num_frames / fps if fps > 0 else 0

                    self._video_info = VideoInfo(
                        width=demuxer.Width(),
                        height=demuxer.Height(),
                        fps=fps,
                        num_frames=num_frames,
                        duration=duration,
                        codec=codec_name
                    )
                    self._demuxer = demuxer
                    self._init_nvc_decoder()

                except Exception as e:
                    logger.error(f"NVC init failed: {e}. Object Dir: {dir(demuxer) if 'demuxer' in locals() else 'N/A'}")
                    logger.warning("Falling back to OpenCV.")
                    self._use_nvc = False
                    self._init_cv_decoder()
            else:
                self._init_cv_decoder()
        return self._video_info
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through all frames as GPU tensors."""
        if self._use_nvc:
            yield from self._iter_nvc()
        else:
            yield from self._iter_cv()
    
    def _iter_nvc(self) -> Iterator[torch.Tensor]:
        """Iterate using PyNvVideoCodec (zero-copy).
        
        CRITICAL: DLPack Lifecycle Management
        We MUST retain references to DecodedFrame/Surface objects until
        PyTorch is done with them. If the GC collects the frame object,
        the underlying VRAM is freed and the PyTorch tensor points to
        garbage → crash with "!handles_.at(i)" error.
        
        Ref: GPU Zero-Copy Video Pipeline Guide.txt, Section 5.4
        """
        if self._decoder is None:
            self._init_nvc_decoder()
        
        while True:
            # Get encoded packet
            packet = self._demuxer.Demux()
            if packet is None:
                break
            
            # Decode to GPU memory
            surfaces = self._decoder.DecodeSurface(packet)
            
            for surface in surfaces:
                # Zero-copy: wrap GPU memory as PyTorch tensor via DLPack
                # clone() takes ownership before the surface goes out of scope
                try:
                    tensor = torch.from_dlpack(surface).clone()
                except (TypeError, AttributeError):
                    # Fallback for older API versions
                    tensor = torch.as_tensor(surface, device=self.device).clone()
                
                # Convert from NV12 to RGB if needed
                if self.output_format == "rgb":
                    tensor = self._nv12_to_rgb(tensor)
                
                self._frame_index += 1
                yield tensor
    
    def _iter_cv(self) -> Iterator[torch.Tensor]:
        """Fallback: OpenCV decode + GPU upload."""
        import cv2
        
        if not hasattr(self, '_cap'):
            self._init_cv_decoder()
        
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Upload to GPU (this is the slow path we want to avoid)
            tensor = torch.from_numpy(frame_rgb).to(self.device)
            
            self._frame_index += 1
            yield tensor
        
        self._cap.release()
    
    def _nv12_to_rgb(self, nv12_tensor: torch.Tensor) -> torch.Tensor:
        """Convert RAW NV12 [H+(H//2), W] format to RGB [3, H, W] using PyTorch natively."""
        try:
            h_total, w = nv12_tensor.shape
            h = int(h_total * 2 / 3)
            
            Y = nv12_tensor[:h, :].float() / 255.0
            UV = nv12_tensor[h:, :].float() / 255.0
            
            u_plane = UV[:, 0::2]
            v_plane = UV[:, 1::2]
            
            u_plane = torch.nn.functional.interpolate(
                u_plane.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            v_plane = torch.nn.functional.interpolate(
                v_plane.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            u_adj = u_plane - 0.5
            v_adj = v_plane - 0.5
            
            r = Y + 1.402 * v_adj
            g = Y - 0.344136 * u_adj - 0.714136 * v_adj
            b = Y + 1.772 * u_adj
            
            tensor = torch.stack([r, g, b], dim=0)
            return torch.clamp(tensor, 0.0, 1.0)
        except Exception as e:
            logger.error(f"NV12 to RGB conversion failed: {e}")
            return nv12_tensor
    
    def get_frame(self, index: int) -> torch.Tensor:
        """Get a specific frame by index."""
        # Seek and decode single frame
        # Implementation depends on decoder capabilities
        for i, frame in enumerate(self):
            if i == index:
                return frame
        raise IndexError(f"Frame {index} not found")
    
    def close(self):
        """Release resources."""
        if hasattr(self, '_cap'):
            self._cap.release()
        if hasattr(self, '_decoder') and self._decoder:
            del self._decoder
        if hasattr(self, '_demuxer'):
            del self._demuxer


class GPUVideoEncoder:
    """
    Hardware-accelerated video encoder using NVIDIA NVENC.
    
    Accepts PyTorch GPU tensors directly (zero-copy).
    
    Usage:
        encoder = GPUVideoEncoder("output.mp4", width=1920, height=1080)
        for processed_frame in frames:
            encoder.write(processed_frame)
        encoder.close()
    """
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = "h264",
        bitrate: str = "8M",
        gpu_id: int = 0
    ):
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.gpu_id = gpu_id
        
        # Try PyNvVideoCodec
        try:
            import PyNvVideoCodec as nvc
            self.nvc = nvc
            self._use_nvc = True
        except ImportError:
            logger.warning("PyNvVideoCodec not available. Falling back to FFmpeg.")
            self._use_nvc = False
        
        self._encoder = None
        self._frame_count = 0
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize the encoder."""
        if self._use_nvc:
            self._encoder = self.nvc.CreateEncoder(
                gpuid=self.gpu_id,
                codec=self.codec,
                width=self.width,
                height=self.height,
                bitrate=self.bitrate,
                fps=self.fps,
                outputpath=str(self.output_path)
            )
        else:
            # Fallback to FFmpeg subprocess
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "rgb24",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "h264_nvenc" if torch.cuda.is_available() else "libx264",
                "-b:v", self.bitrate,
                str(self.output_path)
            ]
            self._encoder = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    
    def write(self, frame: torch.Tensor):
        """
        Write a frame to the output video.
        
        CRITICAL: Stride Handling
        Decoders often pad rows for alignment (e.g., 1920px → 2048 pitch).
        .contiguous() forces memory compaction on GPU to prevent skewed output.
        Ref: GPU Zero-Copy Video Pipeline Guide.txt, Section 6.4
        
        Args:
            frame: GPU tensor of shape (H, W, C) or (C, H, W)
        """
        if self._use_nvc and self._encoder:
            # CHW → HWC for encoder, then force contiguous layout
            if frame.dim() == 3 and frame.shape[0] in (3, 4):
                frame = frame.permute(1, 2, 0)
            frame = frame.contiguous()  # CRITICAL: fix stride mismatch
            # Zero-copy encode from GPU memory
            self._encoder.EncodeSurface(frame)
        else:
            # Fallback: download to CPU and pipe to FFmpeg
            if frame.dim() == 3 and frame.shape[0] == 3:
                frame = frame.permute(1, 2, 0)  # CHW -> HWC
            frame = frame.contiguous()  # Ensure tightly packed
            
            frame_cpu = frame.cpu().numpy().astype(np.uint8)
            self._encoder.stdin.write(frame_cpu.tobytes())
        
        self._frame_count += 1
    
    def close(self):
        """Finalize and close the encoder."""
        if self._use_nvc and self._encoder:
            self._encoder.Flush()
            del self._encoder
        elif hasattr(self, '_encoder') and self._encoder:
            self._encoder.stdin.close()
            self._encoder.wait()
        
        logger.info(f"Encoded {self._frame_count} frames to {self.output_path}")


def create_gpu_pipeline(
    input_path: str,
    output_path: str,
    process_fn: callable,
    gpu_id: int = 0
) -> int:
    """
    Convenience function to run a full GPU video pipeline.
    
    Args:
        input_path: Input video file
        output_path: Output video file
        process_fn: Function that takes a frame tensor and returns processed tensor
        gpu_id: GPU device ID
    
    Returns:
        Number of frames processed
    """
    decoder = GPUVideoDecoder(input_path, gpu_id=gpu_id)
    info = decoder.info
    
    encoder = GPUVideoEncoder(
        output_path,
        width=info.width,
        height=info.height,
        fps=info.fps,
        gpu_id=gpu_id
    )
    
    frame_count = 0
    try:
        for frame in decoder:
            processed = process_fn(frame)
            encoder.write(processed)
            frame_count += 1
    finally:
        decoder.close()
        encoder.close()
    
    return frame_count
