"""
GPU Video Utilities for Zero-Copy Video Processing
Uses FFmpeg piping + Kornia for GPU-resident frame processing
"""

import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json
# Lazy load PyNvVideoCodec later in the decode functions to prevent
# CUDA Context conflicts with PyTorch/Whisper initialization on WSL2.
PYNV_AVAILABLE = True


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe"""
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_data = json.loads(result.stdout)
    
    video_stream = next(s for s in probe_data['streams'] if s['codec_type'] == 'video')
    
    fps_str = video_stream['r_frame_rate']
    try:
        num, den = fps_str.split('/')
        fps_val = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps_val = 30.0

    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps_str': fps_str,
        'fps': fps_val,
        'duration': float(video_stream.get('duration', probe_data['format']['duration'])),
        'codec': video_stream.get('codec_name', 'unknown'),
        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p')
    }


def decode_video_to_tensor(
    video_path: str,
    device: str = 'cuda',
    max_frames: Optional[int] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Decode video using FFmpeg pipe directly to GPU tensor.
    
    Returns:
        Tuple of (frames_tensor, video_info)
        frames_tensor shape: (N, C, H, W) in float32 [0, 1]
    """
    info = get_video_info(video_path)
    width, height = info['width'], info['height']
    
    # Build FFmpeg decode command
    cmd = ['ffmpeg', '-y']
    
    if start_time is not None:
        cmd.extend(['-ss', str(start_time)])
    
    cmd.extend([
        '-hwaccel', 'cuda',        # Use NVDEC if available
        '-hwaccel_output_format', 'cuda',
        '-i', video_path
    ])
    
    if duration is not None:
        cmd.extend(['-t', str(duration)])
    
    if max_frames is not None:
        cmd.extend(['-frames:v', str(max_frames)])
    
    cmd.extend([
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',       # Output RGB
        '-'
    ])
    
    # Run FFmpeg and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    
    frame_size = width * height * 3
    frames = []
    
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            break
        
        # Convert to numpy then torch
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # CHW, [0,1]
        frames.append(tensor)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    process.wait()
    
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    
    # Stack and move to GPU
    frames_tensor = torch.stack(frames).to(device)
    
    return frames_tensor, info

def decode_video_native_stream(
    video_path: str,
    device: str = 'cuda',
    max_frames: Optional[int] = None
):
    """
    Generator for True Zero-Copy Decode using PyNvVideoCodec.
    Yields: (tensor, info)
    """
    if not PYNV_AVAILABLE:
        print("[GPU Utils] PyNvVideoCodec missing, fallback to batched FFmpeg")
        for batch, info in decode_video_batched(video_path, batch_size=1, device=device):
            yield batch[0], info
        return
        
    try:
        import kornia
    except Exception:
        pass # Only needed for GPU resize inside stream? No, used for what?
    
    info = get_video_info(video_path)
    
    # Initialize PyNv components
    import PyNvVideoCodec as nvc
    print(f"[GPU Utils] Opening Demuxer for: {video_path}")
    nvDemux = nvc.CreateDemuxer(filename=video_path)
    try:
        codec = nvDemux.GetNvCodecId()
        nvDec = nvc.CreateDecoder(gpuid=0, codec=codec)
    except Exception as e:
        print(f"[GPU Utils] Encoder init failed ({e}), fallback to FFmpeg")
        for batch, info in decode_video_batched(video_path, batch_size=1, device=device):
            yield batch, info
        return
    
    first_frame = True
    frames_yielded = 0
    
    for packet in nvDemux:
        # Decode returns a LIST of surfaces
        surfaces = nvDec.Decode(packet)
        
        for surface in surfaces:
            try:
                # IMPORTANT: PyNvVideoCodec 2.1.0 nv12_to_rgb() does NOT properly support
                # __dlpack__ encapsulation. Attempting to use it silently yields the parent
                # surface's capsule, causing it to crash downstream on consumption.
                # Therefore, we MUST pull the RAW NV12 surface out, clone it to retain
                # ownership, and handle the color conversion in PyTorch natively.
                
                # Raw NV12 tensor is shape (H * 1.5, W) uint8
                raw_tensor = torch.from_dlpack(surface).clone()
                
            except Exception as e:
                print(f"[GPU Utils] DLPack/Conversion failed: {e}")
                continue
            
            # --- Native NV12 to RGB Conversion (PyTorch) ---
            # NV12 format: 
            # - Top H rows are Y plane: (H, W)
            # - Bottom H/2 rows are interleaved VU plane: (H/2, W) -> (V, U, V, U...)
            
            h_total, w = raw_tensor.shape
            h = int(h_total * 2 / 3)
            
            # Extract Y plane
            Y = raw_tensor[:h, :].float() / 255.0  # (H, W)
            
            # Extract UV plane
            UV = raw_tensor[h:, :].float() / 255.0  # (H/2, W)
            
            # UV is interleaved V U V U
            # U is at even columns, V is at odd columns
            # But in PyNvVideoCodec it usually packs them as (U, V, U, V)
            u_plane = UV[:, 0::2]  # (H/2, W/2)
            v_plane = UV[:, 1::2]  # (H/2, W/2)
            
            # Upsample U and V to full resolution (H, W)
            u_plane = torch.nn.functional.interpolate(
                u_plane.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            v_plane = torch.nn.functional.interpolate(
                v_plane.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            # Standard BT.601 YUV to RGB conversion
            # R = Y + 1.402 * (V - 0.5)
            # G = Y - 0.344136 * (U - 0.5) - 0.714136 * (V - 0.5)
            # B = Y + 1.772 * (U - 0.5)
            
            u_adj = u_plane - 0.5
            v_adj = v_plane - 0.5
            
            r = Y + 1.402 * v_adj
            g = Y - 0.344136 * u_adj - 0.714136 * v_adj
            b = Y + 1.772 * u_adj
            
            # Stack and clamp to create (3, H, W) RGB tensor
            tensor = torch.stack([r, g, b], dim=0)
            tensor = torch.clamp(tensor, 0.0, 1.0)
            if first_frame:
                print(f"[GPU Utils] Native Tensor Shape (Post-Process): {tensor.shape}, Dtype: {tensor.dtype}")
                first_frame = False
            
            yield tensor.to(device), info
            
            frames_yielded += 1
            if max_frames and frames_yielded >= max_frames:
                return


def decode_video_batched(
    video_path: str,
    batch_size: int = 50,
    device: str = 'cuda'
):
    """
    Generator that yields batches of frames as GPU tensors.
    More memory efficient for long videos.
    
    Yields:
        batch_tensor: (B, C, H, W) float32 [0, 1]
    """
    info = get_video_info(video_path)
    width, height = info['width'], info['height']
    
    cmd = ['ffmpeg', '-y']
    
    if device == 'cuda':
        cmd.extend(['-hwaccel', 'cuda'])
        
    cmd.extend([
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-'
    ])
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    
    frame_size = width * height * 3
    batch = []
    
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) < frame_size:
            break
        
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        batch.append(tensor)
        
        if len(batch) >= batch_size:
            yield torch.stack(batch).to(device), info
            batch = []
    
    # Yield remaining frames
    if batch:
        yield torch.stack(batch).to(device), info
    
    process.wait()


def encode_tensor_to_video(
    frames: torch.Tensor,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    use_nvenc: bool = True
) -> bool:
    """
    Encode GPU tensor to video using FFmpeg NVENC.
    
    Args:
        frames: (N, C, H, W) float32 tensor in [0, 1]
        output_path: Output video path
        fps: Frame rate
        width: Output width
        height: Output height
        use_nvenc: Use GPU encoding (h264_nvenc)
    
    Returns:
        True if successful
    """
    encoder = 'h264_nvenc' if use_nvenc else 'libx264'
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', encoder,
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    
    # Convert tensor to raw bytes and pipe
    frames_cpu = frames.cpu()
    for i in range(frames_cpu.shape[0]):
        frame = frames_cpu[i]  # (C, H, W)
        frame_np = (frame.permute(1, 2, 0) * 255).byte().numpy()  # (H, W, C)
        process.stdin.write(frame_np.tobytes())
    
    process.stdin.close()
    process.wait()
    
    return process.returncode == 0


def crop_and_resize_gpu(
    frames: torch.Tensor,
    crop_box: Tuple[int, int, int, int],
    output_size: Tuple[int, int]
) -> torch.Tensor:
    """
    GPU-accelerated crop and resize using Kornia.
    
    Args:
        frames: (N, C, H, W) tensor
        crop_box: (x, y, w, h) crop region
        output_size: (height, width) output dimensions
    
    Returns:
        Cropped and resized tensor (N, C, out_H, out_W)
    """
    x, y, w, h = crop_box
    
    # Crop using slicing (fastest)
    cropped = frames[:, :, y:y+h, x:x+w]
    
    try:
        import kornia
        resized = kornia.geometry.transform.resize(
            cropped,
            output_size,
            interpolation='bilinear',
            antialias=True
        )
    except Exception:
         import torch.nn.functional as F
         resized = F.interpolate(cropped, size=output_size, mode='bilinear', align_corners=False)
    
    return resized


def draw_boxes_gpu(
    frames: torch.Tensor,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[float, float, float] = (0, 1, 0),
    thickness: int = 6
) -> torch.Tensor:
    """
    Draw bounding boxes on GPU tensor.
    
    Args:
        frames: (N, C, H, W) tensor
        boxes: List of (x, y, w, h) boxes per frame, or single box for all
        color: RGB color tuple in [0, 1]
        thickness: Line thickness in pixels
    
    Returns:
        Frames with boxes drawn
    """
    result = frames.clone()
    N, C, H, W = frames.shape
    
    # If single box, apply to all frames
    if len(boxes) == 1 and N > 1:
        boxes = boxes * N
    
    for i, box in enumerate(boxes):
        if box is None:
            continue
        
        x, y, w, h = box
        
        # Draw rectangle (top, bottom, left, right lines)
        # Top line
        result[i, 0, y:y+thickness, x:x+w] = color[0]
        result[i, 1, y:y+thickness, x:x+w] = color[1]
        result[i, 2, y:y+thickness, x:x+w] = color[2]
        
        # Bottom line
        result[i, 0, y+h-thickness:y+h, x:x+w] = color[0]
        result[i, 1, y+h-thickness:y+h, x:x+w] = color[1]
        result[i, 2, y+h-thickness:y+h, x:x+w] = color[2]
        
        # Left line
        result[i, 0, y:y+h, x:x+thickness] = color[0]
        result[i, 1, y:y+h, x:x+thickness] = color[1]
        result[i, 2, y:y+h, x:x+thickness] = color[2]
        
        # Right line
        result[i, 0, y:y+h, x+w-thickness:x+w] = color[0]
        result[i, 1, y:y+h, x+w-thickness:x+w] = color[1]
        result[i, 2, y:y+h, x+w-thickness:x+w] = color[2]
    
    return result


def clear_gpu_cache():
    """Clear CUDA cache to prevent memory buildup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
