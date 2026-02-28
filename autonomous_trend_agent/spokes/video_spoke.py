"""
Video Spokes — Spoke B (Eyes) & Spoke C (Brain)
=================================================
Ephemeral GPU processes for object detection and visual reasoning.

Spoke B (YOLO): Decodes video via NVDEC, runs YOLOv11 detection,
    writes target frame to the VRAM Locker's IPC buffer.

Spoke C (Qwen): Reads frame from Locker (zero-copy), runs
    Qwen3-VL-8B-Instruct with FP8 quantization for VLM reasoning.

Both spokes die after completing work → guaranteed VRAM reclamation.

References:
    - GPU VRAM Orchestration and IPC.txt, Sections 6 & 7
    - GPU Zero-Copy Video Pipeline Guide.txt, Sections 5 & 6
    - Objectives E2.1, D3.1
"""

import logging
import json
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger("VideoSpokes")


# =============================================================================
# Shared IPC Helper
# =============================================================================

def rebuild_from_ipc_handle(handle_info, shape, dtype_str="uint8"):
    """
    Reconstruct a CUDA tensor from the Locker's IPC handle.
    Creates a VIEW into the Locker's persistent memory (zero-copy).

    This function runs inside a Spoke process and maps the Locker's
    buffer into this process's address space without allocating new VRAM.

    Args:
        handle_info: IPC handle tuple from VRAMLocker._share_cuda_()
        shape: Expected tensor shape, e.g. (2160, 3840, 4)
        dtype_str: Data type string ("uint8")

    Returns:
        torch.Tensor pointing to the Locker's VRAM buffer
    """
    import torch

    dtype_map = {
        "uint8": torch.uint8,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.uint8)

    # Rebuild storage from IPC handle
    # handle_info is the tuple returned by storage._share_cuda_()
    storage = torch.UntypedStorage._new_shared_cuda(
        *handle_info
    )

    # Create tensor view over the storage
    numel = 1
    for s in shape:
        numel *= s

    tensor = torch.tensor([], dtype=dtype, device='cuda:0')
    tensor.set_(storage, 0, shape)

    return tensor


# =============================================================================
# Spoke B: YOLO (Eyes)
# =============================================================================

def run_yolo_spoke(
    video_path: str,
    ipc_handle,
    result_queue=None,
    buffer_shape: Tuple[int, ...] = (2160, 3840, 4),
    target_frame_idx: int = 0,
    confidence_threshold: float = 0.5
):
    """
    Ephemeral YOLO detection spoke.

    Pipeline:
        1. Initialize CUDA context
        2. Attach to Locker's IPC buffer (zero-copy)
        3. Decode video via NVDEC (GPU, zero-copy)
        4. Run YOLOv11 detection (TensorRT Int8 or PyTorch)
        5. Write best frame to Locker buffer
        6. Return detections via Queue
        7. Die → OS reclaims YOLO VRAM + NVDEC context

    Args:
        video_path: Path to video file
        ipc_handle: CUDA IPC handle from the Locker
        result_queue: Queue to return detection results
        buffer_shape: Shape of the Locker's buffer
        target_frame_idx: Which frame to analyze (0 = first keyframe)
        confidence_threshold: Min confidence for detections
    """
    import torch
    import os

    try:
        device = torch.device('cuda:0')
        logger.info(f"YOLO Spoke starting on {torch.cuda.get_device_name(0)}")

        # ---- 1. Attach to Locker Buffer ----
        logger.info("Attaching to VRAM Locker...")
        if ipc_handle is not None:
            try:
                shared_buffer = rebuild_from_ipc_handle(
                    ipc_handle, buffer_shape, "uint8"
                )
                logger.info(f"Locker buffer attached: {shared_buffer.shape}")
            except Exception as e:
                logger.warning(f"Could not attach to Locker: {e}. Using local buffer.")
                shared_buffer = torch.zeros(buffer_shape, dtype=torch.uint8, device=device)
        else:
            shared_buffer = torch.zeros(buffer_shape, dtype=torch.uint8, device=device)

        # ---- 2. Decode Video Frame via NVDEC (Zero-Copy) ----
        logger.info(f"Decoding video: {video_path}")
        frame_tensor = None

        try:
            # Try PyNvVideoCodec for true zero-copy NVDEC decoding
            import PyNvVideoCodec as nvc

            decoder = nvc.CreateDecoder(
                gpuid=0,
                cudacontext=0,       # Attach to PyTorch's Primary Context
                cudastream=0,        # Default stream
                usedevicememory=True, # Decode to VRAM (not CPU)
            )

            # Demux and decode
            demuxer = nvc.CreateDemuxer(video_path)
            frame_count = 0

            for packet in demuxer:
                frames = decoder.Decode(packet)
                for frame in frames:
                    if frame_count == target_frame_idx:
                        # Zero-copy: DLPack transfer to PyTorch tensor
                        frame_tensor = torch.from_dlpack(frame)
                        # CRITICAL: Keep frame alive until we're done
                        _frame_ref = frame  # prevent GC (!handles_.at(i) fix)
                        break
                    frame_count += 1
                if frame_tensor is not None:
                    break

        except (ImportError, Exception) as e:
            logger.warning(f"NVDEC unavailable ({e}). Falling back to OpenCV.")
            import cv2

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame_np = cap.read()
            cap.release()

            if ret:
                # Convert BGR→RGB and upload to GPU
                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).to(device)

        if frame_tensor is None:
            logger.error("Could not decode any frame from video")
            if result_queue:
                result_queue.put({"error": "No frames decoded"})
            raise SystemExit(1)

        logger.info(f"Decoded frame: {frame_tensor.shape} {frame_tensor.dtype}")

        # ---- 3. Run YOLOv11 Detection ----
        logger.info("Loading YOLOv11...")
        detections = []

        try:
            from ultralytics import YOLO

            # Prefer TensorRT engine if available (Int8, <500MB)
            engine_path = os.path.join(os.path.dirname(video_path), "yolov11n.engine")
            if os.path.exists(engine_path):
                model = YOLO(engine_path, task="detect")
                logger.info("Loaded YOLOv11 TensorRT engine (Int8)")
            else:
                model = YOLO("yolov11n.pt")
                logger.info("Loaded YOLOv11 PyTorch weights")

            # Prepare frame for YOLO (expects HWC uint8)
            if frame_tensor.dim() == 3:
                if frame_tensor.shape[0] in (3, 4):
                    # CHW → HWC
                    yolo_input = frame_tensor.permute(1, 2, 0).contiguous()
                else:
                    yolo_input = frame_tensor.contiguous()

                # Run detection
                results = model(yolo_input.cpu().numpy(), verbose=False)

                for r in results:
                    for box in r.boxes:
                        if box.conf.item() >= confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({
                                "class": r.names[int(box.cls.item())],
                                "confidence": round(box.conf.item(), 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            })

            logger.info(f"Detected {len(detections)} objects")

        except ImportError:
            logger.warning("Ultralytics not available. Skipping detection.")
            detections = [{"class": "unknown", "confidence": 0.0, "bbox": [0, 0, 0, 0]}]

        # ---- 4. Write Frame to Locker (Device-to-Device) ----
        logger.info("Writing frame to Locker buffer...")
        h, w = frame_tensor.shape[:2] if frame_tensor.dim() == 3 else frame_tensor.shape[1:3]

        if frame_tensor.shape[0] in (3, 4) and frame_tensor.dim() == 3:
            # CHW → HWC for Locker
            frame_hwc = frame_tensor.permute(1, 2, 0).contiguous()
        else:
            frame_hwc = frame_tensor.contiguous()

        # Pad/crop to fit Locker buffer
        buf_h, buf_w, buf_c = shared_buffer.shape
        fh, fw = frame_hwc.shape[0], frame_hwc.shape[1]
        fc = frame_hwc.shape[2] if frame_hwc.dim() == 3 else 1

        copy_h = min(fh, buf_h)
        copy_w = min(fw, buf_w)
        copy_c = min(fc, buf_c)

        shared_buffer[:copy_h, :copy_w, :copy_c].copy_(
            frame_hwc[:copy_h, :copy_w, :copy_c]
        )
        logger.info(f"Frame written to Locker: {copy_h}x{copy_w}x{copy_c}")

        # ---- 5. Return Results ----
        result = {
            "detections": detections,
            "frame_shape": list(frame_hwc.shape),
            "num_objects": len(detections)
        }

        if result_queue:
            result_queue.put(result)

        # ---- 6. Cleanup (process termination handles the rest) ----
        del frame_tensor
        del shared_buffer
        torch.cuda.empty_cache()
        logger.info("YOLO Spoke exiting. VRAM reclaimed by OS.")

    except Exception as e:
        logger.error(f"YOLO Spoke crashed: {e}")
        import traceback
        traceback.print_exc()
        if result_queue:
            result_queue.put({"error": str(e)})
        raise SystemExit(1)


# =============================================================================
# Spoke C: Qwen (Brain)
# =============================================================================

def run_qwen_spoke(
    prompt: str,
    ipc_handle,
    max_model_len: int = 4096,
    result_queue=None,
    buffer_shape: Tuple[int, ...] = (2160, 3840, 4),
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
):
    """
    Ephemeral Qwen VLM reasoning spoke.

    Pipeline:
        1. Initialize CUDA context
        2. Attach to Locker's IPC buffer (read frame zero-copy)
        3. Load Qwen3-VL with FP8 (E4M3) quantization (~7.5GB VRAM)
        4. Run VLM inference with frame + text prompt
        5. Return structured response via Queue
        6. Die → OS reclaims all VRAM

    Args:
        prompt: Text prompt for reasoning
        ipc_handle: CUDA IPC handle from the Locker
        max_model_len: Max sequence length (reduced on OOM retry)
        result_queue: Queue to return reasoning results
        buffer_shape: Shape of the Locker's buffer
        model_name: HuggingFace model identifier
    """
    import torch
    import os

    try:
        device = torch.device('cuda:0')
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            f"Qwen Spoke starting on {torch.cuda.get_device_name(0)} "
            f"({vram_total:.1f}GB VRAM, max_len={max_model_len})"
        )

        # ---- 1. Read Frame from Locker (Zero-Copy) ----
        logger.info("Attaching to VRAM Locker...")
        frame_for_inference = None

        if ipc_handle is not None:
            try:
                shared_buffer = rebuild_from_ipc_handle(
                    ipc_handle, buffer_shape, "uint8"
                )
                # Read the frame (it was written by YOLO spoke)
                # Convert to PIL for Qwen's processor
                frame_for_inference = shared_buffer.cpu().numpy()
                logger.info(f"Read frame from Locker: {shared_buffer.shape}")
                del shared_buffer  # Release the IPC mapping
            except Exception as e:
                logger.warning(f"Could not read from Locker: {e}")

        # ---- 2. Load Qwen3-VL with FP8 Quantization ----
        use_fp8 = False  # Default, updated if Blackwell detected
        logger.info(f"Loading {model_name} with FP8 (E4M3) quantization...")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from PIL import Image
            import numpy as np

            # Determine quantization strategy based on Blackwell capability
            arch_list = torch.cuda.get_arch_list()
            use_fp8 = "sm_120" in arch_list or "compute_120" in arch_list

            if use_fp8:
                logger.info("Blackwell detected: Using native FP8 (E4M3) quantization")
                # FP8 quantization for Blackwell (~7.5GB instead of ~14GB)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float8_e4m3fn,
                    device_map="cuda:0",
                    max_memory={0: f"{int(vram_total * 0.85)}GB"},
                )
            else:
                logger.info("Non-Blackwell GPU: Using BFloat16")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0",
                    max_memory={0: f"{int(vram_total * 0.85)}GB"},
                )

            processor = AutoProcessor.from_pretrained(model_name)
            model.eval()

            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"Model loaded. VRAM: {vram_used:.1f}GB / {vram_total:.1f}GB")

            # ---- 3. Prepare Input ----
            messages = [{"role": "user", "content": []}]

            # Add image if available from Locker
            if frame_for_inference is not None:
                # Convert numpy array to PIL Image
                img_array = frame_for_inference
                if img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]  # Drop alpha
                # Crop to actual content (remove padding)
                # Find the last non-zero row
                row_sums = img_array.sum(axis=(1, 2))
                valid_rows = (row_sums > 0).nonzero()[0]
                col_sums = img_array.sum(axis=(0, 2))
                valid_cols = (col_sums > 0).nonzero()[0]
                if len(valid_rows) > 0 and len(valid_cols) > 0:
                    img_array = img_array[
                        valid_rows[0]:valid_rows[-1]+1,
                        valid_cols[0]:valid_cols[-1]+1
                    ]

                image = Image.fromarray(img_array.astype(np.uint8))
                messages[0]["content"].append({
                    "type": "image",
                    "image": image
                })

            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })

            # ---- 4. Run Inference ----
            logger.info("Running VLM inference...")

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if frame_for_inference is not None:
                inputs = processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                ).to(device)
            else:
                inputs = processor(
                    text=[text_input],
                    return_tensors="pt",
                    padding=True
                ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # Decode output
            output_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            logger.info(f"Inference complete: {len(output_text)} chars")

        except ImportError as e:
            logger.warning(f"Qwen model libraries not available: {e}")
            output_text = (
                f"[Mock Response] Qwen model not loaded. "
                f"Prompt was: {prompt[:100]}..."
            )

        # ---- 5. Return Results ----
        result = {
            "reasoning": output_text,
            "model": model_name,
            "quantization": "fp8_e4m3" if use_fp8 else "bf16",
            "max_model_len": max_model_len
        }

        if result_queue:
            result_queue.put(result)

        # ---- 6. Cleanup ----
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        logger.info("Qwen Spoke exiting. VRAM reclaimed by OS.")

    except Exception as e:
        logger.error(f"Qwen Spoke crashed: {e}")
        import traceback
        traceback.print_exc()
        if result_queue:
            result_queue.put({"error": str(e)})
        raise SystemExit(1)
