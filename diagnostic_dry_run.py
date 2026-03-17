import os
import gc
import time
import torch
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(project_root))

def bytes_to_mb(b):
    return b / (1024 * 1024)

def run_dry_diagnostics():
    print("==================================================")
    print(" AI VIDEO AGENT v2.0 - GOLDEN STACK DIAGNOSTICS")
    print("==================================================")
    
    # 1. Environment Verification
    print("\n[Phase 1] Validating Host & PyTorch Topology")
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available! Ensure you are running inside the GPU passthrough Docker container.")
        sys.exit(1)
        
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Device: {device_name}")
    print(f"Total VRAM: {bytes_to_mb(total_memory):.2f} MB")
    
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
    
    # 2. VRAM Locker Allocation
    print("\n[Phase 2] Hub-and-Spoke Memory Initialization")
    from autonomous_trend_agent.core.ipc_utils import initialize_persistent_hub_buffer
    
    try:
        shared_tensor, _ = initialize_persistent_hub_buffer(100)
        print("[OK] Allocated 100MB Shared Persistent CUDA IPC Tensor.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Hub buffer: {e}")
        
    def check_vram(label):
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        print(f"   [{label}] VRAM -> Allocated: {bytes_to_mb(allocated):.1f}MB | Reserved: {bytes_to_mb(reserved):.1f}MB")

    # 3. Spoke 1: Acoustic Pipeline (WhisperX)
    print("\n[Phase 3] Testing Acoustic Spoke (WhisperX & Pyannote)")
    try:
        check_vram("Pre-Whisper")
        import whisperx
        print("[OK] whisperx module loaded. (Skipping model weights to save time, syntax valid).")
        # In a real dry run, we'd initialize the model, but this requires HF tokens and large downloads.
        # We will just verify imports.
    except ImportError as e:
        print(f"[WARNING] Module not found: {e}. (Expected if not in Docker)")
        
    check_vram("Post-Whisper")

    # 4. Spoke 2: Spatial Tracking (YOLO26s-Pose)
    print("\n[Phase 4] Testing Tracking Spoke (YOLO26s-Pose)")
    try:
        check_vram("Pre-YOLO")
        from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
        # Dry load model
        tracker = YOLOv11Tracker(device="cuda", enable_pose=True)
        # Force load models bypassing lazy setup
        tracker._ensure_loaded()
        print("[OK] YOLO26s-Pose natively instantiated on CUDA.")
        
        # Purge
        del tracker
        gc.collect()
        torch.cuda.empty_cache()
        print("[OK] YOLO module purged.")
    except Exception as e:
        print(f"[ERROR] YOLO diagnostic failed: {e}")
    
    check_vram("Post-YOLO Purge")

    # 5. Spoke 3: Cogitation (Qwen3-4B-INT4)
    print("\n[Phase 5] Testing Cognition Spoke (Qwen3-4B-INT4 vLLM)")
    try:
        check_vram("Pre-vLLM")
        from autonomous_trend_agent.brain.qwen3_video_analyzer import Qwen3VideoAnalyzer
        analyzer = Qwen3VideoAnalyzer()
        print("[OK] Qwen3 Video Analyzer instantiated.")
        print("Note: Skipping actual vLLM engine mount as it explicitly blocks memory.")
        del analyzer
    except Exception as e:
        print(f"[ERROR] Qwen3 diagnostic failed: {e}")
        
    check_vram("Post-vLLM Purge")

    # 6. Spoke 4: Video Hardware Encoding
    print("\n[Phase 6] Testing GPU Extractor Spoke (PyNvVideoCodec & AV1)")
    try:
        import PyNvVideoCodec as nvc
        print("[OK] PyNvVideoCodec detected natively.")
    except ImportError as e:
        print(f"[WARNING] PyNvVideoCodec not found: {e}. (Expected if not compiled/installed via Docker)")
        
    print("\n==================================================")
    print(" DIAGNOSTICS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_dry_diagnostics()
