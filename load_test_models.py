import sys
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

load_dotenv()

if len(sys.argv) < 2:
    print("Please specify a model test: 'yolo', 'whisper', or 'qwen'")
    sys.exit(1)

test_target = sys.argv[1].lower()

try:
    if test_target == "yolo":
        print("[TEST] Initializing YOLO26s-Pose...")
        from autonomous_trend_agent.tracking.yolo_tracker import YOLOv11Tracker
        tracker = YOLOv11Tracker(device="cuda", enable_pose=True)
        tracker._ensure_loaded()
        print("✅ YOLO weights mounted on VRAM successfully.")

    elif test_target == "whisper":
        print("[TEST] Initializing WhisperX (Distil-Large-v3)...")
        import whisperx
        hf_token = os.getenv("HF_TOKEN")
        
        asr_pipeline = whisperx.load_model("distil-large-v3", "cuda", compute_type="float16")
        print("✅ WhisperX ASR weights mounted successfully.")
        
        if hf_token:
            print("[TEST] Initializing Pyannote 4.0 Diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cuda")
            print("✅ Pyannote weights mounted successfully.")
        else:
            print("⚠️ Pyannote skipped (HF_TOKEN not found in .env).")

    elif test_target == "qwen":
        print("[TEST] Initializing Qwen3-4B-INT4 (vLLM Engine)...")
        from autonomous_trend_agent.brain.qwen3_video_analyzer import Qwen3VideoAnalyzer
        analyzer = Qwen3VideoAnalyzer()
        analyzer._ensure_loaded()
        print("✅ vLLM Engine and Qwen3 AWQ weights mounted successfully.")
        
    else:
        print(f"Unknown target: {test_target}")
        sys.exit(1)
        
except Exception as e:
    import traceback
    print(f"\n❌ FATAL ERROR MOUNTING {test_target.upper()}:")
    traceback.print_exc()
    sys.exit(1)
