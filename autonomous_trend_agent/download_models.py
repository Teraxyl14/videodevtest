"""
Model Pre-Downloader — Downloads all required models before first pipeline run.
Run inside Docker: python download_models.py

Models downloaded:
  1. Qwen3-VL-8B-Instruct  (~16GB FP16 safetensors → HuggingFace cache)
  2. Parakeet TDT 0.6B v2   (~1.2GB → NeMo cache)
  3. YOLOv12n               (~6MB → Ultralytics cache)
"""

import os
import sys
import time

def download_qwen3():
    """Download Qwen3-VL-8B-Instruct model weights."""
    print("\n" + "=" * 60)
    print("  [1/3] Downloading Qwen3-VL-8B-Instruct")
    print("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        
        print(f"  Model: {model_id}")
        print(f"  Cache: {cache_dir}")
        print(f"  This will download ~16GB. Please wait...\n")
        
        start = time.time()
        path = snapshot_download(
            model_id,
            cache_dir=os.path.join(cache_dir, "hub"),
            resume_download=True,
        )
        elapsed = time.time() - start
        
        print(f"\n  ✅ Qwen3-VL downloaded to: {path}")
        print(f"     Time: {elapsed:.0f}s")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Qwen3-VL download failed: {e}")
        print("     If 'gated model', run: huggingface-cli login")
        return False


def download_parakeet():
    """Download Parakeet TDT 0.6B v2 via NeMo or HuggingFace fallback."""
    print("\n" + "=" * 60)
    print("  [2/3] Downloading Parakeet TDT 0.6B v2")
    print("=" * 60)
    
    # Try NeMo first (preferred)
    try:
        import nemo.collections.asr as nemo_asr
        
        model_name = "nvidia/parakeet-tdt-0.6b-v3"
        print(f"  Model: {model_name} (via NeMo)")
        print(f"  This will download ~1.2GB. Please wait...\n")
        
        start = time.time()
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        elapsed = time.time() - start
        
        print(f"\n  ✅ Parakeet TDT downloaded successfully (NeMo)")
        print(f"     Time: {elapsed:.0f}s")
        
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except ImportError:
        print("  ⚠️  NeMo not installed. Using HuggingFace fallback...\n")
    except Exception as e:
        print(f"  ⚠️  NeMo download failed: {e}")
        print("  Falling back to HuggingFace download...\n")
    
    # Fallback: download from HuggingFace directly
    try:
        from huggingface_hub import snapshot_download
        
        model_id = "nvidia/parakeet-tdt-0.6b-v3"
        cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        
        print(f"  Model: {model_id} (via HuggingFace)")
        print(f"  Cache: {cache_dir}")
        print(f"  This will download ~1.2GB. Please wait...\n")
        
        start = time.time()
        path = snapshot_download(
            model_id,
            cache_dir=os.path.join(cache_dir, "hub"),
        )
        elapsed = time.time() - start
        
        print(f"\n  ✅ Parakeet TDT downloaded to: {path}")
        print(f"     Time: {elapsed:.0f}s")
        print(f"     Note: NeMo will load this from HF cache on first run.")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Parakeet download failed: {e}")
        return False



def download_yolo():
    """Download YOLOv12n model."""
    print("\n" + "=" * 60)
    print("  [3/3] Downloading YOLOv12n")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        print("  Model: yolo12n.pt")
        print("  This will download ~6MB. Please wait...\n")
        
        start = time.time()
        # This triggers automatic download
        model = YOLO("yolo12n.pt")
        elapsed = time.time() - start
        
        print(f"\n  ✅ YOLOv12n downloaded successfully")
        print(f"     Time: {elapsed:.0f}s")
        
        # Download YOLOv26n (Latest February 2026 stable architecture)
        print("  Downloading YOLO26 Nano (stable)...")
        model_v26 = YOLO("yolo26n.pt")
        print("  ✅ YOLO26 Nano downloaded")
        
        # Download YOLOv12-Face Nano (Path B: face-specific detection)
        print("  Downloading YOLOv12-Face Nano (face detection)...")
        face_model_path = "/app/models/yolov12n-face.pt"
        if not os.path.exists(face_model_path):
            import torch
            os.makedirs("/app/models", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12n-face.pt",
                face_model_path
            )
        print(f"  ✅ YOLOv12-Face Nano downloaded to {face_model_path}")
        
        del model, model_v26
        return True
        
    except Exception as e:
        print(f"\n  ❌ YOLO download failed: {e}")
        return False


def verify_gpu():
    """Check GPU is accessible."""
    print("=" * 60)
    print("  GPU Verification")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ GPU: {name} ({vram:.1f}GB VRAM)")
            return True
        else:
            print("  ⚠️  No GPU detected. Models will still download (CPU mode).")
            return True  # Can still download without GPU
    except Exception as e:
        print(f"  ⚠️  GPU check failed: {e}")
        return True  # Continue anyway


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI VIDEO AGENT — MODEL PRE-DOWNLOADER")
    print("=" * 60)
    
    verify_gpu()
    
    results = {}
    results["Qwen3-VL"] = download_qwen3()
    results["Parakeet"] = download_parakeet()
    results["YOLO"] = download_yolo()
    
    # Summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results.items():
        status = "✅ Ready" if ok else "❌ Failed"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    
    print(f"\n  Gemini 3 Flash: ☁️  Cloud API (no download needed)")
    
    if all_ok:
        print("\n  🚀 All models ready! You can now run the pipeline.")
    else:
        print("\n  ⚠️  Some models failed. Check errors above.")
    
    print("")
    sys.exit(0 if all_ok else 1)
