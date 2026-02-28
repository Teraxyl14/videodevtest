# 🏭 Blackwell Factory: Model Download & Setup Guide

**Architecture:** Hub-and-Spoke (Dual Environment)
**Hardware Target:** RTX 5080 Mobile (16GB VRAM)

---

## 🏗️ Step 1: Create Isolated Environments
Avoid "Dependency Hell" by strictly separating Audio and Video libraries.

### Environment A: `env_video` (Vision + Gen + Tracking)
```powershell
conda create -n env_video python=3.11 -y
conda activate env_video
# Install Torch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install Core Libs
pip install diffusers transformers accelerate opencv-python llama-cpp-python huggingface_hub
```

### Environment B: `env_audio` (ASR + NeMo)
```powershell
conda create -n env_audio python=3.10 -y
conda activate env_audio
# Install Cython first
pip install cython
# Install NeMo Toolkit (This is heavy)
pip install nemo_toolkit[all]
```

---

## 📥 Step 2: Download GGUF Models (The "Pro" Models)
Save all models to: `m:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\autonomous_trend_agent\models\`

### 1. Vision: Qwen2.5-VL-7B (GGUF Q4)
*   **Why:** "Naive Dynamic Resolution" for video. Fits in 6.5GB.
*   **File:** `Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
*   **Source:** HuggingFace `bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF`
*   **Command:**
    ```powershell
    huggingface-cli download bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --local-dir m:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\autonomous_trend_agent\models
    ```

### 2. Generation: Wan2.2-14B (GGUF Q4)
*   **Why:** SOTA 14B model compressed to fit 16GB.
*   **File:** `Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf`
*   **Source:** HuggingFace `QuantStack/Wan2.2-T2V-A14B-GGUF` (HighNoise folder)
*   **Command:**
    ```powershell
    huggingface-cli download QuantStack/Wan2.2-T2V-A14B-GGUF HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf --local-dir m:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\autonomous_trend_agent\models
    ```

### 3. Audio: Parakeet TDT 0.6B
*   **Why:** 3000x Real-time factor + Word Timestamps.
*   **Command (in env_audio):**
    ```powershell
    # Downloads automatically on first run of `spokes/audio.py`
    # or manual:
    import nemo.collections.asr as nemo_asr
    nemo_asr.models.EncDecRNNTModel.from_pretrained("nvidia/parakeet-tdt-0.6b")
    ```

### 4. Tracking: SAM 2.1 Small
*   **Command:**
    ```powershell
    # Download manually to models/ directory
    Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile "m:\Projects\AI_Video\AI_Video-20260116T154526Z-3-001\AI_Video\autonomous_trend_agent\models\sam2.1_hiera_small.pt"
    ```
