# SETUP.md - Week 1 Setup Instructions

## Model Downloads

### 1. Qwen2.5-VL-7B-GGUF (Q4_K_M variant)

**Size:** ~3.5GB  
**Location:** `models/qwen2.5-vl-7b-q4.gguf`

```bash
# Install huggingface-cli
pip install huggingface-hub[cli]

# Download GGUF model
huggingface-cli download \
  Qwen/Qwen2.5-VL-7B-Instruct-GGUF \
  qwen2.5-vl-7b-instruct-q4_k_m.gguf \
  --local-dir models \
  --local-dir-use-symlinks False

# Rename for consistency
mv models/qwen2.5-vl-7b-instruct-q4_k_m.gguf models/qwen2.5-vl-7b-q4.gguf
```

### 2. Install Dependencies

```bash
cd autonomous_trend_agent
pip install -r requirements_v2.txt
```

**Note:** `llama-cpp-python` may require specific build for CUDA:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Verify Installation

```bash
# Test VRAM orchestrator
python core/vram_manager.py

# Expected output:
# Total VRAM: 16.0GB
# Free VRAM: 15.xGB
# ✓ VRAM check passed for analysis: ...
```

## Week 1 Tasks

- [x] Create project structure
- [x] Implement VRAM orchestrator
- [ ] Download Qwen2.5-VL-7B-GGUF
- [ ] Implement Qwen analyzer
- [ ] Integrate Gemini QC
- [ ] Test end-to-end Phase 1

## Troubleshooting

**Issue:** `llama-cpp-python` not detecting CUDA
**Solution:** Set `CMAKE_ARGS` during install (see above)

**Issue:** GGUF model not found
**Solution:** Check `models/` directory, verify filename matches config

**Issue:** VRAM errors during phase transitions
**Solution:** Close other GPU applications, verify 16GB card
