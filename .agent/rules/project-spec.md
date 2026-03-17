---
trigger: always_on
---

AI VIDEO AGENT - PROJECT MANUAL (v2.0)
MISSION: Build an autonomous end-to-end pipeline that discovers trending topics online, sources high-quality long-form video content, intelligently edits into 3-4 viral-ready shorts, and outputs organized ready-to-upload content with metadata.

HARDWARE CONSTRAINTS & EXECUTION ENVIRONMENT
GPU: NVIDIA RTX 5080 Mobile (16GB VRAM, Blackwell sm_120)
CPU: Intel Core Ultra 7 255HX
RAM: 32GB DDR5
OS: WSL2 (Ubuntu 24.04) running on Windows 11
Storage: ABSOLUTE CONSTRAINT. No project data, Docker virtual disks (ext4.vhdx), or HF caches may reside on the C: drive. Everything must be securely mounted to the M: drive.
Container: Custom Docker image based on nvcr.io/nvidia/pytorch:25.03-py3
CRITICAL: Must compile FFmpeg 7.1 from source with --enable-cuda --enable-nvdec --enable-nvenc.
CRITICAL: Must use PyTorch Nightly (cu128) for native sm_120 support.
CRITICAL: Must mount /usr/lib/wsl/lib to /usr/lib/x86_64-linux-gnu for driver visibility.
VRAM ARCHITECTURE (Blackwell-Triad Hub)
Constraint: Maximum 14GB active models. 16GB Total VRAM.
Pattern: Hub-and-Spoke with Persistent Locker.
Hub: Lightweight CPU process (Orchestrator).
Locker: Persistent background process holding a 100MB Shared CUDA Buffer (via IPC Handle) for zero-copy data transfer.
Spokes: Ephemeral processes (ASR, YOLO, Qwen) that attach to the Locker, process data, and terminate to free VRAM.
Model Quantization:
Qwen3-4B-Instruct-AWQ: Must use INT4 quantization via vLLM (approx 3.8GB).
YOLOv11: Int8 TensorRT engine.
FIVE-PHASE PIPELINE
Phase 1 - Trend Discovery
YouTube: Official Data API v3. Monitor Velocity and Acceleration of views.
Twitter/X: Guest Token Scraping via curl_cffi (mimicking browser TLS).
TikTok: Node.js Sidecar service for X-Bogus/msToken signing.
Scoring: Z-Score Virality Algorithm to normalize metrics across platforms.
Validation: Google Trends via pytrends.
Phase 2 - Video Sourcing
Action: Find and download video matching the trend.
Quality Check: >1080p, >30fps, Clear Speech, No Music/Watermarks.
Phase 3 - Analysis
Transcription: WhisperX (Distil-Whisper Large-v3) with wav2vec2 forced alignment for ultra-fast GPU ASR and millisecond-level word timestamps.
Diarization: Pyannote 4.0 natively integrated for speaker clustering.
Visual Analysis: Qwen3-4B. Chain-of-Thought (CoT) inference in INT4 via Dual-Mode vLLM engine (<5.6GB footprint) for scene understanding and segment scoring.
Phase 4 - Editing (Viral Engine)
Reframing: Kalman Filter smoothing on YOLOv11 detections. Target: Eyes at 33% height (Rule of Thirds).
Captions: pycaps with -50ms Perceptual Offset (sync with brain processing time).
Boredom Detector: Monitor TI (Temporal Information) and AE (Audio Excitement). Trigger zooms/cuts if low.
Effects: GPU-native transforms via Kornia.
Phase 5 - Export
Format: 1080x1920 (9:16), 30fps+.
Encoder: PyNvVideoCodec + NVENC AV1 Hardware Encoding.
Metadata: JSON with title, description, tags, and viral score.
TECHNOLOGY STACK (GOLDEN STACK v2.0)
Language: Python 3.10+ (Orchestrator), Node.js (TikTok Sidecar).
AI Models: Qwen3-4B-Instruct-AWQ (Vision/vLLM), WhisperX + Pyannote 4.0 (Audio), YOLOv11 (Tracking), Gemini 3 Flash (Reasoning/Ranking via google-genai).
Video Pipeline: PyNvVideoCodec (v2.0.2), FFmpeg 7.1 (Source), Kornia (Effects).
Infrastructure: Docker, NVIDIA Container Toolkit, WSL2.
DECISION AUTHORITY
Agent: Segment selection, caption styling, reframing, effect application.
User: Web search execution, video download approval, major architectural changes, publishing.
QUALITY STANDARDS
Audio Sync: < 0.1s drift (Target: -0.05s perceptual offset).
Retention: Hook in first 3s. No "boring" sections (>2s low TI/AE).
Visuals: Face never cut off. Smooth camera motion (Kalman).
Performance: Full pipeline < 30 mins for 20-min source.