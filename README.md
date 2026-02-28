# Autonomous Trend Agent — Golden Stack v2.0

An autonomous end-to-end pipeline that discovers trending topics online, sources high-quality long-form video content, intelligently edits it into viral-ready shorts, and outputs organized ready-to-upload content with metadata.

## Overview

The AI Video Pipeline automatically identifies trends and creates vertical short-form videos with zero human intervention. Utilizing modern vision models (Qwen2.5-VL), fast transcription (Parakeet TDT), and robust editing (YOLOv11 tracking + Kornia effects), it transforms 20-minute videos into 3-4 viral-ready shorts.

## Features

- **Trend Discovery:** Youtube Data API, Twitter/X Scraping, TikTok Sidecar, and PyTrends validation.
- **Smart Transcription:** Ultra-fast ASR via Parakeet TDT 0.6B with word-level timestamps.
- **Visual Analysis:** Qwen2.5-VL-7B (FP8 Quantization) for scene understanding.
- **Intelligent Reframing:** YOLOv11 for face tracking and Kalman filter smoothing for cinematic reframing.
- **Dynamic Captions:** Burned-in, word-synced subtitles with a -50ms perceptual offset.
- **Real-Time Dashboard:** Monitor pipeline progress, logs, and system utilization via a local web interface.
- **Crash Hardened:** Watchdogs for thread safety, CUDA out-of-memory guards, and robust race-condition handling.

## Requirements Environment

- **OS:** WSL2 (Ubuntu 24.04) running on Windows 11
- **Hardware:** NVIDIA RTX 5080 Mobile (16GB VRAM, Blackwell sm_120) / Core Ultra 7 255HX / 32GB RAM
- **Docker:** Custom Docker Image based on `nvcr.io/nvidia/pytorch:25.03-py3`
- **FFmpeg:** Compiled from source with CUDA support.

## Project Structure

```text
├── autonomous_trend_agent/ # Core logic (API, Orchestrator, Dashboard, Editors)
├── docker/                 # Build contexts for custom Docker images
├── docs/                   # Full project specs and manuals
├── tests/                  # Scripts for unit testing
├── downloaded_videos/      # (Auto-generated) Ignored in git
├── output/                 # (Auto-generated) Ignored in git
├── models/                 # (Auto-generated) Ignored in git
├── .env.example            # Example configuration keys
├── docker-compose.yml      # Local container orchestration
└── Dockerfile              # Pipeline environment build
```

## Getting Started

1. **Clone the repository.**
2. **Download Models:**
   ```bash
   python autonomous_trend_agent/download_models.py
   ```
3. **Environment Setup:** Create a `.env` file based on `.env.example`.
4. **Build & Run with Docker-Compose:**
   ```bash
   docker-compose up --build -d
   ```
5. **Dashboard Access:** Open your browser to `http://localhost:8080`.

## Architecture Details

- **Hub-and-Spoke VRAM Management:** The system sequences model loading (ASR -> Qwen -> YOLO) to stay under the 14GB VRAM constraint. 
- **Graceful Fault Tolerance:** Built-in safeguards ensure failed stages do not crash the orchestrator, with full thread recovery logic.

## Documentation

Additional documents are located in `docs/project_specs`.

---
*Built as a fully automated agentic pipeline.*
