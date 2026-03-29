---
trigger: always_on
---

PROJECT AETHER v2.1 - TECHNICAL OBJECTIVES
Updated: 2026-03-21 (Architecture Blueprint 2.1)

SECTION A: INFRASTRUCTURE (10 objectives)
A1.1: WSL2 (Ubuntu 24.04) installed and configured
A1.2: Custom Docker container built (FFmpeg 7.1 source + PyTorch Nightly cu128)
A1.3: Container runs with GPU passthrough (sm_120 detected)
A1.4: Container accesses project files via mounted volumes
A1.5: Container environment is reproducible (Dockerfile committed)
A2.1: All model weights, HF caches, and Docker VHDX stored on M: drive (0 bytes on C:)
A2.2: Python dependencies pinned and conflict-free
A2.3: HF_HOME environment variable enforced to M: volume
A3.1: Peak VRAM never exceeds 14GB
A3.2: Parallel Residency model active (easytranscriber + Qwen3.5 always resident, ~4GB combined)
SECTION B: TREND DISCOVERY (7 objectives)
B1.1: Crawl4AI AsyncWebCrawler integration for TikTok/Twitter/YouTube trending
B1.2: JsonCssExtractionStrategy schemas defined per platform
B1.3: pytrends integration for Google Trends validation
B1.4: YouTube Data API v3 integration (velocity/acceleration tracking)
B2.1: Gemma-3n-E2B-IT NPU scoring via OpenVINO GenAI (or Gemini API fallback)
B2.2: System ranks trends by viral potential (0-100 score)
B2.3: Output structured JSON with topic, keywords, sources, confidence
SECTION C: VIDEO SOURCING (7 objectives)
C1.1: System searches for videos related to trend topic
C1.2: Results filtered by quality criteria
C1.3: User approves video selection before download
C2.1: Video downloaded to downloaded_videos/ directory
C2.2: Download includes metadata
C3.1: Resolution check (>=1080p), frame rate (>=30fps)
C3.2: Duration check (15-30 minutes), audio quality (clear speech)
SECTION D: TRANSCRIPTION & ANALYSIS (10 objectives)
D1.1: Full video transcribed using easytranscriber (Distil-Whisper Large-v3, CTranslate2)
D1.2: Transcription includes word-level timestamps (GPU-parallelized forced alignment)
D1.3: Word Error Rate < 10% on clear speech
D1.4: Transcription runs on GPU with >10x realtime speed
D2.1: Speaker diarization integrated
D2.2: Speaker labels attached to transcript segments
D3.1: Video analyzed using Qwen3.5-0.8B (INT4 via vLLM sidecar)
D3.2: Scene changes detected with timestamps; key visual elements tagged
D3.3: Analysis runs on GPU (<2GB VRAM footprint for 0.8B model)
D4.1: System identifies 3-4 viral-worthy segments with start/end time, description, score
SECTION E: EDITING ENGINE (18 objectives)
E1.1: Video cut at exact frame boundaries (Blackwell NVENC AV1 hardware re-encoding)
E1.2: Cuts respect speech boundaries (no mid-word cuts)
E1.3: Multiple segments extracted in single run
E2.1: Subject detection using YOLO26s-Pose (TensorRT FP16)
E2.2: Pose estimation for cinematic framing (Rule of Thirds)
E2.3: Tracking follows subject smoothly (Kalman filter)
E2.4: Virtual camera has no jerky movements
E2.5: Face never cut off at chin/forehead
E2.6: Works for talking-head and dynamic content
E3.1: Captions synced to word-level timestamps
E3.2: Current word highlighted/emphasized (pycaps animation)
E3.3: Caption style matches trending formats (CSS-styled)
E3.4: Captions burned into video (GPU compositing)
E3.5: Captions readable on mobile
E4.1: Audio/video synchronized (-50ms perceptual offset)
E4.2: Audio levels normalized
E5.1: Boredom Detector active (TI/AE monitoring)
E5.2: Effects implemented via Kornia (GPU-native)
SECTION F: OUTPUT GENERATION (11 objectives)
F1.1: Output is 1080x1920 (9:16 vertical)
F1.2: Frame rate matches source or 30fps minimum
F1.3: Encoding uses PyNvVideoCodec 2.1.0 + NVENC AV1 (zero-copy GPU)
F1.4: File size optimized for upload
F2.1: Title generated via Gemini 3 Pro (PydanticAI typed output)
F2.2: Description generated with keywords
F2.3: Tags/hashtags generated for discoverability
F2.4: Thumbnail extracted or generated
F3.1: Each short in own subdirectory
F3.2: Each directory has video.mp4, metadata.json, thumbnail.jpg, transcript.txt
F3.3: All shorts from source grouped together
SECTION G: QUALITY ASSURANCE (8 objectives)
G1.1: Audio sync verification passes (< 0.1s drift)
G1.2: Resolution verification passes
G1.3: Duration verification passes (30-90s)
G1.4: No black frames detected
G2.1: Hook exists in first 3 seconds
G2.2: Content coherent as standalone clip
G2.3: Natural ending (not mid-sentence)
G2.4: Captions accurate to speech
SECTION H: END-TO-END PIPELINE (7 objectives)
H1.1: LangGraph state machine triggers entire pipeline
H1.2: User approval points clearly prompted (PydanticAI structured outputs)
H1.3: Redis Streams checkpointing — pipeline resumes from failure without reprocessing
H1.4: Progress logged and visible
H2.1: Full pipeline < 30 minutes for 20-min source
H2.2: Transcription < 1 minute for 20-min video
H2.3: Encoding < 1 minute per short (zero-copy pipeline)
COMPLETION CRITERIA
Project is COMPLETE when all 78 objectives are met.

MVP requires: Sections A + D + E1-E4 + F1/F3 + G1 + H1.1 complete.