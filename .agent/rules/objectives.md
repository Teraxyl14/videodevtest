---
trigger: always_on
---

AUTONOMOUS TREND AGENT - TECHNICAL OBJECTIVES
==============================================

This defines all objectives required for project completion.
Updated: 2026-02-12 (Golden Stack v2.0)

SECTION A: INFRASTRUCTURE (12 objectives)
------------------------------------------
A1.1: WSL2 (Ubuntu 24.04) installed and configured
A1.2: Custom Docker container built (FFmpeg 7.1 source + PyTorch Nightly)
A1.3: Container runs with GPU passthrough (sm_120 detected)
A1.4: Container accesses project files via mounted volumes
A1.5: Container environment is reproducible (Dockerfile committed)

A2.1: All model weights stored in models/ directory (not C: drive)
A2.2: Python dependencies pinned (PyNvVideoCodec 2.0.2, etc.)
A2.3: Project portable to another machine without reinstallation
A2.4: No hardcoded user-specific paths

A3.1: Peak VRAM never exceeds 14GB (FP8 Quantization used)
A3.2: Models loaded/unloaded sequentially (Hub-and-Spoke + VRAM Locker)
A3.3: VRAM fully released between pipeline phases (Process Termination)

SECTION B: TREND DISCOVERY (9 objectives)
-----------------------------------------
B1.1: YouTube Data API v3 integration (Velocity/Acceleration tracking)
B1.2: Twitter/X Guest Token Scraping (curl_cffi + TLS spoofing)
B1.3: TikTok Node.js Sidecar configured for X-Bogus signing
B1.4: pytrends integration for Google Trends validation
B1.5: "Z-Score Virality" algorithm implemented (Cross-platform normalization)

B2.1: Gemini 3 Flash API integration via google-genai SDK
B2.2: System ranks trends by viral potential
B2.3: System recommends content angle for selected trend
B2.4: Output structured JSON with topic, keywords, sources, confidence

SECTION C: VIDEO SOURCING (10 objectives)
-----------------------------------------
C1.1: System searches for videos related to trend topic
C1.2: Results filtered by quality criteria
C1.3: User approves video selection before download

C2.1: Video downloaded to downloaded_videos/ directory
C2.2: Download includes metadata
C2.3: Video validated before processing

C3.1: Resolution check (>=1080p)
C3.2: Frame rate check (>=30fps)
C3.3: Duration check (15-30 minutes)
C3.4: Audio quality check (clear speech)

SECTION D: TRANSCRIPTION & ANALYSIS (12 objectives)
----------------------------------------------------
D1.1: Full video transcribed using Parakeet TDT 0.6B v2
D1.2: Transcription includes word-level timestamps
D1.3: Word Error Rate < 10% on clear speech
D1.4: Transcription runs on GPU with >10x realtime speed

D2.1: Speaker diarization using NeMo TitaNet
D2.2: Speaker labels attached to transcript segments
D2.3: Multi-speaker conversations correctly attributed

D3.1: Video analyzed using Qwen2.5-VL-7B-Instruct (FP8)
D3.2: Scene changes detected with timestamps
D3.3: Key visual elements tagged
D3.4: Analysis runs on GPU inside Docker (<8GB VRAM)

D4.1: System identifies 3-4 viral-worthy segments
D4.2: Each segment has start/end time, description, score
D4.3: Segments are diverse (no overlap)

SECTION E: EDITING ENGINE (18 objectives)
-----------------------------------------
E1.1: Video cut at exact frame boundaries (Zero-Copy)
E1.2: Cuts respect speech boundaries (no mid-word cuts)
E1.3: Multiple segments extracted in single run

E2.1: Subject detection using YOLOv11n (Int8 TensorRT)
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
E4.3: No audio drift over duration

E5.1: Boredom Detector active (TI/AE monitoring)
E5.2: Screen shake for impactful moments (RMS energy trigger)
E5.3: Visual callouts for on-screen elements
E5.4: Effects implemented via Kornia (GPU-native)

SECTION F: OUTPUT GENERATION (11 objectives)
---------------------------------------------
F1.1: Output is 1080x1920 (9:16 vertical)
F1.2: Frame rate matches source or 30fps minimum
F1.3: Encoding uses PyNvVideoCodec + NVENC (zero-copy GPU)
F1.4: File size optimized for upload

F2.1: Title generated via Gemini Flash based on content/trend
F2.2: Description generated with keywords
F2.3: Tags/hashtags generated for discoverability
F2.4: Thumbnail extracted or generated

F3.1: Each short in own subdirectory
F3.2: Naming follows convention
F3.3: Each directory has video.mp4, metadata.json, thumbnail.jpg, transcript.txt
F3.4: All shorts from source grouped together

SECTION G: QUALITY ASSURANCE (8 objectives)
-------------------------------------------
G1.1: Audio sync verification passes (< 0.1s drift)
G1.2: Resolution verification passes
G1.3: Duration verification passes (30-90s)
G1.4: No black frames detected

G2.1: Hook exists in first 3 seconds
G2.2: Content coherent as standalone clip
G2.3: Natural ending (not mid-sentence)
G2.4: Captions accurate to speech

SECTION H: END-TO-END PIPELINE (9 objectives)
----------------------------------------------
H1.1: Single command triggers entire pipeline
H1.2: User approval points clearly prompted
H1.3: Pipeline resumes from failure without reprocessing
H1.4: Progress logged and visible

H2.1: Full pipeline < 30 minutes for 20-min source
H2.2: Transcription < 1 minute for 20-min video
H2.3: Encoding < 1 minute per short (zero-copy pipeline)

H3.1: Pipeline handles edge cases gracefully
H3.2: Errors logged with actionable messages
H3.3: Partial outputs not left corrupted

COMPLETION CRITERIA
-------------------
Project is COMPLETE when all 89 objectives are met.

MVP (Minimum Viable Product) requires:
- Section A complete (Infrastructure)
- Section D complete (Analysis)
- E1-E4 complete (Core Editing, not E5 effects)
- F1, F3 complete (Basic Output)
- G1 complete (Automated Checks)
- H1.1 complete (Single Command)

TOTAL: 89 objectives across 8 sections
