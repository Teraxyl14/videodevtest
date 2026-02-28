# core/pipeline_config.py
"""
Configuration for the autonomous video factory pipeline
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model paths
# 🏭 Blackwell Factory Configuration (GGUF + Isolation)

# Environment Paths (Conda)
ENV_VIDEO_NAME = "env_video"
ENV_AUDIO_NAME = "env_audio"

# Vision Model (Transformers - Spoke 1)
# Qwen2.5-VL-7B-Instruct (4-bit quantization)
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_CONTEXT_WINDOW = 32768  # 32k context for video analysis

# Generation Model (GGUF - Spoke 3)
# Wan2.2-14B (Q4_K_M)
WAN_MODEL_PATH = MODELS_DIR / "HighNoise" / "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"

# Audio Model (NeMo - Spoke 2)
# Parakeet TDT 0.6B
PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"

# Tracking Model (PyTorch - Spoke 1/3)
SAM2_CHECKPOINT = MODELS_DIR / "sam2.1_hiera_small.pt"
SAM2_CONFIG = "sam2.1_hiera_small.yaml"

# VRAM Configuration
TOTAL_VRAM_GB = 16.0
ENABLE_VRAM_MONITORING = True

# Aggregated Paths for Spokes
MODEL_PATHS = {
    "vision_model": str(QWEN_MODEL_PATH),
    "gen_model": str(WAN_MODEL_PATH),
    "audio_model": PARAKEET_MODEL_NAME,
    "tracking_model": str(SAM2_CHECKPOINT)
}

# Gemini API Configuration
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_DAILY_LIMIT = 20  # RPD
GEMINI_MINUTE_LIMIT = 5  # RPM
GEMINI_TOKEN_LIMIT = 250000  # TPM
ENABLE_GEMINI_QC = True  # Quality control
ENABLE_GEMINI_METADATA = True  # Metadata generation

# Processing Configuration
DEFAULT_NUM_CLIPS = 4
MIN_CLIP_DURATION = 15  # seconds
MAX_CLIP_DURATION = 60  # seconds
TARGET_ASPECT_RATIO = 9/16  # Vertical for TikTok/Reels

# Quality Settings
FFMPEG_VIDEO_BITRATE = "5M"
FFMPEG_PRESET = "p4"  # NVIDIA NVENC preset
ENABLE_GPU_ACCELERATION = True

# Caption Settings
CAPTION_STYLE = "karaoke"  # Word-by-word animation
CAPTION_FONT_SIZE = 48
CAPTION_POSITION = "bottom"  # bottom, top, center

GENERATION_PARAMS = {
    "num_clips": DEFAULT_NUM_CLIPS,
    "min_duration": MIN_CLIP_DURATION,
    "max_duration": MAX_CLIP_DURATION,
    "aspect_ratio": TARGET_ASPECT_RATIO,
    "fps": 24  # Default
}

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
