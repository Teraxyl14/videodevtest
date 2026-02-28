# ============================================================================
# AUTONOMOUS TREND AGENT - V2 GOLDEN STACK DOCKER IMAGE
# ============================================================================
# Single-stage build on NGC PyTorch base for Blackwell (sm_120) support
# Includes: FFmpeg 7.1 (source), PyTorch Nightly (cu128), PyNvVideoCodec 2.0.2
# ============================================================================

FROM nvcr.io/nvidia/pytorch:25.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/lib/python3.12/dist-packages/PyNvVideoCodec/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

# Blackwell-specific: Enable expandable memory segments
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV TORCH_CUDA_ARCH_LIST="12.0"

# Model cache directories (mounted volumes)
ENV TRANSFORMERS_CACHE=/app/models/huggingface
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# =============================================================================
# 1. Install Build Dependencies for FFmpeg 7.1
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    yasm pkg-config libgnutls28-dev libdrm-dev libva-dev \
    libvdpau-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    libssl-dev libx264-dev libx265-dev libnuma-dev \
    libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev \
    libvorbis-dev libgl1-mesa-dev git wget build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Install NVIDIA Codec Headers (Required for NVDEC/NVENC in FFmpeg)
# =============================================================================
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make && make install && \
    cd .. && rm -rf nv-codec-headers

# =============================================================================
# 3. Compile FFmpeg 7.1 from Source with Full NVIDIA Acceleration
# =============================================================================
RUN wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz && \
    tar -xzf ffmpeg-7.1.tar.gz && \
    cd ffmpeg-7.1 && \
    ./configure \
    --prefix=/usr \
    --enable-shared \
    --enable-gpl \
    --enable-nonfree \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvdec \
    --enable-nvenc \
    --enable-libdrm \
    --disable-static \
    --enable-pic && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    cd .. && rm -rf ffmpeg-7.1 ffmpeg-7.1.tar.gz

# =============================================================================
# 4. PyTorch — NGC 25.03 already ships torch 2.7.0 with sm_120 (Blackwell)
#    No nightly needed. If you need bleeding-edge nightly, uncomment below:
# =============================================================================
# RUN pip config set global.constraint '' && \
#     pip install --pre --force-reinstall torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/nightly/cu128

# =============================================================================
# 4b. Install NeMo ASR Toolkit (for Parakeet TDT transcription)
# =============================================================================
RUN pip install --no-cache-dir "nemo_toolkit[asr]"

# =============================================================================
# 5. Install PyNvVideoCodec (Zero-Copy GPU Video Decoder/Encoder) - v2.1.0 required
# =============================================================================
RUN pip install PyNvVideoCodec==2.1.0

# =============================================================================
# 6. CRITICAL FIX: Remove bundled FFmpeg libs and setup WSL2 library paths
#    Prevents symbol lookup errors and fixes Blackwell Error 100 on WSL
# =============================================================================
RUN cd /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec && \
    rm -f libav*.so* libsw*.so* && \
    echo "/usr/lib/wsl/lib" > /etc/ld.so.conf.d/wsl.conf && \
    ldconfig

RUN ln -s /usr/lib/wsl/lib/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 | true && \
    ln -s /usr/lib/wsl/lib/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 | true && \
    ln -s /usr/lib/wsl/lib/libdxcore.so /usr/lib/x86_64-linux-gnu/libdxcore.so | true

# =============================================================================
# 7. Install Project Python Dependencies
# =============================================================================
WORKDIR /app
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# =============================================================================
# 8. Install Playwright Browsers (for Trend Discovery scraping)
# =============================================================================
RUN playwright install chromium && \
    playwright install-deps

# =============================================================================
# Install fontconfig, system fonts, AND caption-ready bold fonts
# - fonts-dejavu-core: reliable fallback
# - fonts-freefont-ttf: includes FreeSansBold (Impact-like heavy sans-serif)
# - fonts-liberation: Liberation Sans (Arial Black substitute, metrically compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig fonts-dejavu-core fonts-freefont-ttf fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Download Montserrat-Black from Google Fonts (open-source, OFL license)
RUN mkdir -p /usr/local/share/fonts/custom && \
    wget -q -O /tmp/montserrat.zip "https://fonts.google.com/download?family=Montserrat" && \
    unzip -q -o /tmp/montserrat.zip -d /tmp/montserrat && \
    cp /tmp/montserrat/static/Montserrat-Black.ttf /usr/local/share/fonts/custom/ 2>/dev/null || \
    cp /tmp/montserrat/Montserrat-Black.ttf /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    rm -rf /tmp/montserrat /tmp/montserrat.zip && \
    chmod 644 /usr/local/share/fonts/custom/* 2>/dev/null || true && \
    fc-cache -fv

# =============================================================================
# 10. Copy Application Code & Config
# =============================================================================
COPY autonomous_trend_agent/ /app/autonomous_trend_agent/
# NOTE: .env is injected via docker-compose env_file, not baked into image

# Copy Supervisor config (if present)
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create runtime directories
RUN mkdir -p /var/log/pipeline /app/checkpoints /app/output

# Mark as Docker environment
ENV DOCKER_CONTAINER=1

# Default: run orchestrator (can be overridden by docker-compose)
CMD ["python", "-m", "autonomous_trend_agent.pipeline.orchestrator"]
