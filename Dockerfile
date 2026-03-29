# ============================================================================
# PROJECT AETHER v2.2 - DOCKER IMAGE (March 2026 Stack)
# ============================================================================
# NGC 26.03 base (native Blackwell sm_120) + vLLM 0.18.0 (pip, Qwen3.5)
# Stack: PyTorch 2.10.0, transformers 5.x, vLLM 0.18.0, easytranscriber
# ============================================================================

FROM nvcr.io/nvidia/pytorch:26.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/lib/python3.12/dist-packages/PyNvVideoCodec/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility,graphics

# Blackwell-specific: Enable expandable memory segments
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
ENV TORCH_CUDA_ARCH_LIST="12.0"

# Model cache directories (mounted volumes)
ENV TRANSFORMERS_CACHE=/app/models/huggingface
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# =============================================================================
# 1. Install Build Dependencies for FFmpeg 7.1 + Redis
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    yasm pkg-config libgnutls28-dev libdrm-dev libva-dev \
    libvdpau-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev \
    libssl-dev libx264-dev libx265-dev libnuma-dev \
    libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev \
    libvorbis-dev libgl1-mesa-dev git wget build-essential \
    supervisor redis-server \
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
# 4. Setup WSL2 library paths
# =============================================================================
RUN echo "/usr/lib/wsl/lib" > /etc/ld.so.conf.d/wsl.conf && ldconfig

RUN ln -sf /usr/lib/wsl/lib/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 || true && \
    ln -sf /usr/lib/wsl/lib/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 || true && \
    ln -sf /usr/lib/wsl/lib/libdxcore.so /usr/lib/x86_64-linux-gnu/libdxcore.so || true

# =============================================================================
# 5. Remove NGC pip constraint pins that block upgrades
#    NGC 26.03 pins torch, transformers, huggingface_hub in /etc/pip/constraint.txt
#    We need to override these for vLLM 0.18.0 (requires PyTorch 2.10.0)
# =============================================================================
RUN sed -i '/^torch[=<>]/d; /^transformers[=<>]/d; /^huggingface.hub[=<>]/d; /^regex[=<>]/d; /^tokenizers[=<>]/d' /etc/pip/constraint.txt || true

# =============================================================================
# 6. Install PyTorch 2.10.0 (cu128) — required by vLLM 0.18.0
# =============================================================================
RUN pip install --no-cache-dir torch==2.10.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# 7. Install vLLM 0.18.0 (pip — native sm_120 + Qwen3.5 support)
#    No more source compilation needed!
# =============================================================================
RUN pip install --no-cache-dir vllm==0.18.0 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# 8. Install transformers 5.x + huggingface_hub for Qwen3.5 (qwen3_5 arch)
# =============================================================================
RUN pip install --no-cache-dir "transformers>=5.2.0" "huggingface_hub>=1.0"

# Patch vLLM 0.18.0 bug: qwen3_5.py passes ignore_keys_at_rope_validation as
# a list but huggingface_hub's dataclass validator does set -= value (needs set)
RUN python3 -c "\
f='/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5.py'; \
t=open(f).read(); \
t=t.replace('ignore_keys_at_rope_validation\"] = [', 'ignore_keys_at_rope_validation\"] = {').replace(\
'\"mrope_interleaved\",\n        ]', '\"mrope_interleaved\",\n        }'); \
open(f,'w').write(t); print('Patched qwen3_5.py: list -> set')"

# =============================================================================
# 9. Install Project Python Dependencies
# =============================================================================
WORKDIR /app
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# =============================================================================
# 10. Install PyNvVideoCodec 2.1.0 (Zero-Copy GPU Video Decoder/Encoder)
# =============================================================================
RUN pip install --no-cache-dir pynvvideocodec==2.1.0 || \
    echo "WARNING: PyNvVideoCodec 2.1.0 failed — may need version bump for CUDA 13.x"

# Clean up bundled PyNvVideoCodec FFmpeg libs that conflict with our build
RUN cd /usr/local/lib/python3.12/dist-packages/PyNvVideoCodec 2>/dev/null && \
    rm -f libav*.so* libsw*.so* && \
    ldconfig || true

# =============================================================================
# 11. Install easytranscriber (--no-deps: torch/torchaudio already installed)
# =============================================================================
RUN pip install --no-cache-dir --no-deps easytranscriber && \
    pip install --no-cache-dir --no-deps easyaligner && \
    pip install --no-cache-dir --no-deps pyannote-audio

# =============================================================================
# 12. Install high-dependency modules (--no-deps: avoids resolver hell)
# =============================================================================
RUN pip install --no-cache-dir --no-deps crawl4ai langgraph pydantic-ai google-genai

# =============================================================================
# 13. Install Playwright Browsers (for Crawl4AI web scraping)
# =============================================================================
RUN playwright install chromium && \
    playwright install-deps

# =============================================================================
# 14. Install caption fonts
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig fonts-dejavu-core fonts-freefont-ttf fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/share/fonts/custom && \
    wget -q -O /tmp/montserrat.zip "https://fonts.google.com/download?family=Montserrat" && \
    unzip -q -o /tmp/montserrat.zip -d /tmp/montserrat && \
    cp /tmp/montserrat/static/Montserrat-Black.ttf /usr/local/share/fonts/custom/ 2>/dev/null || \
    cp /tmp/montserrat/Montserrat-Black.ttf /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    rm -rf /tmp/montserrat /tmp/montserrat.zip && \
    chmod 644 /usr/local/share/fonts/custom/* 2>/dev/null || true && \
    fc-cache -fv

# =============================================================================
# 15. Copy Application Code & Config
# =============================================================================
COPY autonomous_trend_agent/ /app/autonomous_trend_agent/
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create runtime directories
RUN mkdir -p /var/log/pipeline /app/checkpoints /app/output

# Mark as Docker environment
ENV DOCKER_CONTAINER=1

# Start supervisord to manage sidecars and the pipeline
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
