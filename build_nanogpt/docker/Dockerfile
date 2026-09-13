ARG PYTHON_BASE=python:3.12-slim

# ==============================================================================
# nanoGPT Multi-Stage Hermetic Container Build (STD-BLD-001, 016, 017, 019, 021, 022)
# Base: Configurable via PYTHON_BASE build-arg (default: python:3.12-slim)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder Base (Isolated Toolchains & Virtual Environment)
# ------------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS builder-base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    bash \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create in-container virtual environment (STD-BLD-017)
RUN python3 -m venv /opt/venv_nanogpt
ENV VIRTUAL_ENV=/opt/venv_nanogpt \
    PATH="/opt/venv_nanogpt/bin:$PATH" \
    DATASETS_DIR="/workspace/datasets" \
    HF_HOME="/workspace/datasets/huggingface" \
    HF_DATASETS_CACHE="/workspace/datasets/huggingface/datasets" \
    HF_HUB_CACHE="/workspace/datasets/huggingface/hub" \
    TRANSFORMERS_CACHE="/workspace/datasets/huggingface/transformers" \
    TORCH_HOME="/workspace/.cache/torch" \
    DOWNLOAD_CACHE="/workspace/datasets/github_raw" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="utf-8" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

# Configure shell auto-activation on connection for all interactive & login shells (STD-BLD-017)
RUN echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" >> /etc/bash.bashrc && \
    echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" >> /etc/profile && \
    mkdir -p /etc/profile.d && \
    echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" > /etc/profile.d/01-venv.sh && \
    chmod 644 /etc/profile.d/01-venv.sh

# Install fast package manager uv
COPY --from=ghcr.io/astral-sh/uv:0.2.11 /uv /usr/local/bin/uv

# ------------------------------------------------------------------------------
# Stage 2: Builder AI (PyTorch & ML Dependencies via BuildKit Cache Mounts)
# ------------------------------------------------------------------------------
FROM builder-base AS builder-ai

WORKDIR /workspace

# Install core build dependencies and AI runtime libraries
# Utilize BuildKit cache mounts to accelerate iterative rebuilds (STD-BLD-008)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --no-cache \
    wheel \
    build \
    setuptools \
    numpy \
    transformers \
    datasets \
    tiktoken \
    wandb \
    tqdm \
    requests \
    onnx

# Install PyTorch with CPU fallback (or CUDA if building on GPU base)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu

# Optimize image size: prune pycache and strip unneeded symbols from native libraries (.so)
RUN find /opt/venv_nanogpt -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv_nanogpt -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv_nanogpt -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /opt/venv_nanogpt -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 3: Hardened Minimal Runtime (STD-BLD-016, STD-BLD-019, STD-BLD-022)
# ------------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy isolated virtual environment from builder stage
COPY --from=builder-ai /opt/venv_nanogpt /opt/venv_nanogpt

# Configure non-cache environment variables across container runtime
ENV VIRTUAL_ENV=/opt/venv_nanogpt \
    PATH="/opt/venv_nanogpt/bin:$PATH" \
    DATASETS_DIR="/workspace/datasets" \
    HF_HOME="/workspace/datasets/huggingface" \
    HF_DATASETS_CACHE="/workspace/datasets/huggingface/datasets" \
    HF_HUB_CACHE="/workspace/datasets/huggingface/hub" \
    TRANSFORMERS_CACHE="/workspace/datasets/huggingface/transformers" \
    TORCH_HOME="/workspace/.cache/torch" \
    DOWNLOAD_CACHE="/workspace/datasets/github_raw" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="utf-8" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    APP_PORT=8000

# Universal shell auto-switch for runtime containers
RUN echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" >> /etc/bash.bashrc && \
    echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" >> /etc/profile && \
    mkdir -p /etc/profile.d && \
    echo "if [ -f /opt/venv_nanogpt/bin/activate ]; then . /opt/venv_nanogpt/bin/activate; fi" > /etc/profile.d/01-venv.sh && \
    chmod 644 /etc/profile.d/01-venv.sh

# Create non-root app user (STD-BLD-022)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /workspace/dist /workspace/logs /workspace/datasets /workspace/docs && \
    chown -R appuser:appgroup /workspace /opt/venv_nanogpt

WORKDIR /workspace

# Standard container healthcheck probe (STD-BLD-022)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/ || exit 1

EXPOSE 8000

CMD ["/workspace/scripts/deploy.sh"]
