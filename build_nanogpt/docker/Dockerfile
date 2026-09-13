ARG PYTHON_BASE=python:3.12-slim

# ==============================================================================
# nanoGPT Phase 3 Multi-Stage Hermetic Container Build (STD-BLD-001, 016, 017)
# ==============================================================================

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

RUN python3 -m venv /opt/venv_nanogpt
ENV VIRTUAL_ENV=/opt/venv_nanogpt \
    PATH="/opt/venv_nanogpt/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="utf-8" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

COPY --from=ghcr.io/astral-sh/uv:0.2.11 /uv /usr/local/bin/uv

# ------------------------------------------------------------------------------
# Stage 2: Builder AI
# ------------------------------------------------------------------------------
FROM builder-base AS builder-ai

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --no-cache \
    wheel \
    build \
    setuptools \
    numpy \
    tiktoken \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    onnx

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu

RUN find /opt/venv_nanogpt -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv_nanogpt -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv_nanogpt -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /opt/venv_nanogpt -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true

# ------------------------------------------------------------------------------
# Stage 3: Runtime
# ------------------------------------------------------------------------------
FROM ${PYTHON_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-ai /opt/venv_nanogpt /opt/venv_nanogpt

ENV VIRTUAL_ENV=/opt/venv_nanogpt \
    PATH="/opt/venv_nanogpt/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="utf-8" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    APP_PORT=8003

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /workspace/dist /workspace/logs /workspace/docs && \
    chown -R appuser:appgroup /workspace /opt/venv_nanogpt

WORKDIR /workspace

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

EXPOSE 8003
CMD ["/workspace/scripts/deploy.sh"]
