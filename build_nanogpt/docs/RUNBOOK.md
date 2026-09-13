# nanoGPT Phase 3: Production Inference & Serving — Operational Runbook

## 1. Overview & Architecture
`repo_nanogpt-phase3` focuses on low-latency inference serving and edge compression:
- **Stateful Key-Value Caching (KVCache)**: Pre-allocates layer-wise key/value storage, converting $O(N^2)$ quadratic autoregression into $O(1)$ constant step latency per generated token.
- **FastAPI SSE Streaming Daemon**: Implements an OpenAI-compatible `/v1/chat/completions` endpoint supporting asynchronous Server-Sent Events (SSE) token streaming.
- **INT8 Dynamic Quantization**: Post-training quantization of linear projection weights to 8-bit integers, reducing memory footprint for edge devices.
- **Container Health Probes**: Native Docker `HEALTHCHECK` querying `/health` for automated container lifecycle orchestration.

## 2. Container Build Lifecycle (STD-BLD-001 through STD-BLD-022)

```
build_nanogpt/
├── .env
├── .gitignore
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── scripts/
│   ├── build.sh
│   ├── deploy.sh
│   └── verify.sh
└── docs/
    ├── RUNBOOK.md
    └── BUILD_SUMMARY.md
```

### 2.1 Clean
```bash
rm -rf dist/* logs/*
```

### 2.2 Build Images
```bash
docker compose -f docker/docker-compose.yaml build
```

### 2.3 Compile, Quantize & Benchmark
```bash
# Builds wheel, mirrors trained Phase 1 weights, quantizes to INT8, benchmarks KV-cache
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase3-build
```

### 2.4 Verify
```bash
# Audits dist deliverables (model_int8.pt, ckpt.pt), runs local benchmark smoke test
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase3-verify
```

### 2.5 Deploy
```bash
# Starts FastAPI streaming daemon on port 8003
docker compose -f docker/docker-compose.yaml up -d nanogpt-phase3-app

# Probe health
curl -s http://localhost:8003/health

# Test chat completions
curl -s -X POST http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is AI?"}], "max_tokens": 15, "stream": true}'
```
