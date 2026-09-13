# nanoGPT Phase 1: Modern Architectural Evolution — Operational Runbook

## 1. Overview & Architecture
`repo_nanogpt-phase1` transforms baseline nanoGPT into a next-generation generative language model utilizing:
- **Rotary Position Embeddings (RoPE)**: Replaces static learned absolute position tables with real trigonometric rotations (`cos` and `sin`), enabling relative positional distance scaling.
- **Root Mean Square Normalization (RMSNorm)**: Eliminates mean re-centering overhead from LayerNorm, boosting throughput with strictly invariant scale normalization.
- **SwiGLU Non-Linear Feed-Forward**: Employs gated swish linear activation (`silu(gate) * up`) to enhance model parameter capacity without parameter explosion.
- **Hardware-Fused Attention**: Employs `torch.nn.functional.scaled_dot_product_attention` for FlashAttention / memory-efficient causal execution.
- **ONNX Opset 17 Export**: Pure real-valued tensor graph representation ensuring direct exportability to edge and server inference runtimes.

## 2. Container Build Lifecycle (STD-BLD-001 through STD-BLD-022)

All operations run inside isolated Docker containers adhering to `container-build-standard.md`.

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
│   ├── export_onnx.py
│   └── verify.sh
└── docs/
    ├── RUNBOOK.md
    └── BUILD_SUMMARY.md
```

### 2.1 Clean
```bash
# Remove built wheel, checkpoints, and logs
rm -rf dist/* logs/*
```

### 2.2 Build Images
```bash
docker compose -f docker/docker-compose.yaml build
```

### 2.3 Compile & Train
```bash
# Compiles wheel package, executes 40-iteration training on TinyShakespeare, and exports ONNX opset 17
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase1-build
```

### 2.4 Verify
```bash
# Runs hermetic gate checks: venv verification, artifact audit, sample generation, ONNX checker
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase1-verify
```

### 2.5 Deploy
```bash
# Starts HTTP delivery service on port 8001
docker compose -f docker/docker-compose.yaml up -d nanogpt-phase1-app

# Check health and inspect deliverables
curl -s -I http://localhost:8001/
```
