# nanoGPT Phase 2: Training, Tuning & Alignment — Operational Runbook

## 1. Overview & Architecture
`repo_nanogpt-phase2` extends modern nanoGPT with parameter-efficient fine-tuning and human alignment mechanisms:
- **Instruction Supervised Fine-Tuning (SFT)**: Formats prompt/response pairs with strict loss masking: prompt tokens are labeled `-100`, ensuring gradients only propagate across target answers.
- **Low-Rank Adaptation (LoRA)**: Injects rank-4 low-rank factorized matrices ($W = W_0 + \frac{\alpha}{r} B \cdot A$) into attention projection layers. Freezes base weights and trains only 5,792 parameters (0.17% of total model capacity).
- **Direct Preference Optimization (DPO)**: Mathematical alignment eliminating separate reward modeling: directly optimizes policy log-probabilities against a frozen reference prior using implicit log-ratio reward formulations.

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

### 2.3 Compile & Train
```bash
# Builds wheel, injects LoRA into Phase 1 base checkpoint, executes SFT, and verifies DPO
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase2-build
```

### 2.4 Verify
```bash
# Validates venv, inspects lora_adapters.pt deliverable, runs instruction text generation
docker compose -f docker/docker-compose.yaml run --rm nanogpt-phase2-verify
```

### 2.5 Deploy
```bash
# Starts HTTP delivery daemon on port 8002
docker compose -f docker/docker-compose.yaml up -d nanogpt-phase2-app

# Probe endpoint
curl -s -I http://localhost:8002/
```
