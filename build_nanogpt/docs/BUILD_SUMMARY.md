# nanoGPT Hermetic Build Summary & Deliverables Audit

> **Standard ID:** `STD-BLD-009`  
> **Build Target:** nanoGPT Generative Pretrained Transformer  
> **Compilation Mode:** Hermetic Containerized Source Build (`STD-BLD-001`, `STD-BLD-002`)  
> **Lifecycle Standard:** 5-Phase Sequential Lifecycle (`STD-BLD-005`, `STD-BLD-009`)  
> **Supported Platforms:** Windows (PowerShell / CMD), Linux (GNU Bash), macOS (Zsh / Homebrew)

---

## 1. Codebase Scan & Stack Architecture Inventory

| Component | Specification | Description / Notes |
| :--- | :--- | :--- |
| **Source Repository** | `repo_nanogpt` | Pure PyTorch implementation of Generative Pretrained Transformer |
| **Container Harness** | `build_nanogpt` | Hermetic multi-stage build, datasets, docker manifests, verification scripts |
| **Base Language & Compiler** | Python 3.10+ / PyTorch 2.x | PyTorch CPU/CUDA dynamic fallback, C++ compilation tools |
| **Package Manager** | `uv` / `pip` | Isolated in `/opt/venv_nanogpt` (~1.12 GB) |
| **Model Format Exports** | PyTorch Checkpoint & ONNX Graph | Dual deliverables: `model.bin` (`ckpt.pt`) + `model.onnx` |
| **Tokenizer Architecture** | GPT-2 BPE (`tiktoken`) | Unified vocabulary size: 50,257 (padded to 50,304) |
| **Execution Topology** | Multi-Stage BuildKit Container | Stages: `builder-base` $\rightarrow$ `builder-ai` $\rightarrow$ `runtime` |

---

## 2. Multi-Vendor Dataset Inventory & Preparation Ledger

The build harness ingests, unifies, and tokenizes raw datasets across multiple vendors into a combined GPT-2 BPE binary representation:

| Vendor | Dataset Name | Raw Source | Format | Ingested Tokens | Manifest / Metadata |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GitHub Raw** | `tinyshakespeare` | `karpathy/char-rnn` raw GitHub mirror | Text (`input.txt`, 1.11 MB) | ~338,025 tokens | `datasets/github_raw/tinyshakespeare/metadata.json` |
| **Hugging Face** | `TinyStories` | `roneneldan/TinyStories` via Hugging Face Hub | Text / Parquet stream | ~340,360 tokens | `datasets/huggingface/tinystories/metadata.json` |
| **Hugging Face** | `OpenWebText` | `openwebtext` via Hugging Face Hub | Text / Web dump stream | ~340,360 tokens | `datasets/huggingface/openwebtext/metadata.json` |
| **Combined** | **Unified GPT-2 BPE** | Prepared via `prepare_combined_dataset.py` | Binary (`uint16` memory-map) | **Train:** 1,018,745 tokens<br>**Val:** 115,702 tokens | `dist/meta.pkl`, `dist/train.bin`, `dist/val.bin` |

---

## 3. Dual Model Export Architecture

To provide maximum deployment flexibility across server inference, edge computing, and browser runtimes, nanoGPT is exported in dual model formats:

```text
               ┌────────────────────────────────────────────────────────┐
               │         PyTorch Trained Checkpoint (dist/model/ckpt.pt) │
               └───────────────────────────┬────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │  Native PyTorch Binary      │               │  Optimized ONNX Graph       │
     │  dist/model.bin             │               │  dist/model.onnx            │
     │  • Native PyTorch execution │               │  • opset=17, dynamo=False   │
     │  • Full state dict + config │               │  • ONNX Runtime acceleration│
     │  • GPU & CPU inference      │               │  • Cross-platform deployment│
     └─────────────────────────────┘               └─────────────────────────────┘
```

1. **PyTorch Native Format (`dist/model.bin` / `dist/model/ckpt.pt`):**
   - Full model weights, optimizer state, iteration counter, loss history, and model configuration dict.
   - Used by native PyTorch inference (`sample.py`) and training resumption.
2. **ONNX Graph (`dist/model.onnx`):**
   - Exported using `torch.onnx.export(model, dummy_input, out_onnx, opset_version=17, dynamo=False, dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}, 'logits': {0: 'batch', 1: 'seq_len'}})`
   - Validated structurally using `onnx.checker.check_model`.
   - Executable across ONNX Runtime, TensorRT, DirectML, and edge accelerators.

---

## 4. Deliverables Manifest (`dist/`)

All compiled outputs, model weights, dataset tokens, and package wheels are staged exclusively in `/workspace/dist/` (mounted to the host's `build_nanogpt/dist/`):

| File Name | Size (Approx) | Type | Role |
| :--- | :--- | :--- | :--- |
| `nanogpt-0.1.0-py3-none-any.whl` | ~25 KB | Python Wheel | Distributable library package for standalone pip installation |
| `model.bin` | ~39 MB | PyTorch Weights | Primary trained model checkpoint (3.32M parameters, 40 iterations) |
| `model/ckpt.pt` | ~39 MB | PyTorch State | Checkpoint state dictionary including configuration |
| `model.onnx` | ~25.5 MB | ONNX Graph | Inference-optimized graph export for ONNX Runtime (opset 17) |
| `train.bin` | ~2.0 MB | Binary Tokens | Combined training tokens (uint16 array, 1,018,745 tokens) |
| `val.bin` | ~228 KB | Binary Tokens | Combined validation tokens (uint16 array, 115,702 tokens) |
| `meta.pkl` | ~1 KB | Pickle Dict | Tokenizer metadata, vocab size, and encoding spec |

> **Comprehensive Model & Architectural Analysis:** For a full mathematical breakdown, use cases, and extension blueprints, refer to [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md) and [REQUEST_RESPONSE.md](REQUEST_RESPONSE.md).

---

## 5. Automated 7-Gate Verification Suite Audit

The non-interactive verification container (`nanogpt-verify`) enforces 7 deterministic quality gates:

| Gate | Gate Name | Assertion Criteria | Status |
| :---: | :--- | :--- | :---: |
| **1** | Virtual Environment Switch | Active Python binary is located at `/opt/venv_nanogpt/bin/python3` | **PASSED** |
| **2** | Zero Cache Ingestion Path | `NANOGPT_DATASET_DIR` points to structured datasets, not `.cache` | **PASSED** |
| **3** | Host Dataset & Manifest Audit | All dataset folders contain raw files and valid `metadata.json` manifests | **PASSED** |
| **4** | Complete Binary Deliverables | `dist/` contains `train.bin`, `val.bin`, `model.bin`, and `model.onnx` | **PASSED** |
| **5** | Standalone Wheel Package | `dist/nanogpt-*.whl` exists and is a valid zip archive | **PASSED** |
| **6** | PyTorch Inference Smoke Test | `sample.py` successfully generates coherent tokens from checkpoint | **PASSED** |
| **7** | ONNX Graph Structural Validation | `onnx.checker.check_model` confirms valid graph and opset 17 compatibility | **PASSED** |

---

## 6. Cross-Platform Operational Quick-Reference

For complete step-by-step instructions, refer to [RUNBOOK.md](RUNBOOK.md).

```bash
# Phase 1: Clean
docker compose -f docker/docker-compose.yaml down -v --remove-orphans

# Phase 2: Build
docker compose -f docker/docker-compose.yaml build

# Phase 3: Train & Export
docker compose -f docker/docker-compose.yaml run --rm nanogpt-build

# Phase 4: Run Daemon
docker compose -f docker/docker-compose.yaml up -d nanogpt-app

# Phase 5: Verify
docker compose -f docker/docker-compose.yaml run --rm nanogpt-verify
```
