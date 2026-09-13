#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Hermetic Container Build & Packaging Orchestrator (STD-BLD-005)
# Ingests source from /workspace/source, tokenizes datasets, packages Python wheel,
# trains micro-checkpoint, and exports all deliverables to /workspace/dist/.
# ==============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"
readonly SCRIPTS_DIR="${WORKSPACE_ROOT}/scripts"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly LOGS_DIR="${WORKSPACE_ROOT}/logs"
readonly DATASETS_DIR="${WORKSPACE_ROOT}/datasets"

mkdir -p "${DIST_DIR}" "${LOGS_DIR}" "${DATASETS_DIR}"

log_build() {
    local -r stage="$1"
    local -r message="$2"
    local -r timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[${timestamp}] [BUILD_${stage}] ${message}" | tee -a "${LOGS_DIR}/build.log" >&2
}

detect_hardware_target() {
    log_build "TARGET" "Detecting acceleration hardware target (STD-BLD-015)..."
    local device
    device="$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")"
    log_build "TARGET" "Resolved target acceleration device: ${device}"
    echo "${device}"
}

prepare_datasets() {
    log_build "DATA" "Preparing combined dataset (TinyShakespeare + TinyStories) under unified GPT-2 BPE tokenizer..."
    if [[ -f "${SCRIPTS_DIR}/prepare_combined_dataset.py" ]]; then
        python3 "${SCRIPTS_DIR}/prepare_combined_dataset.py" \
            --shakespeare_raw="${DATASETS_DIR}/github_raw/tinyshakespeare/input.txt" \
            --hf_dataset="roneneldan/TinyStories" \
            --datasets_root="${DATASETS_DIR}" \
            --source_data_dir="${SOURCE_DIR}/data/combined" \
            --out_dir="${DIST_DIR}" \
            --quiet
        log_build "DATA" "Unified dataset prepared in ${SOURCE_DIR}/data/combined and staged in ${DIST_DIR}."
    else
        log_build "FAIL" "prepare_combined_dataset.py not found!"
        exit 1
    fi
}

build_python_wheel() {
    log_build "PACKAGE" "Building standalone Python wheel for nanoGPT..."
    cd "${SOURCE_DIR}"
    if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
        log_build "PACKAGE" "Generating standard pyproject.toml for packaging (STD-BLD-012)..."
        cat <<'EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "nanogpt"
version = "0.1.0"
description = "The simplest, fastest repository for training/finetuning medium-sized GPTs."
readme = "README.md"
authors = [{ name = "Andrej Karpathy" }]
license = { text = "MIT" }
requires-python = ">=3.8"
dependencies = [
    "torch",
    "numpy",
    "transformers",
    "datasets",
    "tiktoken",
    "wandb",
    "tqdm",
    "onnx"
]

[tool.setuptools.packages.find]
where = ["."]
include = ["*"]
exclude = ["data*", "config*", "assets*", "build*"]
EOF
    fi
    python3 -m build --wheel --outdir="${DIST_DIR}"
    log_build "PACKAGE" "Wheel built successfully in ${DIST_DIR}."
}

compile_model_checkpoint() {
    local -r target_device="$1"
    local -r max_iters="${NANOGPT_MAX_ITERS:-40}"
    local -r eval_interval=$(( max_iters > 10 ? max_iters / 4 : 5 ))
    log_build "TRAIN" "Executing training run (${max_iters} iters) on device [${target_device}] across combined datasets..."
    cd "${SOURCE_DIR}"
    
    python3 train.py \
        --dataset=combined \
        --n_layer=2 \
        --n_head=2 \
        --n_embd=64 \
        --block_size=64 \
        --batch_size=8 \
        --learning_rate=1e-3 \
        --max_iters="${max_iters}" \
        --lr_decay_iters="${max_iters}" \
        --eval_iters=2 \
        --eval_interval="${eval_interval}" \
        --dropout=0.0 \
        --device="${target_device}" \
        --compile=False \
        --out_dir="${DIST_DIR}/model" 2>&1 | tee -a "${LOGS_DIR}/train.log"

    if [[ -f "${DIST_DIR}/model/ckpt.pt" ]]; then
        cp "${DIST_DIR}/model/ckpt.pt" "${DIST_DIR}/model.bin"
        log_build "TRAIN" "PyTorch checkpoint and model.bin exported to ${DIST_DIR}."

        log_build "ONNX" "Exporting trained model to ONNX format (STD-BLD-012)..."
        python3 "${SCRIPTS_DIR}/export_onnx.py" \
            --ckpt_path="${DIST_DIR}/model/ckpt.pt" \
            --out_onnx="${DIST_DIR}/model.onnx" 2>&1 | tee -a "${LOGS_DIR}/build.log"
        log_build "ONNX" "ONNX export completed: ${DIST_DIR}/model.onnx"
    else
        log_build "WARN" "Model checkpoint not generated; continuing with staged artifacts."
    fi
}

main() {
    log_build "START" "Starting nanoGPT hermetic compilation & packaging..."
    local -r target_device="$(detect_hardware_target)"
    prepare_datasets
    build_python_wheel
    compile_model_checkpoint "${target_device}"
    log_build "COMPLETE" "Build process finished successfully. Artifacts ready in dist/."
}

main "$@"
