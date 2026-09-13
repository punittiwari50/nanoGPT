#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 1 Build & Compilation Pipeline (STD-BLD-005, STD-BLD-012)
# ==============================================================================

readonly WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly LOGS_DIR="${WORKSPACE_ROOT}/logs"
readonly SCRIPTS_DIR="${WORKSPACE_ROOT}/scripts"
readonly DATASETS_DIR="${WORKSPACE_ROOT}/datasets"

mkdir -p "${DIST_DIR}" "${LOGS_DIR}"

log_build() {
    local -r stage="$1"
    local -r msg="$2"
    local -r ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[${ts}] [BUILD_${stage}] ${msg}" | tee -a "${LOGS_DIR}/build.log"
}

build_wheel() {
    log_build "PACKAGE" "Building standalone Python wheel for nanoGPT Phase 1..."
    cd "${SOURCE_DIR}"
    python3 -m build --wheel --outdir="${DIST_DIR}"
    log_build "PACKAGE" "Wheel built successfully in ${DIST_DIR}."
}

train_model() {
    local -r max_iters="${NANOGPT_MAX_ITERS:-40}"
    log_build "TRAIN" "Executing modern architecture training (${max_iters} iterations)..."
    cd "${SOURCE_DIR}"
    python3 train.py \
        --data_dir="${DATASETS_DIR}/combined" \
        --out_dir="${DIST_DIR}/model" \
        --max_iters="${max_iters}" \
        --batch_size=8 \
        --n_embd=64 \
        --n_layer=2 \
        --n_head=2 \
        --block_size=64 \
        --learning_rate=1e-3 \
        --device=cpu 2>&1 | tee -a "${LOGS_DIR}/train.log"

    if [[ -f "${DIST_DIR}/model/ckpt.pt" ]]; then
        cp "${DIST_DIR}/model/ckpt.pt" "${DIST_DIR}/model.bin"
        log_build "TRAIN" "Checkpoint saved and mirrored to ${DIST_DIR}/model.bin"
    fi
}

export_onnx() {
    log_build "ONNX" "Exporting modern transformer to ONNX graph..."
    python3 "${SCRIPTS_DIR}/export_onnx.py" \
        --ckpt_path="${DIST_DIR}/model/ckpt.pt" \
        --out_onnx="${DIST_DIR}/model.onnx" 2>&1 | tee -a "${LOGS_DIR}/build.log"
    log_build "ONNX" "ONNX export completed: ${DIST_DIR}/model.onnx"
}

main() {
    log_build "START" "Starting nanoGPT Phase 1 build lifecycle..."
    build_wheel
    train_model
    export_onnx
    log_build "COMPLETE" "Phase 1 build finished successfully."
}

main "$@"
