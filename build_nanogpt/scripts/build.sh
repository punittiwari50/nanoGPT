#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 2 Build & Tuning Pipeline (STD-BLD-005, STD-BLD-012)
# ==============================================================================

readonly WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly LOGS_DIR="${WORKSPACE_ROOT}/logs"

mkdir -p "${DIST_DIR}" "${LOGS_DIR}"

log_build() {
    local -r stage="$1"
    local -r msg="$2"
    local -r ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[${ts}] [BUILD_${stage}] ${msg}" | tee -a "${LOGS_DIR}/build.log"
}

build_wheel() {
    log_build "PACKAGE" "Building standalone Python wheel for nanoGPT Phase 2..."
    cd "${SOURCE_DIR}"
    python3 -m build --wheel --outdir="${DIST_DIR}"
    log_build "PACKAGE" "Wheel built successfully in ${DIST_DIR}."
}

train_sft() {
    log_build "SFT" "Executing Instruction SFT with LoRA adapters..."
    cd "${SOURCE_DIR}"
    local base_ckpt=""
    if [[ -f "/workspace/phase1_dist/model/ckpt.pt" ]]; then
        base_ckpt="/workspace/phase1_dist/model/ckpt.pt"
    fi

    python3 train_sft.py \
        --base_ckpt="${base_ckpt}" \
        --out_dir="${DIST_DIR}/model" \
        --lora_rank=4 \
        --lora_alpha=8.0 \
        --max_iters=30 \
        --batch_size=4 \
        --learning_rate=2e-3 \
        --device=cpu 2>&1 | tee -a "${LOGS_DIR}/train.log"

    if [[ -f "${DIST_DIR}/model/ckpt.pt" ]]; then
        cp "${DIST_DIR}/model/ckpt.pt" "${DIST_DIR}/model.bin"
        log_build "SFT" "Fine-tuned checkpoint and model.bin exported to ${DIST_DIR}."
    fi
}

verify_dpo() {
    log_build "DPO" "Running Direct Preference Optimization (DPO) mathematical verification step..."
    cd "${SOURCE_DIR}"
    python3 dpo_train.py 2>&1 | tee -a "${LOGS_DIR}/dpo.log"
    log_build "DPO" "DPO alignment step verification passed."
}

main() {
    log_build "START" "Starting nanoGPT Phase 2 tuning lifecycle..."
    build_wheel
    train_sft
    verify_dpo
    log_build "COMPLETE" "Phase 2 build finished successfully."
}

main "$@"
