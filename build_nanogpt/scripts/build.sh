#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 3 Production Inference Build (STD-BLD-005, STD-BLD-012)
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
    log_build "PACKAGE" "Building standalone Python wheel for nanoGPT Phase 3..."
    cd "${SOURCE_DIR}"
    python3 -m build --wheel --outdir="${DIST_DIR}"
    log_build "PACKAGE" "Wheel built successfully in ${DIST_DIR}."
}

stage_model_and_quantize() {
    log_build "STAGE" "Staging model checkpoint and running INT8 Dynamic Quantization..."
    cd "${SOURCE_DIR}"
    local ckpt_path=""
    if [[ -f "/workspace/phase1_dist/model/ckpt.pt" ]]; then
        mkdir -p "${DIST_DIR}/model"
        cp "/workspace/phase1_dist/model/ckpt.pt" "${DIST_DIR}/model/ckpt.pt"
        cp "/workspace/phase1_dist/model.bin" "${DIST_DIR}/model.bin" 2>/dev/null || true
        ckpt_path="${DIST_DIR}/model/ckpt.pt"
        log_build "STAGE" "Mirrored trained Phase 1 checkpoint to Phase 3 dist/model/ckpt.pt"
    fi

    python3 quantize.py --ckpt_path="${ckpt_path}" --out_path="${DIST_DIR}/model/model_int8.pt"
    log_build "QUANTIZE" "INT8 quantization completed: ${DIST_DIR}/model/model_int8.pt"
}

run_kv_benchmark() {
    log_build "BENCH" "Executing KV-Cache latency & throughput benchmark..."
    cd "${SOURCE_DIR}"
    python3 sample.py --ckpt_path="${DIST_DIR}/model/ckpt.pt" --tokens=30 2>&1 | tee -a "${LOGS_DIR}/bench.log"
    log_build "BENCH" "Benchmark completed successfully."
}

main() {
    log_build "START" "Starting nanoGPT Phase 3 build lifecycle..."
    build_wheel
    stage_model_and_quantize
    run_kv_benchmark
    log_build "COMPLETE" "Phase 3 build finished successfully."
}

main "$@"
