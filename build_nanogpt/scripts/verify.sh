#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 1 Verification Protocol (STD-BLD-016, 017, 021)
# ==============================================================================

readonly WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"

log_verify() {
    echo "[$1] $2"
}

main() {
    log_verify "CHECK" "Validating in-container virtual environment..."
    if [[ "${VIRTUAL_ENV}" != "/opt/venv_nanogpt" ]]; then
        log_verify "FAIL" "Virtual environment mismatch!"
        exit 1
    fi
    log_verify "OK" "Virtual environment verified: ${VIRTUAL_ENV}"

    log_verify "CHECK" "Validating deliverables in dist/..."
    for f in model.bin model.onnx; do
        if [[ ! -f "${DIST_DIR}/${f}" ]]; then
            log_verify "FAIL" "Missing deliverable: ${f}"
            exit 1
        fi
        log_verify "OK" "Found deliverable: ${f} ($(du -h "${DIST_DIR}/${f}" | cut -f1))"
    done

    log_verify "CHECK" "Running smoke test text generation from checkpoint..."
    cd "${SOURCE_DIR}"
    python3 sample.py --ckpt_path="${DIST_DIR}/model/ckpt.pt" --start="The future of intelligence" --max_new_tokens=20 --num_samples=1

    log_verify "CHECK" "Validating ONNX graph integrity..."
    python3 -c "import onnx; m = onnx.load('${DIST_DIR}/model.onnx'); onnx.checker.check_model(m); print('ONNX model integrity verified successfully!')"

    log_verify "COMPLETE" "All Phase 1 verification gates passed cleanly!"
}

main "$@"
