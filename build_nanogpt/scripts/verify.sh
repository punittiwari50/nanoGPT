#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 3 Verification Protocol (STD-BLD-016, 017, 021)
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

    log_verify "CHECK" "Validating Phase 3 deliverables in dist/..."
    for f in model/model_int8.pt model/ckpt.pt; do
        if [[ ! -f "${DIST_DIR}/${f}" ]]; then
            log_verify "FAIL" "Missing deliverable: ${f}"
            exit 1
        fi
        log_verify "OK" "Found deliverable: ${f} ($(du -h "${DIST_DIR}/${f}" | cut -f1))"
    done

    log_verify "CHECK" "Executing local KV-Cache benchmark validation..."
    cd "${SOURCE_DIR}"
    python3 sample.py --ckpt_path="${DIST_DIR}/model/ckpt.pt" --tokens=15

    log_verify "COMPLETE" "All Phase 3 verification gates passed cleanly!"
}

main "$@"
