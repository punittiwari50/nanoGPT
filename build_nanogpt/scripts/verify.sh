#!/usr/bin/env bash
set -Eeuo pipefail

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
    for f in model.bin lora_adapters.pt; do
        if [[ ! -f "${DIST_DIR}/model/${f}" && ! -f "${DIST_DIR}/${f}" ]]; then
            log_verify "FAIL" "Missing deliverable: ${f}"
            exit 1
        fi
        log_verify "OK" "Found deliverable: ${f}"
    done

    log_verify "CHECK" "Running smoke test instruction generation from LoRA checkpoint..."
    cd "${SOURCE_DIR}"
    python3 sample.py --ckpt_path="${DIST_DIR}/model/ckpt.pt" --max_new_tokens=25

    log_verify "COMPLETE" "All Phase 2 verification gates passed cleanly!"
}

main "$@"
