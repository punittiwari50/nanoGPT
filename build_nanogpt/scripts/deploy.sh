#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 3 FastAPI Production Server Deploy (STD-BLD-018, STD-BLD-022)
# ==============================================================================

readonly APP_PORT="${APP_PORT:-8003}"
readonly WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly LOGS_DIR="${WORKSPACE_ROOT}/logs"

mkdir -p "${LOGS_DIR}"

log_deploy() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] [DEPLOY_$1] $2" >&2
}

start_fastapi_server() {
    log_deploy "START" "Starting nanoGPT Phase 3 FastAPI daemon on port ${APP_PORT}..."
    cd "${SOURCE_DIR}"
    export NANOGPT_CKPT="${DIST_DIR}/model/ckpt.pt"
    
    python3 -m uvicorn server:app --host 0.0.0.0 --port "${APP_PORT}" &> "${LOGS_DIR}/app.log" &
    local -r srv_pid=$!
    echo "${srv_pid}" > "${LOGS_DIR}/app.pid"

    log_deploy "PROBE" "Polling healthcheck probe at http://localhost:${APP_PORT}/health..."
    local attempts=0
    while ! curl -s -f "http://localhost:${APP_PORT}/health" &>/dev/null; do
        sleep 1
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge 15 ]]; then
            log_deploy "FAIL" "Healthcheck probe timed out!"
            exit 1
        fi
    done
    log_deploy "READY" "nanoGPT Phase 3 API is healthy on port ${APP_PORT} (PID: ${srv_pid})."
}

main() {
    start_fastapi_server
    wait
}

main "$@"
