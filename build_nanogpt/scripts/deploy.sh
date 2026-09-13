#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Phase 1 Production Service Deploy (STD-BLD-018, STD-BLD-022)
# ==============================================================================

readonly APP_PORT="${APP_PORT:-8001}"
readonly WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DIST_DIR="${WORKSPACE_ROOT}/dist"
readonly SOURCE_DIR="${WORKSPACE_ROOT}/source"
readonly LOGS_DIR="${WORKSPACE_ROOT}/logs"

mkdir -p "${LOGS_DIR}"

log_deploy() {
    local -r level="$1"
    local -r message="$2"
    local -r timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[${timestamp}] [DEPLOY_${level}] ${message}" >&2
}

start_service() {
    log_deploy "START" "Starting nanoGPT Phase 1 HTTP daemon on port ${APP_PORT}..."
    cd "${SOURCE_DIR}"
    python3 -m http.server "${APP_PORT}" --directory "${DIST_DIR}" &> "${LOGS_DIR}/app.log" &
    local -r srv_pid=$!
    echo "${srv_pid}" > "${LOGS_DIR}/app.pid"

    log_deploy "PROBE" "Polling readiness probe at http://localhost:${APP_PORT}/..."
    local attempts=0
    while ! curl -s -f "http://localhost:${APP_PORT}/" &>/dev/null; do
        sleep 1
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge 15 ]]; then
            log_deploy "FAIL" "Readiness probe timed out!"
            exit 1
        fi
    done
    log_deploy "READY" "nanoGPT Phase 1 service healthy on port ${APP_PORT} (PID: ${srv_pid})."
}

main() {
    start_service
    wait
}

main "$@"
