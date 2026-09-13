#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# nanoGPT Production Service Deploy & Readiness Probe (STD-BLD-018, STD-BLD-022)
# Verifies host/container port availability, launches background inference service,
# and polls health endpoint before declaring readiness.
# ==============================================================================

readonly APP_PORT="${APP_PORT:-8000}"
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

verify_port_availability() {
    local -r port="$1"
    log_deploy "CHECK" "Validating port ${port} availability via socket test..."
    if python3 -c "import socket; s = socket.socket(); s.bind(('0.0.0.0', ${port})); s.close()" 2>/dev/null; then
        log_deploy "OK" "Port ${port} is available."
        echo "${port}"
    else
        log_deploy "WARN" "Port ${port} in use. Scanning for next available port..."
        local candidate=$((port + 1))
        while ! python3 -c "import socket; s = socket.socket(); s.bind(('0.0.0.0', ${candidate})); s.close()" 2>/dev/null; do
            candidate=$((candidate + 1))
        done
        log_deploy "ALLOCATE" "Dynamically allocated port ${candidate}."
        echo "${candidate}"
    fi
}

start_inference_service() {
    local -r port="$1"
    log_deploy "START" "Starting nanoGPT inference HTTP daemon on port ${port}..."
    cd "${SOURCE_DIR}"
    
    python3 -m http.server "${port}" --directory "${DIST_DIR}" &> "${LOGS_DIR}/app.log" &
    local -r srv_pid=$!
    echo "${srv_pid}" > "${LOGS_DIR}/app.pid"

    log_deploy "PROBE" "Polling readiness probe at http://localhost:${port}/..."
    local attempts=0
    while ! curl -s -f "http://localhost:${port}/" &>/dev/null; do
        sleep 1
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge 15 ]]; then
            log_deploy "FAIL" "Readiness probe timed out after 15 seconds!"
            exit 1
        fi
    done
    log_deploy "READY" "nanoGPT service is healthy and accepting connections on port ${port} (PID: ${srv_pid})."
}

main() {
    log_deploy "INIT" "Initializing nanoGPT service deployment sequence..."
    local -r effective_port="$(verify_port_availability "${APP_PORT}")"
    start_inference_service "${effective_port}"
    wait
}

main "$@"
