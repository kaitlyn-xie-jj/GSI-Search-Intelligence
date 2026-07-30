#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
RUNTIME_DIR="${GSI_VALIDATOR_RUNTIME_DIR:-${ROOT_DIR}/outputs/runtime}"
LOG_DIR="${GSI_VALIDATOR_LOG_DIR:-${ROOT_DIR}/outputs/logs}"
HOST="${GSI_VALIDATOR_HOST:-127.0.0.1}"
PORT="${GSI_VALIDATOR_PORT:-8000}"
PID_FILE="${GSI_VALIDATOR_PID_FILE:-${RUNTIME_DIR}/gsi_validator_${PORT}.pid}"
LOG_FILE="${GSI_VALIDATOR_LOG_FILE:-${LOG_DIR}/gsi_validator_${PORT}.log}"
HEALTH_TIMEOUT="${GSI_VALIDATOR_HEALTH_TIMEOUT:-120}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLEAN_PORT="${GSI_VALIDATOR_CLEAN_PORT:-1}"

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

normalize_gurobi_license_path() {
  if [ -n "${GRB_LICENSE_FILE:-}" ] && [[ "${GRB_LICENSE_FILE}" != /* ]]; then
    export GRB_LICENSE_FILE="${ROOT_DIR}/${GRB_LICENSE_FILE}"
  fi
}

is_running() {
  if [ -f "${PID_FILE}" ]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    kill -0 "${pid}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

find_port_pids() {
  {
    if command_exists ss; then
      ss -ltnp 2>/dev/null \
        | awk -v port=":${PORT}" '$4 ~ port "$" {print $0}' \
        | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p'
    fi

    if command_exists lsof; then
      lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null
    fi

    if command_exists fuser; then
      fuser -n tcp "${PORT}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d'
    fi

    find_port_pids_from_proc
  } | sort -u
}

find_port_pids_from_proc() {
  "${PYTHON_BIN}" - "$PORT" <<'PY' 2>/dev/null || true
import os
import socket
import sys

port = int(sys.argv[1])
target_hex = f"{port:04X}"
listen_state = "0A"
inodes = set()

for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(table, encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_address = parts[1]
                state = parts[3]
                inode = parts[9]
                if state != listen_state:
                    continue
                _, port_hex = local_address.rsplit(":", 1)
                if port_hex.upper() == target_hex:
                    inodes.add(inode)
    except OSError:
        continue

if not inodes:
    sys.exit(0)

for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = os.path.join("/proc", pid, "fd")
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                print(pid)
                break
    except OSError:
        continue
PY
}

port_is_listening() {
  if command_exists ss; then
    ss -ltn 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port "$" {found=1} END {exit found ? 0 : 1}'
    return $?
  fi

  if command_exists lsof; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi

  if command_exists fuser; then
    fuser -n tcp "${PORT}" >/dev/null 2>&1
    return $?
  fi

  [ -n "$(find_port_pids_from_proc)" ]
}

clear_stale_port_listener() {
  if [ "${CLEAN_PORT}" != "1" ]; then
    return 0
  fi

  local current_pid
  local pids
  local pid
  current_pid=""
  if [ -f "${PID_FILE}" ]; then
    current_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  fi

  pids="$(find_port_pids || true)"
  if [ -z "${pids}" ]; then
    return 0
  fi

  echo "[INFO] Port ${HOST}:${PORT} is already in use; clearing stale listener(s): ${pids}"
  for pid in ${pids}; do
    if [ -n "${current_pid}" ] && [ "${pid}" = "${current_pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
      continue
    fi
    kill "${pid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pid in ${pids}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  done

  sleep 1
  if port_is_listening; then
    echo "[ERROR] Port ${HOST}:${PORT} is still in use after cleanup. Run: ss -ltnp | grep ':${PORT}'" >&2
    return 1
  fi
}

wait_for_health() {
  local deadline
  local last_report
  deadline=$((SECONDS + HEALTH_TIMEOUT))
  last_report=0
  echo "[INFO] Waiting for GSI validator health: http://${HOST}:${PORT}/health timeout=${HEALTH_TIMEOUT}s log=${LOG_FILE}"
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl --noproxy "*" -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! is_running; then
      echo "GSI validator process exited before health check passed; log: ${LOG_FILE}" >&2
      return 1
    fi
    if [ $((SECONDS - last_report)) -ge 10 ]; then
      echo "[INFO] Still waiting for GSI validator health; elapsed=$((HEALTH_TIMEOUT - (deadline - SECONDS)))s log=${LOG_FILE}"
      last_report="${SECONDS}"
    fi
    sleep 2
  done
  echo "GSI validator health check failed after ${HEALTH_TIMEOUT}s: http://${HOST}:${PORT}/health" >&2
  echo "Log: ${LOG_FILE}" >&2
  return 1
}

start_server() {
  if is_running; then
    if wait_for_health; then
      echo "GSI validator already running: $(cat "${PID_FILE}") http://${HOST}:${PORT}"
      return 0
    fi
    echo "GSI validator pid exists but health check failed" >&2
    exit 1
  fi
  clear_stale_port_listener
  normalize_gurobi_license_path
  (
    cd "${ROOT_DIR}"
    export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
    exec "${PYTHON_BIN}" -m uvicorn run.plan_validation_server:app --host "${HOST}" --port "${PORT}"
  ) >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  if is_running && wait_for_health; then
    echo "GSI validator started: $(cat "${PID_FILE}") http://${HOST}:${PORT}"
  else
    if is_running; then
      local pid
      pid="$(cat "${PID_FILE}")"
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 1
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    fi
    rm -f "${PID_FILE}"
    echo "GSI validator failed to start; log: ${LOG_FILE}" >&2
    exit 1
  fi
}

stop_server() {
  if ! is_running; then
    rm -f "${PID_FILE}"
    echo "GSI validator is not running"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" >/dev/null 2>&1 || true
  sleep 1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${PID_FILE}"
  echo "GSI validator stopped"
}

case "${1:-status}" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status)
    if is_running; then
      echo "running pid=$(cat "${PID_FILE}") endpoint=http://${HOST}:${PORT}/validate"
    else
      echo "not running endpoint=http://${HOST}:${PORT}/validate"
    fi
    ;;
  logs)
    touch "${LOG_FILE}"
    tail -f "${LOG_FILE}"
    ;;
  *)
    echo "Usage: $0 start|stop|restart|status|logs" >&2
    exit 1
    ;;
esac
