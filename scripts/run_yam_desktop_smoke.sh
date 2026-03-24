#!/bin/bash
#
# run_yam_desktop_smoke.sh — Desktop-local smoke test for the current
# RLinf OpenPI -> RemoteYamEnvWorker -> RobotServer -> YAM follower path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
COMMON_GIT_DIR="$(cd "${REPO_DIR}" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
if [ -n "${COMMON_GIT_DIR}" ]; then
    COMMON_REPO_DIR="$(dirname "${COMMON_GIT_DIR}")"
    ROOT_DIR="$(dirname "${COMMON_REPO_DIR}")"
else
    ROOT_DIR="$(dirname "${REPO_DIR}")"
fi

CONFIG_NAME="yam_ppo_openpi"
MODEL_PATH=""
TASK_DESC=""
ROBOT_SERVER_URL="${ROBOT_SERVER_URL:-localhost:50051}"
MARL_REPO_DIR="${MARL_REPO_DIR:-${ROOT_DIR}/marl}"
MARL_BASE_URL="${MARL_BASE_URL:-http://127.0.0.1:18080}"
OPENPI_REPO_DIR="${OPENPI_REPO_DIR:-${ROOT_DIR}/openpi}"
USE_EDITABLE_OPENPI="${USE_EDITABLE_OPENPI:-false}"
PIN_CHEX="true"
START_MARL="true"
DRY_RUN=""
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_yam_desktop_smoke.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

Run a desktop-local smoke test for the current RLinf remote-YAM stack:

  OpenPI rollout -> RemoteYamEnvWorker -> RobotServerClient -> RobotServer -> YAM follower

This script starts a local dummy marl server by default because
RemoteYamEnvWorker currently requires marl.enabled=true even in eval mode.

Options:
  --config NAME            Hydra config name (default: yam_ppo_openpi)
  --model-path PATH        Model checkpoint path / HF ID for actor + rollout
  --task DESC              Task description override
  --robot-server-url URL   RobotServer gRPC URL (default: localhost:50051)
  --marl-repo-dir PATH     marl repo directory (default: ../marl)
  --marl-base-url URL      marl base URL (default: http://127.0.0.1:18080)
  --openpi-repo-dir PATH   Local openpi checkout (default: ../openpi)
  --editable-openpi        Force `uv run --with-editable <openpi-repo-dir>`
  --no-editable-openpi     Do not use local editable openpi
  --no-chex-pin            Skip `--with chex==0.1.90`
  --no-start-marl          Assume marl is already running; only launch eval
  --dry-run                Print commands without executing them
  --help                   Show this help

Extra Hydra overrides can be passed after `--`.
EOF
    exit 0
}

quote_cmd() {
    printf '%q ' "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)               usage ;;
        --config)             CONFIG_NAME="$2"; shift 2 ;;
        --model-path)         MODEL_PATH="$2"; shift 2 ;;
        --task)               TASK_DESC="$2"; shift 2 ;;
        --robot-server-url)   ROBOT_SERVER_URL="$2"; shift 2 ;;
        --marl-repo-dir)      MARL_REPO_DIR="$2"; shift 2 ;;
        --marl-base-url)      MARL_BASE_URL="$2"; shift 2 ;;
        --openpi-repo-dir)    OPENPI_REPO_DIR="$2"; shift 2 ;;
        --editable-openpi)    USE_EDITABLE_OPENPI="true"; shift ;;
        --no-editable-openpi) USE_EDITABLE_OPENPI="false"; shift ;;
        --no-chex-pin)        PIN_CHEX="false"; shift ;;
        --no-start-marl)      START_MARL="false"; shift ;;
        --dry-run)            DRY_RUN="true"; shift ;;
        --)                   shift; EXTRA_OVERRIDES=("$@"); break ;;
        *)                    echo "Unknown option: $1" >&2; usage ;;
    esac
done

USE_EDITABLE_OPENPI_BOOL="false"
case "${USE_EDITABLE_OPENPI}" in
    true)
        USE_EDITABLE_OPENPI_BOOL="true"
        ;;
    false)
        USE_EDITABLE_OPENPI_BOOL="false"
        ;;
    *)
        echo "Error: unsupported editable-openpi mode: ${USE_EDITABLE_OPENPI}" >&2
        exit 1
        ;;
esac

if [ ! -d "${REPO_DIR}" ]; then
    echo "Error: RLinf repo directory not found: ${REPO_DIR}" >&2
    exit 1
fi

if [ "${START_MARL}" = "true" ] && [ ! -d "${MARL_REPO_DIR}" ]; then
    echo "Error: marl repo directory not found: ${MARL_REPO_DIR}" >&2
    exit 1
fi

if [ "${USE_EDITABLE_OPENPI_BOOL}" = "true" ] && [ ! -d "${OPENPI_REPO_DIR}" ]; then
    echo "Error: requested editable openpi, but directory not found: ${OPENPI_REPO_DIR}" >&2
    exit 1
fi

MARL_URL_NO_SCHEME="${MARL_BASE_URL#http://}"
MARL_URL_NO_SCHEME="${MARL_URL_NO_SCHEME#https://}"
MARL_HOST="${MARL_URL_NO_SCHEME%%:*}"
MARL_PORT="${MARL_URL_NO_SCHEME##*:}"
MARL_PORT="${MARL_PORT%%/*}"
if [ -z "${MARL_HOST}" ] || [ -z "${MARL_PORT}" ]; then
    echo "Error: could not parse host/port from MARL_BASE_URL=${MARL_BASE_URL}" >&2
    exit 1
fi

MARL_CMD=(
    uv run --project "${MARL_REPO_DIR}" --python 3.12
    python -m marl.dummy_server
    --host "${MARL_HOST}"
    --port "${MARL_PORT}"
    --reward-mode global_step
    --log-level info
)

UV_EVAL_CMD=(uv run --project . --extra embodied)
if [ "${PIN_CHEX}" = "true" ]; then
    UV_EVAL_CMD+=(--with "chex==0.1.90")
fi
if [ "${USE_EDITABLE_OPENPI_BOOL}" = "true" ]; then
    UV_EVAL_CMD+=(--with-editable "${OPENPI_REPO_DIR}")
fi

EVAL_CMD=(
    "${UV_EVAL_CMD[@]}"
    python examples/embodiment/eval_embodied_agent_remote_yam.py
    --config-name "${CONFIG_NAME}"
    "env.train.remote_server_url=${ROBOT_SERVER_URL}"
    "env.eval.remote_server_url=${ROBOT_SERVER_URL}"
    "marl.base_url=${MARL_BASE_URL}"
    "marl.planner.interval=0"
    "algorithm.eval_rollout_epoch=1"
    "env.eval.max_steps_per_rollout_epoch=10"
    "env.eval.max_episode_steps=10"
    "cluster.component_placement.rollout.placement=0"
    "cluster.component_placement.env.placement=0"
    "runner.logger.log_path=/tmp/rlinf-yam-desktop-smoke"
)

if [ -n "${MODEL_PATH}" ]; then
    EVAL_CMD+=(
        "actor.model.model_path=${MODEL_PATH}"
        "rollout.model.model_path=${MODEL_PATH}"
    )
fi

if [ -n "${TASK_DESC}" ]; then
    EVAL_CMD+=(
        "env.train.task_description=${TASK_DESC}"
        "env.eval.task_description=${TASK_DESC}"
    )
fi

for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
    EVAL_CMD+=("${override}")
done

echo "=== YAM Desktop Smoke ==="
echo "Config:             ${CONFIG_NAME}"
echo "Model:              ${MODEL_PATH:-<config default>}"
echo "Task:               ${TASK_DESC:-<config default>}"
echo "Robot server URL:   ${ROBOT_SERVER_URL}"
echo "marl repo:          ${MARL_REPO_DIR}"
echo "marl base URL:      ${MARL_BASE_URL}"
echo "Editable openpi:    ${USE_EDITABLE_OPENPI_BOOL}"
echo "openpi repo:        ${OPENPI_REPO_DIR}"
echo "Pin chex==0.1.90:   ${PIN_CHEX}"
echo "Start dummy marl:   ${START_MARL}"
echo ""

if [ -n "${DRY_RUN}" ]; then
    if [ "${START_MARL}" = "true" ]; then
        echo "[dry-run] dummy marl command:"
        quote_cmd "${MARL_CMD[@]}"
        echo ""
    fi
    echo "[dry-run] eval command:"
    quote_cmd "${EVAL_CMD[@]}"
    echo ""
    exit 0
fi

export EMBODIED_PATH="${EMBODIED_PATH:-examples/embodiment}"
export ROBOT_SERVER_URL
export MARL_BASE_URL

if [ "${START_MARL}" = "true" ]; then
    if curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null 2>&1; then
        echo "marl already healthy at ${MARL_BASE_URL}; reusing existing process."
    else
        echo "Starting dummy marl at ${MARL_BASE_URL}..."
        (
            cd "${MARL_REPO_DIR}"
            nohup "${MARL_CMD[@]}" > "${MARL_REPO_DIR}/marl_dummy_server.log" 2>&1 &
        )

        for i in $(seq 1 30); do
            if curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null 2>&1; then
                break
            fi
            sleep 1
            if [ "${i}" = "30" ]; then
                echo "Error: dummy marl failed to become healthy at ${MARL_BASE_URL}" >&2
                exit 1
            fi
        done
        curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null
    fi
fi

echo "Launching desktop smoke eval..."
cd "${REPO_DIR}"
exec "${EVAL_CMD[@]}"
