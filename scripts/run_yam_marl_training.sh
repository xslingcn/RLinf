#!/bin/bash
#
# run_yam_marl_training.sh — Start the single-node Beaker marl sidecar and
# launch the RLinf YAM training loop that consumes it.
#
# Intended topology (single Beaker node, 3 GPUs):
#   GPU 0 — actor
#   GPU 1 — rollout
#   GPU 2 — marl sidecar
#
# This script is safe to reuse in:
#   - Beaker job mode (`submit_yam_training.sh`)
#   - idle-cluster / SSH debugging mode (`submit_yam_beaker_cluster.sh`)
#   - interactive Beaker sessions

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

CONFIG_NAME="yam_ppo_openpi_topreward"
MODEL_PATH=""
TASK_DESC=""
MARL_REPO_DIR="${MARL_REPO_DIR:-${ROOT_DIR}/marl}"
MARL_CONFIG_PATH="${MARL_CONFIG_PATH:-}"
MARL_BASE_URL="${MARL_BASE_URL:-http://127.0.0.1:8080}"
MARL_CUDA_VISIBLE_DEVICES="${MARL_CUDA_VISIBLE_DEVICES:-2}"
OPENPI_REPO_DIR="${OPENPI_REPO_DIR:-${ROOT_DIR}/openpi}"
USE_EDITABLE_OPENPI="${USE_EDITABLE_OPENPI:-false}"
PIN_CHEX="true"
START_MARL="true"
DRY_RUN=""
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_yam_marl_training.sh [OPTIONS] [-- HYDRA_OVERRIDES...]

Start the local marl sidecar for a single-node 3-GPU Beaker run and launch
`train_embodied_agent_marl.py`.

Options:
  --config NAME            Hydra config name (default: yam_ppo_openpi_topreward)
  --model-path PATH        Model checkpoint path / HF ID for actor + rollout
  --task DESC              Task description override
  --marl-repo-dir PATH     marl repo directory (default: ../marl)
  --marl-config PATH       marl YAML config (default: <marl-repo-dir>/marl.yaml)
  --marl-base-url URL      marl base URL (default: http://127.0.0.1:8080)
  --marl-gpu ID            GPU visible to marl (default: 2)
  --openpi-repo-dir PATH   Local openpi checkout (default: ../openpi)
  --editable-openpi        Force `uv run --with-editable <openpi-repo-dir>`
  --no-editable-openpi     Do not use local editable openpi
  --no-chex-pin            Skip `--with chex==0.1.90`
  --no-start-marl          Assume marl is already running; only launch training
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
        --marl-repo-dir)      MARL_REPO_DIR="$2"; shift 2 ;;
        --marl-config)        MARL_CONFIG_PATH="$2"; shift 2 ;;
        --marl-base-url)      MARL_BASE_URL="$2"; shift 2 ;;
        --marl-gpu)           MARL_CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
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

if [ -z "${MARL_CONFIG_PATH}" ]; then
    MARL_CONFIG_PATH="${MARL_REPO_DIR}/marl.yaml"
fi

if [ ! -d "${REPO_DIR}" ]; then
    echo "Error: RLinf repo directory not found: ${REPO_DIR}" >&2
    exit 1
fi

if [ ! -d "${MARL_REPO_DIR}" ]; then
    echo "Error: marl repo directory not found: ${MARL_REPO_DIR}" >&2
    exit 1
fi

if [ ! -f "${MARL_CONFIG_PATH}" ]; then
    echo "Error: marl config not found: ${MARL_CONFIG_PATH}" >&2
    exit 1
fi

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

if [ "${USE_EDITABLE_OPENPI_BOOL}" = "true" ] && [ ! -d "${OPENPI_REPO_DIR}" ]; then
    echo "Error: requested editable openpi, but directory not found: ${OPENPI_REPO_DIR}" >&2
    exit 1
fi

MARL_CMD=(
    env "CUDA_VISIBLE_DEVICES=${MARL_CUDA_VISIBLE_DEVICES}"
    uv run --project "${MARL_REPO_DIR}" --python 3.12
    python -m marl --config "${MARL_CONFIG_PATH}" --log-level info
)

UV_TRAIN_CMD=(uv run --project . --extra embodied)
if [ "${PIN_CHEX}" = "true" ]; then
    UV_TRAIN_CMD+=(--with "chex==0.1.90")
fi
if [ "${USE_EDITABLE_OPENPI_BOOL}" = "true" ]; then
    UV_TRAIN_CMD+=(--with-editable "${OPENPI_REPO_DIR}")
fi

TRAIN_CMD=(
    "${UV_TRAIN_CMD[@]}"
    python examples/embodiment/train_embodied_agent_marl.py
    --config-name "${CONFIG_NAME}"
)

if [ -n "${MODEL_PATH}" ]; then
    TRAIN_CMD+=(
        "actor.model.model_path=${MODEL_PATH}"
        "rollout.model.model_path=${MODEL_PATH}"
    )
fi

if [ -n "${TASK_DESC}" ]; then
    TRAIN_CMD+=(
        "env.train.task_description=${TASK_DESC}"
        "env.eval.task_description=${TASK_DESC}"
    )
fi

for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
    TRAIN_CMD+=("${override}")
done

echo "=== YAM + marl Single-Node Training ==="
echo "Config:             ${CONFIG_NAME}"
echo "Model:              ${MODEL_PATH:-<config default>}"
echo "Task:               ${TASK_DESC:-<config default>}"
echo "marl repo:          ${MARL_REPO_DIR}"
echo "marl config:        ${MARL_CONFIG_PATH}"
echo "marl base URL:      ${MARL_BASE_URL}"
echo "marl GPU:           ${MARL_CUDA_VISIBLE_DEVICES}"
echo "Editable openpi:    ${USE_EDITABLE_OPENPI_BOOL}"
echo "openpi repo:        ${OPENPI_REPO_DIR}"
echo "Pin chex==0.1.90:   ${PIN_CHEX}"
echo "Start marl:         ${START_MARL}"
echo ""

if [ -n "${DRY_RUN}" ]; then
    if [ "${START_MARL}" = "true" ]; then
        echo "[dry-run] marl command:"
        quote_cmd "${MARL_CMD[@]}"
        echo ""
    fi
    echo "[dry-run] training command:"
    quote_cmd "${TRAIN_CMD[@]}"
    echo ""
    exit 0
fi

export MARL_BASE_URL
export EMBODIED_PATH="${EMBODIED_PATH:-examples/embodiment}"

if [ "${START_MARL}" = "true" ]; then
    if curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null 2>&1; then
        echo "marl already healthy at ${MARL_BASE_URL}; reusing existing process."
    else
        echo "Starting marl sidecar on GPU ${MARL_CUDA_VISIBLE_DEVICES}..."
        (
            cd "${MARL_REPO_DIR}"
            nohup "${MARL_CMD[@]}" > "${MARL_REPO_DIR}/marl_server.log" 2>&1 &
        )

        for i in $(seq 1 60); do
            if curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null 2>&1; then
                break
            fi
            sleep 2
            if [ "${i}" = "60" ]; then
                echo "Error: marl failed to become healthy at ${MARL_BASE_URL}" >&2
                exit 1
            fi
        done
        curl -fsS "${MARL_BASE_URL}/healthz" >/dev/null
    fi
fi

echo "Launching RLinf training..."
cd "${REPO_DIR}"
exec "${TRAIN_CMD[@]}"
