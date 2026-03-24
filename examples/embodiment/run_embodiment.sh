#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    echo "Usage: bash examples/embodiment/run_embodiment.sh <config-name> [ROBOT_PLATFORM]" >&2
    exit 1
fi
CONFIG_NAME=$1

# Auto-select entry script:
#   - YAM marl configs use train_embodied_agent_marl.py
#   - explicit staged configs use train_embodied_agent_staged.py
case "$CONFIG_NAME" in
    *marl*)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_marl.py"
        ;;
    *staged*)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent_staged.py"
        ;;
    *)
        SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
        ;;
esac
export SRC_FILE

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
