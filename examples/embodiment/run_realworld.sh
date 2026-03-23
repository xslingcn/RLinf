#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_dummy_franka_sac_cnn"
else
    CONFIG_NAME=$1
fi

# Auto-select entry script:
#   - YAM marl configs use train_embodied_agent_marl.py
#   - explicit staged configs use train_embodied_agent_staged.py
#   - everything else uses train_embodied_agent.py
case "$CONFIG_NAME" in
    yam_ppo_openpi|yam_ppo_openpi_topreward|*marl*)
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

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
