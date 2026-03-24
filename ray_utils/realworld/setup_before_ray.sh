#!/bin/bash

echo "=== setup_before_ray.sh: Starting setup ==="

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "CURRENT_PATH=$CURRENT_PATH"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
echo "REPO_PATH=$REPO_PATH"
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"

# Modify these environment variables as needed
export RLINF_NODE_RANK=0 # Change this to the appropriate node rank if using multiple nodes
echo "RLINF_NODE_RANK=$RLINF_NODE_RANK"
export RLINF_COMM_NET_DEVICES="eth0" # Change this if you use a different network interface
echo "RLINF_COMM_NET_DEVICES=$RLINF_COMM_NET_DEVICES"

# If you are using a docker image or managed environment, activate the desired venv here instead.
source ./.venv/bin/activate # Source your virtual environment here
echo "Virtual environment activated"

# Add any extra controller/runtime setup your robot stack requires here.
