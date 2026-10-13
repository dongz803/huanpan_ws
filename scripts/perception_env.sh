#!/usr/bin/env bash
# Perception environment: ROS 2 Humble + this workspace + the perception venv.
#
#   source scripts/perception_env.sh
#   ros2 launch huanpan_perception perception.launch.py
#
# The venv is prepended to PATH so `python` resolves to the interpreter that
# has torch/ultralytics. Nodes also self-heal via huanpan_perception.launcher
# when started with the system interpreter.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script: source scripts/perception_env.sh" >&2
  exit 2
fi

HUANPAN_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HUANPAN_WS_ROOT

# ROS's setup.bash reads variables such as AMENT_TRACE_SETUP_FILES without a
# default, so sourcing it under `set -u` aborts with "unbound variable".
# Relax -u for the duration of the ROS sourcing and restore it afterwards.
_huanpan_restore_u=0
if [[ $- == *u* ]]; then
  _huanpan_restore_u=1
  set +u
fi

source /opt/ros/humble/setup.bash

if [[ -f "${HUANPAN_WS_ROOT}/install/setup.bash" ]]; then
  source "${HUANPAN_WS_ROOT}/install/setup.bash"
fi

if [[ "${_huanpan_restore_u}" == "1" ]]; then
  set -u
fi
unset _huanpan_restore_u

HUANPAN_PERCEPTION_PYTHON="${HUANPAN_WS_ROOT}/.venv-perception/bin/python"
if [[ ! -x "${HUANPAN_PERCEPTION_PYTHON}" ]]; then
  echo "warning: ${HUANPAN_PERCEPTION_PYTHON} not found." >&2
  echo "  create it with: python3 -m venv .venv-perception" >&2
  echo "  then: .venv-perception/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "  then: .venv-perception/bin/pip install -r perception/requirements-perception.txt" >&2
else
  export HUANPAN_PERCEPTION_PYTHON
  export PATH="$(dirname "${HUANPAN_PERCEPTION_PYTHON}"):${PATH}"
fi

export ROS_LOG_DIR="${HUANPAN_WS_ROOT}/log/ros"
mkdir -p "${ROS_LOG_DIR}"

echo "huanpan_ws ROS 2 perception environment: ${HUANPAN_WS_ROOT}"
