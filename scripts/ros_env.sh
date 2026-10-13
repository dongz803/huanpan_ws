#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script: source scripts/ros_env.sh" >&2
  exit 2
fi

HUANPAN_WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HUANPAN_WS_ROOT

# ROS's setup.bash reads variables such as AMENT_TRACE_SETUP_FILES without a
# default, so sourcing it under `set -u` aborts with "unbound variable".
_huanpan_restore_u=0
if [[ $- == *u* ]]; then
  _huanpan_restore_u=1
  set +u
fi

source /opt/ros/humble/setup.bash

# Development fallback used on this machine until ros-humble-moveit is
# installed system-wide. A normal MoveIt installation already provides it.
HUANPAN_LOCAL_CONTROL_MSGS="${HUANPAN_WS_ROOT}/.deps/control_msgs/opt/ros/humble/share/control_msgs/local_setup.bash"
if ! ros2 pkg prefix control_msgs >/dev/null 2>&1 && [[ -f "${HUANPAN_LOCAL_CONTROL_MSGS}" ]]; then
  source "${HUANPAN_LOCAL_CONTROL_MSGS}"
fi

if [[ -f "${HUANPAN_WS_ROOT}/install/setup.bash" ]]; then
  source "${HUANPAN_WS_ROOT}/install/setup.bash"
fi

if [[ "${_huanpan_restore_u}" == "1" ]]; then
  set -u
fi
unset _huanpan_restore_u

export ROS_LOG_DIR="${HUANPAN_WS_ROOT}/log/ros"
mkdir -p "${ROS_LOG_DIR}"

# Used by huanpan_perception launch files and the re-exec shim.
HUANPAN_PERCEPTION_PYTHON="${HUANPAN_WS_ROOT}/.venv-perception/bin/python"
if [[ -x "${HUANPAN_PERCEPTION_PYTHON}" ]]; then
  export HUANPAN_PERCEPTION_PYTHON
else
  unset HUANPAN_PERCEPTION_PYTHON
fi

unset HUANPAN_LOCAL_CONTROL_MSGS
echo "huanpan_ws ROS 2 environment: ${HUANPAN_WS_ROOT}"
