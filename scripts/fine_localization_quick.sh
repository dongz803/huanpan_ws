#!/usr/bin/env bash
# Start D435 + YOLO coarse ROI + sub-pixel edge refinement + planar IPPE pose.
set -eo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"
source scripts/perception_env.sh

MODEL="${1:-outputs/perception/runs/slot_opening_v1/weights/best.pt}"
SLOT="${2:-0}"
MODE="${3:-edges}"
TARGET_CLASS="${4:-slot_opening}"

if [[ "$MODEL" != /* ]]; then
  MODEL="${WORKSPACE}/${MODEL}"
fi

exec ros2 launch huanpan_perception fine_localization.launch.py \
  model_path:="${MODEL}" \
  target_class:="${TARGET_CLASS}" \
  target_slot:="${SLOT}" \
  refinement_mode:="${MODE}" \
  start_camera:=true
