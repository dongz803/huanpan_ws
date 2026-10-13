#!/usr/bin/env bash
# 一条命令开始采集：启动 D435 + 实时预览窗口
#
#   ./scripts/capture_quick.sh                      # 存到默认数据集目录
#   ./scripts/capture_quick.sh outputs/perception/fine_test
#
# 预览窗口里的按键：
#   SPACE 或 s   保存当前帧（彩色 + 对齐深度）
#   d            切换深度叠加显示
#   q 或 ESC     退出
#
# 深度叠加里红色 = 深度空洞。目标区域一片红，说明深度不可用，
# 精细定位就要改走「平面拟合」或「参数化模型 + PnP」。
# 注意：这里不能用 `set -u`。ROS 的 setup.bash 会引用未设置的
# AMENT_TRACE_SETUP_FILES，在 -u 下会直接中止。
set -eo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE"

OUT_ROOT="${1:-outputs/perception/datasets/rack_grid_v1}"
PREFIX="${2:-capture}"

# 用户可能在 .bashrc 里开了 set -u，perception_env.sh 会自己处理。
source scripts/perception_env.sh

echo "数据集根目录: ${WORKSPACE}/${OUT_ROOT}"
echo "  彩色帧 -> ${WORKSPACE}/${OUT_ROOT}/images"
echo "  深度帧 -> ${WORKSPACE}/${OUT_ROOT}/depth"

exec ros2 launch huanpan_perception capture.launch.py \
  output_dir:="${WORKSPACE}/${OUT_ROOT}" \
  prefix:="${PREFIX}" \
  start_camera:=true \
  save_depth:=true \
  preview:=true \
  max_frames:=100000
