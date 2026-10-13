#!/usr/bin/env bash
# RealSense 环境体检：一条命令查完 Linux 上所有已知的坑
#
#   ./scripts/perception/check_realsense_env.sh
#
# 覆盖的已知问题：
#   1. udev 规则缺失        -> lsusb 看得到，librealsense 认不到
#   2. usbfs_memory_mb 太小 -> 高带宽流分配缓冲失败，几秒掉一次
#   3. USB 自动挂起         -> 设备空闲后被挂起，推流中断
#   4. 链路跑在 USB 2.0     -> 带宽不足，高分辨率打不开
#   5. 设备挂在 hub 后面    -> 供电/转发不稳，偶发掉线
set -uo pipefail

ok()   { printf '  \033[32m✅ %s\033[0m\n' "$1"; }
bad()  { printf '  \033[31m❌ %s\033[0m\n' "$1"; }
warn() { printf '  \033[33m⚠️  %s\033[0m\n' "$1"; }
info() { printf '     %s\n' "$1"; }

FAIL=0

echo "=== 1. 设备是否被 USB 识别 ==="
if lsusb 2>/dev/null | grep -qi '8086:0b3a'; then
  ok "找到 D435i (8086:0b3a)"
  lsusb 2>/dev/null | grep -i '8086:0b3a' | sed 's/^/     /'
else
  bad "USB 层面没找到 RealSense 设备"
  info "换一根 USB 线 / 换一个口，然后重跑本脚本"
  FAIL=1
fi

echo
echo "=== 2. udev 规则 ==="
if ls /etc/udev/rules.d/ 2>/dev/null | grep -qi realsense; then
  ok "已安装: $(ls /etc/udev/rules.d/ | grep -i realsense)"
else
  bad "未安装 udev 规则 —— lsusb 看得到但 librealsense 会认不到"
  info "sudo cp scripts/perception/99-realsense-libusb.rules /etc/udev/rules.d/"
  info "sudo udevadm control --reload-rules && sudo udevadm trigger"
  FAIL=1
fi

echo
echo "=== 3. usbfs 内存上限（最常见的掉线原因）==="
USBFS=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo "?")
echo "     当前值: ${USBFS} MB （内核默认 16；Intel 官方要求 1000）"
if [[ "$USBFS" == "?" ]]; then
  warn "读不到该参数"
elif [[ "$USBFS" -lt 1000 ]]; then
  bad "太小，高带宽流会反复掉线"
  info "临时修复（立即生效）:"
  info "  sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'"
  info "永久修复: 在 /etc/default/grub 的 GRUB_CMDLINE_LINUX_DEFAULT 里加"
  info "  usbcore.usbfs_memory_mb=1000   然后 sudo update-grub && sudo reboot"
  FAIL=1
else
  ok "足够 (${USBFS} MB)"
fi

echo
echo "=== 4. 设备电源管理 / 链路速率 ==="
FOUND=0
for d in /sys/bus/usb/devices/*/; do
  [[ -f "$d/idVendor" ]] || continue
  [[ "$(cat "$d/idVendor" 2>/dev/null)" == "8086" ]] || continue
  FOUND=1
  NAME=$(basename "$d")
  CTRL=$(cat "$d/power/control" 2>/dev/null || echo "?")
  SPEED=$(cat "$d/speed" 2>/dev/null || echo "?")
  echo "     设备 $NAME"
  if [[ "$CTRL" == "on" ]]; then
    ok "    自动挂起已关闭 (control=on)"
  else
    warn "    control=${CTRL} —— 建议关闭自动挂起"
    info "      sudo sh -c 'echo on > $d/power/control'"
  fi
  if [[ "$SPEED" == "5000" || "$SPEED" == "10000" ]]; then
    ok "    链路速率 ${SPEED} Mbps (USB 3.x)"
  else
    bad "    链路速率 ${SPEED} Mbps —— 不是 USB 3.0，高分辨率打不开"
    info "      换 USB 3.0 口 + USB 3.0 线"
  fi
  # 路径形如 2-2.1 表示经过 hub；2-2 表示直连根端口
  if [[ "$NAME" == *.* ]]; then
    warn "    挂在 hub 后面（${NAME}）—— hub 供电/转发不稳，偶发掉线常见诱因"
    info "      条件允许时改插直连口"
  else
    ok "    直连根端口"
  fi
done
[[ "$FOUND" == "1" ]] || warn "没有可检查的 RealSense 设备节点"

echo
echo "=== 5. librealsense 能否枚举 ==="
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  if timeout 15 rs-enumerate-devices -s 2>&1 | grep -qi 'D4\|RealSense'; then
    ok "rs-enumerate-devices 能认到设备"
  else
    bad "rs-enumerate-devices 认不到设备"
    info "先确认第 2、3 项是否已修复，然后拔插一次相机"
    FAIL=1
  fi
else
  warn "没装 rs-enumerate-devices（属于 ros-humble-librealsense2）"
fi

echo
if [[ "$FAIL" == "0" ]]; then
  printf '\033[32m体检通过：可以开始采集\033[0m\n'
  echo "  ./scripts/capture_quick.sh outputs/perception/fine_test fineA"
else
  printf '\033[33m有问题需要修复，见上面带 ❌ 的项\033[0m\n'
fi
