# perception/ — 仿真引导工具（当前未启用）

> **状态：目前不使用。** 你已确认直接上真实 D435，所以这条仿真路线暂停。
> 真实相机 → YOLO 的正式实现和文档在 `src/huanpan_perception/`。

这里保留的是一套**纯 MuJoCo 的离线工具**，用仿真场景自动生成带标注的
槽位图像数据集。它的价值只在将来需要"12 个槽位级别"的大规模预训练数据、
或在真机数据不足时做预训练时才会体现：

- `slotvision/scene.py` — 加载工作单元、随机化机架/相机/光照、渲染 RGB-D
- `slotvision/camera.py` — 针孔模型、投影/反投影、look-at
- `slotvision/generate.py` — 生成 YOLO 格式数据集（含槽位 6D 真值）
- `slotvision/train.py` / `evaluate.py` — 训练与评估
- `tests/test_projection.py` — 投影与标注契约测试

`requirements-perception.txt` **仍然在用**：`src/huanpan_perception/README.md`
第 3 节安装虚拟环境时会读它，不要删除这个文件。

## 如果要重新启用

```bash
cd /home/lzt/huanpan_ws/perception
env -u PYTHONPATH MUJOCO_GL=egl ../.venv/bin/python -m pytest -q
env -u PYTHONPATH MUJOCO_GL=egl ../.venv/bin/python -m slotvision.generate \
  --out outputs/perception/datasets/slot_detect_v1 --train 240 --val 60 --hard 40
```

注意：生成脚本依赖 `src/huanpan_description/.../workcell.xml` 里的槽位几何。
给机架加的 12 个 `slot_NN_entry` / `slot_NN_goal` 坐标系**已保留**，
它们是运动模块将来插入任意槽位时需要的目标坐标系，和相机无关。
仿真相机模型（`perception_cam`）已按你的要求从 `workcell.xml` 移除。
