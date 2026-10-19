# 精选实验输出

本目录只提交经过检查、适合公开且能帮助复现实验结论的少量产物。原始 RGB/深度
数据、包含人员或显示器的画面、视频、缓存、临时预览、训练批次图、`last.pt` 和
重复数据仍由 `.gitignore` 排除。

## 目录内容

```text
outputs/
├── force_traces/                         # MuJoCo 接触力历史 CSV
└── perception/
    ├── eval/slot_opening_v1/metrics.json # YOLO 独立评估摘要
    ├── fine_localization/*/              # 精定位离线结果与摘要
    └── runs/slot_opening_v1/
        ├── weights/best.pt               # 当前单槽口 YOLOv8 最佳权重
        ├── args.yaml                     # Ultralytics 训练参数
        ├── train_invocation.json         # 项目训练入口参数
        ├── results.csv / results.png     # 每轮训练指标
        └── *curve.png / confusion_matrix*.png
```

## YOLO 粗定位基线

- 类别：`slot_opening`，单类别；
- 输入尺寸：640；
- 训练：80 epochs、batch 8、CUDA device 0；
- 数据切分：63 张训练图、16 张验证图；
- `metrics.json` 中的离线验证结果：检测率 0.9375，匹配精度 0.7895，平均匹配
  IoU 0.8672，平均中心误差 7.11 px；
- 权重路径：`perception/runs/slot_opening_v1/weights/best.pt`；
- 权重 SHA-256：
  `d3085194ad3a01f1240109d82f8817483e84415ce5c8c4f08902c036a9bc05d0`。

这些数字来自当前小规模、同一实验环境的数据，只证明现有粗定位基线可以运行，
不能代表跨机柜、跨实验室或实机插入成功率。原始数据集未公开，因此该权重可用于
运行演示，但不能仅凭本仓库完整复现训练数据。

## 精定位离线结果

`perception/fine_localization/` 保留多个历史配置的 `results.json` 和
`summary.json`，用于比较有效率、重投影误差、平移抖动和旋转偏差。它们记录了
算法演进过程，不应只按重投影误差判断几何是否正确：错误物理边缘也可能得到较低
重投影误差。

当前默认运行阶段仍是 `front_view_only: true`：框外深度平面法向负责
pitch/yaw，框内长边负责 roll。严格四边/IPPE 精定位尚未完成 3 mm / 1 deg
重复到位验收。

## MuJoCo 力曲线

`force_traces/*.csv` 是 `/mujoco/drive_wrench` 的历史采样。每行依次为：

```text
stamp.sec, stamp.nanosec, frame_id,
force.x, force.y, force.z, torque.x, torque.y, torque.z
```

这些曲线包含名义插入和故意偏置的超力实验记录，只用于仿真算法分析。当前仿真
阈值不是实机硬盘、背板或连接器的安全限值，不能直接用于真实 UR5e。

## 隐私与复现边界

- 不公开原始相机图像、深度图和视频；
- 不提交 Ultralytics 本机设置、设备 UUID 或绝对缓存路径；
- 不提交 `last.pt`、缓存和重复的 train/val 图像；
- 接入实机前仍需完成相机固定、手眼标定、TF 转换和安全限位验证。

`best.pt` 由 Ultralytics 工具链训练生成；用于再分发或商业用途前，应自行核对对应
Ultralytics 版本及其许可条款。
