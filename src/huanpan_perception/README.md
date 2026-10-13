# huanpan_perception

换盘机器人的**感知识别模块**：在真实 Intel RealSense D435 图像上跑 YOLO，
输出机架/槽位检测结果，供 `huanpan_motion` 做粗定位和接近。

当前主链检测**单个目标槽口**（类别 `slot_opening`，1 类 1 框），再在粗框内
完成四边寻边、四角求交、IPPE PnP 和深度平面检查。手眼标定与机械臂闭环仍属于
后续实机接入阶段。

## 1. 与其它包的关系

```text
D435 ──(realsense2_camera)──► /camera/camera/color/image_raw
                                      │
                                      ▼
                     huanpan_perception / detector_node
                       YOLO → 2D 框 + 类别 + 置信度
                                      │
                                      ▼
                            /detections (vision_msgs)
                                      │
                        （后续）槽位 6D 位姿 → huanpan_motion
```

本包**不含**仿真相机、不做手眼标定、不发 6D 位姿。这些是后续步骤。

## 2. 依赖

| 依赖 | 安装方式 | 说明 |
|---|---|---|
| `ros-humble-realsense2-camera` | `sudo apt install ros-humble-realsense2-camera` | **需要你自己执行**，沙箱里 sudo 不可用 |
| `ros-humble-vision-msgs` | `sudo apt install ros-humble-vision-msgs` | 不装也能跑，见下面 `output_mode` |
| `ultralytics` + `torch`(CPU) | 见第 3 节 | 已装在 `.venv-perception` |

D435 非 root 访问通常还需要一次 udev 规则：

```bash
sudo cp /etc/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/ 2>/dev/null || true
# 或参考 librealsense 的 scripts/setup_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

检查相机是否在线：

```bash
source scripts/perception_env.sh
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
ros2 topic hz /camera/camera/color/image_raw
```

## 3. 环境

推理和训练需要 Torch/Ultralytics，它们不在系统 Python 里，而在工作区虚拟环境：

```bash
cd /home/lzt/huanpan_ws
python3 -m venv .venv-perception
.venv-perception/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
.venv-perception/bin/pip install -r perception/requirements-perception.txt
```

日常使用只需要：

```bash
source scripts/perception_env.sh
```

所有节点都能通过 `ros2 run` 直接启动：入口点内部会检测 Torch 是否可导入，
不可导入时自动用 `.venv-perception` 的解释器重新执行自己
（见 `huanpan_perception/launcher.py`），因此**不需要记住特殊 python 命令**。

## 4. 构建

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select huanpan_perception --symlink-install
source install/setup.bash
```

## 5. 工作流

### 5.1 采集真实图像

#### 最快的方式：一条命令 + 实时预览

```bash
./scripts/capture_quick.sh                              # 默认数据集目录
./scripts/capture_quick.sh outputs/perception/fine_test # 指定目录
./scripts/capture_quick.sh outputs/perception/fine_test fineA  # 指定目录 + 片段名
```

它会启动 D435 并弹出预览窗口。

采集节点会按时间戳同步 RGB、对齐深度和颜色相机 `CameraInfo`。默认允许的
最大时间差为 30 ms；不同步、尺寸不一致或缺少深度的帧不会保存。除
`images/` 和 `depth/` 外，还会生成：

- `metadata/<prefix>_camera_info.yaml`：本场采集使用的颜色相机内参；
- `frames.csv`：三路时间戳、同步误差、分辨率和有效深度比例。

同一目录重复使用同一前缀时会从已有最大编号继续，不会覆盖旧数据。

> **路径约定**：`output_dir` 是**数据集根目录**，彩色帧自动写入 `<root>/images`、
> 深度写入 `<root>/depth`。**不要自己加 `/images`**，否则会变成
> `<root>/images/images`（节点会对此发出警告）。

| 按键 | 作用 |
|---|---|
| **SPACE** 或 `s` | 保存当前帧（彩色 + 对齐深度） |
| `d` | 切换深度叠加显示 |
| `q` / ESC | 退出 |

**深度叠加里红色 = 深度空洞。** 目标准备做精细定位时，先看目标区域是不是一片红——
这直接决定后面走哪条技术路线。

键盘走的是 OpenCV 窗口而不是 stdin，所以在 `ros2 launch` 下也能用。
第三个参数是**片段名**：不同场次用不同名字（`fineA`、`fineB`…），
训练时才能按片段分组切分、避免近重复帧泄漏。

### 5.2 YOLO 粗框后的精细定位

主链不是 AprilTag 检测：现有 YOLOv8 继续负责 L0 粗定位。当前先执行独立的
**正视阶段**：使用黄色粗框外侧的有效深度点拟合局部面板平面，以平面法向计算
相机的 pitch/yaw 修正；同时在黄色框内部提取占优长边，计算 roll 修正。这个阶段
不寻找四角，也不运行 IPPE，线速度始终为零，不会根据框中心调整 X/Y。连续 5 帧
满足“平面倾角不超过 5 度、长边倾角不超过 3 度”后，
`front_view_aligned` 才变为 `true`，随后才运行严格四角精定位。

当前里程碑默认设置 `front_view_only: true`：即使达到 `FRONT_ALIGNED`，也只
显示黄色 YOLO 框和旋转反馈，不运行、不显示绿色 `FINE`。准备开始后续 X/Y
居中时，将该参数改为 `false`，才会在正视后自动进入严格精定位。

正视完成后，节点在原始 RGB
分辨率上为检测框的每一侧保留最多 3 条梯度直线候选，将四侧候选组合成多个
四边形，再用边界距离、边缘强度和 `SOLVEPNP_IPPE` 重投影误差联合选出结果。
这比每侧只取最强边更不容易把反光、槽内纹理误认为物理边缘。槽口已知尺寸仍为
`100 x 25 mm`，输出相机坐标系下的 6D 位姿。

实时节点还会对最近 5 帧做位姿平滑：连续 3 帧有效后才进入 `TRACKING`，短暂
丢失时显示 `HOLDING`，连续 3 帧失败后才进入 `LOST`；单帧平移跳变大于
50 mm 或旋转跳变大于 15 度会被拒绝。对齐深度只拟合槽口周围面板，并在深度
质量足够时检查“PnP 的 Z”与面板中值深度是否相差超过 50 mm。深度采样区域是
绿色四边形外侧紧邻轮廓的一圈面板带，槽口内部和边界附近的混合像素会被排除；
RANSAC 后的中值深度也只统计主平面的内点。上述门限可在
`config/perception.yaml` 中调整。

```bash
cd /home/lzt/huanpan_ws
./scripts/fine_localization_quick.sh
```

查看调试图和数值：

```bash
rqt_image_view /huanpan/vision/debug_image
ros2 topic echo /huanpan/vision/status
ros2 topic echo /huanpan/vision/front_view_correction
ros2 topic echo /huanpan/vision/front_view_aligned
ros2 topic echo /huanpan/vision/slot_pose
```

正视阶段的粗四边形允许最高 8 px 重投影误差，只用于旋转引导；严格绿色精定位
仍使用 2 px 门限。调试图显示 `ORIENTING` 和 `ROT [deg]`。其中 ROT 是相机光学坐标系
中的小角度旋转建议，`MOVE [mm]` 固定为零。它目前只用于手持验证；接入机械臂
前还必须经过手眼变换转换到机器人基坐标系，并增加单步角度和速度限制。
调试图颜色固定为：黄色 `YOLO` 是粗检测框，绿色 `FINE` 是四边寻边结果；
`ORIENTING` 阶段尚未运行精定位，因此只显示黄色框。

把相机手持到希望机器人以后到达的预插入视角，保持静止，然后示教当前
相对位姿：

```bash
ros2 service call /huanpan/vision/teach_reference std_srvs/srv/Trigger "{}"
```

随后移开相机并按调试画面的 `MOVE [mm]`、`ROT [deg]` 提示返回；同时满足
3 mm 和 1 度时 `/huanpan/vision/aligned` 为 `true`。这里的误差是手持引导
量，不会直接控制机械臂。

`/huanpan/vision/status` 的 `tracking_state` 有四种主要状态：`ACQUIRING`
表示正在等待连续有效帧，`TRACKING` 表示稳定输出，`HOLDING` 表示发生了短暂
漏检或异常但暂时保留上一稳定值，`LOST` 表示已经失去目标。示教服务只接受
`TRACKING` 状态下的位姿。

离线回放新采集的数据：

```bash
ros2 run huanpan_perception fine_localize_images \
  --model outputs/perception/runs/slot_opening_v1/weights/best.pt \
  --images outputs/perception/fine_test/images \
  --depth-dir outputs/perception/fine_test/depth \
  --camera-info outputs/perception/fine_test/metadata/fineB_camera_info.yaml
```

现有权重在 `imgsz=640` 训练，所以当前 YOLO 仍按 640 做粗检测；边缘精修
直接处理相机的 1280×720 原图。收集并以1280重新训练后，再同时把训练和
推理的 `imgsz` 改为1280。

可选的 AprilTag 板仅用于精度基准和后续手眼标定，不替代 YOLO 主链：

```bash
ros2 run huanpan_perception generate_tag_board \
  --output outputs/perception/calibration/apriltag_3x2.png
```

按100%打印，成品应为160×110 mm，并刚性固定后再用于标定。

#### 等价的手动命令

```bash
source scripts/perception_env.sh
ros2 launch huanpan_perception capture.launch.py \
  output_dir:=$PWD/outputs/perception/fine_test \
  prefix:=fineA start_camera:=true save_depth:=true preview:=true
```

不需要预览时（无人值守定时采集）：

```bash
ros2 run huanpan_perception capture_node --ros-args \
  -p output_dir:=$PWD/outputs/perception/datasets/rack_grid_v1 \
  -p mode:=interval -p interval_s:=1.0 -p save_depth:=true
```

#### 采完先验深度质量

精细定位能不能做，取决于深度在目标上有没有值。这个检查几分钟出结论：

```bash
analyze_depth \
  --images outputs/perception/fine_test/images \
  --labels outputs/perception/fine_test/labels \
  --output outputs/perception/fine_test/depth_report.png
```

输出：

```
帧数                : 6（有深度 6）
全图深度有效率      : 94.0%
目标框内深度有效率  : 0.0%
结论                : ❌ 目标上基本没有深度，必须用 CAD/参数化模型 + PnP
```

判定阈值：

| 目标框内有效率 | 结论 |
|---|---|
| > 60% | ✅ 深度可直接用于反投影 |
| 25% ~ 60% | ⚠️ 空洞较多，走平面拟合 + 直线求交 |
| < 25% | ❌ 目标上没深度，必须用参数化模型 / CAD + PnP |

同时生成可视化（左彩色 / 右深度，**红色标出空洞**）和 `*_report.json`。

#### 也可以：录视频再抽帧

```bash
extract_frames --video ~/videos/clips/ \
  --output outputs/perception/datasets/rack_grid_v1/images \
  --every-seconds 0.7
```

⚠️ **不要直接对一段视频每帧都截**。相邻帧几乎是同一张图，会导致两个问题：
模型看不到多样性，而且随机切分训练/验证集时，验证集里混进训练集的近重复帧，
**mAP 会虚高**。所以这个工具做了两件事：

1. **按时间间隔抽帧**（默认 0.7 秒一帧），不是每帧都留；
2. **近似去重**：与上一张"已保留"的帧几乎相同的直接丢弃
   （`--min-difference`，默认 0.012，设 0 关闭）。
   实测一段 3 秒静止画面：90 帧 → 30 帧 → **去重后只剩 1 帧**。

抽出来的帧命名为 `<片段名>_00001.jpg`，这样 `train_detector` 能**按片段分组切分**，
保证同一段视频的帧不会同时出现在训练集和验证集里。它会打印用的是哪种切分：

```
[train_detector] split by source group (3 group(s): clipAx4, clipBx4, clipCx4)
                 so near-identical frames from one clip stay on one side
```

如果所有帧名字前缀都一样（例如全是 `capture_00001.png`），无法分组，工具会退化成
随机切分并给出 WARNING。**建议每段视频单独命名**，或者干脆录 5~10 段短片段而不是
一段长视频。

RealSense 录的 `.bag` 先用 `rs-convert -i recording.bag -p out` 转成图片，
再放进 `images/`。

无人值守定时采集（也可用 launch 顺带起相机）：

```bash
ros2 launch huanpan_perception capture.launch.py \
  output_dir:=$PWD/outputs/perception/datasets/rack_grid_v1 \
  max_frames:=200
```

采集建议（第一版只标整块机架网格）：

- 30–60 cm 距离，覆盖正面、略偏左右、略俯仰；
- 包含不同光照、反光、背景杂乱、部分出画的情况；
- 200–400 张起步，其中约 1/4 作为"难例"（大角度、远距离、遮挡）。

### 5.2 标注（单类，画一个框）

```bash
ros2 run huanpan_perception label_dataset \
  --images outputs/perception/datasets/rack_grid_v1/images
```

鼠标拖框，`s` 保存并下一张，`n` 跳过，`d` 删除，`b` 上一张，`q` 退出。
标签写到同级 `labels/` 目录，YOLO 格式 `class cx cy w h`（归一化）。

**目标定义（已确认）**：框住机器人准备插入的**单个 100×25 mm 槽口开口**。
不要框整台服务器、整排硬盘或机柜。一张图只指定一个目标槽口，全部图片保持一致。

### 5.2.1 标注一致性自检（标到几十张以后必用）

10 张时画歪一眼能看出来，300 张时看不出来。这个检查不训练模型，毫秒级完成，
**每次训练前跑一遍**：

```bash
label_dataset --images outputs/perception/datasets/rack_grid_v1/images --check
```

```bash
ros2 run huanpan_perception label_dataset \
  --images outputs/perception/datasets/rack_grid_v1/images --check
```

输出示例：

```
图片总数        : 10
标注框总数      : 9
每个图的框数量  : 0 框 x 1, 1 框 x 9
宽高比中位数    : 2.79
面积占比中位数  : 0.029
未标注          : 1 张
可疑            : 3 张

按可疑程度排序（先看最上面的）:
  [ 10.0] 009_Color.png
          - 没有标签文件
  [  3.1] 004_Color.png
          - 第 1 个框面积占比 0.074，中位数 0.029（2.56x）
  [  1.3] 010_Color.png
          - 第 1 个框宽高比 0.97，中位数 2.79（0.35x）
```

同时把最可疑的样本拼成一张图（按可疑程度排序），方便直奔问题：

```
outputs/perception/label_outliers.png
```

检查项：缺标签 / 空标签 / 框数量与数据集常见值不同 / 宽高比离群 / 面积离群 /
框过小 / 框几乎覆盖整图 / 框中心越界。可用 `--check-output` 改拼图路径。

### 5.3 训练（本机 CPU）

```bash
ros2 run huanpan_perception train_detector \
  --dataset-root outputs/perception/datasets/rack_grid_v1 \
  --class-names slot_opening --name slot_opening_v1 \
  --epochs 60 --imgsz 640 --batch 8 --device 0
```

它会自动按 `--val-ratio`（默认 0.2）切分 `images/` + `labels/`，
生成 `data/data.yaml`，然后微调 `yolov8n`。权重输出到：

```
outputs/perception/runs/slot_opening_v1/weights/best.pt
```

小数据集（<200 张）建议 `--epochs 40 --imgsz 640`；数据多了可以换
`--model yolov8s.pt`。

### 5.4 评估

```bash
ros2 run huanpan_perception evaluate_detector \
  --model outputs/perception/runs/server_unit/weights/best.pt \
  --dataset-root outputs/perception/datasets/rack_grid_v1 \
  --split val --fx 617.16 --depth-m 0.45
```

输出 `outputs/perception/eval/metrics.json` 和预测拼图，包含：

- 检测率、匹配 IoU、平均中心误差（像素）；
- 给了 `--fx` 和 `--depth-m` 时，额外给出**中心误差的毫米值**——
  这才是交接给力控时真正重要的指标。

### 5.5 无相机也能测（假相机）

```bash
ros2 run huanpan_perception fake_camera_node --ros-args \
  -p source:=<任意图片目录> -p image_topic:=/camera/camera/color/image_raw
```

配合 `detector_node` 即可在没有 D435、没有 `realsense2_camera` 的情况下
验证整条 ROS 链路。

### 5.6 运行时检测

```bash
ros2 launch huanpan_perception perception.launch.py
ros2 launch huanpan_perception perception.launch.py start_camera:=true
```

看结果：

```bash
ros2 topic echo /detections --once
ros2 run rqt_image_view rqt_image_view /detections/image
```

## 6. 节点与话题

| 节点 | 可执行文件 | 作用 |
|---|---|---|
| `huanpan_detector` | `detector_node` | 订阅彩色图，YOLO 推理，发布检测 |
| `huanpan_capture` | `capture_node` | 采集数据集（彩色 + 可选对齐深度） |
| `huanpan_fake_camera` | `fake_camera_node` | 用磁盘图片模拟相机流 |
| `huanpan_fine_localizer` | `fine_localization_node` | 正视姿态引导与四角/IPPE精定位 |

| 话题 | 类型 | 方向 |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | 订阅（D435 彩色） |
| `detections` | `vision_msgs/Detection2DArray` | 发布（`output_mode:=vision_msgs`） |
| `detections` | `std_msgs/String` (JSON) | 发布（`output_mode:=json`） |
| `detections/image` | `sensor_msgs/Image` (`rgb8`) | 发布（可视化叠加图） |
| `/huanpan/vision/front_view_correction` | `geometry_msgs/TwistStamped` | 发布（仅角度修正） |
| `/huanpan/vision/front_view_aligned` | `std_msgs/Bool` | 发布（正视阶段完成） |
| `/huanpan/vision/slot_pose` | `geometry_msgs/PoseStamped` | 发布（正视后精定位位姿） |
| `/huanpan/vision/status` | `std_msgs/String` | 发布（JSON状态与质量指标） |

`vision_msgs` 未安装时，把参数改成 `output_mode:=json` 就能立刻跑通，
JSON 里除了全部检测框还有 `target` 字段（默认取 `server_unit` 那一类，
用于后续粗定位交接）。

`/detections` 中的每个 `Detection2D`：

- `id` = 类别名（如 `server_unit`）；
- `results[0].class_id` / `.score` = 类别与置信度；
- `bbox.center.position` / `bbox.size_x/size_y` = 像素框。

## 7. 关键实现说明

- **不使用 `cv_bridge`**。Humble 的 `cv_bridge` 是针对 NumPy 1.x 编译的，
  和虚拟环境里的 NumPy 2.x 一起用会抛
  `AttributeError: _ARRAY_API not found`。`image_utils.py` 手写
  `sensor_msgs/Image` ↔ NumPy 转换，支持 `rgb8 / bgr8 / rgba8 / bgra8 /
  mono8 / mono16 / 16UC1 / 32FC1`，并正确处理行 padding。
- **推理在后端跳过积压帧**：`detector_node` 用非阻塞锁，NPU/CPU 跟不上时
  丢帧而不是排队，保证发布的永远是最新图像的结果。
- **检测预览不阻塞主链**：`/detections/image` 默认缩放到 640 像素宽，采用
  `BEST_EFFORT`、深度 1；没有订阅者时跳过画框和图像序列化。该缩放只影响
  调试显示，不改变 YOLO 输入图像或检测框坐标。
- **类别数自适应**：`RackDetector` 从 `.pt` 里读取 `names`，所以后续把
  数据集扩成 12 个槽位时，无需改代码。
- **测试不依赖 Torch**：`test/` 覆盖图像转换、检测结果数据结构、数据集切分、
  PnP、深度平面和时序状态机，可在没装 Torch 的环境运行。

### 7.1 后续精定位算法选型（GitHub 调研）

当前版本先采用“多边线候选 + IPPE + 时序稳定”，因为它可以直接复用现有
YOLO 框和几十张标注，不需要重新准备像素级掩码或四角关键点数据。后续建议按
下面顺序做独立离线分支，不直接替换当前可运行链路：

1. **示教参考图匹配（优先）**：使用
   [XFeat](https://github.com/verlab/accelerated_features) 提取局部特征，再用
   [LightGlue / Glue Factory](https://github.com/cvg/glue-factory) 匹配当前图与
   示教图，通过 RANSAC 单应矩阵把示教的四角映射到当前图，最后继续使用 IPPE。
   这条路线不需要机柜 CAD，也不需要重新手工标四角，适合“回到示教视角”。
2. **深度直线精修**：用 [DeepLSD](https://github.com/cvg/deeplsd) 替换或精修
   Sobel 直线候选。它的定位是高精度通用线段检测/精修，比直接使用通用
   Transformer 线段检测器更贴合本项目现阶段的四边求交需求。
3. **关键点学习（数据足够后）**：把标注从一个框升级为槽口四角，训练
   YOLO-pose/关键点网络。它能直接针对本机柜学习角点，但几十张图只适合原型，
   还需要更完整的距离、倾角、曝光和遮挡覆盖。
4. **SAM 2 只作辅助**：
   [SAM 2](https://github.com/facebookresearch/sam2) 适合从提示生成并跨视频传播
   掩码，可用于 ROI 稳定或半自动标注；掩码边界本身不是毫米级四角，也不能
   直接输出 6D 位姿，因此不作为当前 PnP 前端的直接替代。

[LETR](https://github.com/mlpc-ucsd/letr) 这类 Transformer 线段检测器也可作为
对照实验，但机柜内部线条很多，仍要解决“哪四条线属于目标槽口”的关联问题。
算法替换前统一使用同一验证集，至少比较检测成功率、重投影误差、静止抖动、
单帧耗时和不同距离/倾角下的有效范围。

## 8. 测试

```bash
cd src/huanpan_perception
env -u PYTHONPATH ../../.venv-perception/bin/python -m pytest test -q

# 也可以用 ROS 工具链
source /home/lzt/huanpan_ws/scripts/perception_env.sh
colcon test --packages-select huanpan_perception
colcon test-result --verbose
```

## 8.5 纯 Python 模式（完全不需要 ROS）

**标注、训练、评估、离线推理这四个工具不依赖 ROS**，只有 `detector_node` /
`capture_node` / `fake_camera_node` 三个节点需要 `rclpy`。所以可以先用纯 Python
把检测跑通，之后再接 ROS 2。

装一次，就有普通命令可用：

```bash
cd /home/lzt/huanpan_ws
.venv-perception/bin/pip install -e src/huanpan_perception
```

之后**不用 source ROS**，也不用 `ros2 run`：

```bash
.venv-perception/bin/label_dataset       --images outputs/perception/datasets/rack_grid_v1/images
.venv-perception/bin/train_detector      --dataset-root outputs/perception/datasets/rack_grid_v1
.venv-perception/bin/evaluate_detector   --model outputs/perception/runs/server_unit/weights/best.pt \
                                         --dataset-root outputs/perception/datasets/rack_grid_v1
.venv-perception/bin/detect_images       --model outputs/perception/runs/server_unit/weights/best.pt \
                                         --source outputs/perception/datasets/rack_grid_v1/images
```

`-e` 是 editable 安装，改代码立即生效，不用重装。同一条命令也仍然可以用
`ros2 run huanpan_perception <名字>` 启动，两套入口指向同一份源码。

### 用 labelImg 标注

labelImg 直接导出 YOLO 格式，和本包格式完全一致，可以混用：

```bash
pip install labelImg    # 建议单独装在别的环境，避免污染 .venv-perception
labelImg
```

设置要点（漏掉第 2 步会把标签写错地方）：

1. 左侧 **Open Dir** → `.../rack_grid_v1/images`
2. **Change Save Dir** → `.../rack_grid_v1/labels`（**必须改**，默认会写到图片旁边）
3. 右下角格式切换到 **YOLO**
4. 拖框后填类别名 `server_unit`（第一张会自动生成 `classes.txt`）
5. `d` 下一张 / `a` 上一张 / `Ctrl+S` 保存

标完用本包自检（能直接读 labelImg 的 txt）：

```bash
.venv-perception/bin/label_dataset \
  --images outputs/perception/datasets/rack_grid_v1/images \
  --preview outputs/perception/label_check.png
```

⚠️ 检查 `labels/` 里第一个 txt：应以类别**索引** `0 ` 开头。若以 `server_unit `
开头，说明导出成了类别名而不是索引，需在 labelImg 里确认导出格式是 YOLO。

## 9. 已知边界与后续步骤

当前**未做**（按你的要求先只做最小 YOLO 检测）：

1. **手眼标定**：手持阶段只输出相机系位姿。实机路线已确定为腕上 D435，
   但最终刚性安装后仍须重新标定 `tool0 -> camera_link`。
2. **深度与6D位姿**：同步采集、框内四边拟合、已知尺寸平面PnP和深度
   面板校验已经接通；当前需要用新采的1280×720数据做实测验收。
3. **槽位级识别**：目前 1 类 = 整块网格。扩到 12 个槽位时重新标注即可，
   代码无需改动（类别名从模型里读）。
4. **真实 D435 数据**：已有35组640×480数据，但没有同步保存的CameraInfo；
   精定位评估需要按新版采集器补采。
5. **模型精度**：当前仅9张人工标注，粗定位链路可用但泛化不足；需要补充
   不同距离、角度和光照的数据。

推荐顺序：同步补采 → YOLO粗框 → 四边精修/PnP回放评估 → 固定相机并
手眼标定 → 视觉小步闭环 → 交给 `huanpan_motion` 的力控插入。
