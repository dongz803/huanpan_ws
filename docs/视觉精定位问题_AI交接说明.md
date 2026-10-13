# 换盘机器人视觉精定位问题：AI 交接说明

**更新日期**：2026-09-18  
**项目路径**：`/home/lzt/huanpan_ws`  
**当前主任务**：从“检测到机架”推进到“输出可供 UR5e 插盘使用的槽位 6D 位姿”

> 本说明面向接手该问题的其他 AI。请严格区分：已经实现、已经验证、只有数据或
> 代码但尚未联调、以及建议中的未来方案。不要把二维检测框当作已经完成的精定位。

---

## 1. 希望接手 AI 解决的问题

当前插入测试仍使用模型中预先写死的 `slot_09_entry` 和 `slot_09_goal` 坐标。
需要建立视觉精定位链路，使系统能够根据 D435 的 RGB-D 数据估计机架位姿，识别
4 列 × 3 行共 12 个槽位，并在机器人基座坐标系下发布指定槽位的插入口位姿。

当前优先目标不是追求通用 6D 物体识别，而是先可靠解决：

1. 在画面中找到目标服务器/硬盘架；
2. 得到机架正面的 6D 位姿；
3. 根据规则的 4×3 几何结构确定 `slot_00`～`slot_11`；
4. 发布 `slot_09_entry`，替代运动程序中的硬编码世界坐标；
5. 给出定位置信度或协方差，定位不可靠时禁止机器人运动；
6. 视觉引导到槽口前，最后几毫米继续由已有力保护和撤退逻辑兜底。

---

## 2. 项目总体情况

### 2.1 技术栈

- Ubuntu + ROS 2 Humble；
- MuJoCo 3.12；
- MoveIt 2（工作区内存在源码、build 和 install）；
- UR5e 六轴机械臂；
- DH Robotics AG95 夹具模型；
- Intel RealSense D435；
- Ultralytics YOLO，当前主要使用 CPU 推理；
- Python 感知节点 + C++ 运动规划节点。

MoveIt 2 源码位于 `src/moveit2`。由于整套源码并行构建曾导致电脑卡顿，项目包编译
通常使用 `--parallel-workers 1`，不要无必要地反复全量重编 MoveIt 2。

### 2.2 主要 ROS 2 包

```text
src/
├── huanpan_description/      UR5e、AG95、硬盘、4×3机架模型
├── huanpan_mujoco/           MuJoCo桥、轨迹控制、扰动、接触力保护
├── huanpan_moveit_config/    MoveIt 2配置、SRDF、OMPL、控制器和目标配置
├── huanpan_motion/           运动学、碰撞、规划、轨迹处理、控制、自动插入测试
└── huanpan_perception/       D435采集、标注、YOLO训练/评估、2D检测节点
```

另有 `perception/slotvision/`：这是以前的 MuJoCo 离线槽位数据生成工具，不是当前
正式 ROS 感知包。它仍可作为离线算法验证材料，但 README 已明确标为“当前未启用”。

---

## 3. 机械结构与坐标约定

### 3.1 当前使用的尺寸

| 对象 | 尺寸 |
|---|---|
| 硬盘架/机箱正面宽度 | 430 mm |
| 硬盘架高度 | 85 mm |
| 硬盘架/硬盘深度方向 | 175 mm |
| 单块硬盘正面宽度 | 100 mm |
| 单块硬盘高度 | 25 mm |
| 槽位排列 | 横向 4 个，纵向 3 个，共 12 个 |

当前模型中的槽位中心间距约为：

- 横向列间距：105 mm；
- 纵向行间距：27.5 mm。

### 3.2 槽位编号与插入方向

- 机架局部坐标系 `+x` 为插入方向；
- `slot_00`～`slot_03` 为同一行的 4 个槽位；
- `slot_04`～`slot_07` 为下一行；
- `slot_08`～`slot_11` 为第三行；
- 当前自动插入测试使用 `slot_09`。

`slot_09` 当前硬编码世界坐标：

```yaml
entry_position: [0.6050, 0.0525, 0.7225]
goal_position:  [0.7875, 0.0525, 0.7225]
orientation_xyzw: [0.70710678, 0.0, 0.70710678, 0.0]
```

机架局部参考点：

```text
slot_09_entry = [-0.1025, 0.0525, -0.0275]
slot_09_goal  = [ 0.0800, 0.0525, -0.0275]
```

需要注意：入口点在机架前平面外侧，因此 `entry → goal` 的轨迹长度不是简单等于
175 mm。视觉模块首先应估计 `rack` 坐标系；12 个 entry/goal 应由机架位姿和已有
局部几何推导，而不是分别学习 12 个任意世界坐标。

关键模型文件：

- `src/huanpan_description/mujoco/universal_robots_ur5e/sa5212_empty_rack.xml`
- `src/huanpan_description/urdf/huanpan_workcell.urdf`
- `src/huanpan_moveit_config/config/insertion_targets.yaml`

---

## 4. 仿真、规划与插入进度

### 4.1 已完成

- UR5e、AG95、携带硬盘和 4×3 空硬盘架已经进入 MuJoCo/URDF；
- 硬盘目前作为机械臂末端固定 link，用同尺寸长方体表示；
- MoveIt 2 可以规划到 `slot_09_entry`；
- 自动测试可生成 `entry → goal` 笛卡尔插入轨迹；
- MuJoCo 桥支持重力补偿和有色关节力矩扰动；
- 已实现接触力发布、滤波、持续超阈值判断、轨迹中止和自动撤退；
- 力保护默认阈值为 25 N；
- 插入测试支持 `nominal` 和故意横向偏置的 `overforce` 两种模式。

### 4.2 最近一次关键问题与修复

曾出现：

```text
MoveGroupInterface::execute() failed or timeout reached
Nominal insertion failed; max force 30.67 N
```

这不是 MoveIt 连接失败，而是力保护真实触发。原因是扰动虽然固定 seed，但扰动从
仿真启动时就开始推进；MoveIt 每次规划耗时不同，插入开始时落在了不同扰动相位。

已经加入：

- `/mujoco/disturbance/reset` 服务；
- 自动插入到达 entry 后重置扰动序列；
- 然后再开启力保护并执行插入。

修复后一次已验证的 nominal 结果：

```text
Disturbance sequence reset to configured seed
Cartesian insertion path fraction: 1.000
Execute request success
NOMINAL INSERTION TEST: PASS; max force 3.02 N
```

### 4.3 测试证据

- `huanpan_mujoco`：19 个 pytest 通过；
- `huanpan_motion`：2 个 gtest 用例通过；
- 当时 `colcon test-result --verbose` 汇总为 22 tests、0 error、0 failure；
- 不应据此声称已完成真实硬盘插入，以上仅证明仿真链路和测试场景通过。

重要代码：

- `src/huanpan_motion/src/insertion_demo_node.cpp`
- `src/huanpan_motion/config/insertion_demo.yaml`
- `src/huanpan_mujoco/huanpan_mujoco/bridge_node.py`
- `src/huanpan_mujoco/huanpan_mujoco/force_guard.py`
- `src/huanpan_mujoco/huanpan_mujoco/disturbance.py`

---

## 5. 当前视觉模块的真实状态

### 5.1 已经实现并验证的能力

正式视觉包：`src/huanpan_perception`

已有工具：

- D435 彩色/对齐深度采集；
- 视频抽帧和近重复帧过滤；
- YOLO 标注及标签一致性检查；
- YOLO 训练、离线推理和评估；
- ROS 2 `detector_node`；
- 无相机时用磁盘图片模拟话题的 `fake_camera_node`；
- 可发布 2D 检测和叠加图。

当前测试结果：

```text
python3 -m pytest -q src/huanpan_perception/test
55 passed
```

### 5.2 当前系统环境

在 2026-09-18 检查：

- `realsense2_camera`：已安装；
- `cv_bridge`：已安装，但项目为了绕过 NumPy ABI 问题使用自写图像转换；
- `image_geometry`、`tf2_ros`：已安装；
- `vision_msgs`：未安装；
- 当前没有 `/dev/video*`，即检查时 D435 未连接或未被系统识别。

旧的 `docs/项目进程说明.md` 写有“RealSense ROS 驱动未安装”，这一点已经过时；
接手时应以现场 `ros2 pkg prefix` 和设备检查结果为准。

### 5.3 真实数据和模型

数据集：`outputs/perception/datasets/rack_grid_v1`

- 彩色图：30 张；
- 标签文件：9 个；
- 训练生成的数据切分：8 张 train、2 张 val；
- 类别：`server_unit`；
- 训练模型存在于：
  `outputs/perception/runs/server_unit_smoke/weights/best.pt`；
- 该模型约 6.25 MB，属于小数据 smoke 模型。

训练 CSV 中曾出现高 mAP，但验证集只有约 2 张，不能用于声称泛化精度。它只证明
训练链路工作、模型能够拟合当前少量数据。

当前配置存在路径不一致：

```text
配置指向：outputs/perception/runs/server_unit/weights/best.pt
实际存在：outputs/perception/runs/server_unit_smoke/weights/best.pt
```

因此直接按默认配置启动 detector 很可能报模型文件不存在。应先统一模型版本和路径，
不要简单复制文件掩盖数据版本关系。

### 5.4 仿真槽位数据

数据集：`outputs/perception/datasets/slot_detect_v1`

- train：240 张；
- val：60 张；
- hard：40 张；
- 总计：340 张；
- 类别 0：`slot`；
- 类别 1：`slot_grid`；
- 标签实例约为 3488 个 slot、340 个 slot_grid；
- val/hard 各保存 8 张深度图，共 16 张；
- `meta.json` 中包含生成时的相机参数和槽位 6D 真值（如文件仍在，应优先复用）。

这套数据和正式运行时目前存在契约不一致：

- 正式 detector 默认 `target_class=server_unit`；
- 仿真数据类别是 `slot` / `slot_grid`；
- 仿真数据生成器期待 MuJoCo 相机 `perception_cam`；
- `perception/README.md` 说明该相机已从当前活动 `workcell.xml` 移除。

因此已有数据可以用于离线几何算法回归测试，但重新生成数据很可能需要先恢复一个
专用视觉场景或给生成器单独提供带相机的 MJCF。不要直接把仿真数据路线误认为当前
ROS 仿真已经发布相机话题。

### 5.5 当前尚未实现

- 没有机架或槽位 6D 位姿节点；
- 没有从 bbox/关键点与深度计算三维位姿；
- 没有手眼标定；
- 相机安装方式尚未最终确认；
- 没有 `camera → world/base_link` 的可靠外参；
- 没有把视觉位姿交给 `huanpan_motion`；
- 没有定位协方差、稳定性和拒绝条件；
- 没有真实 D435 精定位误差统计；
- 没有完成视觉引导下的仿真或真机插入。

---

## 6. 精定位算法的推荐方向

### 6.1 推荐主线

推荐采用：

```text
YOLO/分割得到机架ROI
        ↓
检测机架四角或12个槽口中心/角点
        ↓
RANSAC + IPPE/PnP，求机架初始6D位姿
        ↓
D435对齐深度中拟合机架前平面
        ↓
联合优化重投影、点到平面、4×3间距约束
        ↓
多帧滤波与质量门控
        ↓
rack pose → 推导12个slot entry/goal
```

推荐理由：

- 机架正面近似平面，适合 planar PnP/IPPE；
- 4×3 网格和真实尺寸是强几何先验；
- 不必把 12 个槽位训练成 12 个类别；
- 可解释、可测试、CPU 可运行；
- 深度可以帮助校正平面法向和距离；
- 视觉负责精确接近，力控负责最后接触阶段，符合当前系统结构。

建议目标函数同时包含：

```text
二维关键点重投影误差
+ 深度点到机架前平面的误差
+ 横向105 mm/纵向27.5 mm网格间距误差
+ 多帧位姿变化惩罚
```

### 6.2 深度处理注意事项

不要只读取 bbox 中心深度。硬盘槽内部可能是空洞、黑色表面或反光金属，D435 在这些
区域容易出现 0 值、飞点或背景深度。建议：

- 使用机架前面板/边框的深度，而非槽口内部；
- 对 ROI 做中值与 MAD/分位数离群过滤；
- 使用 RANSAC 拟合平面；
- 输出深度有效率、平面内点比例和残差；
- 深度质量差时保留 PnP 结果但降低置信度，或直接拒绝发布运动目标。

### 6.3 ICP 的正确位置

点云到 CAD 的 point-to-plane ICP 可以作为 PnP 后的局部精修，但不建议独立承担全局
定位。机架前面近似平面且槽位重复，ICP 在初值差时容易进入错误局部极值。

正确顺序：

```text
PnP/几何网格提供初值 → 裁剪机架点云 → point-to-plane ICP → 位姿精修
```

### 6.4 AprilTag 基准线

如果允许暂时在机架边缘贴标记，建议先加 AprilTag 板建立工程基准：

- 验证相机内参、外参、TF方向和单位；
- 为无标记算法提供近似真值；
- 快速验证视觉位姿能否真正驱动 MoveIt；
- 最终方案可再移除标签。

AprilTag 可以作为验证工具，不一定是最终产品方案。

### 6.5 FoundationPose / SAM-6D / MegaPose

这些方法适合基于 CAD 的通用新物体 6D 位姿估计，可以作为论文对比或研究分支，
但不建议作为当前第一落地方案：

- 部署较重，FoundationPose 官方环境依赖 CUDA、PyTorch3D、NVDiffRast；
- 通常仍需要目标掩码、bbox 或候选区域；
- 规则平面和重复槽位存在对称/编号歧义；
- 当前机器是否具有合适 NVIDIA GPU 尚未在本说明中确认；
- 当前最关键问题是几何定位、标定和数据质量，而不是模型类别泛化。

参考：

- OpenCV PnP/IPPE：<https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html>
- Open3D ICP：<https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html>
- FoundationPose：<https://github.com/NVlabs/FoundationPose>
- SAM-6D：<https://arxiv.org/abs/2311.15707>
- MegaPose：<https://arxiv.org/abs/2212.06870>
- AprilTag：<https://github.com/AprilRobotics/apriltag>

---

## 7. 建议的 ROS 接口契约

### 7.1 输入

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
/detections 或内部检测器输出
TF: camera optical frame → world/base_link（完成标定后）
```

### 7.2 输出

建议第一版至少发布：

```text
/perception/rack_pose              geometry_msgs/PoseWithCovarianceStamped
/perception/slot_poses             geometry_msgs/PoseArray
/perception/slot_09_entry          geometry_msgs/PoseStamped
/perception/debug_image            sensor_msgs/Image
/perception/diagnostics            diagnostic_msgs/DiagnosticArray 或 JSON
TF: camera frame → rack
```

如果尚未得到手眼外参，只能发布 camera frame 下的机架位姿；不能把相机坐标直接写成
world 坐标。坐标变换必须明确使用：

```text
T_world_slot = T_world_camera × T_camera_rack × T_rack_slot
```

### 7.3 发布前质量门控

以下阈值可作为初始建议，必须通过真实数据重新标定：

- 至少 4 个有效非退化关键点；
- PnP 正深度，法向方向符合机架朝向先验；
- 重投影 RMSE 初始目标小于约 2 px；
- 平面内点率和深度有效率超过设定阈值；
- 连续约 10 帧位姿稳定；
- 消息时间戳不过期；
- TF 可用且时间一致；
- 超过协方差阈值时不发布可执行目标，只发布诊断。

---

## 8. 建议的实现分阶段

### 阶段 A：纯算法与离线回归

1. 新增不依赖 ROS 的相机投影、深度过滤、平面拟合、PnP 和网格排序模块；
2. 使用 `slot_detect_v1` 中已有 RGB-D/元数据做回归；
3. 与合成 6D 真值比较平移误差、旋转误差和槽位编号正确率；
4. 单独测试异常深度、缺槽位、错误关键点和部分遮挡。

阶段 A 不应直接修改 MoveIt 或插入控制。

### 阶段 B：ROS 2 节点

1. 新建 `slot_pose_node`；
2. 同步 RGB、aligned depth、CameraInfo；
3. 发布 camera frame 下的 rack/slot pose 和诊断信息；
4. 用 bag 或 fake camera 重放测试；
5. 安装或绕过 `vision_msgs`，统一检测消息契约。

### 阶段 C：真实相机与标定

1. 确定 D435 是 eye-to-hand 还是 eye-in-hand；
2. 标定相机内参与相机到机器人基座/末端的外参；
3. 静态机架采集 100 帧，评估重复性、漂移和离群率；
4. 不同距离、角度、光照与反光条件下评估；
5. 必要时用 AprilTag 作为外部基准。

### 阶段 D：与运动模块联调

1. 把视觉 `slot_09_entry` 转换到 world；
2. 只运行到入口前安全距离，先不插入；
3. 检查 RViz、MuJoCo/真机和 TF 三者一致性；
4. 再启用低速插入和 25 N 力保护；
5. 统计多次成功率、最大接触力、视觉误差和撤退次数。

---

## 9. 建议的精度指标

以下是系统工程目标，不是当前已达到的结果：

| 指标 | 初始目标 |
|---|---:|
| 槽位横向/纵向重复定位误差 | P95 ≤ 1.5 mm |
| 机架姿态误差 | P95 ≤ 0.5° |
| 槽位编号正确率 | 接近 100%，错误时必须拒绝而非猜测 |
| 连续帧有效率 | 需在真实数据上统计 |
| 视觉引导插入成功率 | 需在力控开启条件下多次统计 |

0.5° 在 175 mm 插入深度上会造成约 1.5 mm 的末端侧向偏移，因此姿态误差与位置
误差同样重要。最终阈值必须根据真实槽口间隙、夹具柔顺性和力控表现确定。

---

## 10. 接手前必须确认的问题

1. D435 最终固定在机器人底座/外部支架，还是安装在机械臂腕部？
2. 是否允许在开发阶段临时使用 AprilTag？
3. 是否有可用 NVIDIA GPU；若没有，不应把 FoundationPose 作为首个实现目标。
4. 当前机架 CAD/尺寸与真实 SA5212 空架误差是多少？
5. 真实槽位允许的横向、纵向和角度误差是多少？
6. 视觉需要定位整个机架后推导所有槽位，还是还要识别占用/空闲状态？
7. `slot_09` 的编号方向是否已经与真实机柜观察方向最终确认？
8. 是否有 D435 `.bag`、同步 RGB-D 或标定板数据可用于离线开发？

如果这些问题未确认，允许先实现 camera-frame 下的几何算法，但不要擅自写死手眼外参。

---

## 11. 建议阅读顺序

1. `README.md`
2. `docs/SA5212M5_MODEL.md`
3. `docs/AG95_MODEL_SOURCE.md`
4. `src/huanpan_perception/README.md`
5. `src/huanpan_perception/config/perception.yaml`
6. `src/huanpan_perception/huanpan_perception/detector_node.py`
7. `src/huanpan_perception/huanpan_perception/detector.py`
8. `perception/README.md`
9. `perception/slotvision/scene.py`
10. `outputs/perception/datasets/slot_detect_v1/generate_summary.json`
11. `src/huanpan_moveit_config/config/insertion_targets.yaml`
12. `src/huanpan_motion/src/insertion_demo_node.cpp`

注意：`docs/项目进程说明.md` 是较早的阶段记录，部分环境状态已经变化，只能作为背景，
不能替代现场检查。

---

## 12. 可复现检查命令

### 感知单元测试

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m pytest -q src/huanpan_perception/test
```

### 检查 ROS 依赖与相机

```bash
ros2 pkg prefix realsense2_camera
ros2 pkg prefix vision_msgs
ls -l /dev/video*
```

### 构建项目包，避免重编全部 MoveIt 2

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon build --packages-select \
  huanpan_description huanpan_mujoco huanpan_moveit_config \
  huanpan_motion huanpan_perception \
  --symlink-install --parallel-workers 1
```

### 当前自动插入测试入口

终端 A：

```bash
cd /home/lzt/huanpan_ws
source scripts/ros_env.sh
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  show_mujoco:=true use_rviz:=true \
  disturbance_enabled:=true disturbance_scale:=0.2
```

终端 B：

```bash
cd /home/lzt/huanpan_ws
source scripts/ros_env.sh
ros2 launch huanpan_motion insertion_demo.launch.py mode:=nominal
```

这仍然使用硬编码目标，只用于证明运动/力控侧基线，不是视觉精定位测试。

---

## 13. 给接手 AI 的工作原则

- 先读文件和验证当前环境，再修改代码；
- 尽量在现有 `huanpan_perception` 上做小范围增量，不另起一套重复框架；
- 纯算法应与 ROS 节点解耦，便于用合成数据和 bag 做确定性测试；
- 不要把 2D bbox、一次 smoke test、高 mAP 或仿真通过描述为真实精定位完成；
- 不要用提高力阈值掩盖视觉偏差；
- 不要在无标定时把 camera frame 数据伪装成 world frame；
- 不要只给出算法名，应明确输入、输出、依赖、失败条件、测试数据和验收指标；
- 修改后列出文件、测试结果，以及尚未验证的部分。

---

## 14. 一句话状态总结

项目的 UR5e/MoveIt 2/MuJoCo 插入与力保护仿真基线已经建立，视觉侧已完成 D435 数据
采集和 YOLO 二维检测工具链，但目前没有手眼标定、深度几何、机架/槽位 6D 位姿或
视觉到运动的接口；下一阶段应优先实现“关键点/IPPE-PnP + 深度平面 + 4×3 网格约束”
的可验证精定位链路。
