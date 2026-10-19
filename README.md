# huanpan_ws

UR5e 换硬盘机器人的 ROS 2 Humble + MoveIt 2 + MuJoCo 仿真与 D435 视觉定位工作区。

## 当前进度（更新于 2026-10-19）

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 机器人与场景 | 已完成仿真模型 | UR5e、AG95、固定末端硬盘和 4 x 3 空架已接入 MuJoCo/URDF |
| 规划与控制 | 已打通仿真链路 | MoveIt 2 可通过 `FollowJointTrajectory` 驱动 MuJoCo |
| 插入保护 | 已完成仿真原型 | 支持扰动、接触力监测、超限中止和沿历史轨迹回退 |
| YOLO 粗定位 | 已运行实测 | D435 单槽口黄色框检测，支持 CUDA 推理和低延迟调试图 |
| 相机正视反馈 | 当前默认阶段 | 面板深度平面负责 pitch/yaw，框内长边负责 roll，只输出旋转修正 |
| 严格精定位 | 代码已接通，待验收 | 四边精修、四角求交、IPPE/PnP、深度校验和示教比较仍需提升稳定性 |
| 实机闭环 | 尚未完成 | 还需要腕部相机固定、手眼标定、TF 转换和真实 UR5e 小步闭环 |

当前主线先验证“粗定位后把相机调整到近似正视”。配置中
`front_view_only: true`，因此默认只显示黄色 YOLO 框和姿态反馈，不显示绿色严格
精定位框，也不会控制机械臂。已完成、实验中和未完成内容在下文分别说明。

当前已完成：

- UR5e 6 轴机械臂；
- DH Robotics AG95 夹具的完整 14-link 外观与 13-joint 运动链；
- AG95 以 `R1_1_joint` 为唯一控制关节，其余 12 个关节按原 URDF mimic 比例联动；
- 参考浪潮 SA5212M5 正面布局、按实测尺寸建立的 4 列 x 3 行空架；
- 一块 175 x 100 x 25 mm 任务硬盘，默认刚性连接到夹具末端；
- `drive_tip`、`slot_09_entry` 和 `slot_09_goal` 插入参考坐标系；
- `/joint_states` 状态发布；
- MoveIt 兼容的 `FollowJointTrajectory` 机械臂和夹具 Action；
- MoveIt 2 SRDF、KDL、OMPL、关节限制和控制器配置；
- D435 RGB、对齐深度和 CameraInfo 同步采集；
- YOLOv8 单槽口粗定位、相机正视姿态反馈和严格精定位原型。

## 架构

```text
MoveIt 2 / RViz
      |
      | FollowJointTrajectory
      v
huanpan_mujoco/mujoco_bridge
      |-- /ur5e_arm_controller/follow_joint_trajectory
      |-- /gripper_controller/follow_joint_trajectory
      `-- /joint_states
                 |
                 v
       MuJoCo UR5e + 夹具 + 硬盘机柜
```

MoveIt 2 只负责路径规划和轨迹生成。`mujoco_bridge` 暂时充当仿真控制器；后续改成 `ros2_control` 时，MoveIt 侧的 Action 接口可以保持不变。当前 `ur_manipulator` 的规划末端是硬盘前端 `drive_tip`，而不是夹具中心。

视觉链路与运动链路目前保持解耦：

```text
D435 RGB-D
   -> YOLO 黄色槽口框
   -> 框外面板深度 RANSAC + 框内长边
   -> 相机 roll/pitch/yaw 修正
   -> 正视确认
   -> 四边/四角 + IPPE/PnP（实验阶段）
   -> 相机坐标系 6D 位姿与示教误差
```

## ROS 2 包

```text
src/
├── moveit2/               # MoveIt 2 Humble 源码
├── moveit_msgs/           # MoveIt 消息源码
├── moveit_resources/      # MoveIt 测试资源源码
├── huanpan_description/   # URDF、MJCF、夹具、机柜和 RViz 描述
├── huanpan_mujoco/        # MuJoCo 仿真与 ROS 2 轨迹桥
├── huanpan_motion/        # MoveIt 之上的运动规划与执行封装
├── huanpan_perception/    # D435、YOLO、正视反馈与槽口精定位
└── huanpan_moveit_config/ # MoveIt 2 规划和执行配置
```

## 依赖

本机使用 ROS 2 Humble。开发环境中的 MoveIt 2 使用 `humble` 分支源码，位于
`/home/lzt/huanpan_ws/src`，不依赖 `ros-humble-moveit` 二进制元包。为避免把大型
第三方仓库重复上传，`src/moveit2`、`src/moveit_msgs` 和 `src/moveit_resources`
不会提交到本仓库；新环境需要自行安装 MoveIt 2 或检出对应 Humble 源码。

本项目没有使用 xacro，URDF 是静态文件。在 MoveIt 安装前，项目内的 `.deps` 提供了一份本地 `control_msgs` 开发回退，足以启动和测试 MuJoCo 轨迹桥。

纯 MuJoCo 工具使用项目 `.venv`：

```bash
cd /home/lzt/huanpan_ws
python3 -m venv .venv
source .venv/bin/activate
env -u PYTHONPATH python -m pip install -r requirements.txt
```

感知识别使用单独的 `.venv-perception`，避免污染已验证的 `.venv`。下面是 CPU
环境的最小安装方式；RTX 3060 实测使用的是与本机 CUDA 环境匹配的 PyTorch：

```bash
cd /home/lzt/huanpan_ws
python3 -m venv .venv-perception
.venv-perception/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
.venv-perception/bin/pip install -r perception/requirements-perception.txt
```

真实 D435 还需要系统包（需要你自己执行 sudo）：

```bash
sudo apt install ros-humble-realsense2-camera ros-humble-vision-msgs
```

## 构建

建议使用未 `source` 其他 ROS 工作区的新终端：

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 1 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

构建后的常用环境可以通过一条命令加载：

```bash
source scripts/ros_env.sh
```

## 只启动 ROS 2 + MuJoCo

终端 A：

```bash
source scripts/ros_env.sh
ros2 launch huanpan_mujoco simulation.launch.py show_viewer:=true
```

终端 B 执行一次小幅机械臂运动和 AG95 单主关节闭合测试：

```bash
source scripts/ros_env.sh
ros2 run huanpan_mujoco smoke_trajectory
```

无图形界面运行时：

```bash
ros2 launch huanpan_mujoco simulation.launch.py show_viewer:=false
```

## 启动 MoveIt 2 + MuJoCo

完成源码构建后：

```bash
source scripts/ros_env.sh
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py
```

启动项：

- `show_mujoco:=true|false`：是否显示 MuJoCo 交互窗口；
- `use_rviz:=true|false`：是否显示 MoveIt RViz。

在 RViz 的 MotionPlanning 面板中选择 `ur_manipulator`，可设置目标位姿并执行。轨迹会发到 MuJoCo 中的 UR5e。夹具规划组名为 `gripper`，具有 `open` 和 `closed` 命名状态。

当前以 slot 09 作为第一条插入路线：入口坐标系为 `slot_09_entry`，完全插入目标为 `slot_09_goal`，机架局部 `+x` 是插入方向。目标名称、间隙和硬盘朝向见 `src/huanpan_moveit_config/config/insertion_targets.yaml`。

## 插入力保护与自动回退

MuJoCo 桥会把末端固定硬盘 `carried_drive` 的接触合力发布到：

```text
/mujoco/drive_wrench  (geometry_msgs/msg/WrenchStamped, world 坐标系)
```

插入力保护默认关闭。做插入实验时可随 MoveIt 一起启用：

```bash
source scripts/ros_env.sh
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  force_guard_enabled:=true force_guard_threshold_n:=25.0
```

也可以只在到达槽位入口后动态开启，退出插入阶段后关闭：

```bash
ros2 service call /mujoco/force_guard/enable std_srvs/srv/SetBool \
  "{data: true}"
ros2 topic echo /mujoco/drive_wrench
ros2 service call /mujoco/force_guard/enable std_srvs/srv/SetBool \
  "{data: false}"
```

保护开启且机械臂轨迹正在执行时，接触力先经过低通滤波；连续超过阈值
`0.03 s` 后，中止当前 `FollowJointTrajectory`，并沿已执行的关节目标历史回退
`0.25 s`。对应参数集中在 `src/huanpan_mujoco/config/bridge.yaml`。

MuJoCo 机械臂默认开启重力补偿，用来抵消 UR5e、AG95 和固定硬盘负载造成的
静态跟踪偏差；配置的随机扰动力矩会在重力补偿之后叠加。这样 nominal 插入测试
检验的是扰动与槽位误差，而不是由未补偿重力导致的必然下垂。

`25 N` 只是仿真初始值，不是实机硬盘、背板或连接器的已验证安全上限；接入实机前
必须用末端六维力传感器重新标定阈值、滤波、触发时间和回退距离。本阶段实现的是
“过力保护 + 路径回退”，不是恒力跟踪或导纳控制。

无需在 RViz 手动拖动即可运行完整插入测试。先以已验证的非零扰动基线启动场景：

```bash
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  show_mujoco:=true use_rviz:=true \
  disturbance_enabled:=true disturbance_scale:=0.2
```

另开终端自动执行“到入口、再对准、开启力保护、直线插入”：

```bash
source scripts/ros_env.sh
ros2 launch huanpan_motion insertion_demo.launch.py mode:=nominal
```

测试到达入口后会自动重置固定种子的扰动序列，因此不同 OMPL 规划耗时不会改变
插入阶段的扰动相位。

验证超力中止和自动回退时使用仿真专用模式：

```bash
ros2 launch huanpan_motion insertion_demo.launch.py mode:=overforce
```

`overforce` 会故意加入横向偏差，只能用于 MuJoCo，不得用于实机。扰动倍率 `1.0`
在当前毫米级槽位间隙下会触发力保护，属于压力测试配置。

## 感知识别模块

`src/huanpan_perception` 使用 Intel RealSense D435 和单类别 `slot_opening` YOLOv8
模型检测一个目标槽口。当前默认运行正视姿态反馈；严格精定位代码已能够输出相机
坐标系下的 6D 位姿，但在偏角、距离和反光变化下仍需继续验证，尚未接入机械臂闭环。

```text
D435 RGB-D + CameraInfo
  -> YOLO 黄色粗框
  -> 框外面板深度点 RANSAC 平面法向（pitch/yaw）
  -> 框内 Canny/Hough 长边方向（roll）
  -> /huanpan/vision/front_view_correction
  -> 连续 5 帧满足阈值后 FRONT_ALIGNED

后续严格精定位：
  -> Sobel 多边线候选 + Huber 直线拟合
  -> 四线求交得到四角
  -> 已知槽口尺寸 100 x 25 mm 的 IPPE/PnP
  -> 外侧面板深度交叉检查 + 时序滤波 + 示教位姿比较
```

```bash
# 感知环境（ROS 2 + 本工作区 + .venv-perception）
source scripts/perception_env.sh

# 启动 D435 + 正视反馈/精定位节点
./scripts/fine_localization_quick.sh

# 查看画面、姿态修正和状态
ros2 run rqt_image_view rqt_image_view /huanpan/vision/debug_image
ros2 topic echo /huanpan/vision/front_view_correction
ros2 topic echo /huanpan/vision/front_view_aligned
ros2 topic echo /huanpan/vision/status
```

调试图中黄色 `YOLO` 是粗框，绿色 `FINE` 是严格四边精修结果。当前默认
`front_view_only: true`，所以只出现黄色框。若要实验性启用严格精定位，可在
`src/huanpan_perception/config/perception.yaml` 中改为 `false`，重新构建后启动。

采集与训练链路：

```bash
# SPACE 或 s 保存同步的 RGB、对齐深度、CameraInfo 和帧元数据
ros2 launch huanpan_perception capture.launch.py \
  output_dir:=$PWD/outputs/perception/fine_test \
  prefix:=fineA start_camera:=true save_depth:=true preview:=true

# 标注 → 训练 → 评估 → 运行
ros2 run huanpan_perception label_dataset \
  --images outputs/perception/datasets/slot_opening_v1/images
ros2 run huanpan_perception train_detector \
  --dataset-root outputs/perception/datasets/slot_opening_v1 --epochs 60 --imgsz 640
ros2 run huanpan_perception evaluate_detector \
  --model outputs/perception/runs/slot_opening_v1/weights/best.pt \
  --dataset-root outputs/perception/datasets/slot_opening_v1
ros2 launch huanpan_perception perception.launch.py start_camera:=true
```

数据集、训练权重、采集图片和运行输出统一放在 `outputs/`，已被 `.gitignore`
排除，不会上传到公开仓库。克隆仓库后需要自行训练模型，或把 `best.pt` 放到配置中
指定的位置。

`sudo apt install ros-humble-realsense2-camera ros-humble-vision-msgs` 需要手动执行。
详细说明、话题接口和已知边界见 `src/huanpan_perception/README.md`。

### 当前视觉阶段的输出

| ROS 2 接口 | 类型 | 含义 |
|---|---|---|
| `/huanpan/vision/front_view_correction` | `geometry_msgs/TwistStamped` | 相机光学坐标系下的旋转修正，角量单位 rad，线速度为 0 |
| `/huanpan/vision/front_view_aligned` | `std_msgs/Bool` | 正视条件是否已连续稳定满足 |
| `/huanpan/vision/slot_pose` | `geometry_msgs/PoseStamped` | 严格精定位得到的相机系槽口 6D 位姿 |
| `/huanpan/vision/correction` | `geometry_msgs/TwistStamped` | 当前位姿相对示教参考的 XYZ 和旋转误差 |
| `/huanpan/vision/aligned` | `std_msgs/Bool` | 是否达到 3 mm / 1 deg 示教误差阈值 |
| `/huanpan/vision/status` | `std_msgs/String` | 检测、几何质量、深度一致性和耗时 JSON |
| `/huanpan/vision/debug_image` | `sensor_msgs/Image` | 黄色粗框/绿色精修框及状态可视化 |

这些量目前都在相机坐标系中。接入 UR5e 前必须完成 `tool0 -> camera_link` 手眼
标定，并通过 TF 把修正量转换到 `base_link`，同时增加单步位移、角度、速度和失效停止限制。

## 验证

```bash
# MJCF/URDF 关节名称与执行器契约
source .venv/bin/activate
env -u PYTHONPATH python -m pytest -q src/huanpan_mujoco/test

# ROS 2 测试
source scripts/ros_env.sh
colcon test --packages-select huanpan_mujoco
colcon test-result --verbose

# 感知包单元测试（不需要 torch 和相机）
cd src/huanpan_perception
env -u PYTHONPATH ../../.venv-perception/bin/python -m pytest test -q
cd /home/lzt/huanpan_ws

# 无相机时验证整条 ROS 感知链路：假相机 + 检测节点
source scripts/perception_env.sh
ros2 run huanpan_perception fake_camera_node --ros-args \
  -p source:=outputs/perception/datasets/rack_grid_v1/images \
  -p image_topic:=/camera/camera/color/image_raw &
ros2 run huanpan_perception detector_node --ros-args \
  -p model_path:=outputs/perception/weights/yolov8n.pt \
  -p output_mode:=json -p log_every_n_frames:=10

# 离屏渲染工作站
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/render_scene.py \
  --model src/huanpan_description/mujoco/universal_robots_ur5e/workcell.xml \
  --output outputs/ag95_home.png \
  --azimuth -45 --elevation -18 --distance 2.6 --lookat 0.35 0 0.65

# 查看 slot 09 入口/插入完成两个运动学参考姿态
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/render_scene.py \
  --model src/huanpan_description/mujoco/universal_robots_ur5e/workcell.xml \
  --keyframe slot_09_entry --output outputs/ag95_slot_09_entry.png
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python scripts/render_scene.py \
  --model src/huanpan_description/mujoco/universal_robots_ur5e/workcell.xml \
  --keyframe slot_09_inserted --output outputs/ag95_slot_09_inserted.png
```

## 重要限制

- 正视反馈已能输出相机姿态修正，但它是手持视觉验证结果，尚未驱动真实机械臂；
- 严格精定位能够输出相机系 6D 位姿，但对视距、偏角、反光和角点质量仍敏感，
  还没有完成 3 mm / 1 deg 重复到位验收；
- 尚未完成手眼标定和 `camera_color_optical_frame -> base_link` 的 TF 转换；
- 机架采用用户提供的实测包络 430 x 85 x 175 mm（宽 x 高 x 深），
  SA5212M5 只作为 4 x 3 前面板布局参考，不是原厂 CAD；
- 任务硬盘为固定末端 link，本阶段没有抓取、释放或 attach/detach 状态切换；
- AG95 的 STL 在 MuJoCo 中用于高保真显示，接触仍使用简化包围体；
- `slot_09_entry` 和 `slot_09_inserted` 只证明端点运动学可达且初始无碰撞，
  尚未证明 MoveIt 笛卡尔直线插入、接触稳定性或成功率；
- 轨迹桥是简化位置控制；已有仿真接触力超限回退，但未实现真实
  `ros2_control`、完整路径容差或经过实机标定的力控安全策略；
- 这一版只可用于仿真，不能直接控制真实 UR5e。

UR5e MJCF 来源于 MuJoCo Menagerie，已保留原始 `README.md` 和 BSD-3-Clause `LICENSE`。
SA5212M5 建模依据和假设见 `docs/SA5212M5_MODEL.md`。
AG95 的本地参考路径、联动关节与许可边界见 `docs/AG95_MODEL_SOURCE.md`。
