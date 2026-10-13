# huanpan_motion

`huanpan_motion` 是换硬盘机器人在 MoveIt 2 之上的应用层封装。它不复制或修改
`src/moveit2`，而是把运动学、碰撞检测、运动规划、轨迹处理和控制执行拆成清晰模块。

## 设计边界

```text
目标位姿
  -> kinematics
  -> collision_checking
  -> motion_planning
  -> trajectory_processing
  -> control
  -> FollowJointTrajectory
  -> huanpan_mujoco 或后续真机驱动
```

MoveIt 2 负责通用算法，本包负责换盘项目的稳定接口。UR5e、AG95、末端固定硬盘和
4 x 3 硬盘架模型继续由 `huanpan_description` 管理；KDL、OMPL、SRDF 和控制器名称继续
由 `huanpan_moveit_config` 管理；MuJoCo 执行后端继续由 `huanpan_mujoco` 管理。

## 模块说明

### kinematics

- `include/huanpan_motion/kinematics/kinematics_solver.hpp`
- `src/kinematics/kinematics_solver.cpp`

封装 FK、IK、Jacobian 和关节限位检查。后端使用当前 MoveIt 配置中的
`kdl_kinematics_plugin/KDLKinematicsPlugin`。

### collision_checking

- `include/huanpan_motion/collision_checking/collision_checker.hpp`
- `src/collision_checking/collision_checker.cpp`

封装单状态碰撞、整条轨迹采样碰撞和最小碰撞距离。碰撞场景由 MoveIt
`PlanningScene` 提供，硬盘作为末端 link 参与碰撞检查。

### motion_planning

- `include/huanpan_motion/motion_planning/motion_planner.hpp`
- `src/motion_planning/motion_planner.cpp`

封装关节目标、末端位姿目标、OMPL 规划和笛卡尔路径规划。默认规划组是
`ur_manipulator`，规划器由 `huanpan_moveit_config/config/ompl_planning.yaml` 决定。

### trajectory_processing

- `include/huanpan_motion/trajectory_processing/trajectory_processor.hpp`
- `src/trajectory_processing/trajectory_processor.cpp`

将几何路径转换成包含时间、速度和加速度约束的轨迹。当前使用 MoveIt 的
Time-Optimal Trajectory Generation，并检查缩放范围及时间戳单调性。

### control

- `include/huanpan_motion/control/trajectory_executor.hpp`
- `src/control/trajectory_executor.cpp`

只负责通过 `FollowJointTrajectory` 发送、反馈和取消轨迹，不负责 IK、碰撞检测或规划。
默认目标是 MuJoCo Bridge 提供的
`/ur5e_arm_controller/follow_joint_trajectory`；切换真机时保持相同接口。

## 构建

为避免重新构建整套 MoveIt 2，只构建本包：

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
source /home/lzt/huanpan_ws/install/setup.bash
colcon build --packages-select huanpan_motion --symlink-install --executor sequential
```

`--executor sequential` 限制并行编译内存占用。本包只链接工作空间中已安装的
MoveIt 2.5.10，不会触发 MoveIt 2 全量重编译。

## 测试

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
source /home/lzt/huanpan_ws/install/setup.bash
colcon test --packages-select huanpan_motion
colcon test-result --verbose
```

## 启动关系

先启动现有 MoveIt + MuJoCo 系统。下面的命令会同时打开 MuJoCo 和 RViz：

```bash
cd /home/lzt/huanpan_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  show_mujoco:=true use_rviz:=true
```

RViz 默认规划组是 `ur_manipulator`。RViz 使用 UR5e 的 OBJ 外观网格，MuJoCo
使用同一套 UR5e 连杆坐标系和 AG95 STL 网格；默认固定硬盘是 `carried_drive`
末端 link。此启动方式只加载场景和控制接口，不会自动让机械臂运动。

`carried_drive` 当前用 `25 x 100 x 175 mm` 的长方体代替真实硬盘，外观几何在
MuJoCo 的 group 2，碰撞几何在 group 3。若 MuJoCo 中出现大量红、绿、蓝坐标轴，
在左侧 `Rendering` 区域把 `Frame` 设置成 `None`；Bridge 启动时也会默认设置为
`None`。

在 RViz 中拖动末端交互标记后，先观察目标机器人是否出现红色碰撞高亮，再点击
`Plan`。任意拖动到机械臂不可达、硬盘穿入机架或自碰撞的位置仍会正常规划失败。
当前配置已允许 UR5e 固定安装面和 AG95 闭链内部的固有接触，正常初始状态不再被
误判为碰撞。

## 机械臂扰动

MuJoCo Bridge 支持六个 UR5e 关节上的有色随机力矩扰动。扰动使用固定随机种子，
既能模拟结构振动、线缆拖拽等低频外力，又能保证相同配置可重复。无扰动基线默认
保留；开启一级扰动：

```bash
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  show_mujoco:=true use_rviz:=true \
  disturbance_enabled:=true disturbance_scale:=0.2
```

建议先使用 `disturbance_scale:=0.5` 或 `1.0`。更大的值会同时放大标准差和限幅，
可用于压力测试，但不应直接当成真实 UR5e 的标定值。查看当前施加到六个关节的
力矩（单位 N·m）：

```bash
ros2 topic echo /mujoco/disturbance
```

扰动参数和固定随机种子位于 `huanpan_mujoco/config/bridge.yaml`。进行对比实验时，
保持轨迹、目标、随机种子相同，只修改 `disturbance_enabled` 或
`disturbance_scale`。

再启动运动封装节点：

```bash
ros2 launch huanpan_motion motion.launch.py
```

当前 `motion_node` 建立 MoveGroup 和轨迹控制客户端，五个模块作为 C++ 库提供给后续
插入任务状态机。启动文件会把 URDF、SRDF 和 KDL 配置传入该节点，但它不会在启动后
自动驱动机械臂，避免产生未审核运动。

## 自动插入测试

先启动 MoveIt、MuJoCo、扰动模型和可视化：

```bash
ros2 launch huanpan_moveit_config moveit_mujoco.launch.py \
  show_mujoco:=true use_rviz:=true \
  disturbance_enabled:=true disturbance_scale:=1.0
```

另开终端执行正常插入测试，不需要拖动 RViz 交互标记：

```bash
source scripts/ros_env.sh
ros2 launch huanpan_motion insertion_demo.launch.py mode:=nominal
```

该测试会自动规划到 `slot_09_entry`，开启力保护，生成沿机架 `+x` 的笛卡尔路径，
完成时间参数化后插入到 `slot_09_goal`。MuJoCo 中配置的关节扰动会在整个执行阶段
继续生效。到达入口后，测试会调用 `/mujoco/disturbance/reset`，让每次插入从相同
的固定种子扰动序列开始，避免 OMPL 规划耗时变化导致扰动相位漂移。

`disturbance_scale:=0.2` 是当前已经通过 nominal 插入验证的非零扰动基线；
`1.0` 在当前毫米级槽位间隙下会触发力保护，应作为压力测试而不是正常插入配置。

仿真专用的超力回退测试：

```bash
ros2 launch huanpan_motion insertion_demo.launch.py mode:=overforce
```

`overforce` 会故意加入 `4 mm` 横向偏差并允许这条测试路径进入接触，用于确认
`接触力超过阈值 -> FollowJointTrajectory 中止 -> 沿历史轨迹回退`。该模式只用于
MuJoCo 验证，不得直接切换到实机控制器。

## 后续扩展

下一阶段可以在独立任务包中实现：

```text
IDLE -> PRECHECK -> APPROACH_SLOT -> ALIGN -> INSERT -> VERIFY -> DONE/FAILED
```

自由空间阶段调用 `motion_planning`，精密插入阶段调用后续的笛卡尔伺服或导纳控制，
两者都通过 `control` 输出到 MuJoCo 或真机后端。
