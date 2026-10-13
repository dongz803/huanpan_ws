# DH Robotics AG95 模型来源与适配说明

## 本地参考

- URDF：`/home/lzt/JAKA_TRACER-main/src/dh_ag_ros2/src/dh_ag95_ros2/dh_ag_95/urdf/dh_ag_95.urdf`
- 网格：`/home/lzt/JAKA_TRACER-main/src/dh_ag_ros2/src/dh_ag95_ros2/dh_ag_95/meshes/`
- UR5e 参考资产：`/home/lzt/lzt_ros2_ws/src/lzt_mujoco_ros/assets/`

已将 AG95 的 14 个 visual STL 和 14 个 collision STL 复制到
`huanpan_description/meshes/ag95/`。原 UR5e Menagerie 资产与参考工作区中的
UR5e 文件哈希一致，因此保留当前 UR5e，只替换末端夹具。

## 机构适配

- 保留原始 14 links 和 13 joints 的树状机构、关节原点、轴向、质量与惯量。
- `R1_1_joint` 是唯一控制关节：张开为 `0 rad`，闭合为 `-0.94 rad`。
- 其余 12 个关节在 URDF 中使用 `<mimic>`，在 MuJoCo 中使用
  12 条 joint equality constraints 表达相同的倍率。
- 参考 URDF 中 `L2_1_link` 的质量为零，这对 MuJoCo 动力学体无效；
  MuJoCo 版使用其镜像件 `R2_1_link` 的质量和镜像惯量。
- 详细 STL 仅用于 MuJoCo 显示，夹具底座和两个指端使用简化包围盒碰撞体，
  避免复杂三角网格导致接触不稳定。

## 换盘任务约定

- 175 x 100 x 25 mm 硬盘仍通过固定关节连接到 `gripper_tcp`。
- 硬盘局部 `+z` 是插入方向，`drive_tip` 位于硬盘前端面中心。
- slot 09 入口和完全插入关键帧已根据新 TCP 长度重新求解，
  端点和分段直线运动学路径均无 MuJoCo 碰撞。

## 许可边界

参考包 `dh_ag_95/package.xml` 的 license 字段仍是
`TODO: License declaration`。因此这些 AG95 网格目前只按用户提供的本地资产
用于本机内部仿真，在获得清晰许可之前不应对外分发或声称其可重新授权。
