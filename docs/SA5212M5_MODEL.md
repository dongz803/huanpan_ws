# SA5212M5 仿真模型依据

本项目的硬盘架采用用户提供的实测尺寸，浪潮英信 SA5212M5 仅用于参考
12 盘位正面布局。它不是浪潮原厂 CAD，也不应用于制造、装配公差或真实机器人安全认证。

## 官方依据

- 浪潮英信服务器 SA5212M5 用户手册 V2.5：
  <https://www.inspur.com/eportal/fileDir/active_download/userBook/SA5212M5/%E6%B5%AA%E6%BD%AE%E8%8B%B1%E4%BF%A1%E6%9C%8D%E5%8A%A1%E5%99%A8%20SA5212M5%20%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C%20V2.5.pdf>
- 前面板：12 个 3.5/2.5 英寸 SAS/SATA/NVMe 热插拔盘位。
- 盘位编号：上排 0-3，中排 4-7，下排 8-11。

官方整机外形尺寸不再作为当前仿真机架包络；当前模型以实测工装为准。

## 仿真假设

- 机架包络为 430 x 85 x 175 mm（宽 x 高 x 深）。
- 盘位布局为横向 4 个、纵向 3 个；编号仍按上排 0-3、中排 4-7、下排 8-11。
- 任务盘包络为 175 x 100 x 25 mm（深 x 宽 x 高）。
- 12 个盘位均为空；任务盘不再位于机架内，也没有 `drive_slide_joint`。
- 任务盘通过固定关节挂在 `gripper_tcp` 后，形成 `carried_drive` link；其前端为
  MoveIt 规划末端 `drive_tip`。
- 机架前面朝向世界坐标 -x，盘位宽度沿 y、高度沿 z、插入方向为机架局部 +x。
- slot 09 的入口和完全插入目标分别是 `slot_09_entry` 与 `slot_09_goal`。
- 盘位边框使用概念阶段间隙，只服务于首次插入仿真，不代表真实导轨公差。
- 为避免机箱在场景中悬空，仿真额外加入两根落地立柱和底部横梁；它们只是工作站
  安装支架，不属于 SA5212M5 官方外形。

## 关键文件

- `src/huanpan_description/mujoco/universal_robots_ur5e/sa5212_empty_rack.xml`
- `src/huanpan_description/mujoco/universal_robots_ur5e/workcell.xml`
- `src/huanpan_description/urdf/huanpan_workcell.urdf`
- `src/huanpan_moveit_config/config/insertion_targets.yaml`
- `src/huanpan_mujoco/test/test_model_contract.py`
