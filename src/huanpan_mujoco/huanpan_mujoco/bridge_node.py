"""Expose the MuJoCo workcell as standard ROS 2 trajectory controllers."""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import WrenchStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from huanpan_mujoco.disturbance import (
    OrnsteinUhlenbeckTorqueDisturbance,
)
from huanpan_mujoco.force_guard import (
    FilteredForceGuard,
    append_history_sample,
    body_contact_wrench_world,
    select_retreat_target,
)


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINTS = ("R1_1_joint",)
PUBLISHED_JOINTS = ARM_JOINTS + GRIPPER_JOINTS


def duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1.0e-9


class MujocoBridge(Node):
    def __init__(self) -> None:
        super().__init__("mujoco_bridge")
        self.declare_parameter("model_path", "")
        self.declare_parameter("show_viewer", True)
        self.declare_parameter("simulation_rate", 100.0)
        self.declare_parameter("joint_state_rate", 50.0)
        self.declare_parameter("gravity_compensation_enabled", True)
        self.declare_parameter("disturbance_enabled", False)
        self.declare_parameter("disturbance_seed", 7)
        self.declare_parameter("disturbance_scale", 1.0)
        self.declare_parameter("disturbance_correlation_time_s", 0.25)
        self.declare_parameter(
            "disturbance_torque_stddev_nm",
            [0.8, 0.8, 0.5, 0.12, 0.12, 0.08],
        )
        self.declare_parameter(
            "disturbance_torque_limit_nm",
            [2.4, 2.4, 1.5, 0.36, 0.36, 0.24],
        )
        self.declare_parameter("force_guard_enabled", False)
        self.declare_parameter("force_guard_threshold_n", 25.0)
        self.declare_parameter("force_guard_release_n", 15.0)
        self.declare_parameter("force_guard_trip_duration_s", 0.03)
        self.declare_parameter("force_filter_time_constant_s", 0.02)
        self.declare_parameter("force_guard_retreat_lookback_s", 0.25)
        self.declare_parameter("force_guard_retreat_duration_s", 0.50)

        model_path = self.get_parameter("model_path").value
        if not model_path:
            description_share = Path(
                get_package_share_directory("huanpan_description")
            )
            model_path = str(
                description_share
                / "mujoco"
                / "universal_robots_ur5e"
                / "workcell.xml"
            )

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home_id < 0:
            raise RuntimeError("MuJoCo model is missing the home keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        mujoco.mj_forward(self.model, self.data)

        self._lock = threading.RLock()
        self._joint_qpos = {
            name: int(
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
            )
            for name in PUBLISHED_JOINTS
        }
        self._joint_qvel = {
            name: int(
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
            )
            for name in PUBLISHED_JOINTS
        }
        self._arm_dof = np.array(
            [self._joint_qvel[name] for name in ARM_JOINTS], dtype=int
        )
        actuator_names = {
            "shoulder_pan_joint": "shoulder_pan",
            "shoulder_lift_joint": "shoulder_lift",
            "elbow_joint": "elbow",
            "wrist_1_joint": "wrist_1",
            "wrist_2_joint": "wrist_2",
            "wrist_3_joint": "wrist_3",
            "R1_1_joint": "ag95_gripper",
        }
        self._joint_actuator = {
            joint_name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            for joint_name, actuator_name in actuator_names.items()
        }
        if min(self._joint_qpos.values()) < 0 or min(
            self._joint_actuator.values()
        ) < 0:
            raise RuntimeError(
                "MuJoCo model and ROS joint contract do not match"
            )

        self._joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 20
        )
        self._disturbance_pub = self.create_publisher(
            JointState, "/mujoco/disturbance", 20
        )
        self._drive_wrench_pub = self.create_publisher(
            WrenchStamped, "/mujoco/drive_wrench", 20
        )

        disturbance_scale = float(
            self.get_parameter("disturbance_scale").value
        )
        if disturbance_scale < 0.0:
            raise ValueError("disturbance_scale must be non-negative")
        standard_deviation = np.asarray(
            self.get_parameter("disturbance_torque_stddev_nm").value,
            dtype=float,
        )
        torque_limit = np.asarray(
            self.get_parameter("disturbance_torque_limit_nm").value,
            dtype=float,
        )
        if standard_deviation.shape != (len(ARM_JOINTS),):
            raise ValueError(
                "disturbance_torque_stddev_nm must contain six values"
            )
        if torque_limit.shape != (len(ARM_JOINTS),):
            raise ValueError(
                "disturbance_torque_limit_nm must contain six values"
            )
        self._disturbance = None
        self._disturbance_torque = np.zeros(len(ARM_JOINTS))
        self._gravity_compensation_enabled = bool(
            self.get_parameter("gravity_compensation_enabled").value
        )
        if self._gravity_compensation_enabled:
            self.get_logger().info(
                "Arm gravity compensation enabled"
            )
        if bool(self.get_parameter("disturbance_enabled").value):
            self._disturbance = OrnsteinUhlenbeckTorqueDisturbance(
                standard_deviation_nm=standard_deviation * disturbance_scale,
                limit_nm=torque_limit * disturbance_scale,
                correlation_time_s=float(
                    self.get_parameter(
                        "disturbance_correlation_time_s"
                    ).value
                ),
                seed=int(self.get_parameter("disturbance_seed").value),
            )
            self.get_logger().info(
                "Joint torque disturbance enabled: "
                f"scale={disturbance_scale:.3f}, "
                f"seed={self.get_parameter('disturbance_seed').value}"
            )

        carried_drive_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "carried_drive"
        )
        if carried_drive_id < 0:
            raise RuntimeError(
                "MuJoCo model is missing the carried_drive body"
            )
        retreat_lookback_s = float(
            self.get_parameter("force_guard_retreat_lookback_s").value
        )
        retreat_duration_s = float(
            self.get_parameter("force_guard_retreat_duration_s").value
        )
        if retreat_lookback_s < 0.0 or retreat_duration_s <= 0.0:
            raise ValueError(
                "force-guard retreat lookback must be non-negative and "
                "duration must be positive"
            )
        self._carried_drive_id = carried_drive_id
        self._contact_wrench = np.zeros(6, dtype=float)
        self._force_guard = FilteredForceGuard(
            threshold_n=float(
                self.get_parameter("force_guard_threshold_n").value
            ),
            release_n=float(
                self.get_parameter("force_guard_release_n").value
            ),
            trip_duration_s=float(
                self.get_parameter("force_guard_trip_duration_s").value
            ),
            filter_time_constant_s=float(
                self.get_parameter("force_filter_time_constant_s").value
            ),
        )
        self._force_guard_enabled = bool(
            self.get_parameter("force_guard_enabled").value
        )
        self._force_guard_retreat_lookback_s = retreat_lookback_s
        self._force_guard_retreat_duration_s = retreat_duration_s
        self._arm_trajectory_active = False
        self._force_guard_service = self.create_service(
            SetBool,
            "/mujoco/force_guard/enable",
            self._set_force_guard_enabled,
        )
        self._disturbance_reset_service = self.create_service(
            Trigger,
            "/mujoco/disturbance/reset",
            self._reset_disturbance,
        )
        self.get_logger().info(
            "Force guard is "
            f"{'enabled' if self._force_guard_enabled else 'disabled'}; "
            f"threshold={self._force_guard.threshold_n:.1f} N, "
            f"trip_duration={self._force_guard.trip_duration_s:.3f} s"
        )
        simulation_rate = float(self.get_parameter("simulation_rate").value)
        joint_state_rate = float(self.get_parameter("joint_state_rate").value)
        if simulation_rate <= 0.0 or joint_state_rate <= 0.0:
            raise ValueError(
                "simulation_rate and joint_state_rate must be positive"
            )

        self._steps_per_tick = max(
            1, round((1.0 / simulation_rate) / self.model.opt.timestep)
        )
        self._simulation_timer = self.create_timer(
            1.0 / simulation_rate, self._step_simulation
        )
        self._joint_state_timer = self.create_timer(
            1.0 / joint_state_rate, self._publish_joint_states
        )

        action_group = ReentrantCallbackGroup()
        self._action_servers = (
            self._make_action_server(
                "/ur5e_arm_controller/follow_joint_trajectory",
                ARM_JOINTS,
                action_group,
            ),
            self._make_action_server(
                "/gripper_controller/follow_joint_trajectory",
                GRIPPER_JOINTS,
                action_group,
            ),
        )

        self._viewer = None
        if bool(self.get_parameter("show_viewer").value):
            from mujoco import viewer as mujoco_viewer

            self._viewer = mujoco_viewer.launch_passive(
                self.model, self.data
            )
            self._viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        self.get_logger().info(
            f"Loaded {model_path}; trajectory controllers are ready"
        )

    def _make_action_server(
        self,
        action_name: str,
        expected_joints: Sequence[str],
        callback_group: ReentrantCallbackGroup,
    ) -> ActionServer:
        return ActionServer(
            self,
            FollowJointTrajectory,
            action_name,
            execute_callback=self._execute_callback_for(expected_joints),
            goal_callback=self._goal_callback_for(expected_joints),
            cancel_callback=lambda _request: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )

    @staticmethod
    def _goal_callback_for(
        expected_joints: Sequence[str],
    ) -> Callable:
        expected = set(expected_joints)

        def callback(goal_request):
            trajectory = goal_request.trajectory
            if (
                not trajectory.points
                or set(trajectory.joint_names) != expected
            ):
                return GoalResponse.REJECT
            if any(
                len(point.positions) != len(trajectory.joint_names)
                for point in trajectory.points
            ):
                return GoalResponse.REJECT
            times = [
                duration_seconds(point.time_from_start)
                for point in trajectory.points
            ]
            if any(t < 0.0 for t in times) or any(
                later <= earlier for earlier, later in zip(times, times[1:])
            ):
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        return callback

    def _execute_callback_for(
        self, expected_joints: Sequence[str]
    ) -> Callable:
        def callback(goal_handle):
            request = goal_handle.request.trajectory
            source_index = {
                name: request.joint_names.index(name)
                for name in expected_joints
            }
            with self._lock:
                previous_positions = np.array(
                    [
                        self.data.qpos[self._joint_qpos[name]]
                        for name in expected_joints
                    ],
                    dtype=float,
                )
                is_arm_trajectory = tuple(expected_joints) == ARM_JOINTS
                if is_arm_trajectory:
                    self._force_guard.reset_latch()
                    self._arm_trajectory_active = True

            history = deque()
            history_keep_s = self._force_guard_retreat_lookback_s + 1.0
            append_history_sample(
                history,
                time.monotonic(),
                previous_positions,
                history_keep_s,
            )

            try:
                previous_time = 0.0
                for point in request.points:
                    target_time = duration_seconds(point.time_from_start)
                    target_positions = np.array(
                        [
                            point.positions[source_index[name]]
                            for name in expected_joints
                        ],
                        dtype=float,
                    )
                    segment_duration = target_time - previous_time
                    segment_start = time.monotonic()

                    while True:
                        if goal_handle.is_cancel_requested:
                            goal_handle.canceled()
                            result = FollowJointTrajectory.Result()
                            result.error_code = (
                                FollowJointTrajectory.Result.SUCCESSFUL
                            )
                            result.error_string = "Trajectory canceled"
                            return result

                        now = time.monotonic()
                        if is_arm_trajectory:
                            with self._lock:
                                guard_tripped = self._force_guard.tripped
                                force_norm_n = (
                                    self._force_guard.force_norm_n
                                )
                            if guard_tripped:
                                return self._retreat_after_force_trip(
                                    goal_handle,
                                    expected_joints,
                                    history,
                                    now,
                                    force_norm_n,
                                )

                        elapsed = now - segment_start
                        alpha = 1.0 if segment_duration <= 0.0 else min(
                            elapsed / segment_duration, 1.0
                        )
                        desired = previous_positions + alpha * (
                            target_positions - previous_positions
                        )
                        self._set_targets(expected_joints, desired)
                        self._publish_feedback(
                            goal_handle, expected_joints, desired
                        )
                        if (
                            not is_arm_trajectory
                            or force_norm_n
                            <= self._force_guard.release_n
                        ):
                            append_history_sample(
                                history,
                                now,
                                desired,
                                history_keep_s,
                            )
                        if alpha >= 1.0:
                            break
                        time.sleep(
                            min(
                                0.02,
                                max(0.001, segment_duration - elapsed),
                            )
                        )

                    previous_positions = target_positions
                    previous_time = target_time

                settle_end = time.monotonic() + 0.1
                while time.monotonic() < settle_end:
                    if is_arm_trajectory:
                        with self._lock:
                            guard_tripped = self._force_guard.tripped
                            force_norm_n = self._force_guard.force_norm_n
                        if guard_tripped:
                            now = time.monotonic()
                            return self._retreat_after_force_trip(
                                goal_handle,
                                expected_joints,
                                history,
                                now,
                                force_norm_n,
                            )
                    time.sleep(0.01)
                goal_handle.succeed()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "Trajectory executed in MuJoCo"
                return result
            finally:
                if is_arm_trajectory:
                    with self._lock:
                        self._arm_trajectory_active = False

        return callback

    def _retreat_after_force_trip(
        self,
        goal_handle,
        joint_names: Sequence[str],
        history,
        trigger_time: float,
        force_norm_n: float,
    ) -> FollowJointTrajectory.Result:
        retreat_target = select_retreat_target(
            history,
            trigger_time,
            self._force_guard_retreat_lookback_s,
        )
        with self._lock:
            retreat_start = np.array(
                [
                    self.data.qpos[self._joint_qpos[name]]
                    for name in joint_names
                ],
                dtype=float,
            )

        self.get_logger().error(
            "Force guard triggered at "
            f"{force_norm_n:.2f} N; aborting insertion and retreating "
            f"{self._force_guard_retreat_lookback_s:.2f} s along the "
            "recorded joint path"
        )
        retreat_start_time = time.monotonic()
        while True:
            elapsed = time.monotonic() - retreat_start_time
            alpha = min(
                elapsed / self._force_guard_retreat_duration_s, 1.0
            )
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            desired = retreat_start + smooth_alpha * (
                retreat_target - retreat_start
            )
            self._set_targets(joint_names, desired)
            self._publish_feedback(goal_handle, joint_names, desired)
            if alpha >= 1.0:
                break
            time.sleep(0.01)

        goal_handle.abort()
        result = FollowJointTrajectory.Result()
        result.error_code = (
            FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
        )
        result.error_string = (
            f"Force guard triggered at {force_norm_n:.2f} N; "
            "trajectory aborted and retreat completed"
        )
        return result

    def _set_force_guard_enabled(self, request, response):
        with self._lock:
            self._force_guard_enabled = bool(request.data)
            self._force_guard.reset_latch()
        response.success = True
        response.message = (
            "Force guard enabled"
            if request.data
            else "Force guard disabled"
        )
        self.get_logger().info(response.message)
        return response

    def _reset_disturbance(self, _request, response):
        with self._lock:
            self._disturbance_torque.fill(0.0)
            if self._disturbance is not None:
                self._disturbance.reset()
                response.message = (
                    "Disturbance sequence reset to configured seed"
                )
            else:
                response.message = "Disturbance is disabled"
        response.success = True
        self.get_logger().info(response.message)
        return response

    def _set_targets(
        self, joint_names: Sequence[str], positions: np.ndarray
    ) -> None:
        with self._lock:
            for name, position in zip(joint_names, positions):
                actuator_id = self._joint_actuator[name]
                lower, upper = self.model.actuator_ctrlrange[actuator_id]
                self.data.ctrl[actuator_id] = float(
                    np.clip(position, lower, upper)
                )

    def _publish_feedback(
        self, goal_handle, joint_names: Sequence[str], desired: np.ndarray
    ) -> None:
        feedback = FollowJointTrajectory.Feedback()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.joint_names = list(joint_names)
        feedback.desired = JointTrajectoryPoint()
        feedback.actual = JointTrajectoryPoint()
        feedback.error = JointTrajectoryPoint()
        feedback.desired.positions = desired.tolist()
        with self._lock:
            actual = np.array(
                [
                    self.data.qpos[self._joint_qpos[name]]
                    for name in joint_names
                ]
            )
        feedback.actual.positions = actual.tolist()
        feedback.error.positions = (desired - actual).tolist()
        goal_handle.publish_feedback(feedback)

    def _step_simulation(self) -> None:
        newly_tripped = False
        with self._lock:
            for _ in range(self._steps_per_tick):
                applied_torque = np.zeros(len(ARM_JOINTS), dtype=float)
                if self._gravity_compensation_enabled:
                    applied_torque += self.data.qfrc_bias[self._arm_dof]
                if self._disturbance is not None:
                    self._disturbance_torque = self._disturbance.step(
                        self.model.opt.timestep
                    )
                    applied_torque += self._disturbance_torque
                self.data.qfrc_applied[self._arm_dof] = applied_torque
                mujoco.mj_step(self.model, self.data)
                self._contact_wrench = body_contact_wrench_world(
                    self.model, self.data, self._carried_drive_id
                )
                newly_tripped = self._force_guard.update(
                    self._contact_wrench[:3],
                    self.model.opt.timestep,
                    armed=(
                        self._force_guard_enabled
                        and self._arm_trajectory_active
                    ),
                ) or newly_tripped
            if self._viewer is not None:
                if self._viewer.is_running():
                    self._viewer.sync()
                else:
                    self._viewer = None
        if newly_tripped:
            self.get_logger().warn(
                "Sustained drive contact force exceeded the configured "
                "threshold"
            )

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(PUBLISHED_JOINTS)
        with self._lock:
            msg.position = [
                float(self.data.qpos[self._joint_qpos[name]])
                for name in PUBLISHED_JOINTS
            ]
            msg.velocity = [
                float(self.data.qvel[self._joint_qvel[name]])
                for name in PUBLISHED_JOINTS
            ]
        self._joint_state_pub.publish(msg)

        disturbance = JointState()
        disturbance.header = msg.header
        disturbance.name = list(ARM_JOINTS)
        with self._lock:
            disturbance.effort = self._disturbance_torque.tolist()
        self._disturbance_pub.publish(disturbance)

        wrench = WrenchStamped()
        wrench.header = msg.header
        wrench.header.frame_id = "world"
        with self._lock:
            force = self._force_guard.filtered_force.copy()
            torque = self._contact_wrench[3:].copy()
        wrench.wrench.force.x = float(force[0])
        wrench.wrench.force.y = float(force[1])
        wrench.wrench.force.z = float(force[2])
        wrench.wrench.torque.x = float(torque[0])
        wrench.wrench.torque.y = float(torque[1])
        wrench.wrench.torque.z = float(torque[2])
        self._drive_wrench_pub.publish(wrench)

    def destroy_node(self) -> bool:
        if self._viewer is not None:
            self._viewer.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except KeyboardInterrupt:
            # A second SIGINT can arrive while launch is tearing nodes down.
            pass
        if rclpy.ok():
            rclpy.shutdown()
