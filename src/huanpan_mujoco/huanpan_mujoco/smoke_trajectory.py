"""Send one small arm motion and one gripper motion to the MuJoCo bridge."""

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

from .bridge_node import ARM_JOINTS, GRIPPER_JOINTS


class SmokeTrajectory(Node):
    def __init__(self) -> None:
        super().__init__("smoke_trajectory")
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/ur5e_arm_controller/follow_joint_trajectory",
        )
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )

    def send_and_wait(
        self, client, joint_names, positions, seconds: int
    ) -> bool:
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"Action server unavailable: {client._action_name}"
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start.sec = seconds
        goal.trajectory.points = [point]

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error("Trajectory returned no result")
            return False
        result = wrapped_result.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(result.error_string)
            return False
        self.get_logger().info(result.error_string)
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SmokeTrajectory()
    exit_code = 0
    try:
        arm_ok = node.send_and_wait(
            node.arm_client,
            ARM_JOINTS,
            (-1.30, -1.40, 1.45, -1.55, -1.57, 0.10),
            2,
        )
        gripper_ok = node.send_and_wait(
            node.gripper_client,
            GRIPPER_JOINTS,
            (-0.75,),
            1,
        )
        if not (arm_ok and gripper_ok):
            raise RuntimeError("MuJoCo trajectory smoke test failed")
        node.get_logger().info("MuJoCo trajectory smoke test: PASS")
    except Exception as exc:
        node.get_logger().error(str(exc))
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if exit_code:
        raise SystemExit(exit_code)
