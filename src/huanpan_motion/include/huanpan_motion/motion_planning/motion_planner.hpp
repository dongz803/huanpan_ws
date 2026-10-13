#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace huanpan_motion::motion_planning
{
class MotionPlanner
{
public:
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;

  MotionPlanner(
    const rclcpp::Node::SharedPtr & node,
    std::string planning_group,
    std::string move_group_namespace = "",
    double server_wait_seconds = 5.0);
  bool planToPose(
    const geometry_msgs::msg::PoseStamped & target,
    Plan & plan,
    const std::string & end_effector_link = "");
  bool planToJointTarget(const std::vector<double> & joint_values, Plan & plan);
  double computeCartesianPath(
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    moveit_msgs::msg::RobotTrajectory & trajectory,
    double eef_step = 0.002,
    double jump_threshold = 0.0,
    bool avoid_collisions = true);
  void setPlanningTime(double seconds);
  void setPlanningAttempts(unsigned int attempts);
  void setVelocityScaling(double scale);
  void setAccelerationScaling(double scale);
  moveit::planning_interface::MoveGroupInterface & interface();

private:
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
};
}  // namespace huanpan_motion::motion_planning
