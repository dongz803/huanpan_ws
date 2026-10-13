#pragma once

#include <moveit/robot_model/robot_model.h>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <string>

namespace huanpan_motion::trajectory_processing
{
class TrajectoryProcessor
{
public:
  TrajectoryProcessor(
    moveit::core::RobotModelConstPtr robot_model,
    std::string planning_group);
  bool applyTimeOptimalParameterization(
    const moveit_msgs::msg::RobotState & start_state,
    const moveit_msgs::msg::RobotTrajectory & input,
    moveit_msgs::msg::RobotTrajectory & output,
    double velocity_scaling = 0.2,
    double acceleration_scaling = 0.2) const;
  static bool validScalingFactor(double scale);
  static bool hasStrictlyIncreasingTime(
    const moveit_msgs::msg::RobotTrajectory & trajectory);

private:
  moveit::core::RobotModelConstPtr robot_model_;
  std::string planning_group_;
};
}  // namespace huanpan_motion::trajectory_processing
