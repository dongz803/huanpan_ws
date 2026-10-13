#include "huanpan_motion/trajectory_processing/trajectory_processor.hpp"

#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace huanpan_motion::trajectory_processing
{
namespace
{
std::int64_t durationNanoseconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<std::int64_t>(duration.sec) * 1000000000LL + duration.nanosec;
}
}  // namespace

TrajectoryProcessor::TrajectoryProcessor(
  moveit::core::RobotModelConstPtr robot_model,
  std::string planning_group)
: robot_model_(std::move(robot_model)),
  planning_group_(std::move(planning_group))
{
  if (!robot_model_) {
    throw std::invalid_argument("robot_model must not be null");
  }
  if (robot_model_->getJointModelGroup(planning_group_) == nullptr) {
    throw std::invalid_argument("unknown planning group: " + planning_group_);
  }
}

bool TrajectoryProcessor::applyTimeOptimalParameterization(
  const moveit_msgs::msg::RobotState & start_state,
  const moveit_msgs::msg::RobotTrajectory & input,
  moveit_msgs::msg::RobotTrajectory & output,
  const double velocity_scaling,
  const double acceleration_scaling) const
{
  if (!validScalingFactor(velocity_scaling) ||
    !validScalingFactor(acceleration_scaling))
  {
    throw std::invalid_argument("trajectory scaling factors must be in (0, 1]");
  }
  if (input.joint_trajectory.points.empty()) {
    return false;
  }

  moveit::core::RobotState reference_state(robot_model_);
  reference_state.setToDefaultValues();
  if (!moveit::core::robotStateMsgToRobotState(start_state, reference_state)) {
    return false;
  }
  robot_trajectory::RobotTrajectory trajectory(robot_model_, planning_group_);
  trajectory.setRobotTrajectoryMsg(reference_state, input);
  ::trajectory_processing::TimeOptimalTrajectoryGeneration algorithm;
  if (!algorithm.computeTimeStamps(
      trajectory, velocity_scaling, acceleration_scaling))
  {
    return false;
  }
  trajectory.getRobotTrajectoryMsg(output);
  return hasStrictlyIncreasingTime(output);
}

bool TrajectoryProcessor::validScalingFactor(const double scale)
{
  return scale > 0.0 && scale <= 1.0;
}

bool TrajectoryProcessor::hasStrictlyIncreasingTime(
  const moveit_msgs::msg::RobotTrajectory & trajectory)
{
  const auto & points = trajectory.joint_trajectory.points;
  if (points.empty()) {
    return false;
  }
  std::int64_t previous = -1;
  for (const auto & point : points) {
    const std::int64_t current = durationNanoseconds(point.time_from_start);
    if (current <= previous) {
      return false;
    }
    previous = current;
  }
  return true;
}
}  // namespace huanpan_motion::trajectory_processing
