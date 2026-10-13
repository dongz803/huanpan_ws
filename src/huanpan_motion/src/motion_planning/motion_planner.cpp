#include "huanpan_motion/motion_planning/motion_planner.hpp"

#include <stdexcept>
#include <utility>

namespace huanpan_motion::motion_planning
{
MotionPlanner::MotionPlanner(
  const rclcpp::Node::SharedPtr & node,
  std::string planning_group,
  std::string move_group_namespace,
  const double server_wait_seconds)
{
  if (!node) {
    throw std::invalid_argument("node must not be null");
  }
  if (planning_group.empty()) {
    throw std::invalid_argument("planning_group must not be empty");
  }
  if (server_wait_seconds < 0.0) {
    throw std::invalid_argument("server_wait_seconds must be non-negative");
  }
  moveit::planning_interface::MoveGroupInterface::Options options(
    std::move(planning_group), "robot_description", std::move(move_group_namespace));
  move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
    node, options, nullptr, rclcpp::Duration::from_seconds(server_wait_seconds));
}

bool MotionPlanner::planToPose(
  const geometry_msgs::msg::PoseStamped & target,
  Plan & plan,
  const std::string & end_effector_link)
{
  move_group_->setStartStateToCurrentState();
  move_group_->clearPoseTargets();
  if (!move_group_->setPoseTarget(target, end_effector_link)) {
    return false;
  }
  const auto result = move_group_->plan(plan);
  move_group_->clearPoseTargets();
  return static_cast<bool>(result);
}

bool MotionPlanner::planToJointTarget(
  const std::vector<double> & joint_values,
  Plan & plan)
{
  move_group_->setStartStateToCurrentState();
  move_group_->clearPoseTargets();
  if (!move_group_->setJointValueTarget(joint_values)) {
    return false;
  }
  return static_cast<bool>(move_group_->plan(plan));
}

double MotionPlanner::computeCartesianPath(
  const std::vector<geometry_msgs::msg::Pose> & waypoints,
  moveit_msgs::msg::RobotTrajectory & trajectory,
  const double eef_step,
  const double jump_threshold,
  const bool avoid_collisions)
{
  if (eef_step <= 0.0) {
    throw std::invalid_argument("eef_step must be positive");
  }
  move_group_->setStartStateToCurrentState();
  return move_group_->computeCartesianPath(
    waypoints, eef_step, jump_threshold, trajectory, avoid_collisions);
}

void MotionPlanner::setPlanningTime(const double seconds)
{
  if (seconds <= 0.0) {
    throw std::invalid_argument("planning time must be positive");
  }
  move_group_->setPlanningTime(seconds);
}

void MotionPlanner::setPlanningAttempts(const unsigned int attempts)
{
  if (attempts == 0) {
    throw std::invalid_argument("planning attempts must be at least one");
  }
  move_group_->setNumPlanningAttempts(attempts);
}

void MotionPlanner::setVelocityScaling(const double scale)
{
  if (scale <= 0.0 || scale > 1.0) {
    throw std::invalid_argument("velocity scaling must be in (0, 1]");
  }
  move_group_->setMaxVelocityScalingFactor(scale);
}

void MotionPlanner::setAccelerationScaling(const double scale)
{
  if (scale <= 0.0 || scale > 1.0) {
    throw std::invalid_argument("acceleration scaling must be in (0, 1]");
  }
  move_group_->setMaxAccelerationScalingFactor(scale);
}

moveit::planning_interface::MoveGroupInterface & MotionPlanner::interface()
{
  return *move_group_;
}
}  // namespace huanpan_motion::motion_planning
