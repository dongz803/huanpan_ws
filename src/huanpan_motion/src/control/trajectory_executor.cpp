#include "huanpan_motion/control/trajectory_executor.hpp"

#include <stdexcept>
#include <utility>

namespace huanpan_motion::control
{
TrajectoryExecutor::TrajectoryExecutor(
  const rclcpp::Node::SharedPtr & node,
  std::string action_name)
: action_name_(std::move(action_name))
{
  if (!node) {
    throw std::invalid_argument("node must not be null");
  }
  if (action_name_.empty()) {
    throw std::invalid_argument("action_name must not be empty");
  }
  client_ = rclcpp_action::create_client<Action>(node, action_name_);
}

bool TrajectoryExecutor::waitForServer(
  const std::chrono::milliseconds timeout) const
{
  return client_->wait_for_action_server(timeout);
}

TrajectoryExecutor::GoalFuture TrajectoryExecutor::execute(
  const moveit_msgs::msg::RobotTrajectory & trajectory,
  Client::GoalResponseCallback goal_response_callback,
  Client::FeedbackCallback feedback_callback,
  Client::ResultCallback result_callback)
{
  if (trajectory.joint_trajectory.points.empty()) {
    throw std::invalid_argument("trajectory must contain at least one point");
  }
  Action::Goal goal;
  goal.trajectory = trajectory.joint_trajectory;
  Client::SendGoalOptions options;
  options.goal_response_callback = std::move(goal_response_callback);
  options.feedback_callback = std::move(feedback_callback);
  options.result_callback = std::move(result_callback);
  return client_->async_send_goal(goal, options);
}

TrajectoryExecutor::CancelFuture TrajectoryExecutor::cancelAll()
{
  return client_->async_cancel_all_goals();
}

const std::string & TrajectoryExecutor::actionName() const
{
  return action_name_;
}
}  // namespace huanpan_motion::control
