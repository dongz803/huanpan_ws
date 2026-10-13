#pragma once

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <chrono>
#include <future>
#include <string>

namespace huanpan_motion::control
{
class TrajectoryExecutor
{
public:
  using Action = control_msgs::action::FollowJointTrajectory;
  using Client = rclcpp_action::Client<Action>;
  using GoalHandle = rclcpp_action::ClientGoalHandle<Action>;
  using GoalFuture = std::shared_future<GoalHandle::SharedPtr>;
  using CancelFuture = std::shared_future<Client::CancelResponse::SharedPtr>;

  TrajectoryExecutor(
    const rclcpp::Node::SharedPtr & node,
    std::string action_name);
  bool waitForServer(std::chrono::milliseconds timeout) const;
  GoalFuture execute(
    const moveit_msgs::msg::RobotTrajectory & trajectory,
    Client::GoalResponseCallback goal_response_callback = nullptr,
    Client::FeedbackCallback feedback_callback = nullptr,
    Client::ResultCallback result_callback = nullptr);
  CancelFuture cancelAll();
  const std::string & actionName() const;

private:
  std::string action_name_;
  Client::SharedPtr client_;
};
}  // namespace huanpan_motion::control
