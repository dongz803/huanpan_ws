#include "huanpan_motion/control/trajectory_executor.hpp"
#include "huanpan_motion/motion_planning/motion_planner.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "huanpan_motion_node",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  const std::string planning_group = node->get_parameter_or<std::string>(
    "planning_group", "ur_manipulator");
  const std::string controller_action = node->get_parameter_or<std::string>(
    "controller_action", "/ur5e_arm_controller/follow_joint_trajectory");
  const std::string end_effector_link = node->get_parameter_or<std::string>(
    "end_effector_link", "drive_tip");
  const double server_wait_seconds = node->get_parameter_or<double>(
    "server_wait_seconds", 5.0);
  const double planning_time = node->get_parameter_or<double>("planning_time", 5.0);
  const int planning_attempts = node->get_parameter_or<int>("planning_attempts", 3);
  const double velocity_scaling = node->get_parameter_or<double>("velocity_scaling", 0.2);
  const double acceleration_scaling = node->get_parameter_or<double>(
    "acceleration_scaling", 0.2);
  if (planning_attempts <= 0) {
    RCLCPP_FATAL(node->get_logger(), "planning_attempts must be at least one");
    rclcpp::shutdown();
    return 2;
  }

  auto planner = std::make_shared<huanpan_motion::motion_planning::MotionPlanner>(
    node, planning_group, "", server_wait_seconds);
  auto executor = std::make_shared<huanpan_motion::control::TrajectoryExecutor>(
    node, controller_action);
  planner->setPlanningTime(planning_time);
  planner->setPlanningAttempts(static_cast<unsigned int>(planning_attempts));
  planner->setVelocityScaling(velocity_scaling);
  planner->setAccelerationScaling(acceleration_scaling);
  if (!planner->interface().setEndEffectorLink(end_effector_link)) {
    RCLCPP_WARN(
      node->get_logger(), "MoveIt rejected end-effector link: %s",
      end_effector_link.c_str());
  }
  RCLCPP_INFO(
    node->get_logger(), "huanpan_motion ready: group=%s eef=%s controller=%s",
    planning_group.c_str(), end_effector_link.c_str(), executor->actionName().c_str());

  rclcpp::executors::MultiThreadedExecutor ros_executor;
  ros_executor.add_node(node);
  ros_executor.spin();
  planner.reset();
  executor.reset();
  rclcpp::shutdown();
  return 0;
}
