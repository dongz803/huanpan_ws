#include "huanpan_motion/motion_planning/motion_planner.hpp"
#include "huanpan_motion/trajectory_processing/trajectory_processor.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <moveit/robot_state/conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace std::chrono_literals;

geometry_msgs::msg::PoseStamped makePose(
  const rclcpp::Node::SharedPtr & node,
  const std::string & frame_id,
  const std::vector<double> & position,
  const std::vector<double> & orientation)
{
  if (position.size() != 3 || orientation.size() != 4) {
    throw std::invalid_argument(
            "target position must contain 3 values and orientation 4 values");
  }
  geometry_msgs::msg::PoseStamped target;
  target.header.frame_id = frame_id;
  target.header.stamp = node->now();
  target.pose.position.x = position[0];
  target.pose.position.y = position[1];
  target.pose.position.z = position[2];
  target.pose.orientation.x = orientation[0];
  target.pose.orientation.y = orientation[1];
  target.pose.orientation.z = orientation[2];
  target.pose.orientation.w = orientation[3];
  return target;
}

bool setForceGuard(
  const rclcpp::Node::SharedPtr & node,
  const rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr & client,
  const bool enabled,
  const std::chrono::milliseconds timeout)
{
  if (!client->wait_for_service(timeout)) {
    RCLCPP_ERROR(node->get_logger(), "Force-guard service is unavailable");
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = enabled;
  auto future = client->async_send_request(request);
  if (future.wait_for(timeout) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Force-guard service timed out");
    return false;
  }
  const auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(
      node->get_logger(), "Force-guard service rejected request: %s",
      response->message.c_str());
    return false;
  }
  RCLCPP_INFO(node->get_logger(), "%s", response->message.c_str());
  return true;
}

bool resetDisturbance(
  const rclcpp::Node::SharedPtr & node,
  const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr & client,
  const std::chrono::milliseconds timeout)
{
  if (!client->wait_for_service(timeout)) {
    RCLCPP_ERROR(node->get_logger(), "Disturbance-reset service is unavailable");
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future = client->async_send_request(request);
  if (future.wait_for(timeout) != std::future_status::ready) {
    RCLCPP_ERROR(node->get_logger(), "Disturbance-reset service timed out");
    return false;
  }
  const auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(
      node->get_logger(), "Disturbance reset failed: %s",
      response->message.c_str());
    return false;
  }
  RCLCPP_INFO(node->get_logger(), "%s", response->message.c_str());
  return true;
}

int runDemo(const rclcpp::Node::SharedPtr & node)
{
  const std::string mode = node->get_parameter_or<std::string>("mode", "nominal");
  if (mode != "nominal" && mode != "overforce") {
    RCLCPP_ERROR(node->get_logger(), "mode must be nominal or overforce");
    return 2;
  }

  const std::string planning_group = node->get_parameter_or<std::string>(
    "planning_group", "ur_manipulator");
  const std::string end_effector_link = node->get_parameter_or<std::string>(
    "end_effector_link", "drive_tip");
  const std::string reference_frame = node->get_parameter_or<std::string>(
    "reference_frame", "world");
  const double server_wait_seconds = node->get_parameter_or<double>(
    "server_wait_seconds", 10.0);
  const double planning_time = node->get_parameter_or<double>("planning_time", 10.0);
  const int planning_attempts = node->get_parameter_or<int>("planning_attempts", 5);
  const double velocity_scaling = node->get_parameter_or<double>(
    "velocity_scaling", 0.10);
  const double acceleration_scaling = node->get_parameter_or<double>(
    "acceleration_scaling", 0.10);
  const double cartesian_eef_step = node->get_parameter_or<double>(
    "cartesian_eef_step", 0.002);
  const double cartesian_jump_threshold = node->get_parameter_or<double>(
    "cartesian_jump_threshold", 0.0);
  const double minimum_cartesian_fraction = node->get_parameter_or<double>(
    "minimum_cartesian_fraction", 0.995);
  const double fault_lateral_offset_m = node->get_parameter_or<double>(
    "fault_lateral_offset_m", 0.004);
  const double expected_force_threshold_n = node->get_parameter_or<double>(
    "expected_force_threshold_n", 25.0);
  const auto entry_position = node->get_parameter_or<std::vector<double>>(
    "entry_position", {0.6050, 0.0525, 0.7225});
  auto goal_position = node->get_parameter_or<std::vector<double>>(
    "goal_position", {0.7875, 0.0525, 0.7225});
  const auto orientation = node->get_parameter_or<std::vector<double>>(
    "orientation_xyzw", {0.70710678, 0.0, 0.70710678, 0.0});

  if (planning_attempts <= 0 || minimum_cartesian_fraction <= 0.0 ||
    minimum_cartesian_fraction > 1.0)
  {
    RCLCPP_ERROR(node->get_logger(), "Invalid insertion-demo planning parameters");
    return 2;
  }
  if (mode == "overforce") {
    goal_position.at(1) += fault_lateral_offset_m;
  }

  std::atomic<double> maximum_force_n{0.0};
  auto wrench_subscription = node->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "/mujoco/drive_wrench", rclcpp::SensorDataQoS(),
    [&maximum_force_n](const geometry_msgs::msg::WrenchStamped::SharedPtr message) {
      const auto & force = message->wrench.force;
      const double norm = std::sqrt(
        force.x * force.x + force.y * force.y + force.z * force.z);
      double previous = maximum_force_n.load();
      while (norm > previous && !maximum_force_n.compare_exchange_weak(previous, norm)) {
      }
    });
  (void)wrench_subscription;

  auto force_guard_client = node->create_client<std_srvs::srv::SetBool>(
    "/mujoco/force_guard/enable");
  auto disturbance_reset_client = node->create_client<std_srvs::srv::Trigger>(
    "/mujoco/disturbance/reset");
  auto planner = std::make_shared<huanpan_motion::motion_planning::MotionPlanner>(
    node, planning_group, "", server_wait_seconds);
  planner->setPlanningTime(planning_time);
  planner->setPlanningAttempts(static_cast<unsigned int>(planning_attempts));
  planner->setVelocityScaling(velocity_scaling);
  planner->setAccelerationScaling(acceleration_scaling);
  planner->interface().setPoseReferenceFrame(reference_frame);
  planner->interface().setGoalPositionTolerance(0.0005);
  planner->interface().setGoalOrientationTolerance(0.005);
  if (!planner->interface().setEndEffectorLink(end_effector_link)) {
    RCLCPP_ERROR(
      node->get_logger(), "MoveIt rejected end-effector link: %s",
      end_effector_link.c_str());
    return 2;
  }

  if (!setForceGuard(node, force_guard_client, false, 5s)) {
    return 2;
  }
  const auto entry_pose = makePose(
    node, reference_frame, entry_position, orientation);
  huanpan_motion::motion_planning::MotionPlanner::Plan approach_plan;
  RCLCPP_INFO(
    node->get_logger(), "Planning automatically to slot_09_entry at %.4f %.4f %.4f",
    entry_position[0], entry_position[1], entry_position[2]);
  if (!planner->planToPose(entry_pose, approach_plan, end_effector_link)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to plan to slot_09_entry");
    return 1;
  }
  if (!static_cast<bool>(planner->interface().execute(approach_plan))) {
    RCLCPP_ERROR(node->get_logger(), "Failed to execute approach trajectory");
    return 1;
  }
  std::this_thread::sleep_for(300ms);

  if (!resetDisturbance(node, disturbance_reset_client, 5s)) {
    return 2;
  }
  if (!setForceGuard(node, force_guard_client, true, 5s)) {
    return 2;
  }
  maximum_force_n.store(0.0);
  const auto goal_pose = makePose(
    node, reference_frame, goal_position, orientation);
  // Re-align at the entry pose before advancing. Under joint disturbance the
  // measured TCP can be a few millimetres away from the planned endpoint; a
  // diagonal correction while entering the narrow bay can otherwise collide.
  std::vector<geometry_msgs::msg::Pose> waypoints{
    entry_pose.pose, goal_pose.pose};
  moveit_msgs::msg::RobotTrajectory geometric_trajectory;
  const bool avoid_collisions = mode == "nominal";
  const double fraction = planner->computeCartesianPath(
    waypoints, geometric_trajectory, cartesian_eef_step,
    cartesian_jump_threshold, avoid_collisions);
  RCLCPP_INFO(
    node->get_logger(), "Cartesian insertion path fraction: %.3f", fraction);
  if (fraction < minimum_cartesian_fraction) {
    setForceGuard(node, force_guard_client, false, 5s);
    RCLCPP_ERROR(node->get_logger(), "Cartesian insertion path is incomplete");
    return 1;
  }

  auto current_state = planner->interface().getCurrentState(server_wait_seconds);
  if (!current_state) {
    setForceGuard(node, force_guard_client, false, 5s);
    RCLCPP_ERROR(node->get_logger(), "Unable to read current robot state");
    return 2;
  }
  moveit_msgs::msg::RobotState start_state;
  moveit::core::robotStateToRobotStateMsg(*current_state, start_state);
  huanpan_motion::trajectory_processing::TrajectoryProcessor processor(
    planner->interface().getRobotModel(), planning_group);
  moveit_msgs::msg::RobotTrajectory timed_trajectory;
  if (!processor.applyTimeOptimalParameterization(
      start_state, geometric_trajectory, timed_trajectory,
      velocity_scaling, acceleration_scaling))
  {
    setForceGuard(node, force_guard_client, false, 5s);
    RCLCPP_ERROR(node->get_logger(), "Failed to time-parameterize insertion path");
    return 2;
  }

  RCLCPP_INFO(
    node->get_logger(), "Executing %s insertion with disturbance handled by MuJoCo",
    mode.c_str());
  const bool execution_succeeded = static_cast<bool>(
    planner->interface().execute(timed_trajectory));
  std::this_thread::sleep_for(200ms);
  const double observed_force_n = maximum_force_n.load();
  setForceGuard(node, force_guard_client, false, 5s);

  if (mode == "nominal") {
    if (!execution_succeeded) {
      RCLCPP_ERROR(
        node->get_logger(), "Nominal insertion failed; max force %.2f N",
        observed_force_n);
      return 1;
    }
    RCLCPP_INFO(
      node->get_logger(), "NOMINAL INSERTION TEST: PASS; max force %.2f N",
      observed_force_n);
    return 0;
  }

  if (execution_succeeded || observed_force_n < expected_force_threshold_n) {
    RCLCPP_ERROR(
      node->get_logger(),
      "OVERFORCE INSERTION TEST: FAIL; execution_succeeded=%s max force=%.2f N",
      execution_succeeded ? "true" : "false", observed_force_n);
    return 1;
  }
  RCLCPP_INFO(
    node->get_logger(),
    "OVERFORCE INSERTION TEST: PASS; force guard aborted and retreated at %.2f N",
    observed_force_n);
  return 0;
}
}  // namespace

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "insertion_demo",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {executor.spin();});

  int exit_code = 2;
  try {
    exit_code = runDemo(node);
  } catch (const std::exception & error) {
    RCLCPP_FATAL(node->get_logger(), "Insertion demo exception: %s", error.what());
  }

  executor.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return exit_code;
}
