#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <string>

namespace huanpan_motion::kinematics
{
class KinematicsSolver
{
public:
  KinematicsSolver(
    moveit::core::RobotModelConstPtr robot_model,
    std::string planning_group,
    std::string end_effector_link);

  bool solveIK(
    const moveit::core::RobotState & seed_state,
    const Eigen::Isometry3d & target_pose,
    moveit::core::RobotState & solution,
    double timeout = 0.1) const;
  Eigen::Isometry3d computeFK(const moveit::core::RobotState & state) const;
  bool computeJacobian(
    const moveit::core::RobotState & state,
    Eigen::MatrixXd & jacobian) const;
  bool satisfiesBounds(
    const moveit::core::RobotState & state,
    double margin = 0.0) const;
  const moveit::core::JointModelGroup * jointModelGroup() const;
  const std::string & endEffectorLink() const;

private:
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * joint_model_group_;
  const moveit::core::LinkModel * end_effector_link_model_;
  std::string planning_group_;
  std::string end_effector_link_;
};
}  // namespace huanpan_motion::kinematics
