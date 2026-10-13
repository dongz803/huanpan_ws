#include "huanpan_motion/kinematics/kinematics_solver.hpp"

#include <stdexcept>
#include <utility>

namespace huanpan_motion::kinematics
{
KinematicsSolver::KinematicsSolver(
  moveit::core::RobotModelConstPtr robot_model,
  std::string planning_group,
  std::string end_effector_link)
: robot_model_(std::move(robot_model)),
  joint_model_group_(nullptr),
  end_effector_link_model_(nullptr),
  planning_group_(std::move(planning_group)),
  end_effector_link_(std::move(end_effector_link))
{
  if (!robot_model_) {
    throw std::invalid_argument("robot_model must not be null");
  }
  joint_model_group_ = robot_model_->getJointModelGroup(planning_group_);
  if (joint_model_group_ == nullptr) {
    throw std::invalid_argument("unknown planning group: " + planning_group_);
  }
  end_effector_link_model_ = robot_model_->getLinkModel(end_effector_link_);
  if (end_effector_link_model_ == nullptr) {
    throw std::invalid_argument("unknown end-effector link: " + end_effector_link_);
  }
}

bool KinematicsSolver::solveIK(
  const moveit::core::RobotState & seed_state,
  const Eigen::Isometry3d & target_pose,
  moveit::core::RobotState & solution,
  const double timeout) const
{
  if (timeout < 0.0) {
    throw std::invalid_argument("IK timeout must be non-negative");
  }
  solution = seed_state;
  const bool solved = solution.setFromIK(
    joint_model_group_, target_pose, end_effector_link_, timeout);
  if (solved) {
    solution.update();
  }
  return solved && satisfiesBounds(solution);
}

Eigen::Isometry3d KinematicsSolver::computeFK(
  const moveit::core::RobotState & state) const
{
  moveit::core::RobotState updated_state(state);
  updated_state.updateLinkTransforms();
  return updated_state.getGlobalLinkTransform(end_effector_link_model_);
}

bool KinematicsSolver::computeJacobian(
  const moveit::core::RobotState & state,
  Eigen::MatrixXd & jacobian) const
{
  moveit::core::RobotState updated_state(state);
  updated_state.updateLinkTransforms();
  return updated_state.getJacobian(
    joint_model_group_, end_effector_link_model_, Eigen::Vector3d::Zero(), jacobian);
}

bool KinematicsSolver::satisfiesBounds(
  const moveit::core::RobotState & state,
  const double margin) const
{
  return state.satisfiesBounds(joint_model_group_, margin);
}

const moveit::core::JointModelGroup * KinematicsSolver::jointModelGroup() const
{
  return joint_model_group_;
}

const std::string & KinematicsSolver::endEffectorLink() const
{
  return end_effector_link_;
}
}  // namespace huanpan_motion::kinematics
