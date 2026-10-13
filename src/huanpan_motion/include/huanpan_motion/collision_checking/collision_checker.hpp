#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <cstddef>
#include <string>

namespace huanpan_motion::collision_checking
{
class CollisionChecker
{
public:
  explicit CollisionChecker(
    planning_scene::PlanningSceneConstPtr planning_scene,
    std::string planning_group = "ur_manipulator");
  bool isStateCollisionFree(
    const moveit::core::RobotState & state,
    bool verbose = false) const;
  bool isTrajectoryCollisionFree(
    const robot_trajectory::RobotTrajectory & trajectory,
    std::size_t sampling_stride = 1,
    bool verbose = false) const;
  double minimumDistance(const moveit::core::RobotState & state) const;

private:
  planning_scene::PlanningSceneConstPtr planning_scene_;
  std::string planning_group_;
};
}  // namespace huanpan_motion::collision_checking
