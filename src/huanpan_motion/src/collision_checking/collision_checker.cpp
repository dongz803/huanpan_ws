#include "huanpan_motion/collision_checking/collision_checker.hpp"

#include <stdexcept>
#include <utility>

namespace huanpan_motion::collision_checking
{
CollisionChecker::CollisionChecker(
  planning_scene::PlanningSceneConstPtr planning_scene,
  std::string planning_group)
: planning_scene_(std::move(planning_scene)),
  planning_group_(std::move(planning_group))
{
  if (!planning_scene_) {
    throw std::invalid_argument("planning_scene must not be null");
  }
}

bool CollisionChecker::isStateCollisionFree(
  const moveit::core::RobotState & state,
  const bool verbose) const
{
  return !planning_scene_->isStateColliding(state, planning_group_, verbose);
}

bool CollisionChecker::isTrajectoryCollisionFree(
  const robot_trajectory::RobotTrajectory & trajectory,
  const std::size_t sampling_stride,
  const bool verbose) const
{
  if (sampling_stride == 0) {
    throw std::invalid_argument("sampling_stride must be at least one");
  }
  const std::size_t count = trajectory.getWayPointCount();
  if (count == 0) {
    return false;
  }
  for (std::size_t index = 0; index < count; index += sampling_stride) {
    if (!isStateCollisionFree(trajectory.getWayPoint(index), verbose)) {
      return false;
    }
  }
  return (count - 1) % sampling_stride == 0 ||
         isStateCollisionFree(trajectory.getLastWayPoint(), verbose);
}

double CollisionChecker::minimumDistance(
  const moveit::core::RobotState & state) const
{
  return planning_scene_->distanceToCollision(state);
}
}  // namespace huanpan_motion::collision_checking
