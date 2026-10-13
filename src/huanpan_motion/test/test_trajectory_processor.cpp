#include "huanpan_motion/trajectory_processing/trajectory_processor.hpp"

#include <gtest/gtest.h>

#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

TEST(TrajectoryProcessorContract, ScalingFactorRangeIsExplicit)
{
  using huanpan_motion::trajectory_processing::TrajectoryProcessor;
  EXPECT_FALSE(TrajectoryProcessor::validScalingFactor(0.0));
  EXPECT_FALSE(TrajectoryProcessor::validScalingFactor(-0.1));
  EXPECT_TRUE(TrajectoryProcessor::validScalingFactor(0.2));
  EXPECT_TRUE(TrajectoryProcessor::validScalingFactor(1.0));
  EXPECT_FALSE(TrajectoryProcessor::validScalingFactor(1.1));
}

TEST(TrajectoryProcessorContract, RequiresStrictlyIncreasingTime)
{
  using huanpan_motion::trajectory_processing::TrajectoryProcessor;
  moveit_msgs::msg::RobotTrajectory trajectory;
  EXPECT_FALSE(TrajectoryProcessor::hasStrictlyIncreasingTime(trajectory));

  trajectory_msgs::msg::JointTrajectoryPoint first;
  first.time_from_start.nanosec = 100000000;
  trajectory.joint_trajectory.points.push_back(first);
  EXPECT_TRUE(TrajectoryProcessor::hasStrictlyIncreasingTime(trajectory));

  trajectory_msgs::msg::JointTrajectoryPoint duplicate;
  duplicate.time_from_start.nanosec = 100000000;
  trajectory.joint_trajectory.points.push_back(duplicate);
  EXPECT_FALSE(TrajectoryProcessor::hasStrictlyIncreasingTime(trajectory));

  trajectory.joint_trajectory.points.back().time_from_start.nanosec = 200000000;
  EXPECT_TRUE(TrajectoryProcessor::hasStrictlyIncreasingTime(trajectory));
}
