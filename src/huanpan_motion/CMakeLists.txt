cmake_minimum_required(VERSION 3.8)
project(huanpan_motion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/kinematics/kinematics_solver.cpp
  src/control/trajectory_executor.cpp
  src/collision_checking/collision_checker.cpp
  src/motion_planning/motion_planner.cpp
  src/trajectory_processing/trajectory_processor.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}
  control_msgs geometry_msgs moveit_core moveit_msgs
  moveit_ros_planning_interface rclcpp rclcpp_action trajectory_msgs
)

add_executable(motion_node src/motion_node.cpp)
target_link_libraries(motion_node ${PROJECT_NAME})
ament_target_dependencies(motion_node rclcpp)

add_executable(insertion_demo src/insertion_demo_node.cpp)
target_link_libraries(insertion_demo ${PROJECT_NAME})
ament_target_dependencies(
  insertion_demo geometry_msgs moveit_core moveit_msgs
  moveit_ros_planning_interface rclcpp std_srvs
)

install(DIRECTORY include/ DESTINATION include)
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(
  TARGETS ${PROJECT_NAME} motion_node insertion_demo
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  control_msgs geometry_msgs moveit_core moveit_msgs
  moveit_ros_planning_interface rclcpp rclcpp_action std_srvs trajectory_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_trajectory_processor test/test_trajectory_processor.cpp)
  target_link_libraries(test_trajectory_processor ${PROJECT_NAME})
endif()

ament_package()
