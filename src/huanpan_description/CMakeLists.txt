cmake_minimum_required(VERSION 3.8)
project(huanpan_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes mujoco rviz urdf
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
