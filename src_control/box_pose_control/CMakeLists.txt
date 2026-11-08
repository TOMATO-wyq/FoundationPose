cmake_minimum_required(VERSION 3.16)
project(box_pose_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(control_core src/controller.cpp src/serial_port.cpp)
target_include_directories(control_core PUBLIC include)
target_link_libraries(control_core PUBLIC Eigen3::Eigen)

add_executable(box_pose_control_node src/control_node.cpp)
target_link_libraries(box_pose_control_node control_core)
ament_target_dependencies(box_pose_control_node
  rclcpp rcl_interfaces geometry_msgs vision_msgs std_srvs)

install(TARGETS box_pose_control_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/turtlesim_pose_sim.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME turtlesim_pose_sim)
install(PROGRAMS scripts/turtlesim_real_view.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME turtlesim_real_view)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_controller test/test_controller.cpp)
  target_link_libraries(test_controller control_core)
  ament_add_gtest(test_serial_port test/test_serial_port.cpp)
  target_link_libraries(test_serial_port control_core util)
  ament_add_pytest_test(test_ros_node test/test_ros_node.py TIMEOUT 120
    ENV NODE_EXECUTABLE=$<TARGET_FILE:box_pose_control_node>)
  ament_add_pytest_test(test_launch test/test_launch.py TIMEOUT 30)
endif()

ament_package()
