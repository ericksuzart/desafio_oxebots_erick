cmake_minimum_required(VERSION 3.8)
project(desafio_oxebots_erick_interfaces)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  action/MoveBase.action
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
