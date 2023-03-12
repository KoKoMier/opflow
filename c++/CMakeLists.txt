#CMake 最低版本要求
cmake_minimum_required (VERSION 2.8)

#引入opencv
set(OpenCV_DIR "/home/cyh/envirements/opencv-3.4.10/build")
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

MESSAGE(STATUS "OpenCV version: "${OpenCV_VERSION})
MESSAGE(${PROJECT_SOURCE_DIR})
MESSAGE(${PROJECT_BINARY_DIR})

include_directories(include)

set(src_dir src/main.cpp)
add_executable(opflow ${src_dir})

target_link_libraries(opflow ${OpenCV_LIBRARIES})
