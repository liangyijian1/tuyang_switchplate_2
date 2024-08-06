cmake_minimum_required(VERSION 3.28)
project(tuyang_switchplate_2)

set(CMAKE_CXX_STANDARD 20)

# opencv
set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INLCUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(tuyang_switchplate_2 main.cpp)
target_link_libraries(tuyang_switchplate_2 ${OpenCV_LIBS})
target_link_libraries(tuyang_switchplate_2 ${PCL_LIBRARY})
