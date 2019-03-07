cmake_minimum_required(VERSION 2.8.3)
project(model_matching)

set(SUPER4PCS_DIR ${PROJECT_SOURCE_DIR}/include/super4pcs/)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
find_package (PCL)

include_directories(${PROJECT_SOURCE_DIR}/include ${SUPER4PCS_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
add_subdirectory(${SUPER4PCS_DIR}/accelerators)

add_library(${PROJECT_NAME}
  include/super4pcs/pairCreationFunctor.h
  src/rgbd.cpp
  src/pose_clustering.cpp
  src/stocs.cpp)

target_link_libraries (${PROJECT_NAME} super4pcs_accel ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(model_preprocess src/model_preprocess.cpp) 
target_link_libraries (model_preprocess ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

add_executable(stocs_single src/stocs_match_one_object.cpp) 
target_link_libraries (stocs_single ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})