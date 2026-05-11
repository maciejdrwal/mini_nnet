cmake_minimum_required(VERSION 3.14)

project(MiniNeuralNet VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SOURCE_PATH "${PROJECT_SOURCE_DIR}")
set(INCLUDE_PATH "${PROJECT_SOURCE_DIR}")
file(GLOB HPPS "${SOURCE_PATH}/*.h")
file(GLOB CPPS "${SOURCE_PATH}/*.cpp")
list(REMOVE_ITEM CPPS "${SOURCE_PATH}/main.cpp")
						  
add_library(${PROJECT_NAME}_objects OBJECT ${CPPS} ${HPPS})					  
add_executable(${PROJECT_NAME} $<TARGET_OBJECTS:${PROJECT_NAME}_objects> "${SOURCE_PATH}/main.cpp")

target_include_directories(${PROJECT_NAME} PUBLIC "${INCLUDE_PATH}")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
