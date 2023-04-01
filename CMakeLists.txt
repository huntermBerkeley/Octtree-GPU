cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(poggers VERSION 0.1.0 LANGUAGES CXX CUDA)



###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)

#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)

find_package(CUDAToolkit 11.6 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

#package manager
include(cmake/CPM.cmake)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)

## Package Target

set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/downloaded_libraries")

## Package Target

CPMAddPackage(
  NAME poggers
  GITHUB_REPOSITORY huntermBerkeley/poggers
  GIT_TAG origin/main
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 

option(POGGERS_BUILD_TESTS "Configure CMake to build tests" OFF)


ConfigureExecutable(cdpQuadtree.cu "${CMAKE_CURRENT_SOURCE_DIR}/cdpQuadtree.cu" "${CMAKE_BINARY_DIR}")


