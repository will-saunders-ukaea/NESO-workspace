cmake_minimum_required(VERSION 3.14)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

# find MPI
find_package(MPI REQUIRED)

# Include hipSYCL
find_package(hipSYCL 0.9.2 QUIET)
if(NOT hipSYCL_FOUND)
  message(WARNING "hipsycl not found")
  find_package(IntelDPCPP REQUIRED)
else()
  message(STATUS "hipsycl found")
  set(HIPSYCL_TARGETS "omp")
endif(NOT hipSYCL_FOUND)

# hipsycl, trisycl and computecpp all define an "add_sycl_to_target" for the
# compilation of a target
if(NOT COMMAND add_sycl_to_target)
  # Note from hipsycl: "Do not call target_sources after add_sycl_to_target or
  # dependency tracking on compiler flags will break in subtle ways"
  function(add_sycl_to_target)

  endfunction()
endif()


# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(EXECUTABLE global_move_bench)
set(EXECUTABLE_SOURCE main.cpp)


if(NOT NESO_PARTICLES_INCLUDE_DIRECTORIES)
  message(ERROR "NESO_PARTICLES_INCLUDE_DIRECTORIES is not defined.")
endif()


add_executable(${EXECUTABLE} ${EXECUTABLE_SOURCE})

target_include_directories(${EXECUTABLE} PRIVATE ${NESO_PARTICLES_INCLUDE_DIRECTORIES})
#target_link_libraries(${EXECUTABLE} PRIVATE ${LINK_LIBRARIES_COMMON})

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})




