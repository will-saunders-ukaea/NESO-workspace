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

if( NESO_PARTICLES_DEVICE_TYPE STREQUAL CPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
else()
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
endif()

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


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

#add_compile_options(-O3)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG -O3")
message(STATUS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})


set(EXECUTABLE global_move_bench)
set(EXECUTABLE_SOURCE main.cpp)


if(NOT NESO_PARTICLES_ROOT)
  message(ERROR ": NESO_PARTICLES_ROOT is not defined.")
endif()

find_package(MPI REQUIRED)

add_executable(${EXECUTABLE} ${EXECUTABLE_SOURCE})

target_include_directories(${EXECUTABLE} PRIVATE ${NESO_PARTICLES_ROOT}/include
                                                 ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${EXECUTABLE} PRIVATE ${MPI_CXX_LINK_FLAGS}
                                            ${MPI_CXX_LIBRARIES})

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})
