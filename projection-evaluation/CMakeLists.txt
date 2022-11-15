cmake_minimum_required(VERSION 3.14)

project(
  NESO-Workspace-hybrid-move
  VERSION 0.0.1
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# find MPI
find_package(MPI REQUIRED)
find_package(NESO-PARTICLES REQUIRED)

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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

# add_compile_options(-O3) set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG
# -Ofast")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -DNDEBUG -O3")
message(STATUS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

find_package(NESO-PARTICLES REQUIRED)
include_directories(${NESO_PARTICLES_INCLUDE_PATH})

if(NOT NESO_INCLUDE_DIR)
  message(STATUS "NESO_INCLUDE_DIR not set")
endif()
include_directories(${NESO_INCLUDE_DIR})

find_package(Nektar++ REQUIRED)
add_definitions(${NEKTAR++_DEFINITIONS})
# If you're on a system with CMake >=3.12, also add this ADD_COMPILE_DEFINITIONS
# line:
add_compile_definitions(${NEKTAR++_GENERATED_DEFINITIONS})
include_directories(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
# INCLUDE_DIRECTORIES(./nektar)
link_directories(${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(nektar_move_source main.cpp)
set(EXECUTABLE nektar_move.x)
add_executable(${EXECUTABLE} ${nektar_move_source})
set_target_properties(${EXECUTABLE} PROPERTIES COMPILE_FLAGS
                                               "${MPI_COMPILE_FLAGS}")
target_link_libraries(
  ${EXECUTABLE}
  PRIVATE ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})


message(status ${NEKTAR++_DEFINITIONS})
message(status ${NEKTAR++_LIBRARIES} ${NEKTAR++_TP_LIBRARIES} ${MPI_LIBRARIES}
          ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})
message(status ${MPI_COMPILE_FLAGS})
