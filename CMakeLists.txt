cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0148 OLD)

project(robust_semantic)

# set this to path to AirSim root folder if you want your catkin workspace in a custom directory
set(AIRSIM_ROOT /media/david/AirSim)
add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)


set(CMAKE_CXX_STANDARD 17)
set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs 
-l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
-lm -lc -lgcc_s -lgcc  
-lstdc++fs -fmax-errors=10")
# Add -o3 to compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.3.0/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")


## Find system libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
    roscpp
    std_srvs
    sensor_msgs
    ipp_tools
)

catkin_package(
    INCLUDE_DIRS include
    ${EIGEN3_INCLUDE_DIRS}
    LIBRARIES
    CATKIN_DEPENDS roscpp std_srvs sensor_msgs ipp_tools
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${AIRSIM_ROOT}/AirLib/deps/eigen3
    ${AIRSIM_ROOT}/AirLib/include
    ${RPC_LIB_INCLUDES}
    ${AIRSIM_ROOT}/MavLinkCom/include
    ${AIRSIM_ROOT}/MavLinkCom/common_utils
    # Set because Boost is an internal dependency, not transitive.
    ${Boost_INCLUDE_DIRS}
)

add_executable(robust_semantic_node
    src/robust_semantic.cpp
    src/robust_semantic_node.cpp
)

target_link_libraries(robust_semantic_node
    ${catkin_LIBRARIES}
    Eigen3::Eigen
    AirLib
)

