cmake_minimum_required(VERSION 3.16)
project(ascend_smoke LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.0.0" CACHE PATH "Actual CANN installation")
set(SOC_VERSION "Ascend910B1" CACHE STRING "Value verified using aclrtGetSocName")
set(RUN_MODE "npu" CACHE STRING "Real NPU execution only")
if(NOT RUN_MODE STREQUAL "npu")
    message(FATAL_ERROR "This probe only supports RUN_MODE=npu; CPU/simulator output is not accepted")
endif()
set(ASCENDC_CMAKE_FILE "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    CACHE FILEPATH "Kernel Launch CMake entry observed in the installed SDK")
if(NOT EXISTS "${ASCENDC_CMAKE_FILE}")
    message(FATAL_ERROR "Ascend C CMake entry not found: ${ASCENDC_CMAKE_FILE}")
endif()
include("${ASCENDC_CMAKE_FILE}")

ascendc_library(ascend_smoke_kernel STATIC kernel.cpp)
add_executable(ascend_smoke main.cpp)
target_link_libraries(ascend_smoke PRIVATE ascend_smoke_kernel host_intf_pub)

