cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)
project(matmul_lyc_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

set(MATMUL_LYC_KERNEL op_kernel/matmul_lyc_custom_kernel.cpp)
# .cpp defaults to CXX; this file contains AscendC device code.
set_source_files_properties(${MATMUL_LYC_KERNEL} PROPERTIES LANGUAGE ASC)

add_library(matmul_lyc_custom_ops SHARED
    ${MATMUL_LYC_KERNEL}
    op_host/matmul_lyc_custom_host.cpp
    op_host/matmul_lyc_custom_tiling.cpp
    op_extension/matmul_lyc_custom_torch.cpp
    op_extension/register.cpp
)

target_include_directories(matmul_lyc_custom_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${ACL_INCLUDE_DIR}
)

target_link_directories(matmul_lyc_custom_ops PRIVATE
    ${ACL_LIB_DIR}
    "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64"
    ${TORCH_NPU_LIBRARIES}
)

target_link_libraries(matmul_lyc_custom_ops PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
    ascendcl
    ascendc_runtime
)

target_compile_options(matmul_lyc_custom_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2002>
)

message(STATUS "Building libmatmul_lyc_custom_ops.so for Ascend310P3 (DAV_2002)")
