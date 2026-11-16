cmake_minimum_required(VERSION 3.21)
project(manor_m_multi_gpu LANGUAGES CXX CUDA)

# Build a separate library. The existing target's source files and numerical
# flags are reused; its original build and installed library remain unchanged.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." single_gpu)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
option(MANOR_M_MULTI_CACHED_GRAM "Reuse per-state tangent Gram entries in readout" OFF)
option(MANOR_M_MULTI_PERSISTENT_READOUT "Retain readout buffers through a fit" OFF)
if(MANOR_M_MULTI_PERSISTENT_READOUT)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_MULTI_PERSISTENT_READOUT=1)
endif()
if(MANOR_M_MULTI_CACHED_GRAM)
    set(multi_readout "${CMAKE_CURRENT_BINARY_DIR}/multi_cached_readout.cuh")
    execute_process(COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/readout_source.py"
        "${CMAKE_CURRENT_LIST_DIR}/../../matlab/cuda_mex/manor_m_phase2_mex.cu"
        "${multi_readout}" COMMAND_ERROR_IS_FATAL ANY)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_MULTI_CACHED_GRAM=1)
    target_include_directories(manor_m_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/readout_source.py"
        "${CMAKE_CURRENT_LIST_DIR}/../../matlab/cuda_mex/manor_m_phase2_mex.cu")
endif()
set(multi_phase1 "${CMAKE_CURRENT_BINARY_DIR}/multi_phase1.cu")
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/phase1_source.py"
    "${CMAKE_CURRENT_LIST_DIR}/../core/manor_m_phase1_port.cu"
    "${multi_phase1}" "${CMAKE_CURRENT_LIST_DIR}/phase1_multi.cuh"
    COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/phase1_source.py"
    "${CMAKE_CURRENT_LIST_DIR}/phase1_multi.cuh"
    "${CMAKE_CURRENT_LIST_DIR}/../core/manor_m_phase1_port.cu")
set_property(TARGET manor_m_cuda PROPERTY SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/../core/manor_m_cuda.cu"
    "${multi_phase1}"
    "${CMAKE_CURRENT_LIST_DIR}/phase2.cu")
set_target_properties(manor_m_cuda PROPERTIES OUTPUT_NAME manor_m_cuda_multi)
find_package(Threads REQUIRED)
target_link_libraries(manor_m_cuda PRIVATE Threads::Threads)
