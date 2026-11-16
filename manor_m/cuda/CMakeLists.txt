cmake_minimum_required(VERSION 3.21)

project(manor_m_cuda LANGUAGES CXX CUDA)

set(MANOR_M_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures for this target (for example 89 or 90)")
option(MANOR_M_BUILD_MATLAB "Build the MATLAB MEX adapter" OFF)
option(MANOR_M_STATIC_GNU_RUNTIME
    "Link the GNU C++ runtime into the shared core/adapters for cluster portability"
    OFF)
option(MANOR_M_REGRESSION_SHARED_SOLVE
    "Keep the Phase-2 small double solve in per-block shared memory"
    ON)
option(MANOR_M_REGRESSION_WARP_SYNC
    "Experimental single-warp regression synchronization; arithmetic unchanged"
    OFF)
option(MANOR_M_REGRESSION_COOPERATIVE_SOLVE
    "Experimental ordered Cholesky columns and independent RHS solves across lanes"
    OFF)
option(MANOR_M_REGRESSION_PARALLEL_SYSTEM
    "Experimental independent regression matrix/RHS assembly across lanes"
    OFF)
option(MANOR_M_REGRESSION_SHUFFLE_INPUTS
    "Experimental ordered neighbour regression with single-warp register exchange"
    OFF)
option(MANOR_M_REGRESSION_WARP_ENTRIES
    "Experimental D10/D20 regression covariance entries distributed across a full warp"
    OFF)
option(MANOR_M_AGGREGATION_WARP_GATHER
    "Experimental warp exchange of gradient columns; accumulation order unchanged"
    OFF)
option(MANOR_M_AGGREGATION_WARP_ENTRIES
    "Experimental D10/D20 covariance entries distributed across complete warps"
    OFF)
option(MANOR_M_AGGREGATION_TILED_MASK
    "Rebuild exact neighbor masks inside bounded aggregation-owner tiles"
    ON)
option(MANOR_M_GENERATED_PREPARED_RESPONSE
    "Reuse a prepared response-kNN layout inside generated Phase 2"
    OFF)
option(MANOR_M_HOST_INDEX_VECTOR_CHECK
    "Experimental vectorizable full range check for borrowed resident int32 indices"
    OFF)
option(MANOR_M_STREAM_OWNER_ROWS
    "Gather compact-owner index rows directly into bounded pinned tiles"
    ON)
option(MANOR_M_DEVTAN_ORTHONORMAL_FP64
    "Versioned DevTan v2: orthonormalize predicted evaluator basis in FP64"
    ON)
# Fixed-order metric sums are mandatory in all new core/MEX builds.
# Fail explicitly on an obsolete cache or command-line opt-out.
if(DEFINED MANOR_M_EVAL_FIXED_ORDER AND NOT MANOR_M_EVAL_FIXED_ORDER)
    message(FATAL_ERROR "Unordered FP32 metric sums were retired; remove the OFF override")
endif()
set(MANOR_M_EVAL_MIN_BLOCKS_PER_SM "3" CACHE STRING
    "Experimental launch-bound floor for the 256-thread fused evaluator (0 disables it)")
set(MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM "24" CACHE STRING
    "Experimental Phase-1 selector launch-bound floor; 24 preserves the verified default")
set(MANOR_M_GNU_RUNTIME_LIBRARY_DIR "" CACHE PATH
    "Optional directory containing static libstdc++.a/libgcc.a archives")

if(NOT MANOR_M_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
        "Set MANOR_M_CUDA_ARCHITECTURES explicitly (for example 90-real). "
        "Real/SASS targets avoid driver-dependent PTX JIT surprises.")
endif()

add_library(manor_m_cuda SHARED
    core/manor_m_cuda.cu
    core/manor_m_phase1_port.cu
    core/manor_m_phase2_port.cu)
target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_CUDA_BUILDING_LIBRARY=1)
if(MANOR_M_REGRESSION_SHARED_SOLVE)
    target_compile_definitions(manor_m_cuda PRIVATE
        MANOR_M_REGRESSION_SHARED_SOLVE=1)
endif()
if(MANOR_M_REGRESSION_WARP_SYNC)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_REGRESSION_WARP_SYNC=1)
endif()
if(MANOR_M_REGRESSION_COOPERATIVE_SOLVE)
    if(NOT MANOR_M_REGRESSION_SHARED_SOLVE)
        message(FATAL_ERROR "Cooperative regression requires shared solve storage")
    endif()
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_REGRESSION_COOPERATIVE_SOLVE=1)
endif()
if(MANOR_M_REGRESSION_SHUFFLE_INPUTS)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_REGRESSION_SHUFFLE_INPUTS=1)
endif()
if(MANOR_M_REGRESSION_WARP_ENTRIES)
    if(NOT MANOR_M_REGRESSION_SHUFFLE_INPUTS)
        message(FATAL_ERROR "Warp-entry regression requires shuffled regression inputs")
    endif()
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_REGRESSION_WARP_ENTRIES=1)
endif()
if(MANOR_M_REGRESSION_PARALLEL_SYSTEM)
    if(NOT MANOR_M_REGRESSION_COOPERATIVE_SOLVE)
        message(FATAL_ERROR "Parallel regression assembly requires cooperative solve")
    endif()
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_REGRESSION_PARALLEL_SYSTEM=1)
endif()
if(MANOR_M_AGGREGATION_WARP_GATHER)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_AGGREGATION_WARP_GATHER=1)
endif()
if(MANOR_M_AGGREGATION_WARP_ENTRIES)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_AGGREGATION_WARP_ENTRIES=1)
endif()
if(MANOR_M_AGGREGATION_TILED_MASK)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_AGGREGATION_TILED_MASK=1)
endif()
if(MANOR_M_GENERATED_PREPARED_RESPONSE)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_GENERATED_PREPARED_RESPONSE=1)
endif()
if(MANOR_M_HOST_INDEX_VECTOR_CHECK)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_HOST_INDEX_VECTOR_CHECK=1)
endif()
if(MANOR_M_STREAM_OWNER_ROWS)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_STREAM_OWNER_ROWS=1)
endif()
if(MANOR_M_DEVTAN_ORTHONORMAL_FP64)
    target_compile_definitions(manor_m_cuda PRIVATE MANOR_M_DEVTAN_ORTHONORMAL_FP64=1)
endif()
if(NOT MANOR_M_EVAL_MIN_BLOCKS_PER_SM STREQUAL "0")
    target_compile_definitions(manor_m_cuda PRIVATE
        MANOR_M_EVAL_MIN_BLOCKS_PER_SM=${MANOR_M_EVAL_MIN_BLOCKS_PER_SM})
endif()
if(NOT MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM MATCHES "^[0-9]+$" OR
   MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM LESS 1 OR
   MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM GREATER 32)
    message(FATAL_ERROR "MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM must be an integer in [1,32]")
endif()
target_compile_definitions(manor_m_cuda PRIVATE
    MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM=${MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM})
target_include_directories(manor_m_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>)
target_compile_features(manor_m_cuda PUBLIC cxx_std_17)
# Numerical parity is defined against build_manor_m_mex.m, whose production
# NVCC contract is -O3 --use_fast_math -lineinfo.  Keep the shared core on the
# same contract: changing these flags changes Phase-1 owner aggregation even
# when every discrete selection decision is identical.
target_compile_options(manor_m_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>)
set_target_properties(manor_m_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden)
set_property(TARGET manor_m_cuda PROPERTY
    CUDA_ARCHITECTURES "${MANOR_M_CUDA_ARCHITECTURES}")

# MATLAB R2024a ships libstdc++ through GLIBCXX_3.4.28, while several current
# clusters build CUDA host code with GCC 11+.  A MEX linked to that newer
# dynamic runtime compiles successfully and then fails at load time.  The
# opt-in static runtime keeps the adapter self-contained without replacing or
# preloading MATLAB's process-wide libstdc++.  Hide archive symbols so they do
# not interpose on MATLAB or another extension.
if(MANOR_M_STATIC_GNU_RUNTIME)
    if(MANOR_M_GNU_RUNTIME_LIBRARY_DIR)
        if(NOT EXISTS "${MANOR_M_GNU_RUNTIME_LIBRARY_DIR}/libstdc++.a")
            message(FATAL_ERROR
                "MANOR_M_GNU_RUNTIME_LIBRARY_DIR lacks libstdc++.a: "
                "${MANOR_M_GNU_RUNTIME_LIBRARY_DIR}")
        endif()
        target_link_directories(manor_m_cuda BEFORE PRIVATE
            "${MANOR_M_GNU_RUNTIME_LIBRARY_DIR}")
    endif()
    target_link_options(manor_m_cuda PRIVATE
        -static-libstdc++
        -static-libgcc
        LINKER:--exclude-libs,ALL)
endif()

if(MANOR_M_BUILD_MATLAB)
    find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY MEX_COMPILER)
    matlab_add_mex(
        NAME manor_m_cuda_mex
        SRC matlab/manor_m_cuda_mex.cpp
        LINK_TO manor_m_cuda)
    target_include_directories(manor_m_cuda_mex PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core)
    set_target_properties(manor_m_cuda_mex PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
    if(MANOR_M_STATIC_GNU_RUNTIME)
        if(MANOR_M_GNU_RUNTIME_LIBRARY_DIR)
            target_link_directories(manor_m_cuda_mex BEFORE PRIVATE
                "${MANOR_M_GNU_RUNTIME_LIBRARY_DIR}")
        endif()
        target_link_options(manor_m_cuda_mex PRIVATE
            -static-libstdc++
            -static-libgcc
            LINKER:--exclude-libs,ALL)
    endif()
    add_custom_command(TARGET manor_m_cuda_mex POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:manor_m_cuda>
            $<TARGET_FILE_DIR:manor_m_cuda_mex>)
endif()

install(TARGETS manor_m_cuda LIBRARY DESTINATION lib)
install(FILES core/manor_m_cuda.h DESTINATION include)
