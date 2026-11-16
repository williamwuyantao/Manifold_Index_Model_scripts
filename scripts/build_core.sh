#!/usr/bin/env bash
set -euo pipefail
repository=$(cd "$(dirname "$0")/.." && pwd)
build_directory=${BUILD_DIR:-"$repository/build"}
cuda_compiler=${NVCC:-$(command -v nvcc || true)}
if [[ -z "$cuda_compiler" && -x /usr/local/cuda/bin/nvcc ]]; then
  cuda_compiler=/usr/local/cuda/bin/nvcc
fi
if [[ -z "$cuda_compiler" ]]; then
  echo 'CUDA compiler not found. Set NVCC to the nvcc executable.' >&2
  exit 1
fi
cmake -S "$repository/manor_m/cuda" -B "$build_directory" \
  -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
  -DMANOR_M_CUDA_ARCHITECTURES="${CUDA_ARCH:-native}" \
  -DMANOR_M_BUILD_MATLAB=OFF -DMANOR_M_STATIC_GNU_RUNTIME="${STATIC_GNU_RUNTIME:-OFF}" \
  -DMANOR_M_REGRESSION_SHARED_SOLVE=ON -DMANOR_M_REGRESSION_WARP_SYNC=ON \
  -DMANOR_M_REGRESSION_COOPERATIVE_SOLVE=ON -DMANOR_M_REGRESSION_PARALLEL_SYSTEM=ON \
  -DMANOR_M_REGRESSION_SHUFFLE_INPUTS=ON -DMANOR_M_REGRESSION_WARP_ENTRIES=ON \
  -DMANOR_M_AGGREGATION_WARP_ENTRIES=ON -DMANOR_M_AGGREGATION_TILED_MASK=ON \
  -DMANOR_M_AGGREGATION_WARP_GATHER=OFF -DMANOR_M_GENERATED_PREPARED_RESPONSE=OFF \
  -DMANOR_M_HOST_INDEX_VECTOR_CHECK=ON -DMANOR_M_STREAM_OWNER_ROWS=ON \
  -DMANOR_M_DEVTAN_ORTHONORMAL_FP64=ON \
  -DMANOR_M_EVAL_MIN_BLOCKS_PER_SM=5 -DMANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM=24 \
  '-DCMAKE_CUDA_FLAGS=--ptxas-options=-v -DMANOR_M_JACOBI_READ_BARRIER=1'
cmake --build "$build_directory" --parallel "${BUILD_JOBS:-4}"
"${PYTHON_BIN:-python3}" "$repository/scripts/write_runtime.py" \
  --library "$build_directory/libmanor_m_cuda.so" \
  --output "$build_directory/runtime.json"
