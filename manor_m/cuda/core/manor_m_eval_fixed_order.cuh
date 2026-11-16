#pragma once

#include <cstddef>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define MANOR_EVAL_HD __host__ __device__ __forceinline__
#else
#define MANOR_EVAL_HD inline
#endif


/* Deterministic metric reduction: each query contribution has one writer,
   and the fixed 256-lane tree preserves the configured accumulation order. */
constexpr int manor_eval_sum_threads = 256;

MANOR_EVAL_HD std::size_t manor_eval_point_offset(
    std::size_t slot, int metric, int outputs, std::size_t queries, int query)
{
    return (slot * (static_cast<std::size_t>(outputs) + 4) + metric) *
        queries + static_cast<std::size_t>(query);
}

MANOR_EVAL_HD float manor_eval_sum_add(float left, float right)
{
#ifdef __CUDA_ARCH__
    return __fadd_rn(left, right);
#else
    // Host tests use round-to-nearest and no fast-math/reassociation.
    volatile float rounded = left + right;
    return rounded;
#endif
}

MANOR_EVAL_HD float manor_eval_lane_sum(
    const float* values, std::size_t count, int lane)
{
    float total = 0.0f;
    for (std::size_t query = static_cast<std::size_t>(lane); query < count;
            query += manor_eval_sum_threads)
        total = manor_eval_sum_add(total, values[query]);
    return total;
}

#ifdef __CUDACC__
/* One block per channel; launch dimensions are deliberately part of the
   numerical contract. No global floating-point atomics or dynamic tree. */
__global__ void manor_eval_fixed_order_sum_kernel(
    const float* points, std::size_t queries, int outputs,
    float* sums, float* coordinate_sums, const int* stopped,
    unsigned int metric_mask = ~0u)
{
    if (*stopped) return;
    const std::size_t channel = blockIdx.x;
    const std::size_t width = static_cast<std::size_t>(outputs) + 4;
    const std::size_t slot = channel / width;
    const int metric = static_cast<int>(channel % width);
    // Separate readout families may share a model and destination sums. Each
    // still uses its original query/lane order and writes only its own fields.
    if ((metric_mask & (1u << metric)) == 0) return;
    __shared__ float partial[manor_eval_sum_threads];
    const int lane = threadIdx.x;
    partial[lane] = manor_eval_lane_sum(points + channel * queries,
        queries, lane);
    __syncthreads();
    for (int stride = manor_eval_sum_threads / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            partial[lane] = manor_eval_sum_add(partial[lane],
                partial[lane + stride]);
        __syncthreads();
    }
    if (lane == 0) {
        if (metric < 4) sums[slot * 4 + metric] = partial[0];
        else if (coordinate_sums != nullptr)
            coordinate_sums[slot * outputs + metric - 4] = partial[0];
    }
}
#endif

#undef MANOR_EVAL_HD
