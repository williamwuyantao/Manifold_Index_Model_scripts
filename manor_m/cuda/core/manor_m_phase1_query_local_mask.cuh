#pragma once

#include <cstddef>
#include <cstdint>

/* Query-local Phase-1 neighborhood reconstruction.
 *
 * The first Phase-1 pass still computes exactly the same selected model and
 * neighborhood mask for each active query block. It additionally saves the
 * exact float32 radius limit used by adaptive_final_extract_kernel. After the
 * global owner rule is known, the owner aggregation pass reconstructs each
 * query's mask predicate directly from that radius and the same response-index
 * row. Candidate positions are consumed in increasing order, matching the
 * word/ffs traversal of owner_aggregate_kernel_mm. This removes the global
 * N*ceil(k/32) mask without changing model arithmetic or accumulation order.
 */

__global__ void phase1_capture_radius_limit_kernel(
    const float* __restrict__ top_dist2,
    const uint32_t* __restrict__ selected_cut,
    int q0, int B, int tmax,
    float* __restrict__ radius_limit)
{
    int b = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const int anchor = q0 + b;
    const uint32_t cut = selected_cut[anchor];
    if (cut == 0u || cut > static_cast<uint32_t>(tmax)) {
        radius_limit[anchor] = 0.0f;
        return;
    }
    const float radius2 = fmaxf(
        top_dist2[static_cast<size_t>(b) * tmax + cut - 1u], 0.0f);
    radius_limit[anchor] =
        radius2 + 1.0e-6f * fmaxf(1.0f, radius2);
}

__device__ __forceinline__ int phase1_query_local_idx_at(
    const int* index, int anchor, int position, int k,
    bool shared, int anchor_base)
{
    return shared ? index[position]
                  : index[static_cast<size_t>(anchor - anchor_base) * k + position];
}

__global__ __launch_bounds__(32, 4)
void phase1_owner_aggregate_from_radius_kernel_mm(
    const float* __restrict__ radius_limit,
    const int32_t* __restrict__ selected_scale,
    const int32_t* __restrict__ response_counts,
    const uint8_t* __restrict__ owner_mask,
    const float* __restrict__ tangent_raw,
    const float* __restrict__ center_raw,
    const float* __restrict__ x,
    const int* __restrict__ index,
    int D, int MM, int N, int k,
    bool index_is_shared_slice,
    int q0, int B,
    float* __restrict__ covariance_out,
    float* __restrict__ center_sum_out,
    int32_t* __restrict__ count_out,
    int index_anchor_base = 0)
{
    const unsigned full_mask = 0xffffffffu;
    const int b = static_cast<int>(blockIdx.x);
    if (b >= B) return;
    const int lane = threadIdx.x;
    const int anchor = q0 + b;
    float* covariance = covariance_out + static_cast<size_t>(b) * D * D;
    float* center_sum = center_sum_out + static_cast<size_t>(b) * D;

    if (!owner_mask[anchor]) {
        for (int element = lane; element < D * D; element += 32)
            covariance[element] = 0.0f;
        for (int d = lane; d < D; d += 32) center_sum[d] = 0.0f;
        if (lane == 0) count_out[b] = 0;
        return;
    }

    extern __shared__ float dynamic_shared[];
    float* shared_covariance = dynamic_shared;
    float* shared_center = shared_covariance + D * D;
    __shared__ int shared_count;
    for (int element = lane; element < D * D; element += 32)
        shared_covariance[element] = 0.0f;
    for (int d = lane; d < D; d += 32) shared_center[d] = 0.0f;
    if (lane == 0) shared_count = 0;
    __syncwarp();

    const int scale = selected_scale[anchor];
    const int response_count = scale >= 0 ? response_counts[scale] : 0;
    const float limit = radius_limit[anchor];
    const float* query = x + static_cast<size_t>(anchor) * D;
    int local_count = 0;
    for (int base = 0; base < response_count; base += 32) {
        const int candidate = base + lane;
        bool included = false;
        if (candidate < response_count) {
            const int source = phase1_query_local_idx_at(
                index, anchor, candidate, k, index_is_shared_slice,
                index_anchor_base);
            float distance2 = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float difference =
                    x[d + static_cast<size_t>(source) * D] - query[d];
                distance2 += difference * difference;
            }
            included = distance2 <= limit;
        }
        uint32_t bits = __ballot_sync(full_mask, included);
        while (bits) {
            const int bit = __ffs(static_cast<int>(bits)) - 1;
            bits &= bits - 1u;
            const int position = base + bit;
            const int source = phase1_query_local_idx_at(
                index, anchor, position, k, index_is_shared_slice,
                index_anchor_base);
            if (static_cast<unsigned>(source) >= static_cast<unsigned>(N)
                    || !owner_mask[source]) {
                continue;
            }
            ++local_count;
            for (int d = lane; d < D; d += 32)
                shared_center[d] +=
                    center_raw[d + static_cast<size_t>(source) * D];
            for (int column = 0; column < MM; ++column) {
                const float* source_tangent = tangent_raw
                    + static_cast<size_t>(D)
                        * (column + static_cast<size_t>(MM) * source);
                for (int element = lane; element < D * D; element += 32) {
                    const int row = element / D;
                    const int other = element - row * D;
                    shared_covariance[element] +=
                        source_tangent[row] * source_tangent[other];
                }
            }
        }
    }
    if (lane == 0) shared_count = local_count;
    __syncwarp();

    for (int element = lane; element < D * D; element += 32)
        covariance[element] = shared_covariance[element];
    for (int d = lane; d < D; d += 32)
        center_sum[d] = shared_center[d];
    if (lane == 0) count_out[b] = shared_count;
}
