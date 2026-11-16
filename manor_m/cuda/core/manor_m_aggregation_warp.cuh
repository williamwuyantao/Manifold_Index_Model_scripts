#pragma once

// Experimental work redistribution, not a reduction over neighbours. Each
// covariance entry still consumes exactly the original (neighbour, output)
// sequence; only the lane that owns that entry changes. D=10/20 are dispatched
// explicitly. Other dimensions keep the legacy kernel.
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#define MANOR_M_AGG_HD __host__ __device__
#else
#define MANOR_M_AGG_HD
#endif

template <int D>
struct OwnerAggregationWarpLayout {
    static_assert(D > 0 && D <= 32, "One gradient vector must fit in a warp");
    static constexpr int entries_per_lane = (D * D + 31) / 32;
    MANOR_M_AGG_HD static constexpr int entry(int lane, int slot) {
        return lane + 32 * slot;
    }
    MANOR_M_AGG_HD static constexpr bool valid(int lane, int slot) {
        return entry(lane, slot) < D * D;
    }
};

#undef MANOR_M_AGG_HD

#if defined(__CUDACC__)
template <int D, int WarpsPerBlock>
__global__ void owner_aggregation_warp_accumulate_kernel(
    const float* __restrict__ grad, const float* __restrict__ center_pre,
    const int* __restrict__ owner_index, const int* __restrict__ owner_lookup,
    const uint8_t* __restrict__ heavy, const uint32_t* __restrict__ neigh_mask_bits,
    const int* __restrict__ idx_gpu, const int* __restrict__ index_rows,
    int M, int actual_D, int MY, int kSlice,
    int mask_words, bool idx_shared_slice, int* __restrict__ agg_count,
    float* __restrict__ agg_cacc, float* __restrict__ agg_cov,
    const int* __restrict__ stopped)
{
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 8, "Invalid block size");
    const int lane = threadIdx.x & 31;
    const size_t warp_owner = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + (threadIdx.x >> 5);
    // These exits are uniform across each complete warp. Other warps in the
    // block may proceed; there is deliberately no block-wide barrier.
    if (*stopped || warp_owner >= static_cast<size_t>(M) || actual_D != D) return;
    const int q = static_cast<int>(warp_owner);
    const int index_anchor = index_rows ? index_rows[q] : q;
    (void)owner_index;
    using Layout = OwnerAggregationWarpLayout<D>;
    float covariance[Layout::entries_per_lane];
    int rows[Layout::entries_per_lane], columns[Layout::entries_per_lane];
#pragma unroll
    for (int slot = 0; slot < Layout::entries_per_lane; ++slot) {
        const int entry = Layout::entry(lane, slot);
        covariance[slot] = 0.0f;
        // Even inactive tail lanes participate in both shuffles. Give them
        // valid source lanes, then suppress only their arithmetic and stores.
        rows[slot] = entry < D * D ? entry / D : 0;
        columns[slot] = entry < D * D ? entry % D : 0;
    }
    float center = 0.0f;
    int count = 0;
    const uint32_t* mask = neigh_mask_bits + static_cast<size_t>(q) * mask_words;
    for (int word_index = 0; word_index < mask_words; ++word_index) {
        uint32_t word = mask[word_index];
        while (word) {
            const int bit = __ffs(static_cast<int>(word)) - 1;
            const int s = (word_index << 5) + bit;
            word &= word - 1u;
            if (s >= kSlice) continue;
            const int j = idx_gpu[idx_shared_slice ? s :
                static_cast<size_t>(index_anchor) * kSlice + s];
            if (!heavy[j]) continue;
            const int source_q = owner_lookup[j];
            if (source_q < 0) continue;
            if (lane == 0) ++count;
            if (lane < D) center += center_pre[static_cast<size_t>(source_q) * D + lane];
            for (int out = 0; out < MY; ++out) {
                const float value = lane < D ? grad[
                    static_cast<size_t>(source_q) * MY * D + static_cast<size_t>(out) * D + lane] : 0.0f;
#pragma unroll
                for (int slot = 0; slot < Layout::entries_per_lane; ++slot) {
                    const float row = __shfl_sync(0xffffffffu, value, rows[slot]);
                    const float column = __shfl_sync(0xffffffffu, value, columns[slot]);
                    if (Layout::valid(lane, slot))
                        covariance[slot] += row * column;
                }
            }
        }
    }
    if (lane < D) agg_cacc[static_cast<size_t>(q) * D + lane] = center;
#pragma unroll
    for (int slot = 0; slot < Layout::entries_per_lane; ++slot) {
        if (Layout::valid(lane, slot))
            agg_cov[static_cast<size_t>(q) * D * D + Layout::entry(lane, slot)] = covariance[slot];
    }
    if (lane == 0) agg_count[q] = count;
}
#endif
