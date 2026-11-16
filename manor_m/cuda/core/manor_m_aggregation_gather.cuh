#pragma once

// Called by every row < D, and only those rows, in an owner block. Each row
// loads its own gradient once; the first D lanes exchange exactly those bits.
// Neighbour/output/column accumulation order and FMA expressions are unchanged.
template <int MAXD, bool WarpGather>
__device__ __forceinline__ void owner_accumulate_gradient_row(
    float* covariance, const float* gradient, int row, int D)
{
    const float value = gradient[row];
    if constexpr (WarpGather && MAXD <= 32) {
        // D is validated in [1, MAXD]. A partial warp names no absent lanes;
        // lanes >= D are not named and need not execute this function.
        const unsigned mask = 0xffffffffu >> (32 - D);
        for (int c = 0; c < D; ++c) {
            const float column = __shfl_sync(mask, value, c);
            covariance[c] += value * column;
        }
    } else {
        // Preserve the original load path for larger dimension tiers.
        for (int c = 0; c < D; ++c)
            covariance[c] += value * gradient[c];
    }
}
