#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Candidate path for the D10/D20 regression tiers.  The launch remains one
// warp per owner, but all 32 lanes own covariance entries instead of leaving
// the physical warp's tail idle.  Each entry still receives exactly one
// update per neighbour in the original neighbour order; no dot-product or
// cross-neighbour reduction is repartitioned.
template <int MAXD, int MAXMY>
__device__ __forceinline__ void owner_regression_warp_entry_accumulate(
    const float* Xrm, const float* Yrm, const int* indices,
    const float* tangent, float xi, int local_q, int D, int MM, int MY,
    int kSlice, bool shared_index, float hk2_inv, float beta,
    int regression_mode, uint32_t* mask_row, float& a01_row,
    float& cacc_row, float (&bb1_row)[MAXMY],
    float (&cov_entries)[(MAXD * MAXD + 31) / 32],
    float& aa00, float& wsum, int& nnei, float (&bb0)[MAXMY])
{
    static_assert(MAXD > 0 && MAXD <= 32, "one physical warp is required");
    constexpr unsigned mask = 0xffffffffu;
    constexpr int cov_per_lane = (MAXD * MAXD + 31) / 32;
    const int lane = threadIdx.x;
    for (int s = 0; s < kSlice; ++s) {
        const int j = indices[shared_index ? static_cast<size_t>(s) :
            static_cast<size_t>(local_q) * kSlice + s];
        const float xj = lane < D ? Xrm[static_cast<size_t>(j) * D + lane] : 0.0f;
        const float dx_lane = lane < D ? xj - xi : 0.0f;
        float projection = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float dx = __shfl_sync(mask, dx_lane, d);
            if (lane < MM) projection += tangent[lane * MAXD + d] * dx;
        }
        float z_lane = dx_lane;
        for (int mm = 0; mm < MM; ++mm) {
            const float projected = __shfl_sync(mask, projection, mm);
            if (lane < D) z_lane += beta * projected * tangent[mm * MAXD + lane];
        }
        float squared = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float z = __shfl_sync(mask, z_lane, d);
            if (lane == 0) squared += z * z;
        }
        squared = __shfl_sync(mask, squared * hk2_inv, 0);
        if (squared >= 1.0f) continue;
        if (lane == 0 && mask_row)
            mask_row[s >> 5] |= (1u << (s & 31));
        const float weight = (1.0f - squared) * (1.0f - squared);
        const float y_lane = lane < MY ? Yrm[static_cast<size_t>(j) * MY + lane] : 0.0f;
        const float xreg_lane = regression_mode == 1 ?
            (lane < MM ? projection : 0.0f) : z_lane;
        if (lane == 0) {
            aa00 += weight;
            wsum += weight;
            ++nnei;
        }
        if (lane < D) {
            a01_row = __fadd_rn(a01_row, __fmul_rn(weight, xreg_lane));
            cacc_row += xj;
        }
        for (int out = 0; out < MY; ++out) {
            const float y = __shfl_sync(mask, y_lane, out);
            if (lane == 0) bb0[out] += weight * y;
            if (lane < D) bb1_row[out] += weight * y * xreg_lane;
        }
        // Every lane executes every shuffle. Invalid tail entries still name
        // lane zero but do not update storage.
        for (int p = 0; p < cov_per_lane; ++p) {
            const int entry = lane + 32 * p;
            const bool valid = entry < D * D;
            const int row = valid ? entry / D : 0;
            const int column = valid ? entry - row * D : 0;
            const float xr = __shfl_sync(mask, xreg_lane, row);
            const float xc = __shfl_sync(mask, xreg_lane, column);
            if (valid) cov_entries[p] += weight * xr * xc;
        }
    }
}
