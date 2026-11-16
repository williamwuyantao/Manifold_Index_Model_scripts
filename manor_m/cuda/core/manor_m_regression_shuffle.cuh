#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Candidate only: exchange the same neighbour values through registers in a
// single (possibly partial) warp. Every dot product and neighbour accumulator
// keeps the legacy order. The tangent tile is immutable after the caller's
// initial barrier; no shared scratch is reused between neighbours here.
template <int MAXD, int MAXMY>
__device__ __forceinline__ void owner_regression_shuffle_accumulate(
    const float* Xrm, const float* Yrm, const int* indices,
    const float* tangent, float xi, int local_q, int D, int MM, int MY,
    int kSlice, bool shared_index, float hk2_inv, float beta,
    int regression_mode, uint32_t* mask_row, float& a01_row,
    float& cacc_row, float (&bb1_row)[MAXMY], float (&cov_row)[MAXD],
    float& aa00, float& wsum, int& nnei, float (&bb0)[MAXMY])
{
    static_assert(MAXD > 0 && MAXD <= 32, "Only one launched warp is supported");
    constexpr unsigned mask = 0xffffffffu >> (32 - MAXD);
    const int row = threadIdx.x;
    for (int s = 0; s < kSlice; ++s) {
        const int j = indices[shared_index ? static_cast<size_t>(s) :
            static_cast<size_t>(local_q) * kSlice + s];
        const float xj = row < D ? Xrm[static_cast<size_t>(j) * D + row] : 0.0f;
        const float dx_row = row < D ? xj - xi : 0.0f;
        float projection = 0.0f;
        for (int d = 0; d < D; ++d) {
            // All named lanes execute each shuffle, including row >= MM.
            const float dx = __shfl_sync(mask, dx_row, d);
            if (row < MM) projection += tangent[row * MAXD + d] * dx;
        }
        float z_row = dx_row;
        for (int mm = 0; mm < MM; ++mm) {
            const float p = __shfl_sync(mask, projection, mm);
            if (row < D) z_row += beta * p * tangent[mm * MAXD + row];
        }
        float squared = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float z = __shfl_sync(mask, z_row, d);
            if (row == 0) squared += z * z;
        }
        squared = __shfl_sync(mask, squared * hk2_inv, 0);
        // This branch is warp-uniform. No source lane drops out of a shuffle.
        if (squared >= 1.0f) continue;
        if (row == 0 && mask_row)
            mask_row[s >> 5] |= (1u << (s & 31));
        const float weight = (1.0f - squared) * (1.0f - squared);
        const float y_row = row < MY ? Yrm[static_cast<size_t>(j) * MY + row] : 0.0f;
        const float xreg = regression_mode == 1 ?
            (row < MM ? projection : 0.0f) : z_row;
        if (row == 0) {
            aa00 += weight;
            wsum += weight;
            ++nnei;
        }
        if (row < D) {
            // The frozen sm90 kernel reuses rounded weight*xreg in covariance
            // and adds it to a01 with FADD, not FFMA. Register exchange can
            // otherwise contract this expression even with identical C++ order.
            a01_row = __fadd_rn(a01_row, __fmul_rn(weight, xreg));
            cacc_row += xj;
        }
        for (int out = 0; out < MY; ++out) {
            const float y = __shfl_sync(mask, y_row, out);
            if (row == 0) bb0[out] += weight * y;
            if (row < D) bb1_row[out] += weight * y * xreg;
        }
        for (int c = 0; c < D; ++c) {
            const float zc = __shfl_sync(mask, xreg, c);
            if (row < D) cov_row[c] += weight * xreg * zc;
        }
    }
}
