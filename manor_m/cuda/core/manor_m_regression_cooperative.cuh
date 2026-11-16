#pragma once

#ifndef MANOR_M_REGRESSION_PARALLEL_SYSTEM
#define MANOR_M_REGRESSION_PARALLEL_SYSTEM 0
#endif

// Each matrix/RHS entry has one writer. The only reduction (the trace) keeps
// the original lane-zero order. aa00 and bb0 are valid only in lane zero.
template <int MAXD, bool Retry>
__device__ __forceinline__ void owner_regression_assemble_parallel(
    double* matrix, double* rhs, const float* gathered, int D, int MY,
    float aa00, const float* bb0, float lambda_ridge, float jitter_scale,
    double* shared_jitter)
{
    constexpr unsigned mask = 0xffffffffu >> (32 - MAXD);
    const int lane = threadIdx.x, nsys = D + 1;
    if constexpr (!Retry) {
        if (lane == 0) {
            double trace = aa00;
            for (int a = 0; a < D; ++a) for (int c = 0; c < D; ++c) {
                const double value = gathered[2 * D + a * D + c];
                if (a == c) trace += value;
            }
            *shared_jitter = lambda_ridge + jitter_scale * trace / fmax((double)nsys, 1.0);
        }
    }
    __syncwarp(mask);
    const double jitter = *shared_jitter;
    double diagonal_add = jitter;
    if constexpr (Retry) {
        const double add = 100.0 * jitter + 1e-8;
        diagonal_add = jitter + add;
    }
    // Every entry is assigned, so the old serial zero-fill is unnecessary.
    for (int t = lane; t < nsys * nsys; t += MAXD) {
        const int a = t / nsys, c = t % nsys;
        double value;
        if (t == 0) value = aa00; // t==0 is owned by lane zero.
        else if (a == 0) value = gathered[c - 1];
        else if (c == 0) value = gathered[a - 1];
        else value = gathered[2 * D + (a - 1) * D + c - 1];
        matrix[t] = value;
        if (a == c) matrix[t] += diagonal_add;
    }
    if (lane == 0) for (int out = 0; out < MY; ++out) rhs[out] = bb0[out];
    for (int t = lane; t < D * MY; t += MAXD)
        rhs[MY + t] = gathered[2 * D + D * D + t];
    // The caller must synchronize before the Cholesky solve consumes writes.
}

// Candidate only. Each dot product keeps the serial MEX k-loop order.
// Parallelism is across independent Cholesky column entries and RHS columns,
// not across summands. Partial warps name exactly the launched MAXD lanes.
template <int MAXD>
__device__ __forceinline__ bool owner_cholesky_cooperative(
    double* A, double* B, int nsys, int nrhs, int* failed)
{
    static_assert(MAXD <= 32, "cooperative solve requires a single warp");
    constexpr unsigned mask = 0xffffffffu >> (32 - MAXD);
    const int lane = threadIdx.x;
    if (lane == 0) *failed = 0;
    __syncwarp(mask);
    for (int j = 0; j < nsys; ++j) {
        if (lane == 0) {
            double sum = A[j * nsys + j];
            for (int k = 0; k < j; ++k)
                sum -= A[j * nsys + k] * A[j * nsys + k];
            if (!(sum > 1e-30)) *failed = 1;
            else A[j * nsys + j] = sqrt(sum);
        }
        __syncwarp(mask);
        if (*failed) return false;
        for (int i = j + 1 + lane; i < nsys; i += MAXD) {
            double s = A[i * nsys + j];
            for (int k = 0; k < j; ++k)
                s -= A[i * nsys + k] * A[j * nsys + k];
            A[i * nsys + j] = s / A[j * nsys + j];
        }
        __syncwarp(mask);
    }
    for (int rhs = lane; rhs < nrhs; rhs += MAXD) {
        for (int i = 0; i < nsys; ++i) {
            double s = B[i * nrhs + rhs];
            for (int k = 0; k < i; ++k)
                s -= A[i * nsys + k] * B[k * nrhs + rhs];
            B[i * nrhs + rhs] = s / A[i * nsys + i];
        }
        for (int i = nsys - 1; i >= 0; --i) {
            double s = B[i * nrhs + rhs];
            for (int k = i + 1; k < nsys; ++k)
                s -= A[k * nsys + i] * B[k * nrhs + rhs];
            B[i * nrhs + rhs] = s / A[i * nsys + i];
        }
    }
    __syncwarp(mask);
    return true;
}


template <int MAXD, bool ParallelSystem = (MANOR_M_REGRESSION_PARALLEL_SYSTEM != 0)>
__device__ __forceinline__ void owner_regression_cooperative_solve(
    unsigned char* storage, const float* gathered, const float* tangent,
    int D, int MM, int MY, int local_q, int owner, float aa00,
    const float* bb0, float wsum, int nnei, float beta,
    float lambda_ridge, float jitter_scale, int regression_mode,
    const float* centers, float* weight_sum, int32_t* num_neigh,
    uint8_t* solver_failed, float* center_pre, float* grad_out)
{
    static_assert(MAXD <= 32, "fallback to serial solve for larger dimensions");
    constexpr unsigned mask = 0xffffffffu >> (32 - MAXD);
    const int lane = threadIdx.x, nsys = D + 1;
    __shared__ int has_system, failed;
    __shared__ double shared_jitter;
    if (lane == 0) {
        weight_sum[local_q] = wsum;
        num_neigh[local_q] = nnei;
        solver_failed[local_q] = 0u;
        for (int d = 0; d < D; ++d)
            center_pre[(size_t)local_q * D + d] = nnei > 0 ?
                gathered[D + d] / (float)nnei : centers[(size_t)owner * D + d];
        has_system = nnei >= 2 * D;
    }
    __syncwarp(mask);
    if (!has_system) return;
    const size_t gathered_bytes = sizeof(float) * (
        (size_t)2 * D + (size_t)D * D + (size_t)D * MY);
    const size_t solve_offset = (gathered_bytes + alignof(double) - 1) &
        ~(static_cast<size_t>(alignof(double)) - 1);
    double* matrix = reinterpret_cast<double*>(storage + solve_offset);
    double* rhs = matrix + (size_t)nsys * nsys;
    if constexpr (ParallelSystem) {
        owner_regression_assemble_parallel<MAXD, false>(matrix, rhs, gathered,
            D, MY, aa00, bb0, lambda_ridge, jitter_scale, &shared_jitter);
    } else
    if (lane == 0) {
        for (int t = 0; t < nsys * nsys; ++t) matrix[t] = 0.0;
        matrix[0] = aa00;
        double trace = aa00;
        for (int d = 0; d < D; ++d) {
            matrix[d + 1] = gathered[d];
            matrix[(d + 1) * nsys] = gathered[d];
        }
        for (int a = 0; a < D; ++a) for (int c = 0; c < D; ++c) {
            const double value = gathered[2 * D + a * D + c];
            matrix[(a + 1) * nsys + c + 1] = value;
            if (a == c) trace += value;
        }
        shared_jitter = lambda_ridge + jitter_scale * trace / fmax((double)nsys, 1.0);
        for (int t = 0; t < nsys; ++t) matrix[t * nsys + t] += shared_jitter;
        for (int out = 0; out < MY; ++out) rhs[out] = bb0[out];
        for (int d = 0; d < D; ++d) for (int out = 0; out < MY; ++out)
            rhs[(d + 1) * MY + out] = gathered[2 * D + D * D + d * MY + out];
    }
    __syncwarp(mask);
    bool solved = owner_cholesky_cooperative<MAXD>(matrix, rhs, nsys, MY, &failed);
    if (!solved) {
        if constexpr (ParallelSystem) {
            owner_regression_assemble_parallel<MAXD, true>(matrix, rhs, gathered,
                D, MY, aa00, bb0, lambda_ridge, jitter_scale, &shared_jitter);
        } else
        if (lane == 0) {
            const double add = 100.0 * shared_jitter + 1e-8;
            for (int t = 0; t < nsys * nsys; ++t) matrix[t] = 0.0;
            matrix[0] = aa00;
            for (int d = 0; d < D; ++d) {
                matrix[d + 1] = gathered[d];
                matrix[(d + 1) * nsys] = gathered[d];
            }
            for (int a = 0; a < D; ++a) for (int c = 0; c < D; ++c)
                matrix[(a + 1) * nsys + c + 1] = gathered[2 * D + a * D + c];
            for (int t = 0; t < nsys; ++t) matrix[t * nsys + t] += shared_jitter + add;
            for (int out = 0; out < MY; ++out) rhs[out] = bb0[out];
            for (int d = 0; d < D; ++d) for (int out = 0; out < MY; ++out)
                rhs[(d + 1) * MY + out] = gathered[2 * D + D * D + d * MY + out];
        }
        __syncwarp(mask);
        solved = owner_cholesky_cooperative<MAXD>(matrix, rhs, nsys, MY, &failed);
    }
    if (!solved) {
        if (lane == 0) solver_failed[local_q] = 1u;
        for (int t = lane; t < MY * D; t += MAXD)
            grad_out[(size_t)local_q * MY * D + t] = 0.0f;
        return;
    }
    for (int out = lane; out < MY; out += MAXD) {
        float gradient[MAXD];
        if (regression_mode == 1) {
            for (int d = 0; d < D; ++d) {
                float total = 0.0f;
                for (int mm = 0; mm < MM; ++mm)
                    total += tangent[mm * MAXD + d] * (float)rhs[(mm + 1) * MY + out];
                gradient[d] = total;
            }
        } else {
            for (int d = 0; d < D; ++d) gradient[d] = (float)rhs[(d + 1) * MY + out];
            float coefficient[MAX_MM_MANOR];
            for (int mm = 0; mm < MM; ++mm) {
                coefficient[mm] = 0.0f;
                for (int d = 0; d < D; ++d)
                    coefficient[mm] += tangent[mm * MAXD + d] * gradient[d];
            }
            for (int d = 0; d < D; ++d) {
                float correction = 0.0f;
                for (int mm = 0; mm < MM; ++mm)
                    correction += tangent[mm * MAXD + d] * coefficient[mm];
                gradient[d] += beta * correction;
            }
        }
        for (int d = 0; d < D; ++d)
            grad_out[(size_t)local_q * MY * D + (size_t)out * D + d] = gradient[d];
    }
}
