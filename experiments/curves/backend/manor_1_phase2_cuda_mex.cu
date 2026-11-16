#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:cuda", "%s (%s:%d)", cudaGetErrorString(code), file, line);
    }
}

#ifndef MAX_D_MANOR
#define MAX_D_MANOR 32
#endif

// Current production path is direct full-slice computation with exact-slice
// cache disabled and shared-sort cache builder disabled. Earlier cache-era
// branches should not be treated as the preferred speed path unless they are
// configured for the current workloads.

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void fullslice_shared_sort_kernel_u16(
    const float* __restrict__ Xrm,
    int D, int N, int q0, int B, int kSlice,
    float* __restrict__ dist2_out,
    uint16_t* __restrict__ rel_out);

struct Phase2Opts {
    float gpu_mem_fraction = 0.92f;
    float reserve_fraction = 0.08f;
    double reserve_bytes   = 6.0 * 1024.0 * 1024.0 * 1024.0;
    float heavy_factor     = 0.05f;
    bool  return_debug     = false;
    float jitter           = 1e-6f;
    int   power_iter_steps = 0; // ignored in exact-Jacobi vector aggregation path
    int   min_query_block  = 512;
    int   max_query_block  = 131072;
    int   exact_slice_cache_mode = 2; // production path hard-codes exact-slice cache off
    double exact_slice_cache_max_bytes = 72.0 * 1024.0 * 1024.0 * 1024.0;
    bool  use_cuda_graphs = true;
    int   shared_sort_mode = 2; // production path hard-codes shared-sort cache builder off
    int   shared_sort_max_kslice = 6144;
    int   warps_per_block = 2;  // supported: 2 or 4
    int   build_block_threads = 256;
    bool  center_update_guard_enable = false;
    float center_update_guard_ratio = 0.90f;
    bool  center_update_guard_use_ess = false;
    bool  center_reduce_tangent_final = true;
    bool  center_kernel_weighted = true;
    int   center_weight_source_mode = 1; // hardcoded: raw X samples
    bool  vector_update_guard_enable = false;
    float vector_update_guard_ratio = 0.90f;
    bool  vector_update_guard_use_ess = false;
    bool  vector_kernel_weighted = false;
    bool  profile_timers = false;
    bool  silent = false;
};

static Phase2Opts parse_opts(const mxArray* opts)
{
    Phase2Opts o;
    if (!mxIsStruct(opts)) return o;
    auto getf = [&](const char* name) -> const mxArray* { return mxGetField(opts, 0, name); };
    if (const mxArray* a = getf("gpu_mem_fraction")) o.gpu_mem_fraction = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("reserve_fraction")) o.reserve_fraction = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("reserve_bytes"))    o.reserve_bytes = mxGetScalar(a);
    if (const mxArray* a = getf("heavy_factor"))     o.heavy_factor = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("return_debug"))     o.return_debug = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("jitter"))           o.jitter = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("power_iter_steps")) o.power_iter_steps = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("min_query_block"))  o.min_query_block = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("max_query_block"))  o.max_query_block = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("exact_slice_cache_mode")) o.exact_slice_cache_mode = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("exact_slice_cache_max_bytes")) o.exact_slice_cache_max_bytes = mxGetScalar(a);
    if (const mxArray* a = getf("use_cuda_graphs")) o.use_cuda_graphs = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("shared_sort_mode")) o.shared_sort_mode = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("shared_sort_max_kslice")) o.shared_sort_max_kslice = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("warps_per_block")) o.warps_per_block = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("build_block_threads")) o.build_block_threads = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("center_update_guard_enable")) o.center_update_guard_enable = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("center_update_guard_ratio")) o.center_update_guard_ratio = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("center_reduce_tangent_final")) o.center_reduce_tangent_final = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("vector_update_guard_enable")) o.vector_update_guard_enable = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("vector_update_guard_ratio")) o.vector_update_guard_ratio = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("profile_timers")) o.profile_timers = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("silent")) o.silent = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (o.warps_per_block != 2 && o.warps_per_block != 4) o.warps_per_block = 4;
    if (o.build_block_threads != 128 && o.build_block_threads != 256 && o.build_block_threads != 512) o.build_block_threads = 256;
    if (!std::isfinite(o.center_update_guard_ratio)) o.center_update_guard_ratio = 0.90f;
    o.center_update_guard_ratio = fminf(1.0f, fmaxf(0.0f, o.center_update_guard_ratio));
    if (!std::isfinite(o.vector_update_guard_ratio)) o.vector_update_guard_ratio = 0.90f;
    o.vector_update_guard_ratio = fminf(1.0f, fmaxf(0.0f, o.vector_update_guard_ratio));
    o.center_kernel_weighted = true;
    o.center_weight_source_mode = 1;
    return o;
}

static inline int clamp_int(int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); }

static size_t compute_usable_bytes(const Phase2Opts& opts)
{
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    double reserve = std::max(opts.reserve_bytes, (double)total_b * (double)opts.reserve_fraction);
    double usable = std::max(0.0, ((double)free_b - reserve) * (double)opts.gpu_mem_fraction);
    return (size_t)usable;
}

static int choose_query_block(int D, int N, const Phase2Opts& opts)
{
    size_t usable_b = compute_usable_bytes(opts);
    double bytes_per_query = 1.35 * (
        sizeof(float) * (1.0 * D * D + 5.0 * D + 4.0) +
        sizeof(int)   * 2.0 +
        sizeof(uint8_t)
    );
    int BQ = static_cast<int>(std::floor((double)usable_b / std::max(bytes_per_query, 1.0)));
    BQ = clamp_int(BQ, opts.min_query_block, opts.max_query_block);
    BQ = std::min(BQ, N);
    return BQ;
}


template <typename OffsetT>
__device__ __forceinline__ void pair_swap_device(float* keys, OffsetT* vals, int a, int b)
{
    float tk = keys[a]; keys[a] = keys[b]; keys[b] = tk;
    OffsetT tv = vals[a]; vals[a] = vals[b]; vals[b] = tv;
}

template <typename OffsetT>
__device__ int partition_pairs_device(float* keys, OffsetT* vals, int left, int right, int pivot_idx)
{
    float pivot = keys[pivot_idx];
    pair_swap_device(keys, vals, pivot_idx, right);
    OffsetT pivot_v = vals[right];
    int store = left;
    for (int i = left; i < right; ++i) {
        if (keys[i] < pivot || (keys[i] == pivot && vals[i] < pivot_v)) {
            pair_swap_device(keys, vals, i, store);
            ++store;
        }
    }
    pair_swap_device(keys, vals, store, right);
    return store;
}

template <typename OffsetT>
__device__ void insertion_sort_pairs_range_device(float* keys, OffsetT* vals, int left, int right)
{
    for (int i = left + 1; i <= right; ++i) {
        float k = keys[i];
        OffsetT v = vals[i];
        int j = i - 1;
        while (j >= left && (keys[j] > k || (keys[j] == k && vals[j] > v))) {
            keys[j + 1] = keys[j];
            vals[j + 1] = vals[j];
            --j;
        }
        keys[j + 1] = k;
        vals[j + 1] = v;
    }
}

template <typename OffsetT>
__device__ void quicksort_pairs_device(float* keys, OffsetT* vals, int n)
{
    if (n <= 1) return;
    int lstack[64], rstack[64];
    int top = 0;
    lstack[0] = 0; rstack[0] = n - 1;
    while (top >= 0) {
        int left = lstack[top];
        int right = rstack[top];
        --top;
        while (left < right) {
            if (right - left <= 24) {
                insertion_sort_pairs_range_device(keys, vals, left, right);
                break;
            }
            int mid = left + ((right - left) >> 1);
            int piv = partition_pairs_device(keys, vals, left, right, mid);
            if (piv - left < right - piv) {
                if (piv + 1 < right) { ++top; lstack[top] = piv + 1; rstack[top] = right; }
                right = piv - 1;
            } else {
                if (left < piv - 1) { ++top; lstack[top] = left; rstack[top] = piv - 1; }
                left = piv + 1;
            }
        }
    }
}

__device__ __forceinline__ float warp_reduce_sum_float(float v)
{
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ int warp_reduce_sum_int(int v)
{
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

template <int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void fullslice_shared_sort_kernel_u16(
    const float* __restrict__ Xrm,
    int D, int N, int q0, int B, int kSlice,
    float* __restrict__ dist2_out,
    uint16_t* __restrict__ rel_out)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int i = q0 + b;

    extern __shared__ unsigned char smem[];
    float* s_keys = reinterpret_cast<float*>(smem);
    uint16_t* s_vals = reinterpret_cast<uint16_t*>(s_keys + kSlice);

    __shared__ float xi_sh[MAX_D_MANOR];
    for (int d = threadIdx.x; d < D; d += BLOCK_THREADS) xi_sh[d] = Xrm[(size_t)i * (size_t)D + (size_t)d];
    __syncthreads();

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    for (int s = threadIdx.x; s < kSlice; s += BLOCK_THREADS) {
        int j = start + s;
        float acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < MAX_D_MANOR; ++d) {
            if (d < D) {
                float dx = Xrm[(size_t)j * (size_t)D + (size_t)d] - xi_sh[d];
                acc += dx * dx;
            }
        }
        s_keys[s] = acc;
        s_vals[s] = (uint16_t)s;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        quicksort_pairs_device(s_keys, s_vals, kSlice);

        float* dist_i = dist2_out + (size_t)i * (size_t)kSlice;
        uint16_t* rel_i = rel_out + (size_t)i * (size_t)kSlice;
        for (int p = 0; p < kSlice; ++p) {
            dist_i[p] = s_keys[p];
            rel_i[p] = s_vals[p];
        }
    }
}

__global__ void fullslice_dist_rel_kernel(
    const float* __restrict__ Xrm,
    int D, int N, int q0, int B, int kSlice,
    float* __restrict__ keys_out,
    int*   __restrict__ vals_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * kSlice;
    if (tid >= total) return;

    int b = tid / kSlice;
    int s = tid - b * kSlice;
    int i = q0 + b;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;
    int j = start + s;

    const float* xi = Xrm + (size_t)i * (size_t)D;
    const float* xj = Xrm + (size_t)j * (size_t)D;
    float acc = 0.0f;
    #pragma unroll
    for (int d = 0; d < MAX_D_MANOR; ++d) {
        if (d < D) {
            float dx = xj[d] - xi[d];
            acc += dx * dx;
        }
    }
    keys_out[tid] = acc;
    vals_out[tid] = s;
}

__global__ void cast_int_to_u16_kernel(const int* __restrict__ in, uint16_t* __restrict__ out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = (uint16_t)in[tid];
}


static __device__ __forceinline__ void normalize_vec_device(float* v, int D)
{
    float nrm2 = 0.0f;
    for (int d = 0; d < D; ++d) nrm2 += v[d] * v[d];
    float inv = rsqrtf(fmaxf(nrm2, 1e-30f));
    for (int d = 0; d < D; ++d) v[d] *= inv;
}

static __device__ __forceinline__ bool cholesky_solve_device_double(double* A, double* b, int n)
{
    for (int j = 0; j < n; ++j) {
        double sum = A[j * n + j];
        for (int k = 0; k < j; ++k) sum -= A[j * n + k] * A[j * n + k];
        if (!(sum > 1e-30)) return false;
        A[j * n + j] = sqrt(sum);
        for (int i = j + 1; i < n; ++i) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / A[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < i; ++k) b[i] -= A[i * n + k] * b[k];
        b[i] /= A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int k = i + 1; k < n; ++k) b[i] -= A[k * n + i] * b[k];
        b[i] /= A[i * n + i];
    }
    return true;
}

template <int MAXD>
static __device__ __forceinline__ void top_eigenvector_symmetric_jacobi_device(
    const float cov[MAXD][MAXD], int D, const float* v_ref, float* v_out)
{
    float A[MAXD][MAXD];
    float V[MAXD][MAXD];
    double tr = 0.0;
    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float symv = (float)(0.5 * ((double)cov[a][c] + (double)cov[c][a]));
            A[a][c] = symv;
            V[a][c] = (a == c) ? 1.0f : 0.0f;
        }
        tr += A[a][a];
    }

    if (!(tr > 1e-30)) {
        double nrm2 = 0.0;
        if (v_ref) {
            for (int d = 0; d < D; ++d) {
                v_out[d] = v_ref[d];
                nrm2 += (double)v_out[d] * (double)v_out[d];
            }
        }
        if (!(nrm2 > 1e-30)) {
            for (int d = 0; d < D; ++d) v_out[d] = (d == 0) ? 1.0f : 0.0f;
        } else {
            normalize_vec_device(v_out, D);
        }
        return;
    }

    const int max_sweeps = 12 * MAXD;
    const double tol = 1e-10 * fmax(tr, 1.0);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int p = 0, q = 1;
        double max_off = 0.0;
        for (int a = 0; a < D; ++a) {
            for (int c = a + 1; c < D; ++c) {
                double v = fabs((double)A[a][c]);
                if (v > max_off) {
                    max_off = v;
                    p = a;
                    q = c;
                }
            }
        }
        if (max_off <= tol) break;

        double app = (double)A[p][p];
        double aqq = (double)A[q][q];
        double apq = (double)A[p][q];
        if (fabs(apq) <= tol) continue;

        double tau = (aqq - app) / (2.0 * apq);
        double t = ((tau >= 0.0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        for (int k = 0; k < D; ++k) {
            if (k == p || k == q) continue;
            double akp = (double)A[k][p];
            double akq = (double)A[k][q];
            float new_kp = (float)(c * akp - s * akq);
            float new_kq = (float)(s * akp + c * akq);
            A[k][p] = new_kp; A[p][k] = new_kp;
            A[k][q] = new_kq; A[q][k] = new_kq;
        }

        double app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        double aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][p] = (float)app_new;
        A[q][q] = (float)aqq_new;
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;

        for (int k = 0; k < D; ++k) {
            double vkp = (double)V[k][p];
            double vkq = (double)V[k][q];
            V[k][p] = (float)(c * vkp - s * vkq);
            V[k][q] = (float)(s * vkp + c * vkq);
        }
    }

    int best = 0;
    for (int a = 1; a < D; ++a) {
        if ((double)A[a][a] > (double)A[best][best]) best = a;
    }

    double nrm2 = 0.0;
    for (int d = 0; d < D; ++d) {
        v_out[d] = (float)V[d][best];
        nrm2 += (double)v_out[d] * (double)v_out[d];
    }
    if (!(nrm2 > 1e-30)) {
        for (int d = 0; d < D; ++d) v_out[d] = (v_ref ? v_ref[d] : ((d == 0) ? 1.0f : 0.0f));
    }
    normalize_vec_device(v_out, D);

    if (v_ref) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += v_out[d] * v_ref[d];
        if (dot < 0.0f) {
            for (int d = 0; d < D; ++d) v_out[d] = -v_out[d];
        }
    }
}




template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void regression_solve_kernel_full_warp(
    const float* __restrict__ Xrm,
    const float* __restrict__ Y,
    const float* __restrict__ Vec,
    const float* __restrict__ Cen,
    int D, int N, int q0, int B, int kSlice,
    float hk, float rhok, float jitter,
    int center_kernel_weighted, int center_weight_source_mode,
    uint32_t* __restrict__ neigh_mask_bits,
    int mask_words_per_row,
    float* __restrict__ grad,
    float* __restrict__ center_pre,
    float* __restrict__ weight_sum,
    float* __restrict__ weight_sum_sq,
    int*   __restrict__ num_neigh)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;
    const int nsys = D + 1;

    float xi_lane = 0.0f;
    float vi_lane = 0.0f;
    if (lane < D) {
        xi_lane = Xrm[(size_t)i * (size_t)D + (size_t)lane];
        vi_lane = Vec[(size_t)i * (size_t)D + (size_t)lane];
    }

    float vi_host[MAXD];
    double aa00 = 0.0, bb0 = 0.0, wsum = 0.0, wsum_sq = 0.0, center_wsum = 0.0;
    int nnei = 0;

    double a01[MAXD];
    double bb1[MAXD];
    double cacc[MAXD];
    double row[MAXD][MAXD];

    float a01_lane = 0.0f;
    float bb1_lane = 0.0f;
    float cacc_lane = 0.0f;
    float row_lane[MAXD];

    if (lane == 0) {
        for (int d = 0; d < D; ++d) {
            vi_host[d] = Vec[(size_t)i * (size_t)D + (size_t)d];
        }
        for (int d = 0; d < MAXD; ++d) {
            a01[d] = 0.0f;
            bb1[d] = 0.0f;
            cacc[d] = 0.0f;
            for (int c = 0; c < MAXD; ++c) row[d][c] = 0.0f;
        }
    }
    for (int c = 0; c < MAXD; ++c) row_lane[c] = 0.0f;

    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    uint32_t* mask_row = neigh_mask_bits + (size_t)i * (size_t)mask_words_per_row;
    for (int w = lane; w < mask_words_per_row; w += 32) mask_row[w] = 0u;
    __syncwarp();

    for (int s = 0; s < kSlice; ++s) {
        int j = start + s;

        float dx_lane = 0.0f;
        if (lane < D) {
            dx_lane = Xrm[(size_t)j * (size_t)D + (size_t)lane] - xi_lane;
        }

        float proj = warp_reduce_sum_float((lane < D) ? (vi_lane * dx_lane) : 0.0f);
        proj = __shfl_sync(0xffffffffu, proj, 0);

        float z_lane = 0.0f;
        if (lane < D) z_lane = dx_lane + beta * proj * vi_lane;

        float sq = warp_reduce_sum_float((lane < D) ? (z_lane * z_lane) : 0.0f);
        sq = __shfl_sync(0xffffffffu, sq, 0);
        sq *= hk2_inv;

        if (sq < 1.0f) {
            if (lane == 0) {
                mask_row[s >> 5] |= (uint32_t(1) << (s & 31));
            }
            float rt = 1.0f - sq;
            float w = rt * rt;
            float yj = Y[j];

            if (lane == 0) {
                aa00 += (double)w;
                bb0 += (double)w * (double)yj;
                wsum += (double)w;
                wsum_sq += (double)w * (double)w;
                center_wsum += 1.0;
                ++nnei;
            }

            if (lane < D) {
                float csrc = Xrm[(size_t)j * (size_t)D + (size_t)lane];
                a01_lane += w * z_lane;
                bb1_lane += w * yj * z_lane;
                cacc_lane += csrc;
            }

            #pragma unroll
            for (int c = 0; c < MAXD; ++c) {
                float z_c = __shfl_sync(0xffffffffu, z_lane, c);
                if (lane < D && c < D) {
                    row_lane[c] += w * z_lane * z_c;
                }
            }
        }
    }

    float a01_lane_all = a01_lane;
    float bb1_lane_all = bb1_lane;
    float cacc_lane_all = cacc_lane;
    for (int d = 0; d < D; ++d) {
        float a01_d = __shfl_sync(0xffffffffu, a01_lane_all, d);
        float bb1_d = __shfl_sync(0xffffffffu, bb1_lane_all, d);
        float cacc_d = __shfl_sync(0xffffffffu, cacc_lane_all, d);
        if (lane == 0) {
            a01[d] = (double)a01_d;
            bb1[d] = (double)bb1_d;
            cacc[d] = (double)cacc_d;
        }
    }

    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float row_ac = __shfl_sync(0xffffffffu, row_lane[c], a);
            if (lane == 0) row[a][c] = (double)row_ac;
        }
    }

    if (lane != 0) return;

    if (nnei < 2 * D) {
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] = 0.0f;
            center_pre[(size_t)i * (size_t)D + (size_t)d] =
                Cen[(size_t)i * (size_t)D + (size_t)d];
        }
        if (weight_sum) weight_sum[i] = wsum;
        if (weight_sum_sq) weight_sum_sq[i] = wsum_sq;
        num_neigh[i] = nnei;
        return;
    }

    double rhs[MAXD + 1];
    double Mat[(MAXD + 1) * (MAXD + 1)];
    for (int t = 0; t < nsys * nsys; ++t) Mat[t] = 0.0;

    Mat[0] = aa00;
    rhs[0] = bb0;
    double tr = aa00;

    for (int d = 0; d < D; ++d) {
        Mat[d + 1] = a01[d];
        Mat[(size_t)(d + 1) * (size_t)nsys] = a01[d];
        rhs[d + 1] = bb1[d];
    }
    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            double v = row[a][c];
            Mat[(size_t)(a + 1) * (size_t)nsys + (size_t)(c + 1)] = v;
            if (a == c) tr += v;
        }
    }

    double jit = (double)jitter * tr / fmax((double)nsys, 1.0);
    for (int t = 0; t < nsys; ++t) {
        Mat[(size_t)t * (size_t)nsys + (size_t)t] += jit;
    }

    bool ok = cholesky_solve_device_double(Mat, rhs, nsys);
    if (!ok) {
        double add = 100.0 * jit + 1e-8;
        for (int t = 0; t < nsys; ++t) {
            Mat[(size_t)t * (size_t)nsys + (size_t)t] += add;
        }
        ok = cholesky_solve_device_double(Mat, rhs, nsys);
    }

    double dotgv = 0.0;
    if (ok) {
        for (int d = 0; d < D; ++d) dotgv += rhs[d + 1] * vi_host[d];
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] =
                (float)(rhs[d + 1] + (double)beta * dotgv * (double)vi_host[d]);
        }
    } else {
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] = 0.0f;
        }
    }

    for (int d = 0; d < D; ++d) {
        float cp = (center_wsum > 0.0)
            ? (float)(cacc[d] / center_wsum)
            : Cen[(size_t)i * (size_t)D + (size_t)d];
        center_pre[(size_t)i * (size_t)D + (size_t)d] = cp;
    }

    if (weight_sum) weight_sum[i] = (float)wsum;
    if (weight_sum_sq) weight_sum_sq[i] = (float)wsum_sq;
    num_neigh[i] = nnei;
}



template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void regression_solve_kernel_cached_warp(
    const float* __restrict__ Xrm,
    const float* __restrict__ Y,
    const float* __restrict__ Vec,
    const float* __restrict__ Cen,
    const float* __restrict__ dist2_cache,
    const uint16_t* __restrict__ rel_cache,
    int D, int N, int q0, int B, int kSlice,
    float hk, float rhok, float jitter,
    int center_kernel_weighted, int center_weight_source_mode,
    float* __restrict__ grad,
    float* __restrict__ center_pre,
    float* __restrict__ weight_sum,
    float* __restrict__ weight_sum_sq,
    int*   __restrict__ num_neigh)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;
    const int nsys = D + 1;

    float xi_lane = 0.0f;
    float vi_lane = 0.0f;
    if (lane < D) {
        xi_lane = Xrm[(size_t)i * (size_t)D + (size_t)lane];
        vi_lane = Vec[(size_t)i * (size_t)D + (size_t)lane];
    }

    float vi_host[MAXD];
    float aa00 = 0.0f, bb0 = 0.0f, wsum = 0.0f, wsum_sq = 0.0f, center_wsum = 0.0f;
    int nnei = 0;

    float a01[MAXD];
    float bb1[MAXD];
    float cacc[MAXD];
    float row[MAXD][MAXD];

    float a01_lane = 0.0f;
    float bb1_lane = 0.0f;
    float cacc_lane = 0.0f;
    float row_lane[MAXD];

    if (lane == 0) {
        for (int d = 0; d < D; ++d) {
            vi_host[d] = Vec[(size_t)i * (size_t)D + (size_t)d];
        }
        for (int d = 0; d < MAXD; ++d) {
            a01[d] = 0.0f;
            bb1[d] = 0.0f;
            cacc[d] = 0.0f;
            for (int c = 0; c < MAXD; ++c) row[d][c] = 0.0f;
        }
    }
    for (int c = 0; c < MAXD; ++c) row_lane[c] = 0.0f;

    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const uint16_t* rel_i = rel_cache + (size_t)i * (size_t)kSlice;
    const float* dist2_i = dist2_cache + (size_t)i * (size_t)kSlice;
    const float hk2_thresh = hk * hk;
    int m = 0, lo = 0, hi = kSlice;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (dist2_i[mid] <= hk2_thresh) lo = mid + 1;
        else hi = mid;
    }
    m = lo;

    for (int p = 0; p < m; ++p) {
        int j = start + (int)rel_i[p];

        float dx_lane = 0.0f;
        if (lane < D) {
            dx_lane = Xrm[(size_t)j * (size_t)D + (size_t)lane] - xi_lane;
        }

        float proj = warp_reduce_sum_float((lane < D) ? (vi_lane * dx_lane) : 0.0f);
        proj = __shfl_sync(0xffffffffu, proj, 0);

        float z_lane = 0.0f;
        if (lane < D) z_lane = dx_lane + beta * proj * vi_lane;

        float sq = warp_reduce_sum_float((lane < D) ? (z_lane * z_lane) : 0.0f);
        sq = __shfl_sync(0xffffffffu, sq, 0);
        sq *= hk2_inv;

        if (sq < 1.0f) {
            float rt = 1.0f - sq;
            float w = rt * rt;
            float yj = Y[j];

            if (lane == 0) {
                aa00 += w;
                bb0 += w * yj;
                wsum += w;
                wsum_sq += w * w;
                center_wsum += 1.0f;
                ++nnei;
            }

            if (lane < D) {
                float csrc = Xrm[(size_t)j * (size_t)D + (size_t)lane];
                a01_lane += w * z_lane;
                bb1_lane += w * yj * z_lane;
                cacc_lane += csrc;
            }

            #pragma unroll
            for (int c = 0; c < MAXD; ++c) {
                float z_c = __shfl_sync(0xffffffffu, z_lane, c);
                if (lane < D && c < D) {
                    row_lane[c] += w * z_lane * z_c;
                }
            }
        }
    }

    float a01_lane_all = a01_lane;
    float bb1_lane_all = bb1_lane;
    float cacc_lane_all = cacc_lane;
    for (int d = 0; d < D; ++d) {
        float a01_d = __shfl_sync(0xffffffffu, a01_lane_all, d);
        float bb1_d = __shfl_sync(0xffffffffu, bb1_lane_all, d);
        float cacc_d = __shfl_sync(0xffffffffu, cacc_lane_all, d);
        if (lane == 0) {
            a01[d] = a01_d;
            bb1[d] = bb1_d;
            cacc[d] = cacc_d;
        }
    }

    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float row_ac = __shfl_sync(0xffffffffu, row_lane[c], a);
            if (lane == 0) row[a][c] = row_ac;
        }
    }

    if (lane != 0) return;

    if (nnei < 2 * D) {
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] = 0.0f;
            center_pre[(size_t)i * (size_t)D + (size_t)d] =
                Cen[(size_t)i * (size_t)D + (size_t)d];
        }
        if (weight_sum) weight_sum[i] = wsum;
        if (weight_sum_sq) weight_sum_sq[i] = wsum_sq;
        num_neigh[i] = nnei;
        return;
    }

    float rhs[MAXD + 1];
    float Mat[(MAXD + 1) * (MAXD + 1)];
    for (int t = 0; t < nsys * nsys; ++t) Mat[t] = 0.0f;

    Mat[0] = aa00;
    rhs[0] = bb0;
    float tr = aa00;

    for (int d = 0; d < D; ++d) {
        Mat[d + 1] = a01[d];
        Mat[(size_t)(d + 1) * (size_t)nsys] = a01[d];
        rhs[d + 1] = bb1[d];
    }
    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float v = row[a][c];
            Mat[(size_t)(a + 1) * (size_t)nsys + (size_t)(c + 1)] = v;
            if (a == c) tr += v;
        }
    }

    float jit = jitter * tr / fmaxf((float)nsys, 1.0f);
    for (int t = 0; t < nsys; ++t) {
        Mat[(size_t)t * (size_t)nsys + (size_t)t] += jit;
    }

    bool ok = cholesky_solve_device(Mat, rhs, nsys);
    if (!ok) {
        float add = 100.0f * jit + 1e-8f;
        for (int t = 0; t < nsys; ++t) {
            Mat[(size_t)t * (size_t)nsys + (size_t)t] += add;
        }
        ok = cholesky_solve_device(Mat, rhs, nsys);
    }

    float dotgv = 0.0f;
    if (ok) {
        for (int d = 0; d < D; ++d) dotgv += rhs[d + 1] * vi_host[d];
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] =
                rhs[d + 1] + beta * dotgv * vi_host[d];
        }
    } else {
        for (int d = 0; d < D; ++d) {
            grad[(size_t)i * (size_t)D + (size_t)d] = 0.0f;
        }
    }

    for (int d = 0; d < D; ++d) {
        float cp = (center_wsum > 0.0f)
            ? (cacc[d] / center_wsum)
            : Cen[(size_t)i * (size_t)D + (size_t)d];
        center_pre[(size_t)i * (size_t)D + (size_t)d] = cp;
    }

    if (weight_sum) weight_sum[i] = wsum;
    if (weight_sum_sq) weight_sum_sq[i] = wsum_sq;
    num_neigh[i] = nnei;
}


template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void aggregation_eigen_kernel_full_warp(
    const float* __restrict__ Vec,
    const float* __restrict__ grad,
    const float* __restrict__ center_pre,
    const uint8_t* __restrict__ heavy,
    const uint32_t* __restrict__ neigh_mask_bits,
    int D, int N, int q0, int B, int kSlice,
    int mask_words_per_row,
    float* __restrict__ Vec_next,
    float* __restrict__ Cen_next)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;

    float vi[MAXD];
    if (lane == 0) {
        for (int d = 0; d < D; ++d) {
            vi[d] = Vec[(size_t)i * (size_t)D + (size_t)d];
        }
    }

    int cnt = 0;
    float cacc[MAXD];
    float cov[MAXD][MAXD];

    float cacc_lane = 0.0f;
    float cov_lane[MAXD];

    if (lane == 0) {
        for (int d = 0; d < MAXD; ++d) {
            cacc[d] = 0.0f;
            for (int c = 0; c < MAXD; ++c) cov[d][c] = 0.0f;
        }
    }
    for (int c = 0; c < MAXD; ++c) cov_lane[c] = 0.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const uint32_t* mask_row = neigh_mask_bits + (size_t)i * (size_t)mask_words_per_row;
    for (int s = 0; s < kSlice; ++s) {
        uint32_t word = mask_row[s >> 5];
        if (((word >> (s & 31)) & 1u) == 0u) continue;

        int j = start + s;
        if (!heavy[j]) continue;

        if (lane == 0) ++cnt;

        float ga_lane = 0.0f;
        if (lane < D) {
            cacc_lane += center_pre[(size_t)j * (size_t)D + (size_t)lane];
            ga_lane = grad[(size_t)j * (size_t)D + (size_t)lane];
        }

        #pragma unroll
        for (int c = 0; c < MAXD; ++c) {
            float gc = __shfl_sync(0xffffffffu, ga_lane, c);
            if (lane < D && c < D) {
                cov_lane[c] += ga_lane * gc;
            }
        }
    }

    float cacc_lane_all = cacc_lane;
    for (int d = 0; d < D; ++d) {
        float cacc_d = __shfl_sync(0xffffffffu, cacc_lane_all, d);
        if (lane == 0) cacc[d] = cacc_d;
    }

    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float cov_ac = __shfl_sync(0xffffffffu, cov_lane[c], a);
            if (lane == 0) cov[a][c] = cov_ac;
        }
    }

    if (lane != 0) return;
    if (cnt <= 0) return;

    float v[MAXD];
    top_eigenvector_symmetric_jacobi_device<MAXD>(cov, D, vi, v);

    float invcnt = 1.0f / (float)cnt;
    for (int d = 0; d < D; ++d) {
        Vec_next[(size_t)i * (size_t)D + (size_t)d] = v[d];
        Cen_next[(size_t)i * (size_t)D + (size_t)d] = cacc[d] * invcnt;
    }
}



template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void regression_accumulate_kernel_split(
    const float* __restrict__ Xrm,
    const float* __restrict__ Y,
    const float* __restrict__ Vec,
    int D, int N, int q0, int B, int kSlice,
    float hk, float rhok,
    uint32_t* __restrict__ neigh_mask_bits,
    int mask_words_per_row,
    double* __restrict__ stat_aa00,
    double* __restrict__ stat_bb0,
    float* __restrict__ stat_a01,
    float* __restrict__ stat_bb1,
    float* __restrict__ stat_cacc,
    float* __restrict__ stat_row,
    float* __restrict__ weight_sum,
    float* __restrict__ weight_sum_sq,
    int*   __restrict__ num_neigh)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;

    float xi_lane = 0.0f;
    float vi_lane = 0.0f;
    if (lane < D) {
        xi_lane = Xrm[(size_t)i * (size_t)D + (size_t)lane];
        vi_lane = Vec[(size_t)i * (size_t)D + (size_t)lane];
    }

    double aa00 = 0.0, bb0 = 0.0;
    float wsum = 0.0f, wsum_sq = 0.0f;
    int nnei = 0;

    float a01_lane = 0.0f;
    float bb1_lane = 0.0f;
    float cacc_lane = 0.0f;
    float row_lane[MAXD];
    #pragma unroll
    for (int c = 0; c < MAXD; ++c) row_lane[c] = 0.0f;

    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    uint32_t* mask_row = neigh_mask_bits + (size_t)i * (size_t)mask_words_per_row;
    for (int w = lane; w < mask_words_per_row; w += 32) mask_row[w] = 0u;
    __syncwarp();

    for (int s = 0; s < kSlice; ++s) {
        int j = start + s;

        float dx_lane = 0.0f;
        if (lane < D) {
            dx_lane = Xrm[(size_t)j * (size_t)D + (size_t)lane] - xi_lane;
        }

        float proj = warp_reduce_sum_float((lane < D) ? (vi_lane * dx_lane) : 0.0f);
        proj = __shfl_sync(0xffffffffu, proj, 0);

        float z_lane = 0.0f;
        if (lane < D) z_lane = dx_lane + beta * proj * vi_lane;

        float sq = warp_reduce_sum_float((lane < D) ? (z_lane * z_lane) : 0.0f);
        sq = __shfl_sync(0xffffffffu, sq, 0);
        sq *= hk2_inv;

        if (sq < 1.0f) {
            if (lane == 0) {
                mask_row[s >> 5] |= (uint32_t(1) << (s & 31));
            }
            float rt = 1.0f - sq;
            float w = rt * rt;
            float yj = Y[j];

            if (lane == 0) {
                aa00 += (double)w;
                bb0 += (double)w * (double)yj;
                wsum += w;
                wsum_sq += w * w;
                ++nnei;
            }

            if (lane < D) {
                float csrc = Xrm[(size_t)j * (size_t)D + (size_t)lane];
                a01_lane += w * z_lane;
                bb1_lane += w * yj * z_lane;
                cacc_lane += csrc;
            }

            #pragma unroll
            for (int c = 0; c < MAXD; ++c) {
                float z_c = __shfl_sync(0xffffffffu, z_lane, c);
                if (lane < D && c < D) {
                    row_lane[c] += w * z_lane * z_c;
                }
            }
        }
    }

    if (lane < D) {
        size_t vbase = (size_t)i * (size_t)D + (size_t)lane;
        stat_a01[vbase] = a01_lane;
        stat_bb1[vbase] = bb1_lane;
        stat_cacc[vbase] = cacc_lane;

        size_t rbase = ((size_t)i * (size_t)D + (size_t)lane) * (size_t)D;
        #pragma unroll
        for (int c = 0; c < MAXD; ++c) {
            if (c < D) stat_row[rbase + (size_t)c] = row_lane[c];
        }
    }

    if (lane == 0) {
        stat_aa00[i] = aa00;
        stat_bb0[i] = bb0;
        weight_sum[i] = wsum;
        weight_sum_sq[i] = wsum_sq;
        num_neigh[i] = nnei;
    }
}

template <int MAXD>
__global__ __launch_bounds__(128, 2) void regression_finalize_kernel_split_beta(
    const float* __restrict__ Vec,
    const float* __restrict__ Cen,
    int D, int N,
    float jitter,
    float rhok,
    const double* __restrict__ stat_aa00,
    const double* __restrict__ stat_bb0,
    const float* __restrict__ stat_a01,
    const float* __restrict__ stat_bb1,
    const float* __restrict__ stat_cacc,
    const float* __restrict__ stat_row,
    const int* __restrict__ num_neigh,
    float* __restrict__ grad,
    float* __restrict__ center_pre)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int nnei = num_neigh[i];
    size_t base = (size_t)i * (size_t)D;
    if (nnei < 2 * D) {
        for (int d = 0; d < D; ++d) {
            grad[base + (size_t)d] = 0.0f;
            center_pre[base + (size_t)d] = Cen[base + (size_t)d];
        }
        return;
    }

    const int nsys = D + 1;
    double rhs[MAXD + 1];
    double Mat[(MAXD + 1) * (MAXD + 1)];
    float vi_host[MAXD];
    for (int d = 0; d < D; ++d) vi_host[d] = Vec[base + (size_t)d];
    for (int t = 0; t < nsys * nsys; ++t) Mat[t] = 0.0;

    Mat[0] = stat_aa00[i];
    rhs[0] = stat_bb0[i];
    double tr = stat_aa00[i];
    for (int d = 0; d < D; ++d) {
        double a01 = (double)stat_a01[base + (size_t)d];
        double bb1 = (double)stat_bb1[base + (size_t)d];
        Mat[d + 1] = a01;
        Mat[(size_t)(d + 1) * (size_t)nsys] = a01;
        rhs[d + 1] = bb1;
    }
    for (int a = 0; a < D; ++a) {
        size_t rbase = ((size_t)i * (size_t)D + (size_t)a) * (size_t)D;
        for (int c = 0; c < D; ++c) {
            double v = (double)stat_row[rbase + (size_t)c];
            Mat[(size_t)(a + 1) * (size_t)nsys + (size_t)(c + 1)] = v;
            if (a == c) tr += v;
        }
    }

    double jit = (double)jitter * tr / fmax((double)nsys, 1.0);
    for (int t = 0; t < nsys; ++t) {
        Mat[(size_t)t * (size_t)nsys + (size_t)t] += jit;
    }

    bool ok = cholesky_solve_device_double(Mat, rhs, nsys);
    if (!ok) {
        double add = 100.0 * jit + 1e-8;
        for (int t = 0; t < nsys; ++t) {
            Mat[(size_t)t * (size_t)nsys + (size_t)t] += add;
        }
        ok = cholesky_solve_device_double(Mat, rhs, nsys);
    }

    const double beta = (double)(1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f);
    if (ok) {
        double dotgv = 0.0;
        for (int d = 0; d < D; ++d) dotgv += rhs[d + 1] * (double)vi_host[d];
        for (int d = 0; d < D; ++d) {
            grad[base + (size_t)d] = (float)(rhs[d + 1] + beta * dotgv * (double)vi_host[d]);
        }
    } else {
        for (int d = 0; d < D; ++d) grad[base + (size_t)d] = 0.0f;
    }

    float invcnt = 1.0f / fmaxf((float)nnei, 1.0f);
    for (int d = 0; d < D; ++d) {
        center_pre[base + (size_t)d] = stat_cacc[base + (size_t)d] * invcnt;
    }
}

template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void aggregation_accumulate_kernel_split(
    const float* __restrict__ grad,
    const float* __restrict__ center_pre,
    const uint8_t* __restrict__ heavy,
    const uint32_t* __restrict__ neigh_mask_bits,
    int D, int N, int q0, int B, int kSlice,
    int mask_words_per_row,
    int* __restrict__ agg_count,
    float* __restrict__ agg_cacc,
    float* __restrict__ agg_cov)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;

    int cnt = 0;
    float cacc_lane = 0.0f;
    float cov_lane[MAXD];
    #pragma unroll
    for (int c = 0; c < MAXD; ++c) cov_lane[c] = 0.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const uint32_t* mask_row = neigh_mask_bits + (size_t)i * (size_t)mask_words_per_row;
    for (int w = 0; w < mask_words_per_row; ++w) {
        uint32_t word = mask_row[w];
        while (word) {
            uint32_t bit = (uint32_t)(__ffs((int)word) - 1);
            int s = (w << 5) + (int)bit;
            word &= (word - 1u);
            if (s >= kSlice) continue;

            int j = start + s;
            if (!heavy[j]) continue;

            if (lane == 0) ++cnt;

            float ga_lane = 0.0f;
            if (lane < D) {
                cacc_lane += center_pre[(size_t)j * (size_t)D + (size_t)lane];
                ga_lane = grad[(size_t)j * (size_t)D + (size_t)lane];
            }

            #pragma unroll
            for (int c = 0; c < MAXD; ++c) {
                float gc = __shfl_sync(0xffffffffu, ga_lane, c);
                if (lane < D && c < D) {
                    cov_lane[c] += ga_lane * gc;
                }
            }
        }
    }

    if (lane < D) {
        size_t base = (size_t)i * (size_t)D + (size_t)lane;
        agg_cacc[base] = cacc_lane;
        size_t rbase = ((size_t)i * (size_t)D + (size_t)lane) * (size_t)D;
        #pragma unroll
        for (int c = 0; c < MAXD; ++c) {
            if (c < D) agg_cov[rbase + (size_t)c] = cov_lane[c];
        }
    }
    if (lane == 0) agg_count[i] = cnt;
}

template <int MAXD>
__global__ __launch_bounds__(128, 2) void aggregation_finalize_kernel_split(
    const float* __restrict__ Vec,
    int D, int N,
    const int* __restrict__ agg_count,
    const float* __restrict__ agg_cacc,
    const float* __restrict__ agg_cov,
    float* __restrict__ Vec_next,
    float* __restrict__ Cen_next)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cnt = agg_count[i];
    if (cnt <= 0) return;

    float vi[MAXD];
    float v[MAXD];
    float cacc[MAXD];
    float cov[MAXD][MAXD];
    size_t base = (size_t)i * (size_t)D;
    for (int d = 0; d < D; ++d) {
        vi[d] = Vec[base + (size_t)d];
        cacc[d] = agg_cacc[base + (size_t)d];
    }
    for (int a = 0; a < D; ++a) {
        size_t rbase = ((size_t)i * (size_t)D + (size_t)a) * (size_t)D;
        for (int c = 0; c < D; ++c) {
            cov[a][c] = agg_cov[rbase + (size_t)c];
        }
    }

    top_eigenvector_symmetric_jacobi_device<MAXD>(cov, D, vi, v);

    float invcnt = 1.0f / (float)cnt;
    for (int d = 0; d < D; ++d) {
        Vec_next[base + (size_t)d] = v[d];
        Cen_next[base + (size_t)d] = cacc[d] * invcnt;
    }
}

template <int MAXD, int WARPS_PER_BLOCK>
__global__ __launch_bounds__(WARPS_PER_BLOCK * 32, 2) void aggregation_eigen_kernel_cached_warp(
    const float* __restrict__ Xrm,
    const float* __restrict__ Vec,
    const float* __restrict__ grad,
    const float* __restrict__ center_pre,
    const uint8_t* __restrict__ heavy,
    const float* __restrict__ dist2_cache,
    const uint16_t* __restrict__ rel_cache,
    int D, int N, int q0, int B, int kSlice,
    float hk, float rhok, int power_iter_steps,
    int vector_kernel_weighted,
    float* __restrict__ Vec_next,
    float* __restrict__ Cen_next)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int local_q = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (local_q >= B) return;

    int i = q0 + local_q;

    float xi_lane = (lane < D) ? Xrm[(size_t)i * (size_t)D + (size_t)lane] : 0.0f;
    float vi_lane = (lane < D) ? Vec[(size_t)i * (size_t)D + (size_t)lane] : 0.0f;

    float xi[MAXD];
    float vi[MAXD];
    if (lane == 0) {
        for (int d = 0; d < D; ++d) {
            xi[d] = Xrm[(size_t)i * (size_t)D + (size_t)d];
            vi[d] = Vec[(size_t)i * (size_t)D + (size_t)d];
        }
    }

    int cnt = 0;
    float cacc[MAXD];
    float cov[MAXD][MAXD];

    float cacc_lane = 0.0f;
    float cov_lane[MAXD];

    if (lane == 0) {
        for (int d = 0; d < MAXD; ++d) {
            cacc[d] = 0.0f;
            for (int c = 0; c < MAXD; ++c) cov[d][c] = 0.0f;
        }
    }
    for (int c = 0; c < MAXD; ++c) cov_lane[c] = 0.0f;

    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const uint16_t* rel_i = rel_cache + (size_t)i * (size_t)kSlice;
    const float* dist2_i = dist2_cache + (size_t)i * (size_t)kSlice;
    const float hk2_thresh = hk * hk;
    int m = 0, lo = 0, hi = kSlice;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (dist2_i[mid] <= hk2_thresh) lo = mid + 1;
        else hi = mid;
    }
    m = lo;

    for (int p = 0; p < m; ++p) {
        int j = start + (int)rel_i[p];
        if (!heavy[j]) continue;

        float dx_lane = (lane < D) ? (Xrm[(size_t)j * (size_t)D + (size_t)lane] - xi_lane) : 0.0f;

        float proj = warp_reduce_sum_float((lane < D) ? (vi_lane * dx_lane) : 0.0f);
        proj = __shfl_sync(0xffffffffu, proj, 0);

        float z_lane = dx_lane + beta * proj * vi_lane;
        float sq = warp_reduce_sum_float((lane < D) ? (z_lane * z_lane) : 0.0f);
        sq = __shfl_sync(0xffffffffu, sq, 0);
        sq *= hk2_inv;

        if (sq < 1.0f) {
            float vw = 1.0f;
            if (lane == 0) ++cnt;

            float ga_lane = 0.0f;
            if (lane < D) {
                cacc_lane += center_pre[(size_t)j * (size_t)D + (size_t)lane];
                ga_lane = grad[(size_t)j * (size_t)D + (size_t)lane];
            }

            #pragma unroll
            for (int c = 0; c < MAXD; ++c) {
                float gc = __shfl_sync(0xffffffffu, ga_lane, c);
                if (lane < D && c < D) {
                    cov_lane[c] += vw * ga_lane * gc;
                }
            }
        }
    }

    float cacc_lane_all = cacc_lane;
    for (int d = 0; d < D; ++d) {
        float cacc_d = __shfl_sync(0xffffffffu, cacc_lane_all, d);
        if (lane == 0) cacc[d] = cacc_d;
    }

    for (int a = 0; a < D; ++a) {
        for (int c = 0; c < D; ++c) {
            float cov_ac = __shfl_sync(0xffffffffu, cov_lane[c], a);
            if (lane == 0) cov[a][c] = cov_ac;
        }
    }

    if (lane != 0) return;
    if (cnt <= 0) return;

    float v[MAXD];
    top_eigenvector_symmetric_jacobi_device<MAXD>(cov, D, vi, v);

    float invcnt = 1.0f / (float)cnt;
    for (int d = 0; d < D; ++d) {
        Vec_next[(size_t)i * (size_t)D + (size_t)d] = v[d];
        Cen_next[(size_t)i * (size_t)D + (size_t)d] = cacc[d] * invcnt;
    }
}



__global__ void set_heavy_mask_from_sum_kernel(
    const int* __restrict__ sum_num_hist,
    int iter, int D, int N, float heavy_factor,
    const int* __restrict__ num_neigh,
    uint8_t* __restrict__ heavy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float mean_num = (float)sum_num_hist[iter] / fmaxf((float)N, 1.0f);
        float thr = fmaxf(2.0f * (float)D, heavy_factor * mean_num);
        heavy[tid] = (uint8_t)(num_neigh[tid] > thr);
    }
}

__global__ void fallback_heavy_if_empty_kernel(
    const int* __restrict__ sum_heavy,
    int N,
    uint8_t* __restrict__ heavy)
{
    if (sum_heavy[0] != 0) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) heavy[tid] = (uint8_t)1;
}


__global__ void copy_vec_center_kernel(
    const float* __restrict__ Vec,
    const float* __restrict__ center_pre,
    int total,
    float* __restrict__ Vec_next,
    float* __restrict__ Cen_next)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        Vec_next[tid] = Vec[tid];
        Cen_next[tid] = center_pre[tid];
    }
}

__global__ void copy_center_kernel(
    const float* __restrict__ Cen_src,
    int total,
    float* __restrict__ Cen_dst)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        Cen_dst[tid] = Cen_src[tid];
    }
}

__device__ __forceinline__ float center_guard_metric_device(
    int nnei,
    float wsum,
    float wsum_sq,
    int use_ess)
{
    if (use_ess == 0) return (float)nnei;
    if (!(wsum_sq > 0.0f)) return 0.0f;
    float ess = (wsum * wsum) / fmaxf(wsum_sq, 1.0e-20f);
    ess = fminf(ess, (float)nnei);
    return fmaxf(ess, 0.0f);
}

__global__ void update_center_guard_hist_only_kernel(
    const int* __restrict__ num_neigh,
    const float* __restrict__ weight_sum,
    const float* __restrict__ weight_sum_sq,
    int N,
    int use_ess,
    int* __restrict__ num_neigh_hist_max,
    float* __restrict__ center_guard_hist_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cur = num_neigh[i];
    int hist = num_neigh_hist_max[i];
    if (cur > hist) hist = cur;
    num_neigh_hist_max[i] = hist;

    float cur_metric = center_guard_metric_device(cur, weight_sum[i], weight_sum_sq[i], use_ess);
    float hist_metric = center_guard_hist_max[i];
    if (cur_metric > hist_metric) hist_metric = cur_metric;
    center_guard_hist_max[i] = hist_metric;
}

__global__ void center_guard_update_kernel(
    const int* __restrict__ num_neigh,
    const float* __restrict__ weight_sum,
    const float* __restrict__ weight_sum_sq,
    int D, int N,
    float accept_ratio,
    int use_ess,
    int* __restrict__ num_neigh_hist_max,
    float* __restrict__ center_guard_hist_max,
    const float* __restrict__ Cen_prev,
    uint8_t* __restrict__ center_accept,
    float* __restrict__ Cen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cur = num_neigh[i];
    int hist = num_neigh_hist_max[i];
    if (cur > hist) hist = cur;
    num_neigh_hist_max[i] = hist;

    float cur_metric = center_guard_metric_device(cur, weight_sum[i], weight_sum_sq[i], use_ess);
    float hist_metric = center_guard_hist_max[i];
    if (cur_metric > hist_metric) hist_metric = cur_metric;
    center_guard_hist_max[i] = hist_metric;

    bool accept = (cur_metric >= accept_ratio * hist_metric);
    center_accept[i] = (uint8_t)(accept ? 1 : 0);
    if (!accept) {
        size_t base = (size_t)i * (size_t)D;
        for (int d = 0; d < D; ++d) {
            Cen[base + (size_t)d] = Cen_prev[base + (size_t)d];
        }
    }
}


__global__ void update_vector_guard_hist_only_kernel(
    const int* __restrict__ num_neigh,
    const float* __restrict__ weight_sum,
    const float* __restrict__ weight_sum_sq,
    int N,
    int use_ess,
    int* __restrict__ num_neigh_hist_max,
    float* __restrict__ vector_guard_hist_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cur = num_neigh[i];
    int hist = num_neigh_hist_max[i];
    if (cur > hist) hist = cur;
    num_neigh_hist_max[i] = hist;

    float cur_metric = center_guard_metric_device(cur, weight_sum[i], weight_sum_sq[i], use_ess);
    float hist_metric = vector_guard_hist_max[i];
    if (cur_metric > hist_metric) hist_metric = cur_metric;
    vector_guard_hist_max[i] = hist_metric;
}

__global__ void vector_guard_update_kernel(
    const int* __restrict__ num_neigh,
    const float* __restrict__ weight_sum,
    const float* __restrict__ weight_sum_sq,
    int D, int N,
    float accept_ratio,
    int use_ess,
    int* __restrict__ num_neigh_hist_max,
    float* __restrict__ vector_guard_hist_max,
    const float* __restrict__ grad_prev,
    uint8_t* __restrict__ vector_accept,
    float* __restrict__ grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cur = num_neigh[i];
    int hist = num_neigh_hist_max[i];
    if (cur > hist) hist = cur;
    num_neigh_hist_max[i] = hist;

    float cur_metric = center_guard_metric_device(cur, weight_sum[i], weight_sum_sq[i], use_ess);
    float hist_metric = vector_guard_hist_max[i];
    if (cur_metric > hist_metric) hist_metric = cur_metric;
    vector_guard_hist_max[i] = hist_metric;

    bool accept = (cur_metric >= accept_ratio * hist_metric);
    vector_accept[i] = (uint8_t)(accept ? 1 : 0);
    if (!accept) {
        size_t base = (size_t)i * (size_t)D;
        for (int d = 0; d < D; ++d) {
            grad[base + (size_t)d] = grad_prev[base + (size_t)d];
        }
    }
}

__global__ __launch_bounds__(256, 2) void project_center_off_tangent_kernel(
    const float* __restrict__ Xrm,
    const float* __restrict__ Vec,
    int D, int N,
    float* __restrict__ Cen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dotv = 0.0f;
    for (int d = 0; d < D; ++d) {
        dotv += Vec[(size_t)i * (size_t)D + (size_t)d] *
                (Cen[(size_t)i * (size_t)D + (size_t)d] - Xrm[(size_t)i * (size_t)D + (size_t)d]);
    }
    for (int d = 0; d < D; ++d) {

        Cen[(size_t)i * (size_t)D + (size_t)d] -= dotv * Vec[(size_t)i * (size_t)D + (size_t)d];
    }
}


static void column_major_from_row_major(const std::vector<float>& Xrm, int D, int N, std::vector<float>& Xcm)
{
    Xcm.assign((size_t)D * (size_t)N, 0.0f);
    for (int j = 0; j < N; ++j) {
        for (int d = 0; d < D; ++d) Xcm[d + j * D] = Xrm[(size_t)j * (size_t)D + (size_t)d];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs >= 1 && mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:cmd",
            "String command inputs are no longer supported in this cache-off build.");
    }

    if (nrhs != 8) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:nrhs",
            "Expected 8 inputs: X_sorted, Y_sorted, Vec_pp, Cen_pp, hk, rhok, kSlice, opts");
    }
    if (nlhs > 1) mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:nlhs", "Returns exactly one struct output.");

    const mxArray* mxX = prhs[0];
    const mxArray* mxY = prhs[1];
    const mxArray* mxVec = prhs[2];
    const mxArray* mxCen = prhs[3];
    const mxArray* mxhk = prhs[4];
    const mxArray* mxrhok = prhs[5];
    if (!mxIsSingle(mxX) || !mxIsSingle(mxY) || !mxIsSingle(mxVec) ||
        !mxIsSingle(mxCen) || !mxIsSingle(mxhk) || !mxIsSingle(mxrhok)) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:type",
            "X_sorted, Y_sorted, Vec_pp, Cen_pp, hk, rhok must all be single.");
    }

    const int D = (int)mxGetM(mxX);
    const int N = (int)mxGetN(mxX);
    const int Ny = (int)mxGetNumberOfElements(mxY);
    const int kSlice = (int)mxGetScalar(prhs[6]);
    Phase2Opts opts = parse_opts(prhs[7]);

    if (D <= 0 || N <= 0) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "Invalid D or N.");
    }
    if (D > MAX_D_MANOR) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:maxd",
            "D=%d exceeds MAX_D_MANOR=%d", D, MAX_D_MANOR);
    }
    if (Ny != N) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "Y_sorted length must equal size(X_sorted,2).");
    }
    if ((int)mxGetM(mxVec) != D || (int)mxGetN(mxVec) != N) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "Vec_pp must have size [D, N].");
    }
    if ((int)mxGetM(mxCen) != D || (int)mxGetN(mxCen) != N) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "Cen_pp must have size [D, N].");
    }
    if (kSlice <= 0 || kSlice > N) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "kSlice must satisfy 1 <= kSlice <= N.");
    }
    if ((kSlice & 1) == 0) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "kSlice must be odd.");
    }
    if (kSlice > 65535) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:args",
            "This file uses uint16 slice offsets, so kSlice must be <= 65535.");
    }

    size_t Kh = mxGetNumberOfElements(mxhk);
    size_t Kr = mxGetNumberOfElements(mxrhok);
    size_t K = std::max(Kh, Kr);
    if (K == 0) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:iters",
            "hk and rhok must be non-empty.");
    }
    if (!(Kh == 1 || Kh == K) || !(Kr == 1 || Kr == K)) {
        mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:iters",
            "hk and rhok must be scalars or vectors of same length.");
    }

    const float* hk_in = (const float*)mxGetData(mxhk);
    const float* rhok_in = (const float*)mxGetData(mxrhok);
    std::vector<float> hk_iter(K), rhok_iter(K);
    for (size_t k = 0; k < K; ++k) {
        hk_iter[k] = hk_in[(Kh == 1) ? 0 : k];
        rhok_iter[k] = rhok_in[(Kr == 1) ? 0 : k];

        if (!std::isfinite(hk_iter[k]) || hk_iter[k] <= 0.0f) {
            mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:iters",
                "All hk values must be finite and positive.");
        }
        if (!std::isfinite(rhok_iter[k]) || rhok_iter[k] <= 0.0f) {
            mexErrMsgIdAndTxt("manor_1_phase2_cuda_mex:iters",
                "All rhok values must be finite and positive.");
        }
    }

    const float* Veccm = (const float*)mxGetData(mxVec);
    const float* Cencm = (const float*)mxGetData(mxCen);
    const float* Yh = (const float*)mxGetData(mxY);

    bool y_non_decreasing = true;
    for (int j = 1; j < N; ++j) {
        if (Yh[j] < Yh[j - 1]) { y_non_decreasing = false; break; }
    }
    if (!y_non_decreasing) {
        mexWarnMsgIdAndTxt("manor_1_phase2_cuda_mex:ysort",
            "Y_sorted is not nondecreasing. This CUDA implementation assumes Y-sorted slice neighborhoods.");
    }

    const bool direct_path = true;

    std::vector<float> Vecrm((size_t)N * (size_t)D, 0.0f), Cenrm((size_t)N * (size_t)D, 0.0f);
    for (int j = 0; j < N; ++j) {
        for (int d = 0; d < D; ++d) {
            Vecrm[(size_t)j * (size_t)D + (size_t)d] = Veccm[d + j * D];
            Cenrm[(size_t)j * (size_t)D + (size_t)d] = Cencm[d + j * D];
        }
    }

    std::vector<float> Xrm, Y;
    float *d_Xrm=nullptr, *d_Y=nullptr, *d_Vec=nullptr, *d_Cen=nullptr, *d_Vec_next=nullptr;
    const bool cache_enabled = false;
    const double setup_sec = 0.0;

    const float* Xcm = (const float*)mxGetData(mxX);
    Xrm.assign((size_t)N * (size_t)D, 0.0f);
    Y.assign((size_t)N, 0.0f);
    for (int j = 0; j < N; ++j) {
        Y[(size_t)j] = Yh[j];
        for (int d = 0; d < D; ++d) {
            Xrm[(size_t)j * (size_t)D + (size_t)d] = Xcm[d + j * D];
        }
    }
    CUDA_CHECK(cudaMalloc(&d_Xrm, sizeof(float) * Xrm.size()));
    CUDA_CHECK(cudaMalloc(&d_Y, sizeof(float) * Y.size()));
    CUDA_CHECK(cudaMemcpy(d_Xrm, Xrm.data(), sizeof(float) * Xrm.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, Y.data(), sizeof(float) * Y.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_Vec, sizeof(float) * Vecrm.size()));
    CUDA_CHECK(cudaMalloc(&d_Cen, sizeof(float) * Cenrm.size()));
    CUDA_CHECK(cudaMalloc(&d_Vec_next, sizeof(float) * Vecrm.size()));
    CUDA_CHECK(cudaMemcpy(d_Vec, Vecrm.data(), sizeof(float) * Vecrm.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Cen, Cenrm.data(), sizeof(float) * Cenrm.size(), cudaMemcpyHostToDevice));

    using clock_type = std::chrono::steady_clock;
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    auto phase2_total_start = clock_type::now();

    double copyout_sec = qnan;
    std::vector<double> iter_total_sec((size_t)K, qnan);
    std::vector<double> regression_sec((size_t)K, qnan);
    std::vector<double> project_pre_sec((size_t)K, qnan);
    std::vector<double> heavy_reduce_sec((size_t)K, qnan);
    std::vector<double> copy_pre_sec((size_t)K, qnan);
    std::vector<double> aggregation_sec((size_t)K, qnan);
    std::vector<double> project_final_sec((size_t)K, qnan);
    bool used_cuda_graphs = false;

    float *d_keys_in=nullptr, *d_keys_out=nullptr;
    int *d_vals_in=nullptr, *d_vals_out=nullptr, *d_offsets=nullptr;
    void* d_sort_temp=nullptr;

    int BQ = choose_query_block(D, N, opts);
    if (!opts.silent) {

        mexPrintf("[phase2_cuda_extreme] D=%d N=%d kSlice=%d hk=%.6g rhok=%.6g BQ=%d direct=%d graphs=%d warps_per_block=%d\n",

                  D, N, kSlice, hk_iter[0], rhok_iter[0], BQ, (int)direct_path, (int)opts.use_cuda_graphs, opts.warps_per_block);

    }
    mexEvalString("drawnow;");

    float *d_grad=nullptr, *d_grad_prev=nullptr, *d_center_pre=nullptr, *d_weight_sum=nullptr, *d_weight_sum_sq=nullptr, *d_Cen_prev=nullptr, *d_center_guard_hist_max=nullptr, *d_vector_guard_hist_max=nullptr;
    double *d_phase2_scalar0=nullptr, *d_phase2_scalar1=nullptr;
    float *d_phase2_mat_accum=nullptr, *d_phase2_vecbuf0=nullptr, *d_phase2_vecbuf1=nullptr, *d_phase2_vecbuf2=nullptr;
    int *d_phase2_count_accum=nullptr;
    uint32_t* d_phase2_neigh_mask_bits=nullptr;
    uint8_t *d_heavy=nullptr, *d_center_accept=nullptr, *d_vector_accept=nullptr;
    int *d_num_neigh=nullptr, *d_num_neigh_hist_max=nullptr, *d_sum_heavy=nullptr, *d_sum_num=nullptr;
    void* d_reduce_temp=nullptr;
    size_t temp_num=0, temp_heavy=0, temp_reduce=0;

    CUDA_CHECK(cudaMalloc(&d_grad, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMemset(d_grad, 0, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMalloc(&d_grad_prev, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMemset(d_grad_prev, 0, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMalloc(&d_center_pre, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMalloc(&d_Cen_prev, sizeof(float) * (size_t)N * (size_t)D));
    CUDA_CHECK(cudaMalloc(&d_weight_sum, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_weight_sum_sq, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_num_neigh, sizeof(int) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_num_neigh_hist_max, sizeof(int) * (size_t)N));
    CUDA_CHECK(cudaMemset(d_num_neigh_hist_max, 0, sizeof(int) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_center_guard_hist_max, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMemset(d_center_guard_hist_max, 0, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_vector_guard_hist_max, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMemset(d_vector_guard_hist_max, 0, sizeof(float) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_heavy, sizeof(uint8_t) * (size_t)N));
    if (opts.center_update_guard_enable) CUDA_CHECK(cudaMalloc(&d_center_accept, sizeof(uint8_t) * (size_t)N));
    if (opts.vector_update_guard_enable) CUDA_CHECK(cudaMalloc(&d_vector_accept, sizeof(uint8_t) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_sum_num, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_heavy, sizeof(int)));
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_num, d_num_neigh, d_sum_num, N));
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_heavy, d_heavy, d_sum_heavy, N));
    temp_reduce = std::max(temp_num, temp_heavy);
    CUDA_CHECK(cudaMalloc(&d_reduce_temp, temp_reduce));

    const int phase2_mask_words_per_row = (kSlice + 31) >> 5;
    const size_t phase2_mask_word_count = (size_t)N * (size_t)phase2_mask_words_per_row;
    CUDA_CHECK(cudaMalloc(&d_phase2_neigh_mask_bits, sizeof(uint32_t) * phase2_mask_word_count));

    const size_t phase2_vec_count = (size_t)N * (size_t)D;
    const size_t phase2_mat_count = phase2_vec_count * (size_t)D;
    CUDA_CHECK(cudaMalloc(&d_phase2_scalar0, sizeof(double) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_phase2_scalar1, sizeof(double) * (size_t)N));
    CUDA_CHECK(cudaMalloc(&d_phase2_vecbuf0, sizeof(float) * phase2_vec_count));
    CUDA_CHECK(cudaMalloc(&d_phase2_vecbuf1, sizeof(float) * phase2_vec_count));
    CUDA_CHECK(cudaMalloc(&d_phase2_vecbuf2, sizeof(float) * phase2_vec_count));
    CUDA_CHECK(cudaMalloc(&d_phase2_mat_accum, sizeof(float) * phase2_mat_count));
    CUDA_CHECK(cudaMalloc(&d_phase2_count_accum, sizeof(int) * (size_t)N));

    std::vector<int32_t> num_neigh_iter((size_t)K, 0);
    std::vector<float> heavy_frac_iter((size_t)K, 0.0f);
    std::vector<float> center_accept_frac_iter((size_t)K, 1.0f);
    std::vector<float> vector_accept_frac_iter((size_t)K, 1.0f);

    int *d_sum_num_hist=nullptr, *d_sum_heavy_hist=nullptr, *d_sum_center_accept_hist=nullptr, *d_sum_vector_accept_hist=nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_num_hist, sizeof(int) * (size_t)K));
    CUDA_CHECK(cudaMalloc(&d_sum_heavy_hist, sizeof(int) * (size_t)K));
    if (opts.center_update_guard_enable) CUDA_CHECK(cudaMalloc(&d_sum_center_accept_hist, sizeof(int) * (size_t)K));
    if (opts.vector_update_guard_enable) CUDA_CHECK(cudaMalloc(&d_sum_vector_accept_hist, sizeof(int) * (size_t)K));

    const int threadsN = 256;
    const int totalVec = N * D;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto launch_regression = [&](cudaStream_t s, int q0, int B, float hk, float rhok) {
        int warps = (opts.warps_per_block == 2 ? 2 : 4);
        dim3 grid_scan((B + warps - 1) / warps);
        const int threads_finalize = 128;
        dim3 grid_finalize((B + threads_finalize - 1) / threads_finalize);
        if (warps == 2) {
            if (D <= 16) {
                regression_accumulate_kernel_split<16, 2><<<grid_scan, dim3(64), 0, s>>>(
                    d_Xrm, d_Y, d_Vec,
                    D, N, q0, B, kSlice, hk, rhok,
                    d_phase2_neigh_mask_bits, phase2_mask_words_per_row,
                    d_phase2_scalar0, d_phase2_scalar1,
                    d_phase2_vecbuf0, d_phase2_vecbuf1, d_phase2_vecbuf2, d_phase2_mat_accum,
                    d_weight_sum, d_weight_sum_sq, d_num_neigh);
                regression_finalize_kernel_split_beta<16><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D, d_Cen + (size_t)q0 * (size_t)D,
                    D, B, opts.jitter, rhok,
                    d_phase2_scalar0 + q0, d_phase2_scalar1 + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf1 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf2 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_num_neigh + q0,
                    d_grad + (size_t)q0 * (size_t)D,
                    d_center_pre + (size_t)q0 * (size_t)D);
            } else {
                regression_accumulate_kernel_split<32, 2><<<grid_scan, dim3(64), 0, s>>>(
                    d_Xrm, d_Y, d_Vec,
                    D, N, q0, B, kSlice, hk, rhok,
                    d_phase2_neigh_mask_bits, phase2_mask_words_per_row,
                    d_phase2_scalar0, d_phase2_scalar1,
                    d_phase2_vecbuf0, d_phase2_vecbuf1, d_phase2_vecbuf2, d_phase2_mat_accum,
                    d_weight_sum, d_weight_sum_sq, d_num_neigh);
                regression_finalize_kernel_split_beta<32><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D, d_Cen + (size_t)q0 * (size_t)D,
                    D, B, opts.jitter, rhok,
                    d_phase2_scalar0 + q0, d_phase2_scalar1 + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf1 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf2 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_num_neigh + q0,
                    d_grad + (size_t)q0 * (size_t)D,
                    d_center_pre + (size_t)q0 * (size_t)D);
            }
        } else {
            if (D <= 16) {
                regression_accumulate_kernel_split<16, 4><<<grid_scan, dim3(128), 0, s>>>(
                    d_Xrm, d_Y, d_Vec,
                    D, N, q0, B, kSlice, hk, rhok,
                    d_phase2_neigh_mask_bits, phase2_mask_words_per_row,
                    d_phase2_scalar0, d_phase2_scalar1,
                    d_phase2_vecbuf0, d_phase2_vecbuf1, d_phase2_vecbuf2, d_phase2_mat_accum,
                    d_weight_sum, d_weight_sum_sq, d_num_neigh);
                regression_finalize_kernel_split_beta<16><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D, d_Cen + (size_t)q0 * (size_t)D,
                    D, B, opts.jitter, rhok,
                    d_phase2_scalar0 + q0, d_phase2_scalar1 + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf1 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf2 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_num_neigh + q0,
                    d_grad + (size_t)q0 * (size_t)D,
                    d_center_pre + (size_t)q0 * (size_t)D);
            } else {
                regression_accumulate_kernel_split<32, 4><<<grid_scan, dim3(128), 0, s>>>(
                    d_Xrm, d_Y, d_Vec,
                    D, N, q0, B, kSlice, hk, rhok,
                    d_phase2_neigh_mask_bits, phase2_mask_words_per_row,
                    d_phase2_scalar0, d_phase2_scalar1,
                    d_phase2_vecbuf0, d_phase2_vecbuf1, d_phase2_vecbuf2, d_phase2_mat_accum,
                    d_weight_sum, d_weight_sum_sq, d_num_neigh);
                regression_finalize_kernel_split_beta<32><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D, d_Cen + (size_t)q0 * (size_t)D,
                    D, B, opts.jitter, rhok,
                    d_phase2_scalar0 + q0, d_phase2_scalar1 + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf1 + (size_t)q0 * (size_t)D,
                    d_phase2_vecbuf2 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_num_neigh + q0,
                    d_grad + (size_t)q0 * (size_t)D,
                    d_center_pre + (size_t)q0 * (size_t)D);
            }
        }
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto launch_aggregation = [&](cudaStream_t s, int q0, int B, float hk, float rhok) {
        (void)hk; (void)rhok;
        int warps = (opts.warps_per_block == 2 ? 2 : 4);
        dim3 grid_scan((B + warps - 1) / warps);
        const int threads_finalize = 128;
        dim3 grid_finalize((B + threads_finalize - 1) / threads_finalize);
        if (warps == 2) {
            if (D <= 16) {
                aggregation_accumulate_kernel_split<16, 2><<<grid_scan, dim3(64), 0, s>>>(
                    d_grad, d_center_pre, d_heavy, d_phase2_neigh_mask_bits,
                    D, N, q0, B, kSlice, phase2_mask_words_per_row,
                    d_phase2_count_accum, d_phase2_vecbuf0, d_phase2_mat_accum);
                aggregation_finalize_kernel_split<16><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D,
                    D, B,
                    d_phase2_count_accum + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_Vec_next + (size_t)q0 * (size_t)D,
                    d_Cen + (size_t)q0 * (size_t)D);
            } else {
                aggregation_accumulate_kernel_split<32, 2><<<grid_scan, dim3(64), 0, s>>>(
                    d_grad, d_center_pre, d_heavy, d_phase2_neigh_mask_bits,
                    D, N, q0, B, kSlice, phase2_mask_words_per_row,
                    d_phase2_count_accum, d_phase2_vecbuf0, d_phase2_mat_accum);
                aggregation_finalize_kernel_split<32><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D,
                    D, B,
                    d_phase2_count_accum + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_Vec_next + (size_t)q0 * (size_t)D,
                    d_Cen + (size_t)q0 * (size_t)D);
            }
        } else {
            if (D <= 16) {
                aggregation_accumulate_kernel_split<16, 4><<<grid_scan, dim3(128), 0, s>>>(
                    d_grad, d_center_pre, d_heavy, d_phase2_neigh_mask_bits,
                    D, N, q0, B, kSlice, phase2_mask_words_per_row,
                    d_phase2_count_accum, d_phase2_vecbuf0, d_phase2_mat_accum);
                aggregation_finalize_kernel_split<16><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D,
                    D, B,
                    d_phase2_count_accum + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_Vec_next + (size_t)q0 * (size_t)D,
                    d_Cen + (size_t)q0 * (size_t)D);
            } else {
                aggregation_accumulate_kernel_split<32, 4><<<grid_scan, dim3(128), 0, s>>>(
                    d_grad, d_center_pre, d_heavy, d_phase2_neigh_mask_bits,
                    D, N, q0, B, kSlice, phase2_mask_words_per_row,
                    d_phase2_count_accum, d_phase2_vecbuf0, d_phase2_mat_accum);
                aggregation_finalize_kernel_split<32><<<grid_finalize, threads_finalize, 0, s>>>(
                    d_Vec + (size_t)q0 * (size_t)D,
                    D, B,
                    d_phase2_count_accum + q0,
                    d_phase2_vecbuf0 + (size_t)q0 * (size_t)D,
                    d_phase2_mat_accum + (size_t)q0 * (size_t)D * (size_t)D,
                    d_Vec_next + (size_t)q0 * (size_t)D,
                    d_Cen + (size_t)q0 * (size_t)D);
            }
        }
        CUDA_CHECK(cudaPeekAtLastError());
    };

    auto launch_iteration_sequence = [&](cudaStream_t s) {
        for (size_t iter = 0; iter < K; ++iter) {
            float hk = hk_iter[iter];
            float rhok = rhok_iter[iter];

            int blocksCopy = (totalVec + threadsN - 1) / threadsN;
            if (opts.vector_update_guard_enable) {
                copy_center_kernel<<<blocksCopy, threadsN, 0, s>>>(d_grad, totalVec, d_grad_prev);
                CUDA_CHECK(cudaPeekAtLastError());
            }

            for (int q0 = 0; q0 < N; q0 += BQ) {
                int B = std::min(BQ, N - q0);
                launch_regression(s, q0, B, hk, rhok);
            }

            int blocksN = (N + threadsN - 1) / threadsN;

            if (opts.vector_update_guard_enable) {
                vector_guard_update_kernel<<<blocksN, threadsN, 0, s>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, D, N, opts.vector_update_guard_ratio,
                    0,
                    d_num_neigh_hist_max, d_vector_guard_hist_max, d_grad_prev, d_vector_accept, d_grad);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_vector_accept, d_sum_vector_accept_hist + iter, N, s));
            } else {
                update_vector_guard_hist_only_kernel<<<blocksN, threadsN, 0, s>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, N,
                    0,
                    d_num_neigh_hist_max, d_vector_guard_hist_max);
                CUDA_CHECK(cudaPeekAtLastError());
            }

            project_center_off_tangent_kernel<<<blocksN, threadsN, 0, s>>>(d_Xrm, d_Vec, D, N, d_center_pre);
            CUDA_CHECK(cudaPeekAtLastError());

            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_num, d_num_neigh, d_sum_num_hist + iter, N, s));
            set_heavy_mask_from_sum_kernel<<<blocksN, threadsN, 0, s>>>(d_sum_num_hist, (int)iter, D, N, opts.heavy_factor, d_num_neigh, d_heavy);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_heavy, d_sum_heavy_hist + iter, N, s));
            fallback_heavy_if_empty_kernel<<<blocksN, threadsN, 0, s>>>(d_sum_heavy_hist + iter, N, d_heavy);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_heavy, d_sum_heavy_hist + iter, N, s));

            if (opts.center_update_guard_enable) {
                copy_center_kernel<<<blocksCopy, threadsN, 0, s>>>(d_Cen, totalVec, d_Cen_prev);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            copy_vec_center_kernel<<<blocksCopy, threadsN, 0, s>>>(d_Vec, d_center_pre, totalVec, d_Vec_next, d_Cen);
            CUDA_CHECK(cudaPeekAtLastError());

            for (int q0 = 0; q0 < N; q0 += BQ) {
                int B = std::min(BQ, N - q0);
                launch_aggregation(s, q0, B, hk, rhok);
            }

            std::swap(d_Vec, d_Vec_next);

            if (opts.center_reduce_tangent_final) {
                project_center_off_tangent_kernel<<<blocksN, threadsN, 0, s>>>(d_Xrm, d_Vec, D, N, d_Cen);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            if (opts.center_update_guard_enable) {
                center_guard_update_kernel<<<blocksN, threadsN, 0, s>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, D, N, opts.center_update_guard_ratio,
                    0,
                    d_num_neigh_hist_max, d_center_guard_hist_max, d_Cen_prev, d_center_accept, d_Cen);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_center_accept, d_sum_center_accept_hist + iter, N, s));
            } else {
                update_center_guard_hist_only_kernel<<<blocksN, threadsN, 0, s>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, N,
                    0,
                    d_num_neigh_hist_max, d_center_guard_hist_max);
                CUDA_CHECK(cudaPeekAtLastError());
            }
        }
    };

    if (opts.profile_timers) {
        for (size_t iter = 0; iter < K; ++iter) {
            float hk = hk_iter[iter];
            float rhok = rhok_iter[iter];
            auto t_iter = clock_type::now();

            auto t_reg = clock_type::now();
            int blocksCopy = (totalVec + threadsN - 1) / threadsN;
            if (opts.vector_update_guard_enable) {
                copy_center_kernel<<<blocksCopy, threadsN, 0, stream>>>(d_grad, totalVec, d_grad_prev);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            for (int q0 = 0; q0 < N; q0 += BQ) {
                int B = std::min(BQ, N - q0);
                launch_regression(stream, q0, B, hk, rhok);
            }
            int blocksN = (N + threadsN - 1) / threadsN;
            if (opts.vector_update_guard_enable) {
                vector_guard_update_kernel<<<blocksN, threadsN, 0, stream>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, D, N, opts.vector_update_guard_ratio,
                    0,
                    d_num_neigh_hist_max, d_vector_guard_hist_max, d_grad_prev, d_vector_accept, d_grad);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_vector_accept, d_sum_vector_accept_hist + iter, N, stream));
            } else {
                update_vector_guard_hist_only_kernel<<<blocksN, threadsN, 0, stream>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, N,
                    0,
                    d_num_neigh_hist_max, d_vector_guard_hist_max);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            regression_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_reg).count();

            auto t_proj_pre = clock_type::now();
            project_center_off_tangent_kernel<<<blocksN, threadsN, 0, stream>>>(d_Xrm, d_Vec, D, N, d_center_pre);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            project_pre_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_proj_pre).count();

            auto t_heavy = clock_type::now();
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_num, d_num_neigh, d_sum_num_hist + iter, N, stream));
            set_heavy_mask_from_sum_kernel<<<blocksN, threadsN, 0, stream>>>(d_sum_num_hist, (int)iter, D, N, opts.heavy_factor, d_num_neigh, d_heavy);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_heavy, d_sum_heavy_hist + iter, N, stream));
            fallback_heavy_if_empty_kernel<<<blocksN, threadsN, 0, stream>>>(d_sum_heavy_hist + iter, N, d_heavy);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_heavy, d_sum_heavy_hist + iter, N, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            heavy_reduce_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_heavy).count();

            auto t_copy_pre = clock_type::now();
            if (opts.center_update_guard_enable) {
                copy_center_kernel<<<blocksCopy, threadsN, 0, stream>>>(d_Cen, totalVec, d_Cen_prev);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            copy_vec_center_kernel<<<blocksCopy, threadsN, 0, stream>>>(d_Vec, d_center_pre, totalVec, d_Vec_next, d_Cen);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            copy_pre_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_copy_pre).count();

            auto t_agg = clock_type::now();
            for (int q0 = 0; q0 < N; q0 += BQ) {
                int B = std::min(BQ, N - q0);
                launch_aggregation(stream, q0, B, hk, rhok);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            aggregation_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_agg).count();

            std::swap(d_Vec, d_Vec_next);

            auto t_proj_final = clock_type::now();
            if (opts.center_reduce_tangent_final) {
                project_center_off_tangent_kernel<<<blocksN, threadsN, 0, stream>>>(d_Xrm, d_Vec, D, N, d_Cen);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            if (opts.center_update_guard_enable) {
                center_guard_update_kernel<<<blocksN, threadsN, 0, stream>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, D, N, opts.center_update_guard_ratio,
                    0,
                    d_num_neigh_hist_max, d_center_guard_hist_max, d_Cen_prev, d_center_accept, d_Cen);
                CUDA_CHECK(cudaPeekAtLastError());
                CUDA_CHECK(cub::DeviceReduce::Sum(d_reduce_temp, temp_heavy, d_center_accept, d_sum_center_accept_hist + iter, N, stream));
            } else {
                update_center_guard_hist_only_kernel<<<blocksN, threadsN, 0, stream>>>(
                    d_num_neigh, d_weight_sum, d_weight_sum_sq, N,
                    0,
                    d_num_neigh_hist_max, d_center_guard_hist_max);
                CUDA_CHECK(cudaPeekAtLastError());
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            project_final_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_proj_final).count();

            iter_total_sec[iter] = std::chrono::duration<double>(clock_type::now() - t_iter).count();
        }
    } else if (opts.use_cuda_graphs && K > 0) {
        used_cuda_graphs = true;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_iteration_sequence(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    } else {
        launch_iteration_sequence(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<int> sum_num_hist((size_t)K, 0), sum_heavy_hist((size_t)K, 0), sum_center_accept_hist((size_t)K, 0), sum_vector_accept_hist((size_t)K, 0);
    CUDA_CHECK(cudaMemcpy(sum_num_hist.data(), d_sum_num_hist, sizeof(int) * (size_t)K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sum_heavy_hist.data(), d_sum_heavy_hist, sizeof(int) * (size_t)K, cudaMemcpyDeviceToHost));
    if (opts.center_update_guard_enable) {
        CUDA_CHECK(cudaMemcpy(sum_center_accept_hist.data(), d_sum_center_accept_hist, sizeof(int) * (size_t)K, cudaMemcpyDeviceToHost));
    }
    if (opts.vector_update_guard_enable) {
        CUDA_CHECK(cudaMemcpy(sum_vector_accept_hist.data(), d_sum_vector_accept_hist, sizeof(int) * (size_t)K, cudaMemcpyDeviceToHost));
    }
    for (size_t iter = 0; iter < K; ++iter) {
        double mean_num = (double)sum_num_hist[iter] / std::max(N, 1);
        num_neigh_iter[iter] = (int32_t)std::llround(mean_num);
        heavy_frac_iter[iter] = (float)((double)sum_heavy_hist[iter] / std::max(N, 1));
        center_accept_frac_iter[iter] = opts.center_update_guard_enable
            ? (float)((double)sum_center_accept_hist[iter] / std::max(N, 1))
            : 1.0f;
        vector_accept_frac_iter[iter] = opts.vector_update_guard_enable
            ? (float)((double)sum_vector_accept_hist[iter] / std::max(N, 1))
            : 1.0f;
    }

    std::vector<float> Vec_rm((size_t)N * (size_t)D), Cen_rm((size_t)N * (size_t)D);
    std::vector<int32_t> Num(N), NumHistMax(N);
    std::vector<uint8_t> Heavy(N);

    auto t_copyout = clock_type::now();
    CUDA_CHECK(cudaMemcpy(Vec_rm.data(), d_Vec, sizeof(float) * Vec_rm.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Cen_rm.data(), d_Cen, sizeof(float) * Cen_rm.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Num.data(), d_num_neigh, sizeof(int32_t) * Num.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(NumHistMax.data(), d_num_neigh_hist_max, sizeof(int32_t) * NumHistMax.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Heavy.data(), d_heavy, sizeof(uint8_t) * Heavy.size(), cudaMemcpyDeviceToHost));
    copyout_sec = std::chrono::duration<double>(clock_type::now() - t_copyout).count();

    std::vector<float> Vec_cm, Cen_cm;
    column_major_from_row_major(Vec_rm, D, N, Vec_cm);
    column_major_from_row_major(Cen_rm, D, N, Cen_cm);

    std::vector<float> Grad_cm;
    std::vector<float> WSum((size_t)N), WSumSq((size_t)N), CenterGuardHistMax((size_t)N), CenterGuardMetric((size_t)N), VectorGuardHistMax((size_t)N), VectorGuardMetric((size_t)N);

    if (opts.return_debug) {
        std::vector<float> Grad_rm((size_t)N * (size_t)D);
        CUDA_CHECK(cudaMemcpy(Grad_rm.data(), d_grad, sizeof(float) * Grad_rm.size(), cudaMemcpyDeviceToHost));
        column_major_from_row_major(Grad_rm, D, N, Grad_cm);
    }

    CUDA_CHECK(cudaMemcpy(WSum.data(), d_weight_sum, sizeof(float) * WSum.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(WSumSq.data(), d_weight_sum_sq, sizeof(float) * WSumSq.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(CenterGuardHistMax.data(), d_center_guard_hist_max, sizeof(float) * CenterGuardHistMax.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(VectorGuardHistMax.data(), d_vector_guard_hist_max, sizeof(float) * VectorGuardHistMax.size(), cudaMemcpyDeviceToHost));
    for (size_t idx = 0; idx < CenterGuardMetric.size(); ++idx) {
        float cur_n = (float)Num[idx];
        if (opts.center_update_guard_use_ess) {
            float denom = fmaxf(WSumSq[idx], 1.0e-20f);
            float ess = (denom > 0.0f) ? (WSum[idx] * WSum[idx]) / denom : 0.0f;
            ess = fminf(ess, cur_n);
            CenterGuardMetric[idx] = fmaxf(ess, 0.0f);
        } else {
            CenterGuardMetric[idx] = cur_n;
        }
        if (opts.vector_update_guard_use_ess) {
            float denom_v = fmaxf(WSumSq[idx], 1.0e-20f);
            float ess_v = (denom_v > 0.0f) ? (WSum[idx] * WSum[idx]) / denom_v : 0.0f;
            ess_v = fminf(ess_v, cur_n);
            VectorGuardMetric[idx] = fmaxf(ess_v, 0.0f);
        } else {
            VectorGuardMetric[idx] = cur_n;
        }
    }

    if (opts.return_debug) {
        const char* fields[] = {
            "Vec_next","Cen_next","num_neigh","num_neigh_hist_max","heavy_mask",
            "grad","weight_sum","weight_sum_sq","center_guard_hist_max","center_guard_metric",
            "vector_guard_hist_max","vector_guard_metric",
            "num_neigh_iter","heavy_frac_iter","center_accept_frac_iter","vector_accept_frac_iter","cache_enabled","profile"
        };
        plhs[0] = mxCreateStructMatrix(1, 1, 18, fields);
    } else {
        const char* fields[] = {
            "Vec_next","Cen_next","num_neigh","num_neigh_hist_max","heavy_mask",
            "weight_sum","weight_sum_sq","center_guard_hist_max","center_guard_metric",
            "vector_guard_hist_max","vector_guard_metric",
            "num_neigh_iter","heavy_frac_iter","center_accept_frac_iter","vector_accept_frac_iter","cache_enabled","profile"
        };
        plhs[0] = mxCreateStructMatrix(1, 1, 17, fields);
    }

    mxArray* mxVecOut = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCenOut = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxNum = mxCreateNumericMatrix(1, N, mxINT32_CLASS, mxREAL);
    mxArray* mxNumHistMax = mxCreateNumericMatrix(1, N, mxINT32_CLASS, mxREAL);
    mxArray* mxHeavy = mxCreateNumericMatrix(1, N, mxUINT8_CLASS, mxREAL);
    mxArray* mxWSum = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxWSumSq = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCenterGuardHistMax = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCenterGuardMetric = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxVectorGuardHistMax = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxVectorGuardMetric = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxNumIter = mxCreateNumericMatrix(1, (mwSize)K, mxINT32_CLASS, mxREAL);
    mxArray* mxHeavyIter = mxCreateNumericMatrix(1, (mwSize)K, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCenterAcceptIter = mxCreateNumericMatrix(1, (mwSize)K, mxSINGLE_CLASS, mxREAL);
    mxArray* mxVectorAcceptIter = mxCreateNumericMatrix(1, (mwSize)K, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCache = mxCreateDoubleScalar(cache_enabled ? 1.0 : 0.0);

    std::memcpy(mxGetData(mxVecOut), Vec_cm.data(), sizeof(float) * Vec_cm.size());
    std::memcpy(mxGetData(mxCenOut), Cen_cm.data(), sizeof(float) * Cen_cm.size());
    std::memcpy(mxGetData(mxNum), Num.data(), sizeof(int32_t) * Num.size());
    std::memcpy(mxGetData(mxNumHistMax), NumHistMax.data(), sizeof(int32_t) * NumHistMax.size());
    std::memcpy(mxGetData(mxHeavy), Heavy.data(), sizeof(uint8_t) * Heavy.size());
    std::memcpy(mxGetData(mxWSum), WSum.data(), sizeof(float) * WSum.size());
    std::memcpy(mxGetData(mxWSumSq), WSumSq.data(), sizeof(float) * WSumSq.size());
    std::memcpy(mxGetData(mxCenterGuardHistMax), CenterGuardHistMax.data(), sizeof(float) * CenterGuardHistMax.size());
    std::memcpy(mxGetData(mxCenterGuardMetric), CenterGuardMetric.data(), sizeof(float) * CenterGuardMetric.size());
    std::memcpy(mxGetData(mxVectorGuardHistMax), VectorGuardHistMax.data(), sizeof(float) * VectorGuardHistMax.size());
    std::memcpy(mxGetData(mxVectorGuardMetric), VectorGuardMetric.data(), sizeof(float) * VectorGuardMetric.size());
    std::memcpy(mxGetData(mxNumIter), num_neigh_iter.data(), sizeof(int32_t) * num_neigh_iter.size());
    std::memcpy(mxGetData(mxHeavyIter), heavy_frac_iter.data(), sizeof(float) * heavy_frac_iter.size());
    std::memcpy(mxGetData(mxCenterAcceptIter), center_accept_frac_iter.data(), sizeof(float) * center_accept_frac_iter.size());
    std::memcpy(mxGetData(mxVectorAcceptIter), vector_accept_frac_iter.data(), sizeof(float) * vector_accept_frac_iter.size());

    mxSetField(plhs[0], 0, "Vec_next", mxVecOut);
    mxSetField(plhs[0], 0, "Cen_next", mxCenOut);
    mxSetField(plhs[0], 0, "num_neigh", mxNum);
    mxSetField(plhs[0], 0, "num_neigh_hist_max", mxNumHistMax);
    mxSetField(plhs[0], 0, "heavy_mask", mxHeavy);
    mxSetField(plhs[0], 0, "weight_sum", mxWSum);
    mxSetField(plhs[0], 0, "weight_sum_sq", mxWSumSq);
    mxSetField(plhs[0], 0, "center_guard_hist_max", mxCenterGuardHistMax);
    mxSetField(plhs[0], 0, "center_guard_metric", mxCenterGuardMetric);
    mxSetField(plhs[0], 0, "vector_guard_hist_max", mxVectorGuardHistMax);
    mxSetField(plhs[0], 0, "vector_guard_metric", mxVectorGuardMetric);
    mxSetField(plhs[0], 0, "num_neigh_iter", mxNumIter);
    mxSetField(plhs[0], 0, "heavy_frac_iter", mxHeavyIter);
    mxSetField(plhs[0], 0, "center_accept_frac_iter", mxCenterAcceptIter);
    mxSetField(plhs[0], 0, "vector_accept_frac_iter", mxVectorAcceptIter);
    mxSetField(plhs[0], 0, "cache_enabled", mxCache);

    const char* profile_fields[] = {
        "setup_sec","copyout_sec","total_sec",
        "iter_total_sec","regression_sec","project_pre_sec","heavy_reduce_sec",
        "copy_pre_sec","aggregation_sec","project_final_sec",
        "regression_sec_sum","project_pre_sec_sum","heavy_reduce_sec_sum",
        "copy_pre_sec_sum","aggregation_sec_sum","project_final_sec_sum",
        "BQ","direct_path","graphs_enabled","warps_per_block"
    };
    mxArray* mxProfile = mxCreateStructMatrix(1, 1, 20, profile_fields);

    auto make_row_double = [](const std::vector<double>& v) -> mxArray* {
        mxArray* a = mxCreateDoubleMatrix(1, (mwSize)v.size(), mxREAL);
        if (!v.empty()) std::memcpy(mxGetPr(a), v.data(), sizeof(double) * v.size());
        return a;
    };
    auto sum_not_nan = [](const std::vector<double>& v) -> double {
        double s = 0.0;
        bool any = false;
        for (double x : v) {
            if (std::isfinite(x)) { s += x; any = true; }
        }
        return any ? s : std::numeric_limits<double>::quiet_NaN();
    };

    mxSetField(mxProfile, 0, "setup_sec", mxCreateDoubleScalar(setup_sec));
    mxSetField(mxProfile, 0, "copyout_sec", mxCreateDoubleScalar(copyout_sec));
    mxSetField(mxProfile, 0, "total_sec", mxCreateDoubleScalar(std::chrono::duration<double>(clock_type::now() - phase2_total_start).count()));
    mxSetField(mxProfile, 0, "iter_total_sec", make_row_double(iter_total_sec));
    mxSetField(mxProfile, 0, "regression_sec", make_row_double(regression_sec));
    mxSetField(mxProfile, 0, "project_pre_sec", make_row_double(project_pre_sec));
    mxSetField(mxProfile, 0, "heavy_reduce_sec", make_row_double(heavy_reduce_sec));
    mxSetField(mxProfile, 0, "copy_pre_sec", make_row_double(copy_pre_sec));
    mxSetField(mxProfile, 0, "aggregation_sec", make_row_double(aggregation_sec));
    mxSetField(mxProfile, 0, "project_final_sec", make_row_double(project_final_sec));
    mxSetField(mxProfile, 0, "regression_sec_sum", mxCreateDoubleScalar(sum_not_nan(regression_sec)));
    mxSetField(mxProfile, 0, "project_pre_sec_sum", mxCreateDoubleScalar(sum_not_nan(project_pre_sec)));
    mxSetField(mxProfile, 0, "heavy_reduce_sec_sum", mxCreateDoubleScalar(sum_not_nan(heavy_reduce_sec)));
    mxSetField(mxProfile, 0, "copy_pre_sec_sum", mxCreateDoubleScalar(sum_not_nan(copy_pre_sec)));
    mxSetField(mxProfile, 0, "aggregation_sec_sum", mxCreateDoubleScalar(sum_not_nan(aggregation_sec)));
    mxSetField(mxProfile, 0, "project_final_sec_sum", mxCreateDoubleScalar(sum_not_nan(project_final_sec)));
    mxSetField(mxProfile, 0, "BQ", mxCreateDoubleScalar((double)BQ));
    mxSetField(mxProfile, 0, "direct_path", mxCreateDoubleScalar(direct_path ? 1.0 : 0.0));
    mxSetField(mxProfile, 0, "graphs_enabled", mxCreateDoubleScalar(used_cuda_graphs ? 1.0 : 0.0));
    mxSetField(mxProfile, 0, "warps_per_block", mxCreateDoubleScalar((double)opts.warps_per_block));

    mxSetField(plhs[0], 0, "profile", mxProfile);

    if (opts.return_debug) {
        mxArray* mxGrad = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
        std::memcpy(mxGetData(mxGrad), Grad_cm.data(), sizeof(float) * Grad_cm.size());
        mxSetField(plhs[0], 0, "grad", mxGrad);
    }

    if (d_sort_temp) cudaFree(d_sort_temp);
    if (d_offsets) cudaFree(d_offsets);
    if (d_vals_out) cudaFree(d_vals_out);
    if (d_vals_in) cudaFree(d_vals_in);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_keys_in) cudaFree(d_keys_in);

    cudaFree(d_Xrm); cudaFree(d_Y);
    cudaFree(d_Vec); cudaFree(d_Cen); cudaFree(d_Vec_next); cudaFree(d_Cen_prev);
    cudaFree(d_grad); cudaFree(d_grad_prev); cudaFree(d_center_pre); cudaFree(d_weight_sum); cudaFree(d_weight_sum_sq);
    cudaFree(d_num_neigh); cudaFree(d_num_neigh_hist_max); cudaFree(d_center_guard_hist_max); cudaFree(d_vector_guard_hist_max);
    cudaFree(d_heavy); cudaFree(d_center_accept); cudaFree(d_vector_accept);
    cudaFree(d_sum_num); cudaFree(d_sum_heavy); cudaFree(d_reduce_temp); cudaFree(d_phase2_neigh_mask_bits);
    cudaFree(d_phase2_scalar0); cudaFree(d_phase2_scalar1);
    cudaFree(d_phase2_vecbuf0); cudaFree(d_phase2_vecbuf1); cudaFree(d_phase2_vecbuf2);
    cudaFree(d_phase2_mat_accum); cudaFree(d_phase2_count_accum);
    cudaFree(d_sum_num_hist); cudaFree(d_sum_heavy_hist); cudaFree(d_sum_center_accept_hist); cudaFree(d_sum_vector_accept_hist);
    if (stream) cudaStreamDestroy(stream);
}
