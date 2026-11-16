#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:cuda", "%s (%s:%d)", cudaGetErrorString(code), file, line);
    }
}

#ifndef MAX_D_MANOR
#define MAX_D_MANOR 32
#endif

#ifndef MAX_R_MANOR
#define MAX_R_MANOR 64
#endif

#ifndef PHASE1_TILE_TOPK_LOCAL_MAX
#define PHASE1_TILE_TOPK_LOCAL_MAX 4096
#endif

// Current production path uses radix exact selection.
// Earlier shared/tile top-k experiments are retained only for compatibility
// in profile/output fields and should not be treated as the preferred speed path.

struct Phase1Opts {
    float gpu_mem_fraction = 0.92f;
    float reserve_fraction = 0.08f;
    double reserve_bytes   = 6.0 * 1024.0 * 1024.0 * 1024.0;
    float heavy_factor     = 0.05f;
    float jitter           = 1e-6f;
    int   power_iter_steps = 12;
    int   min_query_block  = 128;
    int   max_query_block  = 4096;
    bool  return_debug      = false;
    bool  gpu_pass2         = true;
    bool  fullsort_fallback = true; // retained for profile/console compatibility
    bool  profile_timers = false;
    bool  silent         = false;
};

static Phase1Opts parse_opts(const mxArray* opts)
{
    Phase1Opts o;
    if (!mxIsStruct(opts)) return o;
    auto getf = [&](const char* name) -> const mxArray* { return mxGetField(opts, 0, name); };
    if (const mxArray* a = getf("gpu_mem_fraction")) o.gpu_mem_fraction = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("reserve_fraction")) o.reserve_fraction = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("reserve_bytes"))    o.reserve_bytes    = mxGetScalar(a);
    if (const mxArray* a = getf("heavy_factor"))     o.heavy_factor     = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("jitter"))           o.jitter           = static_cast<float>(mxGetScalar(a));
    if (const mxArray* a = getf("power_iter_steps")) o.power_iter_steps = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("min_query_block"))  o.min_query_block  = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("max_query_block"))  o.max_query_block  = static_cast<int>(mxGetScalar(a));
    if (const mxArray* a = getf("return_debug"))     o.return_debug      = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("gpu_pass2"))        o.gpu_pass2         = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("fullsort_fallback"))o.fullsort_fallback = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("profile_timers")) o.profile_timers = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (const mxArray* a = getf("silent")) o.silent = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    return o;
}

static inline int clamp_int(int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); }


static int choose_query_block(int D, int kSlice, int tmax, int R, const Phase1Opts& opts)
{
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    double reserve = std::max(opts.reserve_bytes, static_cast<double>(total_b) * static_cast<double>(opts.reserve_fraction));
    double usable  = std::max(0.0, (static_cast<double>(free_b) - reserve) * static_cast<double>(opts.gpu_mem_fraction));

    double per_query =
        2.1 * (
            2.0 * static_cast<double>(kSlice) * sizeof(float) +
            2.0 * static_cast<double>(kSlice) * sizeof(uint16_t) +
            1.0 * static_cast<double>(tmax)   * sizeof(float) +
            1.0 * static_cast<double>(tmax)   * sizeof(uint16_t) +
            1.0 * static_cast<double>(R)      * sizeof(int) +
            1.0 * static_cast<double>(D)      * static_cast<double>(D) * sizeof(float) +
            1.0 * static_cast<double>(D)      * sizeof(float) * 8.0 +
            1.0 * static_cast<double>(R)      * static_cast<double>(D) * sizeof(float) +
            1.0 * static_cast<double>(R)      * static_cast<double>(D) * static_cast<double>(D) * sizeof(float)
        );

    int BQ = static_cast<int>(std::floor(usable / std::max(per_query, 1.0)));
    BQ = clamp_int(BQ, opts.min_query_block, opts.max_query_block);
    return BQ;
}


// -----------------------------------------------------------------------------
// Packed-neighborhood bit helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ void bitset_zero_row_u32(uint32_t* row, int words)
{
    for (int w = 0; w < words; ++w) row[w] = 0u;
}

__device__ __forceinline__ void bitset_set_u32(uint32_t* row, int pos)
{
    row[pos >> 5] |= (uint32_t(1) << (pos & 31));
}

__device__ __forceinline__ bool bitset_get_u32(const uint32_t* row, int pos)
{
    return ((row[pos >> 5] >> (pos & 31)) & uint32_t(1)) != 0u;
}

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

__device__ static void jacobi_eigen_device(const float* A_in, int D, float* eigvals, float* eigvecs)
{
    float A[MAX_D_MANOR * MAX_D_MANOR];
    for (int i = 0; i < D * D; ++i) A[i] = A_in[i];
    for (int i = 0; i < D * D; ++i) eigvecs[i] = 0.0f;
    for (int i = 0; i < D; ++i) eigvecs[i * D + i] = 1.0f;

    const int max_sweeps = 32;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float max_off = 0.0f;
        int p = 0, q = (D > 1 ? 1 : 0);
        for (int i = 0; i < D; ++i) {
            for (int j = i + 1; j < D; ++j) {
                float v = fabsf(A[i * D + j]);
                if (v > max_off) { max_off = v; p = i; q = j; }
            }
        }
        if (max_off < 1e-10f) break;

        float app = A[p * D + p];
        float aqq = A[q * D + q];
        float apq = A[p * D + q];
        float denom = 2.0f * apq;
        if (fabsf(denom) < 1.0e-30f) break;
        float tau = (aqq - app) / denom;
        float t = (tau >= 0.0f)
                ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
        float c = rsqrtf(1.0f + t * t);
        float s = t * c;

        for (int k = 0; k < D; ++k) {
            if (k == p || k == q) continue;
            float aik = A[p * D + k];
            float aqk = A[q * D + k];
            A[p * D + k] = c * aik - s * aqk;
            A[k * D + p] = A[p * D + k];
            A[q * D + k] = s * aik + c * aqk;
            A[k * D + q] = A[q * D + k];
        }

        float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        A[p * D + p] = new_app;
        A[q * D + q] = new_aqq;
        A[p * D + q] = 0.0f;
        A[q * D + p] = 0.0f;

        for (int k = 0; k < D; ++k) {
            float vip = eigvecs[k * D + p];
            float viq = eigvecs[k * D + q];
            eigvecs[k * D + p] = c * vip - s * viq;
            eigvecs[k * D + q] = s * vip + c * viq;
        }
    }

    for (int i = 0; i < D; ++i) eigvals[i] = A[i * D + i];
    for (int i = 0; i < D - 1; ++i) {
        int best = i;
        for (int j = i + 1; j < D; ++j) if (eigvals[j] < eigvals[best]) best = j;
        if (best != i) {
            float tv = eigvals[i]; eigvals[i] = eigvals[best]; eigvals[best] = tv;
            for (int k = 0; k < D; ++k) {
                float tmp = eigvecs[k * D + i];
                eigvecs[k * D + i] = eigvecs[k * D + best];
                eigvecs[k * D + best] = tmp;
            }
        }
    }
}

__device__ static inline void normalize_vec_device(float* v, int D)
{
    float nrm2 = 0.0f;
    for (int d = 0; d < D; ++d) nrm2 += v[d] * v[d];
    float inv = rsqrtf(fmaxf(nrm2, 1.0e-30f));
    for (int d = 0; d < D; ++d) v[d] *= inv;
}

__device__ static inline int median3_index_device(int a, int b, int c)
{
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
}

// -----------------------------------------------------------------------------
// Phase-1 kernels
// -----------------------------------------------------------------------------

template <typename OffsetT>
__global__ __launch_bounds__(256, 2) void phase1_dist_offset_kernel_cm_shared(
    const float* __restrict__ Xcm,
    int D, int N,
    int q0, int B, int kSlice,
    float* __restrict__ keys_out,
    OffsetT* __restrict__ vals_out)
{
    int b = blockIdx.y;
    if (b >= B) return;
    int i = q0 + b;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= kSlice) return;

    extern __shared__ float xi_sh[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) xi_sh[d] = Xcm[d + i * D];
    __syncthreads();

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;

    if (start > N - kSlice) start = N - kSlice;
    int j = start + s;

    float acc = 0.0f;
#pragma unroll
    for (int d = 0; d < MAX_D_MANOR; ++d) {
        if (d < D) {
            float dx = Xcm[d + j * D] - xi_sh[d];
            acc += dx * dx;
        }
    }

    size_t tid = static_cast<size_t>(b) * static_cast<size_t>(kSlice) + static_cast<size_t>(s);
    keys_out[tid] = acc;
    vals_out[tid] = static_cast<OffsetT>(s);
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
    int store = left;
    for (int i = left; i < right; ++i) {
        if (keys[i] < pivot || (keys[i] == pivot && vals[i] < vals[right])) {
            pair_swap_device(keys, vals, i, store);
            ++store;
        }
    }
    pair_swap_device(keys, vals, store, right);
    return store;
}

template <typename OffsetT>
__device__ void quickselect_pairs_device(float* keys, OffsetT* vals, int n, int kth)
{
    if (kth < 0 || kth >= n) return;
    int left = 0, right = n - 1;
    while (left < right) {
        int mid = left + ((right - left) >> 1);
        int pivot = partition_pairs_device(keys, vals, left, right, mid);
        if (pivot == kth) return;
        if (kth < pivot) right = pivot - 1;
        else left = pivot + 1;
    }
}

template <typename OffsetT>
__device__ void insertion_sort_pairs_device(float* keys, OffsetT* vals, int n)
{
    for (int i = 1; i < n; ++i) {
        float k = keys[i];
        OffsetT v = vals[i];
        int j = i - 1;
        while (j >= 0 && (keys[j] > k || (keys[j] == k && vals[j] > v))) {
            keys[j + 1] = keys[j];
            vals[j + 1] = vals[j];
            --j;
        }
        keys[j + 1] = k;
        vals[j + 1] = v;
    }
}

template <typename OffsetT, int BLOCK_THREADS>
__global__ __launch_bounds__(BLOCK_THREADS, 2) void phase1_dist_topk_shared_kernel(
    const float* __restrict__ Xcm,
    int D, int N,
    int q0, int B, int kSlice, int tmax,
    float* __restrict__ top_keys,
    OffsetT* __restrict__ top_vals)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int i = q0 + b;

    extern __shared__ unsigned char smem[];
    float* s_keys = reinterpret_cast<float*>(smem);
    OffsetT* s_vals = reinterpret_cast<OffsetT*>(s_keys + kSlice);

    __shared__ float xi_sh[MAX_D_MANOR];
    for (int d = threadIdx.x; d < D; d += BLOCK_THREADS) xi_sh[d] = Xcm[d + i * D];
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
                float dx = Xcm[d + j * D] - xi_sh[d];
                acc += dx * dx;
            }
        }
        s_keys[s] = acc;
        s_vals[s] = static_cast<OffsetT>(s);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        quickselect_pairs_device(s_keys, s_vals, kSlice, tmax - 1);
        insertion_sort_pairs_device(s_keys, s_vals, tmax);
        float* outk = top_keys + (size_t)b * (size_t)tmax;
        OffsetT* outv = top_vals + (size_t)b * (size_t)tmax;
        for (int p = 0; p < tmax; ++p) {
            outk[p] = s_keys[p];
            outv[p] = s_vals[p];
        }
    }
}


template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int LOCAL_KEEP_MAX>
__global__ __launch_bounds__(BLOCK_THREADS, 1) void phase1_dist_topk_tile_exact_kernel(
    const float* __restrict__ Xcm,
    int D, int N,
    int q0, int B, int kSlice, int tmax,
    float* __restrict__ top_keys,
    uint16_t* __restrict__ top_vals)
{
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;
    using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, uint16_t>;
    struct SharedStorage {
        typename BlockSort::TempStorage sort;
        float tile_keys[TILE_ITEMS];
        uint16_t tile_vals[TILE_ITEMS];
    };
    __shared__ SharedStorage sh;
    __shared__ float xi_sh[MAX_D_MANOR];
    __shared__ int tile_valid;

    int b = blockIdx.x;
    if (b >= B) return;
    int i = q0 + b;

    for (int d = threadIdx.x; d < D; d += BLOCK_THREADS) xi_sh[d] = Xcm[d + i * D];
    __syncthreads();

    const int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    float outk_tmp[ITEMS_PER_THREAD];
    uint16_t outv_tmp[ITEMS_PER_THREAD];
    float* outk = top_keys + (size_t)b * (size_t)tmax;
    uint16_t* outv = top_vals + (size_t)b * (size_t)tmax;
    int cur_count = 0;

    for (int tile0 = 0; tile0 < kSlice; tile0 += TILE_ITEMS) {
        const int base_rel = tile0 + threadIdx.x * ITEMS_PER_THREAD;
#pragma unroll
        for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
            const int rel = base_rel + it;
            if (rel < kSlice) {
                const int j = start + rel;
                float acc = 0.0f;
#pragma unroll
                for (int d = 0; d < MAX_D_MANOR; ++d) {
                    if (d < D) {
                        float dx = Xcm[d + j * D] - xi_sh[d];
                        acc += dx * dx;
                    }
                }
                outk_tmp[it] = acc;
                outv_tmp[it] = static_cast<uint16_t>(rel);
            } else {
                outk_tmp[it] = INFINITY;
                outv_tmp[it] = 0;
            }
        }
        __syncthreads();
        BlockSort(sh.sort).Sort(outk_tmp, outv_tmp);
        __syncthreads();
#pragma unroll
        for (int it = 0; it < ITEMS_PER_THREAD; ++it) {
            int idx = threadIdx.x * ITEMS_PER_THREAD + it;
            sh.tile_keys[idx] = outk_tmp[it];
            sh.tile_vals[idx] = outv_tmp[it];
        }
        if (threadIdx.x == 0) {
            tile_valid = min(TILE_ITEMS, kSlice - tile0);
            const int tile_keep = min(tile_valid, tmax);
            if (cur_count == 0) {
                for (int p = 0; p < tile_keep; ++p) {
                    outk[p] = sh.tile_keys[p];
                    outv[p] = sh.tile_vals[p];
                }
                cur_count = tile_keep;
            } else {
                float merge_keys[LOCAL_KEEP_MAX];
                uint16_t merge_vals[LOCAL_KEEP_MAX];
                const int max_out = min(tmax, cur_count + tile_keep);
                int a = 0, c = 0, o = 0;
                while (o < max_out) {
                    bool take_old = (c >= tile_keep) || (a < cur_count && outk[a] <= sh.tile_keys[c]);
                    if (take_old) {
                        merge_keys[o] = outk[a];
                        merge_vals[o] = outv[a];
                        ++a;
                    } else {
                        merge_keys[o] = sh.tile_keys[c];
                        merge_vals[o] = sh.tile_vals[c];
                        ++c;
                    }
                    ++o;
                }
                for (int p = 0; p < max_out; ++p) {
                    outk[p] = merge_keys[p];
                    outv[p] = merge_vals[p];
                }
                cur_count = max_out;
            }
        }
        __syncthreads();
    }
}

template <typename OffsetT>
__global__ void compact_topk_kernel_offset(
    const float* __restrict__ sorted_keys,
    const OffsetT* __restrict__ sorted_vals,
    int B, int kSlice, int tmax,
    float* __restrict__ top_keys,
    OffsetT* __restrict__ top_vals)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * tmax;
    if (tid >= total) return;
    int b = tid / tmax;
    int p = tid - b * tmax;
    top_keys[tid] = sorted_keys[b * kSlice + p];
    top_vals[tid] = sorted_vals[b * kSlice + p];
}



__global__ __launch_bounds__(32, 4) void phase1_chunkstats_from_top_kernel(
    const float* __restrict__ Xcm,
    const uint16_t* __restrict__ top_off,
    const int* __restrict__ cuts,
    int D, int N, int kSlice, int tmax, int R,
    int q0, int B,
    float* __restrict__ chunk_sum1,
    float* __restrict__ chunk_sum2)
{
    int b = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (b >= B) return;

    const int i = q0 + b;
    const int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const float* xi = Xcm + (size_t)i * (size_t)D;
    const uint16_t* off_b = top_off + (size_t)b * (size_t)tmax;

    __shared__ float sh_dx[MAX_D_MANOR];

    for (int r = 0; r < R; ++r) {
        int prev_cut = (r == 0 ? 0 : cuts[r - 1]);
        int tsel = cuts[r];
        if (tsel < 1) tsel = 1;
        if (tsel > tmax) tsel = tmax;
        if (prev_cut < 0) prev_cut = 0;
        if (prev_cut > tsel) prev_cut = tsel;

        float s1 = 0.0f;
        float s2row[MAX_D_MANOR];
        if (lane < D) {
#pragma unroll
            for (int c = 0; c < MAX_D_MANOR; ++c) s2row[c] = 0.0f;
        }

        for (int p = prev_cut; p < tsel; ++p) {
            int rel = (int)off_b[p];
            int j = start + rel;
            if (lane < D) {
                sh_dx[lane] = Xcm[(size_t)j * (size_t)D + (size_t)lane] - xi[lane];
            }
            __syncwarp();
            if (lane < D) {
                float xa = sh_dx[lane];
                s1 += xa;
#pragma unroll
                for (int c = 0; c < MAX_D_MANOR; ++c) {
                    if (c < D) s2row[c] += xa * sh_dx[c];
                }
            }
            __syncwarp();
        }

        if (lane < D) {
            size_t base1 = ((size_t)b * (size_t)R + (size_t)r) * (size_t)D;
            chunk_sum1[base1 + (size_t)lane] = s1;
            size_t base2 = ((((size_t)b * (size_t)R + (size_t)r) * (size_t)D) + (size_t)lane) * (size_t)D;
            for (int c = 0; c < D; ++c) chunk_sum2[base2 + (size_t)c] = s2row[c];
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(32, 4) void phase1_select_from_sorted_kernel(
    const float* __restrict__ Xcm,
    const float* __restrict__ top_dist2,
    const uint16_t* __restrict__ top_off,
    const int* __restrict__ cuts,
    const float* __restrict__ chunk_sum1,
    const float* __restrict__ chunk_sum2,
    int D, int N, int kSlice, int tmax, int R,
    int q0, int B,
    float jitter,
    float* __restrict__ Vec_raw,
    float* __restrict__ Cen_raw,
    float* __restrict__ Rad0,
    float* __restrict__ axial_tan0,
    int32_t* __restrict__ num_neigh0,
    float* __restrict__ geomtype0,
    float* __restrict__ SlocMin0,
    float* __restrict__ SlocMid0,
    float* __restrict__ SlocMax0,
    uint32_t* __restrict__ neigh_mask0_bits,
    int mask_words_per_row)
{
    int b = blockIdx.x;
    if (b >= B || threadIdx.x != 0) return;

    const int i = q0 + b;
    const int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

	    struct Candidate {
	        int r;
	        float geom, score, smin, smid, smax;
	        float mean[MAX_D_MANOR];
	        float v[MAX_D_MANOR];
	    } best[3];
	    for (int k = 0; k < 3; ++k) {
	        best[k].r = -1;
	        best[k].geom = -1.0e30f;
	        best[k].score = -1.0f;
	        best[k].smin = best[k].smid = best[k].smax = 0.0f;
	        for (int d = 0; d < D; ++d) { best[k].mean[d] = 0.0f; best[k].v[d] = 0.0f; }
	    }

    float mean_cur[MAX_D_MANOR];
    float sum1[MAX_D_MANOR];
    float cov[MAX_D_MANOR * MAX_D_MANOR];
    float sum2[MAX_D_MANOR * MAX_D_MANOR];
    float eigvals[MAX_D_MANOR];
    float eigvecs[MAX_D_MANOR * MAX_D_MANOR];
    float v_cur[MAX_D_MANOR];

    for (int d = 0; d < D; ++d) sum1[d] = 0.0f;
    for (int t = 0; t < D * D; ++t) sum2[t] = 0.0f;

    for (int r = 0; r < R; ++r) {
        int tsel = cuts[r];
        if (tsel < 1) tsel = 1;
        if (tsel > tmax) tsel = tmax;

        size_t base1 = ((size_t)b * (size_t)R + (size_t)r) * (size_t)D;
        size_t base2 = ((size_t)b * (size_t)R + (size_t)r) * (size_t)D * (size_t)D;
        for (int d = 0; d < D; ++d) sum1[d] += chunk_sum1[base1 + (size_t)d];
        for (int t = 0; t < D * D; ++t) sum2[t] += chunk_sum2[base2 + (size_t)t];

        float inv_t = 1.0f / static_cast<float>(tsel);
        for (int d = 0; d < D; ++d) mean_cur[d] = sum1[d] * inv_t;

        for (int a = 0; a < D; ++a) {
            float ma = mean_cur[a];
            for (int c = 0; c < D; ++c) {
                float val = sum2[a * D + c] * inv_t - ma * mean_cur[c];
                if (a == c) val += jitter;
                cov[a * D + c] = val;
            }
        }

        jacobi_eigen_device(cov, D, eigvals, eigvecs);
        for (int kk = 0; kk < D; ++kk) eigvals[kk] = fmaxf(eigvals[kk], 1.0e-12f);

        float smin_cur = eigvals[0];
        float smax_cur = eigvals[D - 1];
        float lo_num = eigvals[min(1, D - 1)];
        float hi_num = eigvals[max(D - 2, 0)];
        float smid_cur = sqrtf(fmaxf(lo_num * hi_num, 1.0e-20f));
        float geom_cur = logf(fmaxf((smid_cur * smid_cur) / fmaxf(smin_cur * smax_cur, 1.0e-20f), 1.0e-20f));
        int choose_col = (geom_cur > 0.0f ? 0 : (D - 1));
        for (int d = 0; d < D; ++d) v_cur[d] = eigvecs[d * D + choose_col];
        normalize_vec_device(v_cur, D);

	        float score_cur = fabsf(geom_cur);
	        int slot = -1;
	        if (best[0].r < 0 || score_cur > best[0].score) slot = 0;
	        else if (best[1].r < 0 || score_cur > best[1].score) slot = 1;
	        else if (best[2].r < 0 || score_cur > best[2].score) slot = 2;
	        if (slot >= 0) {
	            for (int mm = 2; mm > slot; --mm) best[mm] = best[mm - 1];
	            best[slot].r = r;
	            best[slot].geom = geom_cur;
	            best[slot].score = score_cur;
	            best[slot].smin = smin_cur;
	            best[slot].smid = smid_cur;
	            best[slot].smax = smax_cur;
            for (int d = 0; d < D; ++d) {
                best[slot].mean[d] = mean_cur[d];
                best[slot].v[d] = v_cur[d];
            }
        }
    }

    int chosen = 0;
    if (R >= 3 && best[2].r >= 0) {
        int medr = median3_index_device(best[0].r, best[1].r, best[2].r);
        if (best[1].r == medr) chosen = 1;
        else if (best[2].r == medr) chosen = 2;
        else chosen = 0;
    }
    const Candidate& sel = best[chosen];
    int optidr = (sel.r >= 0 ? sel.r : 0);
    int tsel = cuts[optidr];
    if (tsel < 1) tsel = 1;
    if (tsel > tmax) tsel = tmax;

    for (int d = 0; d < D; ++d) Vec_raw[d + i * D] = sel.v[d];

    const float rad2 = fmaxf(top_dist2[(size_t)b * (size_t)tmax + (size_t)(tsel - 1)], 0.0f);
    const float tie_tol = 1.0e-6f * fmaxf(1.0f, rad2);
    const float rad2_lim = rad2 + tie_tol;
    uint32_t* mask_i = neigh_mask0_bits + (size_t)i * (size_t)mask_words_per_row;
    bitset_zero_row_u32(mask_i, mask_words_per_row);

    float mean_tie[MAX_D_MANOR];
    for (int d = 0; d < D; ++d) mean_tie[d] = 0.0f;
    int cnt_tie = 0;

    const float* xi = Xcm + (size_t)i * (size_t)D;

    int p_top = 0;
    for (; p_top < tmax; ++p_top) {
        float dist2p = top_dist2[(size_t)b * (size_t)tmax + (size_t)p_top];
        if (dist2p > rad2_lim) break;
        int rel = (int)top_off[(size_t)b * (size_t)tmax + (size_t)p_top];
        bitset_set_u32(mask_i, rel);
        ++cnt_tie;
        int j = start + rel;
        for (int d = 0; d < D; ++d) mean_tie[d] += Xcm[d + j * D] - xi[d];
    }

    for (int pos = 0; pos < kSlice; ++pos) {
        if (bitset_get_u32(mask_i, pos)) continue;
        int j = start + pos;
        float dist2 = 0.0f;
        for (int d = 0; d < D; ++d) {
            float z = Xcm[d + j * D] - xi[d];
            dist2 += z * z;
        }
        if (dist2 <= rad2_lim) {
            bitset_set_u32(mask_i, pos);
            ++cnt_tie;
            for (int d = 0; d < D; ++d) mean_tie[d] += Xcm[d + j * D] - xi[d];
        }
    }

    float inv_cnt_tie = 1.0f / static_cast<float>(cnt_tie);
    for (int d = 0; d < D; ++d) {
        mean_tie[d] *= inv_cnt_tie;
        Cen_raw[d + i * D] = xi[d] + mean_tie[d];
    }

    num_neigh0[i] = static_cast<int32_t>(cnt_tie);
    Rad0[i]       = sqrtf(rad2);
    if (geomtype0) geomtype0[i] = sel.geom;
    if (SlocMin0)  SlocMin0[i]  = sel.smin;
    if (SlocMid0)  SlocMid0[i]  = sel.smid;
    if (SlocMax0)  SlocMax0[i]  = sel.smax;

    float maxproj = 0.0f;
    for (int w = 0; w < mask_words_per_row; ++w) {
        uint32_t bits = mask_i[w];
        while (bits) {
            uint32_t lsb = bits & (~bits + 1u);
            int bit = __ffs((int)bits) - 1;
            bits ^= lsb;
            int pos = (w << 5) + bit;
            if (pos >= kSlice) continue;
            int j = start + pos;
            float proj = 0.0f;
            for (int d = 0; d < D; ++d) {
                float z = (Xcm[d + j * D] - xi[d]) - mean_tie[d];
                proj += sel.v[d] * z;
            }
            maxproj = fmaxf(maxproj, fabsf(proj));
        }
    }
    axial_tan0[i] = maxproj;
}

__global__ void heavy_mask_kernel(const int32_t* __restrict__ num_neigh0, int N, int heavy_thr, uint8_t* __restrict__ heavy_mask0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    heavy_mask0[i] = (num_neigh0[i] > heavy_thr) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

__global__ __launch_bounds__(256, 2) void pass2_aggregate_kernel_u16(
    const uint32_t* __restrict__ neigh_mask0_bits,
    const uint8_t*  __restrict__ heavy_mask0,
    const float*    __restrict__ Vec_raw,
    const float*    __restrict__ Cen_raw,
    int D, int N, int kSlice, int mask_words_per_row,
    int q0, int B,
    float* __restrict__ cov_out,
    float* __restrict__ cen_sum_out,
    int32_t* __restrict__ cnt_out)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int i = q0 + b;
    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    float* cov_b = cov_out + static_cast<size_t>(b) * static_cast<size_t>(D) * static_cast<size_t>(D);
    float* cen_b = cen_sum_out + static_cast<size_t>(b) * static_cast<size_t>(D);
    for (int t = 0; t < D * D; ++t) cov_b[t] = 0.0f;
    for (int d = 0; d < D; ++d) cen_b[d] = 0.0f;

    int cnt = 0;
    const uint32_t* mask_i = neigh_mask0_bits + static_cast<size_t>(i) * static_cast<size_t>(mask_words_per_row);
    for (int w = 0; w < mask_words_per_row; ++w) {
        uint32_t bits = mask_i[w];
        while (bits) {
            uint32_t lsb = bits & (~bits + 1u);
            int bit = __ffs((int)bits) - 1;
            bits ^= lsb;
            int pos = (w << 5) + bit;
            if (pos >= kSlice) continue;
            int j = start + pos;
            if ((unsigned)j >= (unsigned)N) continue;
            if (!heavy_mask0[j]) continue;
            ++cnt;
            const float* vj = Vec_raw + static_cast<size_t>(j) * static_cast<size_t>(D);
            const float* cj = Cen_raw + static_cast<size_t>(j) * static_cast<size_t>(D);
            for (int d = 0; d < D; ++d) cen_b[d] += cj[d];
            for (int a = 0; a < D; ++a) {
                float va = vj[a];
                for (int c = 0; c < D; ++c) cov_b[a * D + c] += va * vj[c];
            }
        }
    }
    cnt_out[b] = static_cast<int32_t>(cnt);
}

__global__ __launch_bounds__(256, 2) void pass2_finalize_kernel(
    const float* __restrict__ cov_in,
    const float* __restrict__ cen_sum_in,
    const int32_t* __restrict__ cnt_in,
    const float* __restrict__ Xcm,
    const float* __restrict__ Vec_raw,
    const float* __restrict__ Cen_raw,
    int D, int N, int q0, int B, int power_iter_steps,
    float* __restrict__ Vec0,
    float* __restrict__ Cen0)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int i = q0 + b;
    (void)power_iter_steps;

    const float* cov_b = cov_in + static_cast<size_t>(b) * static_cast<size_t>(D) * static_cast<size_t>(D);
    const float* cen_b = cen_sum_in + static_cast<size_t>(b) * static_cast<size_t>(D);
    int cnt = static_cast<int>(cnt_in[b]);

    float* v0_i = Vec0 + static_cast<size_t>(i) * static_cast<size_t>(D);
    float* c0_i = Cen0 + static_cast<size_t>(i) * static_cast<size_t>(D);

    const float* vraw_i = Vec_raw + static_cast<size_t>(i) * static_cast<size_t>(D);
    const float* craw_i = Cen_raw + static_cast<size_t>(i) * static_cast<size_t>(D);

    if (cnt <= 0) {
        for (int d = 0; d < D; ++d) {
            v0_i[d] = vraw_i[d];
            c0_i[d] = craw_i[d];
        }
        return;
    }

    float eigvals[MAX_D_MANOR];
    float eigvecs[MAX_D_MANOR * MAX_D_MANOR];
    jacobi_eigen_device(cov_b, D, eigvals, eigvecs);

    float v[MAX_D_MANOR];
    int top = D - 1;
    float top_eval = eigvals[top];
    if (!(top_eval > 0.0f)) {
        for (int d = 0; d < D; ++d) v[d] = vraw_i[d];
        normalize_vec_device(v, D);
    } else {
        for (int d = 0; d < D; ++d) v[d] = eigvecs[d * D + top];
        normalize_vec_device(v, D);
        float sgn = 0.0f;
        for (int d = 0; d < D; ++d) sgn += v[d] * vraw_i[d];
        if (sgn < 0.0f) {
            for (int d = 0; d < D; ++d) v[d] = -v[d];
        }
    }

    float invcnt = 1.0f / static_cast<float>(cnt);
    float dotv = 0.0f;
    for (int d = 0; d < D; ++d) {
        v0_i[d] = v[d];
        c0_i[d] = cen_b[d] * invcnt;
        dotv += v[d] * (c0_i[d] - Xcm[d + i * D]);
    }
    for (int d = 0; d < D; ++d) c0_i[d] -= dotv * v[d];
}

__global__ __launch_bounds__(256, 2) void pass2_recompute_axial_kernel(
    const uint32_t* __restrict__ neigh_mask0_bits,
    const uint8_t* __restrict__ heavy_mask0,
    const float* __restrict__ Vec0,
    const float* __restrict__ Xcm,
    int D, int N, int kSlice, int mask_words_per_row,
    float* __restrict__ axial_tan0)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int kHalf = kSlice / 2;
    int start = i - kHalf;
    if (start < 0) start = 0;
    if (start > N - kSlice) start = N - kSlice;

    const uint32_t* mask_i = neigh_mask0_bits + static_cast<size_t>(i) * static_cast<size_t>(mask_words_per_row);
    const float* v0_i = Vec0 + static_cast<size_t>(i) * static_cast<size_t>(D);

    float mean_x[MAX_D_MANOR];
    for (int d = 0; d < D; ++d) mean_x[d] = 0.0f;

    int cnt = 0;
    for (int w = 0; w < mask_words_per_row; ++w) {
        uint32_t bits = mask_i[w];
        while (bits) {
            uint32_t lsb = bits & (~bits + 1u);
            int bit = __ffs((int)bits) - 1;
            bits ^= lsb;
            int pos = (w << 5) + bit;
            if (pos >= kSlice) continue;
            int j = start + pos;
            if ((unsigned)j >= (unsigned)N) continue;
            if (!heavy_mask0[j]) continue;
            ++cnt;
            for (int d = 0; d < D; ++d) mean_x[d] += Xcm[d + j * D];
        }
    }

    if (cnt <= 0) return;

    float inv_cnt = 1.0f / static_cast<float>(cnt);
    for (int d = 0; d < D; ++d) mean_x[d] *= inv_cnt;

    float maxproj = 0.0f;
    for (int w = 0; w < mask_words_per_row; ++w) {
        uint32_t bits = mask_i[w];
        while (bits) {
            uint32_t lsb = bits & (~bits + 1u);
            int bit = __ffs((int)bits) - 1;
            bits ^= lsb;
            int pos = (w << 5) + bit;
            if (pos >= kSlice) continue;
            int j = start + pos;
            if ((unsigned)j >= (unsigned)N) continue;
            if (!heavy_mask0[j]) continue;

            float proj = 0.0f;
            for (int d = 0; d < D; ++d) proj += v0_i[d] * (Xcm[d + j * D] - mean_x[d]);
            maxproj = fmaxf(maxproj, fabsf(proj));
        }
    }
    axial_tan0[i] = maxproj;
}


// -----------------------------------------------------------------------------
// mexFunction
// -----------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:nrhs", "Expected 5 inputs: X_sorted, Y_sorted, kSlice, cuts, opts");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:nlhs", "Returns exactly one struct output.");
    }

    const mxArray* mxX      = prhs[0];
    const mxArray* mxY      = prhs[1];
    const mxArray* mxKSlice = prhs[2];
    const mxArray* mxCuts   = prhs[3];
    const mxArray* mxOpts   = prhs[4];

    if (!mxIsSingle(mxX) || !mxIsSingle(mxY) || !mxIsInt32(mxCuts)) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:type",
            "X_sorted and Y_sorted must be single; cuts must be int32.");
    }

    const int D = static_cast<int>(mxGetM(mxX));
    const int N = static_cast<int>(mxGetN(mxX));
    const int Ny = static_cast<int>(mxGetNumberOfElements(mxY));
    const int kSlice = static_cast<int>(mxGetScalar(mxKSlice));
    const int R = static_cast<int>(mxGetNumberOfElements(mxCuts));

    if (D <= 0 || N <= 0 || Ny != N || kSlice <= 0 || R <= 0) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "Invalid D, N, Y_sorted length, kSlice, or cuts.");
    }
    if (D > MAX_D_MANOR || R > MAX_R_MANOR) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "D or R exceeds compile-time limit.");
    }
    if (kSlice > N) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "kSlice must satisfy 1 <= kSlice <= N.");
    }
    if ((kSlice & 1) == 0) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "kSlice must be odd.");
    }
    if (kSlice > 65535) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "This file uses uint16 slice offsets, so kSlice must be <= 65535.");
    }

    Phase1Opts opts = parse_opts(mxOpts);
    const float* Xcm_host = static_cast<const float*>(mxGetData(mxX));
    const float* Yh = static_cast<const float*>(mxGetData(mxY));
    const int* cuts_h = static_cast<const int*>(mxGetData(mxCuts));

    bool y_non_decreasing = true;
    for (int j = 1; j < N; ++j) {
        if (Yh[j] < Yh[j - 1]) { y_non_decreasing = false; break; }
    }
    if (!y_non_decreasing) {
        mexWarnMsgIdAndTxt("manor_1_phase1_cuda_mex:ysort",
            "Y_sorted is not nondecreasing. This CUDA implementation assumes Y-sorted slice neighborhoods.");
    }

    for (int r = 0; r < R; ++r) {
        if (cuts_h[r] <= 0 || cuts_h[r] > kSlice) {
            mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
                "All cuts entries must satisfy 1 <= cuts(r) <= kSlice.");
        }
        if (r > 0 && cuts_h[r] < cuts_h[r - 1]) {
            mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
                "cuts must be nondecreasing.");
        }
    }

    int tmax = cuts_h[R - 1];
    if (tmax <= 0 || tmax > kSlice) {
        mexErrMsgIdAndTxt("manor_1_phase1_cuda_mex:args",
            "cuts(end) must satisfy 1 <= cuts(end) <= kSlice.");
    }

    auto t_total_start = std::chrono::steady_clock::now();
    double t_upload_sec = std::numeric_limits<double>::quiet_NaN();
    double t_topk_select_sec = std::numeric_limits<double>::quiet_NaN();
    double t_reduce_heavy_sec = std::numeric_limits<double>::quiet_NaN();
    double t_pass2_sec = std::numeric_limits<double>::quiet_NaN();
    double t_copyout_sec = std::numeric_limits<double>::quiet_NaN();

    float* d_Xcm = nullptr;
    int*   d_cuts = nullptr;
    auto t_upload_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&d_Xcm, sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMemcpy(d_Xcm, Xcm_host, sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(N), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cuts, sizeof(int) * static_cast<size_t>(R)));
    CUDA_CHECK(cudaMemcpy(d_cuts, cuts_h, sizeof(int) * static_cast<size_t>(R), cudaMemcpyHostToDevice));
    t_upload_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_upload_start).count();

    int BQ = choose_query_block(D, kSlice, tmax, R, opts);
    // phase1_dist_offset_kernel_cm_shared launches with grid.y = B, so keep the
    // query block within CUDA's conventional grid-y limit.
    BQ = std::min(BQ, 65535);
    BQ = std::min(BQ, N);
    if (!opts.silent) {
        mexPrintf("[phase1_cuda_extreme] D=%d N=%d kSlice=%d R=%d tmax=%d BQ=%d gpu_pass2=%d radix_select_exact=1 fullsort_fallback=%d\n",
                  D, N, kSlice, R, tmax, BQ, (int)opts.gpu_pass2, (int)opts.fullsort_fallback);
    }

    const size_t max_items = static_cast<size_t>(BQ) * static_cast<size_t>(kSlice);
    const size_t max_top   = static_cast<size_t>(BQ) * static_cast<size_t>(tmax);

    float*    d_keys_in   = nullptr;
    float*    d_keys_out  = nullptr;
    uint16_t* d_vals_in   = nullptr;
    uint16_t* d_vals_out  = nullptr;
    float*    d_top_keys  = nullptr;
    uint16_t* d_top_vals  = nullptr;
    float*    d_chunk_sum1 = nullptr;
    float*    d_chunk_sum2 = nullptr;
    int*      d_offsets   = nullptr;
    void*     d_temp_sort = nullptr;
    size_t    temp_bytes_sort = 0;

    float*    d_VecRaw    = nullptr;
    float*    d_CenRaw    = nullptr;
    float*    d_Rad0      = nullptr;
    float*    d_AxialTan0 = nullptr;
    int32_t*  d_NumNeigh0 = nullptr;
    uint8_t*  d_Heavy0    = nullptr;
    float*    d_Geom0     = nullptr;
    float*    d_SMin0     = nullptr;
    float*    d_SMid0     = nullptr;
    float*    d_SMax0     = nullptr;
    const int mask_words_per_row = (kSlice + 31) / 32;
    const size_t mask_words_total = static_cast<size_t>(N) * static_cast<size_t>(mask_words_per_row);
    uint32_t* d_neigh_mask0 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_keys_in,   sizeof(float)    * max_items));
    CUDA_CHECK(cudaMalloc(&d_keys_out,  sizeof(float)    * max_items));
    CUDA_CHECK(cudaMalloc(&d_vals_in,   sizeof(uint16_t) * max_items));
    CUDA_CHECK(cudaMalloc(&d_vals_out,  sizeof(uint16_t) * max_items));
    CUDA_CHECK(cudaMalloc(&d_top_keys,  sizeof(float)    * max_top));
    CUDA_CHECK(cudaMalloc(&d_top_vals,  sizeof(uint16_t) * max_top));
    CUDA_CHECK(cudaMalloc(&d_chunk_sum1, sizeof(float) * static_cast<size_t>(BQ) * static_cast<size_t>(R) * static_cast<size_t>(D)));
    CUDA_CHECK(cudaMalloc(&d_chunk_sum2, sizeof(float) * static_cast<size_t>(BQ) * static_cast<size_t>(R) * static_cast<size_t>(D) * static_cast<size_t>(D)));
    CUDA_CHECK(cudaMalloc(&d_offsets,   sizeof(int)      * static_cast<size_t>(BQ + 1)));


    CUDA_CHECK(cudaMalloc(&d_VecRaw,    sizeof(float)    * static_cast<size_t>(D) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_CenRaw,    sizeof(float)    * static_cast<size_t>(D) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_Rad0,      sizeof(float)    * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_AxialTan0, sizeof(float)    * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_NumNeigh0, sizeof(int32_t)  * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_Heavy0,    sizeof(uint8_t)  * static_cast<size_t>(N)));
    if (opts.return_debug) {
        CUDA_CHECK(cudaMalloc(&d_Geom0,     sizeof(float)    * static_cast<size_t>(N)));
        CUDA_CHECK(cudaMalloc(&d_SMin0,     sizeof(float)    * static_cast<size_t>(N)));
        CUDA_CHECK(cudaMalloc(&d_SMid0,     sizeof(float)    * static_cast<size_t>(N)));
        CUDA_CHECK(cudaMalloc(&d_SMax0,     sizeof(float)    * static_cast<size_t>(N)));
    }
    CUDA_CHECK(cudaMalloc(&d_neigh_mask0, sizeof(uint32_t) * mask_words_total));

    CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_bytes_sort,
        d_keys_in, d_keys_out,
        d_vals_in, d_vals_out,
        static_cast<int>(max_items), BQ,
        d_offsets, d_offsets + 1));
    CUDA_CHECK(cudaMalloc(&d_temp_sort, temp_bytes_sort));

    std::vector<int> h_offsets(static_cast<size_t>(BQ + 1), 0);
    for (int b = 0; b <= BQ; ++b) h_offsets[b] = b * kSlice;
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * static_cast<size_t>(BQ + 1), cudaMemcpyHostToDevice));

    const int threads = 256;
    auto t_topk_start = std::chrono::steady_clock::now();

    for (int q0 = 0; q0 < N; q0 += BQ) {
        int B = std::min(BQ, N - q0);

        dim3 distBlock(threads, 1, 1);
        dim3 distGrid((kSlice + threads - 1) / threads, B, 1);
        phase1_dist_offset_kernel_cm_shared<uint16_t><<<distGrid, distBlock, sizeof(float) * static_cast<size_t>(D)>>>(
            d_Xcm, D, N, q0, B, kSlice, d_keys_in, d_vals_in);
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_sort, temp_bytes_sort,
            d_keys_in, d_keys_out,
            d_vals_in, d_vals_out,
            B * kSlice, B,
            d_offsets, d_offsets + 1));

        int compact_total = B * tmax;
        int compact_blocks = (compact_total + threads - 1) / threads;
        compact_topk_kernel_offset<uint16_t><<<compact_blocks, threads>>>(
            d_keys_out, d_vals_out, B, kSlice, tmax, d_top_keys, d_top_vals);
        CUDA_CHECK(cudaPeekAtLastError());

        phase1_chunkstats_from_top_kernel<<<B, 32>>>(
            d_Xcm, d_top_vals, d_cuts,
            D, N, kSlice, tmax, R, q0, B,
            d_chunk_sum1, d_chunk_sum2);
        CUDA_CHECK(cudaPeekAtLastError());

        phase1_select_from_sorted_kernel<<<B, 32>>>(
            d_Xcm, d_top_keys, d_top_vals, d_cuts,
            d_chunk_sum1, d_chunk_sum2,
            D, N, kSlice, tmax, R, q0, B, opts.jitter,
            d_VecRaw, d_CenRaw, d_Rad0, d_AxialTan0, d_NumNeigh0, d_Geom0, d_SMin0, d_SMid0, d_SMax0,
            d_neigh_mask0, mask_words_per_row);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    if (opts.profile_timers) CUDA_CHECK(cudaDeviceSynchronize());
    t_topk_select_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_topk_start).count();

    if (d_temp_sort) { cudaFree(d_temp_sort); d_temp_sort = nullptr; }
    if (d_offsets)   { cudaFree(d_offsets);   d_offsets = nullptr; }
    if (d_top_vals)  { cudaFree(d_top_vals);  d_top_vals = nullptr; }
    if (d_top_keys)  { cudaFree(d_top_keys);  d_top_keys = nullptr; }
    if (d_chunk_sum2){ cudaFree(d_chunk_sum2); d_chunk_sum2 = nullptr; }
    if (d_chunk_sum1){ cudaFree(d_chunk_sum1); d_chunk_sum1 = nullptr; }
    if (d_vals_out)  { cudaFree(d_vals_out);  d_vals_out = nullptr; }
    if (d_vals_in)   { cudaFree(d_vals_in);   d_vals_in = nullptr; }
    if (d_keys_out)  { cudaFree(d_keys_out);  d_keys_out = nullptr; }
    if (d_keys_in)   { cudaFree(d_keys_in);   d_keys_in = nullptr; }

    auto t_reduce_start = std::chrono::steady_clock::now();
    long long* d_sum_num = nullptr;
    void* d_temp_reduce = nullptr;
    size_t temp_bytes_reduce = 0;
    CUDA_CHECK(cudaMalloc(&d_sum_num, sizeof(long long)));
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes_reduce, d_NumNeigh0, d_sum_num, N));
    CUDA_CHECK(cudaMalloc(&d_temp_reduce, temp_bytes_reduce));
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_reduce, temp_bytes_reduce, d_NumNeigh0, d_sum_num, N));
    long long h_sum_num = 0;
    CUDA_CHECK(cudaMemcpy(&h_sum_num, d_sum_num, sizeof(long long), cudaMemcpyDeviceToHost));
    double avgNum = static_cast<double>(h_sum_num) / static_cast<double>(std::max(N, 1));
    const int heavy_thr = std::max(2 * D, static_cast<int>(std::floor(opts.heavy_factor * avgNum)));
    int mask_blocks = (N + threads - 1) / threads;
    heavy_mask_kernel<<<mask_blocks, threads>>>(d_NumNeigh0, N, heavy_thr, d_Heavy0);
    CUDA_CHECK(cudaPeekAtLastError());
    if (opts.profile_timers) CUDA_CHECK(cudaDeviceSynchronize());
    t_reduce_heavy_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_reduce_start).count();

    float* d_Vec0 = d_VecRaw;
    float* d_Cen0 = d_CenRaw;
    float* d_cov_blk = nullptr;
    float* d_censum_blk = nullptr;
    int32_t* d_cnt_blk = nullptr;

    auto t_pass2_start = std::chrono::steady_clock::now();
    if (opts.gpu_pass2) {
        CUDA_CHECK(cudaMalloc(&d_Vec0, sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(N)));
        CUDA_CHECK(cudaMalloc(&d_Cen0, sizeof(float) * static_cast<size_t>(D) * static_cast<size_t>(N)));
        CUDA_CHECK(cudaMalloc(&d_cov_blk, sizeof(float) * static_cast<size_t>(BQ) * static_cast<size_t>(D) * static_cast<size_t>(D)));
        CUDA_CHECK(cudaMalloc(&d_censum_blk, sizeof(float) * static_cast<size_t>(BQ) * static_cast<size_t>(D)));
        CUDA_CHECK(cudaMalloc(&d_cnt_blk, sizeof(int32_t) * static_cast<size_t>(BQ)));

        for (int q0 = 0; q0 < N; q0 += BQ) {
            int B = std::min(BQ, N - q0);
            int blocks = (B + threads - 1) / threads;
            pass2_aggregate_kernel_u16<<<blocks, threads>>>(
                d_neigh_mask0, d_Heavy0, d_VecRaw, d_CenRaw,
                D, N, kSlice, mask_words_per_row, q0, B,
                d_cov_blk, d_censum_blk, d_cnt_blk);
            CUDA_CHECK(cudaPeekAtLastError());
            pass2_finalize_kernel<<<blocks, threads>>>(
                d_cov_blk, d_censum_blk, d_cnt_blk, d_Xcm, d_VecRaw, d_CenRaw,
                D, N, q0, B, opts.power_iter_steps,
                d_Vec0, d_Cen0);
            CUDA_CHECK(cudaPeekAtLastError());
        }

        int axial_blocks = (N + threads - 1) / threads;
        pass2_recompute_axial_kernel<<<axial_blocks, threads>>>(
            d_neigh_mask0, d_Heavy0, d_Vec0, d_Xcm,
            D, N, kSlice, mask_words_per_row, d_AxialTan0);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    if (opts.profile_timers) CUDA_CHECK(cudaDeviceSynchronize());
    t_pass2_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pass2_start).count();

    if (d_neigh_mask0) { cudaFree(d_neigh_mask0); d_neigh_mask0 = nullptr; }
    if (opts.gpu_pass2) {
        if (d_VecRaw) { cudaFree(d_VecRaw); d_VecRaw = nullptr; }
        if (d_CenRaw) { cudaFree(d_CenRaw); d_CenRaw = nullptr; }
    }

    auto t_copyout_start = std::chrono::steady_clock::now();

    if (opts.return_debug) {
        const char* fields[] = {
            "Vec0","Cen0","Rad0","axial_tan0","num_neigh0","heavy_mask0",
            "geomtype0","SlocMin0","SlocMid0","SlocMax0","profile","memory"
        };
        plhs[0] = mxCreateStructMatrix(1, 1, 12, fields);
    } else {
        const char* fields[] = {
            "Vec0","Cen0","Rad0","axial_tan0","num_neigh0","heavy_mask0","profile","memory"
        };
        plhs[0] = mxCreateStructMatrix(1, 1, 8, fields);
    }

    mxArray* mxVec0  = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxCen0  = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxRad0  = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxAxial = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mxNum   = mxCreateNumericMatrix(1, N, mxINT32_CLASS,  mxREAL);
    mxArray* mxHeavy = mxCreateNumericMatrix(1, N, mxUINT8_CLASS,  mxREAL);

    CUDA_CHECK(cudaMemcpy(mxGetData(mxVec0),  d_Vec0,      sizeof(float)   * static_cast<size_t>(D) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(mxCen0),  d_Cen0,      sizeof(float)   * static_cast<size_t>(D) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(mxRad0),  d_Rad0,      sizeof(float)   * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(mxAxial), d_AxialTan0, sizeof(float)   * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(mxNum),   d_NumNeigh0, sizeof(int32_t) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mxGetData(mxHeavy), d_Heavy0,    sizeof(uint8_t) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));

    mxSetField(plhs[0], 0, "Vec0",       mxVec0);
    mxSetField(plhs[0], 0, "Cen0",       mxCen0);
    mxSetField(plhs[0], 0, "Rad0",       mxRad0);
    mxSetField(plhs[0], 0, "axial_tan0", mxAxial);
    mxSetField(plhs[0], 0, "num_neigh0", mxNum);
    mxSetField(plhs[0], 0, "heavy_mask0",mxHeavy);

    t_copyout_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_copyout_start).count();
    double t_total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total_start).count();

    {
        const char* pf[] = {"upload_sec","topk_select_sec","reduce_heavy_sec","pass2_sec","copyout_sec","total_sec","BQ","shared_topk_enabled","tile_topk_enabled","shared_topk_mode","tile_topk_mode","shared_requested","tile_requested","shared_feasible_raw","tile_feasible","shared_shmem_need","shared_shmem_limit","topk_block_threads","tile_sort_block_threads","tile_items_per_thread"};
        mxArray* mxProf = mxCreateStructMatrix(1, 1, 20, pf);
        mxSetField(mxProf, 0, "upload_sec", mxCreateDoubleScalar(t_upload_sec));
        mxSetField(mxProf, 0, "topk_select_sec", mxCreateDoubleScalar(t_topk_select_sec));
        mxSetField(mxProf, 0, "reduce_heavy_sec", mxCreateDoubleScalar(t_reduce_heavy_sec));
        mxSetField(mxProf, 0, "pass2_sec", mxCreateDoubleScalar(t_pass2_sec));
        mxSetField(mxProf, 0, "copyout_sec", mxCreateDoubleScalar(t_copyout_sec));
        mxSetField(mxProf, 0, "total_sec", mxCreateDoubleScalar(t_total_sec));
        mxSetField(mxProf, 0, "BQ", mxCreateDoubleScalar((double)BQ));
        mxSetField(mxProf, 0, "shared_topk_enabled", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "tile_topk_enabled", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "shared_topk_mode", mxCreateDoubleScalar(2.0));
        mxSetField(mxProf, 0, "tile_topk_mode", mxCreateDoubleScalar(2.0));
        mxSetField(mxProf, 0, "shared_requested", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "tile_requested", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "shared_feasible_raw", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "tile_feasible", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "shared_shmem_need", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "shared_shmem_limit", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "topk_block_threads", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "tile_sort_block_threads", mxCreateDoubleScalar(0.0));
        mxSetField(mxProf, 0, "tile_items_per_thread", mxCreateDoubleScalar(0.0));
        mxSetField(plhs[0], 0, "profile", mxProf);

        const char* mf[] = {"mask_bytes","sort_workspace_bytes","topk_bytes","chunkstats_bytes","raw_vec_cen_bytes","bitpack_ratio"};
        mxArray* mxMem = mxCreateStructMatrix(1, 1, 6, mf);
        mxSetField(mxMem, 0, "mask_bytes", mxCreateDoubleScalar((double)(sizeof(uint32_t) * mask_words_total)));
        mxSetField(mxMem, 0, "sort_workspace_bytes", mxCreateDoubleScalar((double)temp_bytes_sort));
        mxSetField(mxMem, 0, "topk_bytes", mxCreateDoubleScalar((double)(sizeof(float) * (double)max_top + sizeof(uint16_t) * (double)max_top)));
        mxSetField(mxMem, 0, "chunkstats_bytes", mxCreateDoubleScalar((double)(sizeof(float) * (double)BQ * (double)R * (double)D * (1.0 + (double)D))));
        mxSetField(mxMem, 0, "raw_vec_cen_bytes", mxCreateDoubleScalar((double)(2.0 * sizeof(float) * (double)D * (double)N)));
        mxSetField(mxMem, 0, "bitpack_ratio", mxCreateDoubleScalar(((double)sizeof(uint32_t) * (double)mask_words_total) / ((double)sizeof(uint8_t) * (double)N * (double)kSlice)));
        mxSetField(plhs[0], 0, "memory", mxMem);
    }

    if (opts.return_debug) {
        mxArray* mxGeom  = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
        mxArray* mxSMin  = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
        mxArray* mxSMid  = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
        mxArray* mxSMax  = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);

        CUDA_CHECK(cudaMemcpy(mxGetData(mxGeom),  d_Geom0, sizeof(float) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mxGetData(mxSMin),  d_SMin0, sizeof(float) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mxGetData(mxSMid),  d_SMid0, sizeof(float) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mxGetData(mxSMax),  d_SMax0, sizeof(float) * static_cast<size_t>(N), cudaMemcpyDeviceToHost));

        mxSetField(plhs[0], 0, "geomtype0", mxGeom);
        mxSetField(plhs[0], 0, "SlocMin0",  mxSMin);
        mxSetField(plhs[0], 0, "SlocMid0",  mxSMid);
        mxSetField(plhs[0], 0, "SlocMax0",  mxSMax);
    }

    if (d_cnt_blk) cudaFree(d_cnt_blk);
    if (d_censum_blk) cudaFree(d_censum_blk);
    if (d_cov_blk) cudaFree(d_cov_blk);
    if (opts.gpu_pass2) {
        cudaFree(d_Cen0);
        cudaFree(d_Vec0);
    }
    if (d_temp_reduce) cudaFree(d_temp_reduce);
    if (d_sum_num) cudaFree(d_sum_num);
    if (d_neigh_mask0) cudaFree(d_neigh_mask0);
    if (d_SMax0) cudaFree(d_SMax0);
    if (d_SMid0) cudaFree(d_SMid0);
    if (d_SMin0) cudaFree(d_SMin0);
    if (d_Geom0) cudaFree(d_Geom0);
    if (d_Heavy0) cudaFree(d_Heavy0);
    if (d_NumNeigh0) cudaFree(d_NumNeigh0);
    if (d_AxialTan0) cudaFree(d_AxialTan0);
    if (d_Rad0) cudaFree(d_Rad0);
    if (d_CenRaw) cudaFree(d_CenRaw);
    if (d_VecRaw) cudaFree(d_VecRaw);
    if (d_temp_sort) cudaFree(d_temp_sort);
    if (d_offsets) cudaFree(d_offsets);
    if (d_top_vals) cudaFree(d_top_vals);
    if (d_top_keys) cudaFree(d_top_keys);
    if (d_vals_out) cudaFree(d_vals_out);
    if (d_vals_in) cudaFree(d_vals_in);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_keys_in) cudaFree(d_keys_in);
    if (d_cuts) cudaFree(d_cuts);
    if (d_Xcm) cudaFree(d_Xcm);
}
