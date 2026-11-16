/* MANOR CUDA adapter for the multiscale response-neighborhood and local-regression pipeline. */

#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
#include "../../cuda/core/manor_m_minimex.h"
#include "../../cuda/core/manor_m_cuda_internal.h"
#include "../../cuda/core/manor_m_cuda_resident.h"
#if MANOR_M_HOST_INDEX_VECTOR_CHECK
#include "../../cuda/core/manor_m_index_validation.h"
#endif
#else
#include "mex.h"
#include "matrix.h"
#endif

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

static bool owner_resident_finite_f32(const void* host, size_t bytes)
{
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
    return manor_m_resident_finite_f32(host, bytes);
#else
    (void)host; (void)bytes;
    return false;
#endif
}

static bool owner_resident_index_validated(
    const void* host, size_t bytes, int64_t upper_bound)
{
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
    return manor_m_resident_index_range_validated(host, bytes, upper_bound);
#else
    (void)host; (void)bytes; (void)upper_bound;
    return false;
#endif
}

#ifndef MAX_D_MANOR
#define MAX_D_MANOR 48
#endif
#ifndef MAX_MM_MANOR
#define MAX_MM_MANOR 8
#endif
#ifndef MAX_MY_MANOR
#define MAX_MY_MANOR 8
#endif
#ifndef MANOR_M_REGRESSION_SHARED_SOLVE
#define MANOR_M_REGRESSION_SHARED_SOLVE 0
#endif
#ifndef MANOR_M_EVAL_MIN_BLOCKS_PER_SM
#define MANOR_M_EVAL_MIN_BLOCKS_PER_SM 0
#endif
#ifndef MANOR_M_AGGREGATION_TILED_MASK
#define MANOR_M_AGGREGATION_TILED_MASK 0
#endif
#if defined(MANOR_M_EVAL_FIXED_ORDER) && (MANOR_M_EVAL_FIXED_ORDER != 1)
#error "Unordered FP32 metric sums were retired; fixed-order sums are mandatory."
#endif
// Native MEX and the language-neutral core use one mandatory numerical
// contract: unique per-query FP32 contributions, then the fixed 256-lane tree.
// Unordered sums are excluded from the deterministic reduction.
#include "../../cuda/core/manor_m_eval_fixed_order.cuh"
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_COOPERATIVE_SOLVE
#include "../../cuda/core/manor_m_regression_cooperative.cuh"
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_SHUFFLE_INPUTS
#include "../../cuda/core/manor_m_regression_shuffle.cuh"
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
#include "../../cuda/core/manor_m_regression_warp_entries.cuh"
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_WARP_GATHER
#include "../../cuda/core/manor_m_aggregation_gather.cuh"
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_WARP_ENTRIES
#include "../../cuda/core/manor_m_aggregation_warp.cuh"
#endif

/* Dimension-templated iteration kernels are instantiated on a small
   geometric ladder and clamped at MAX_D_MANOR. */
static_assert(MAX_D_MANOR >= MAX_MM_MANOR && MAX_D_MANOR >= MAX_MY_MANOR,
    "A tier is a launch block size and must cover the row < MM and row < MY "
    "guards in owner_regression_kernel.");

static constexpr int owner_maxd_tier(int tier)
{
    return tier < MAX_D_MANOR ? tier : MAX_D_MANOR;
}

template <typename Fn>
static void owner_dispatch_maxd(int D, Fn&& fn)
{
    if (D <= 12) { fn(std::integral_constant<int, owner_maxd_tier(12)>{}); return; }
    if (D <= 20) { fn(std::integral_constant<int, owner_maxd_tier(20)>{}); return; }
    if (D <= 32) { fn(std::integral_constant<int, owner_maxd_tier(32)>{}); return; }
    fn(std::integral_constant<int, MAX_D_MANOR>{});
}

struct FusedEvalOpts {
    float sum_sparse_threshold   = 1.0e1f;
    int   count_sparse_threshold = 0;
    float fallback_exp_scale     = 2.0f;
    float eps_weight             = 1.0e-20f;
    int   warps_per_block        = 4;
    bool  mse_only_eval          = false;
    bool  tangent_only_eval      = false;
    /* The truth-array validator fills these switches.  They are deliberately
       independent: array presence, not a monolithic evaluation mode, decides
       which metric-specific work is launched. */
    bool  response_eval          = false;
    bool  tangent_metric_eval    = false;
    bool  center_eval            = false;
    bool  center_tangent_metric_eval = false;
    float center_eval_rho_scale  = 4.0f;
    float center_eval_h_scale    = 1.25f;
    std::vector<float> eval_rho_scale = {1.0f};
    std::vector<float> eval_h_scale   = {1.0f};
    std::vector<int>   eval_k_arr;
    bool  eval_k0 = false;  /* evaluate the pre-iteration (Phase-1) state */
};

static std::vector<float> read_real_vector_field(const mxArray* a, const char* field_name)
{
    if (!a || mxGetNumberOfElements(a) == 0) return {};
    if (mxIsComplex(a) || !(mxIsSingle(a) || mxIsDouble(a))) {
        mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
            "%s must be a real single/double vector.", field_name);
    }
    const mwSize n = mxGetNumberOfElements(a);
    std::vector<float> out((size_t)n);
    if (mxIsSingle(a)) {
        const float* p = static_cast<const float*>(mxGetData(a));
        std::copy(p, p + n, out.begin());
    } else {
        const double* p = mxGetPr(a);
        for (mwSize i = 0; i < n; ++i) out[(size_t)i] = static_cast<float>(p[i]);
    }
    return out;
}

static std::vector<int> read_int_vector_field(const mxArray* a, const char* field_name)
{
    if (!a || mxGetNumberOfElements(a) == 0) return {};
    if (mxIsComplex(a) || !(mxIsSingle(a) || mxIsDouble(a) || mxIsInt32(a) || mxIsUint32(a))) {
        mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
            "%s must be a real numeric vector.", field_name);
    }
    const mwSize n = mxGetNumberOfElements(a);
    std::vector<int> out;
    out.reserve((size_t)n);
    if (mxIsInt32(a)) {
        const int32_t* p = static_cast<const int32_t*>(mxGetData(a));
        for (mwSize i = 0; i < n; ++i) out.push_back((int)p[i]);
    } else if (mxIsUint32(a)) {
        const uint32_t* p = static_cast<const uint32_t*>(mxGetData(a));
        for (mwSize i = 0; i < n; ++i) out.push_back((int)p[i]);
    } else if (mxIsSingle(a)) {
        const float* p = static_cast<const float*>(mxGetData(a));
        for (mwSize i = 0; i < n; ++i) out.push_back((int)std::lround((double)p[i]));
    } else {
        const double* p = mxGetPr(a);
        for (mwSize i = 0; i < n; ++i) out.push_back((int)std::lround(p[i]));
    }
    return out;
}

static FusedEvalOpts parse_fused_eval_opts(const mxArray* opts)
{
    FusedEvalOpts o;
    if (!opts || !mxIsStruct(opts)) return o;
    auto getf = [&](const char* name) -> const mxArray* { return mxGetField(opts, 0, name); };
    if (auto a = getf("sum_sparse_threshold"))   o.sum_sparse_threshold = (float)mxGetScalar(a);
    if (auto a = getf("count_sparse_threshold")) o.count_sparse_threshold = (int)mxGetScalar(a);
    if (auto a = getf("fallback_exp_scale"))     o.fallback_exp_scale = (float)mxGetScalar(a);
    if (auto a = getf("eps_weight"))             o.eps_weight = (float)mxGetScalar(a);
    if (auto a = getf("warps_per_block"))        o.warps_per_block = (int)mxGetScalar(a);
    if (auto a = getf("mse_only_eval"))
        o.mse_only_eval = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (auto a = getf("tangent_only_eval"))
        o.tangent_only_eval = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (o.mse_only_eval && o.tangent_only_eval) {
        mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
            "mse_only_eval and tangent_only_eval are mutually exclusive.");
    }
    if (auto a = getf("eval_rho_scale"))         o.eval_rho_scale = read_real_vector_field(a, "eval_rho_scale");
    if (auto a = getf("eval_h_scale"))           o.eval_h_scale   = read_real_vector_field(a, "eval_h_scale");
    if (auto a = getf("hyper_param_rho"))        o.eval_rho_scale = read_real_vector_field(a, "hyper_param_rho");
    if (auto a = getf("hyper_param_R"))          o.eval_h_scale   = read_real_vector_field(a, "hyper_param_R");
    if (auto a = getf("eval_k_arr"))             o.eval_k_arr     = read_int_vector_field(a, "eval_k_arr");
    if (auto a = getf("eval_K_arr"))             o.eval_k_arr     = read_int_vector_field(a, "eval_K_arr");
    if (auto a = getf("center_eval_rho_scale")) o.center_eval_rho_scale = (float)mxGetScalar(a);
    if (auto a = getf("center_eval_h_scale"))   o.center_eval_h_scale = (float)mxGetScalar(a);
    if (o.warps_per_block != 2 && o.warps_per_block != 4) o.warps_per_block = 4;
    o.sum_sparse_threshold = std::max(0.0f, o.sum_sparse_threshold);
    o.count_sparse_threshold = std::max(0, o.count_sparse_threshold);
    o.fallback_exp_scale = std::max(0.0f, o.fallback_exp_scale);
    o.eps_weight = std::max(1.0e-30f, o.eps_weight);
    if (o.eval_rho_scale.empty()) o.eval_rho_scale = {1.0f};
    if (o.eval_h_scale.empty()) o.eval_h_scale = {1.0f};
    for (float v : o.eval_rho_scale) {
        if (!std::isfinite(v) || v <= 0.0f) {
            mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
                "All eval_rho_scale values must be finite and positive.");
        }
    }
    for (float v : o.eval_h_scale) {
        if (!std::isfinite(v) || v <= 0.0f) {
            mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
                "All eval_h_scale values must be finite and positive.");
        }
    }
    if (!std::isfinite(o.center_eval_rho_scale) || o.center_eval_rho_scale <= 0.0f ||
            !std::isfinite(o.center_eval_h_scale) || o.center_eval_h_scale <= 0.0f) {
        mexErrMsgIdAndTxt("manor_m_phase2_mex:fused_eval",
            "center_eval_rho_scale and center_eval_h_scale must be finite and positive.");
    }
    o.eval_k0 = std::find(o.eval_k_arr.begin(), o.eval_k_arr.end(), 0) !=
        o.eval_k_arr.end();
    o.eval_k_arr.erase(std::remove_if(o.eval_k_arr.begin(), o.eval_k_arr.end(),
        [](int k) { return k <= 0; }), o.eval_k_arr.end());
    std::sort(o.eval_k_arr.begin(), o.eval_k_arr.end());
    o.eval_k_arr.erase(std::unique(o.eval_k_arr.begin(), o.eval_k_arr.end()), o.eval_k_arr.end());
    return o;
}

static bool fused_eval_should_run_k(const FusedEvalOpts& opts, int k_actual)
{
    if (opts.eval_k_arr.empty()) return true;
    return std::binary_search(opts.eval_k_arr.begin(), opts.eval_k_arr.end(), k_actual);
}

__device__ __forceinline__ void normalize_vec_device(float* v, int D)
{
    float nrm2 = 0.0f;
    for (int d = 0; d < D; ++d) nrm2 += v[d] * v[d];
    float inv = rsqrtf(fmaxf(nrm2, 1e-30f));
    for (int d = 0; d < D; ++d) v[d] *= inv;
}

/* Cholesky decomposition + forward/back-solve for nsys × nrhs system.
   A is row-major nsys×nsys, B is row-major nsys×nrhs.  A is overwritten with L.
   Returns false if factorisation fails. */
static __device__ bool cholesky_solve_device_multi(double* A, double* B,
                                                   int nsys, int nrhs)
{
    /* L L' = A */
    for (int j = 0; j < nsys; ++j) {
        double sum = A[j * nsys + j];
        for (int k = 0; k < j; ++k)
            sum -= A[j * nsys + k] * A[j * nsys + k];
        if (!(sum > 1e-30)) return false;
        A[j * nsys + j] = sqrt(sum);
        for (int i = j + 1; i < nsys; ++i) {
            double s = A[i * nsys + j];
            for (int k = 0; k < j; ++k)
                s -= A[i * nsys + k] * A[j * nsys + k];
            A[i * nsys + j] = s / A[j * nsys + j];
        }
    }
    /* Forward solve  L y = B  (overwrite B) */
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        for (int i = 0; i < nsys; ++i) {
            double s = B[i * nrhs + rhs];
            for (int k = 0; k < i; ++k)
                s -= A[i * nsys + k] * B[k * nrhs + rhs];
            B[i * nrhs + rhs] = s / A[i * nsys + i];
        }
    }
    /* Back solve  L' x = y */
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        for (int i = nsys - 1; i >= 0; --i) {
            double s = B[i * nrhs + rhs];
            for (int k = i + 1; k < nsys; ++k)
                s -= A[k * nsys + i] * B[k * nrhs + rhs];
            B[i * nrhs + rhs] = s / A[i * nsys + i];
        }
    }
    return true;
}

/* Warp-cooperative shared-memory cyclic Jacobi for Phase 2 aggregation finalize.
   Replaces top_mm_eigenvectors_jacobi_device which had per-thread spill of
   ~15 KB local memory and used the slow find-max single-rotation strategy. */
template <int MAXD>
__device__ static void shared_jacobi_top_mm_warp(
    float* sh_A,         /* [D*D] shared, destroyed */
    float* sh_V,         /* [D*D] shared */
    int D, int MM,
    const float* v_ref,  /* [D*MM] in registers/local */
    float* V_out)        /* [D*MM] in registers/local */
{
    const unsigned full_mask = 0xffffffffu;
    const int tid = threadIdx.x & 31;  /* lane id within warp */

    /* Initialize V to identity */
    for (int idx = tid; idx < D * D; idx += 32) sh_V[idx] = 0.0f;
    __syncwarp();
    for (int d = tid; d < D; d += 32) sh_V[d * D + d] = 1.0f;
    __syncwarp();

    /* Symmetrize sh_A in place. */
    for (int idx = tid; idx < D * D; idx += 32) {
        int a = idx / D, c = idx - a * D;
        if (a < c) {
            float v = 0.5f * (sh_A[a * D + c] + sh_A[c * D + a]);
            sh_A[a * D + c] = v;
            sh_A[c * D + a] = v;
        }
    }
    __syncwarp();

    /* Compute trace for tolerance. */
    float local_tr = 0.0f;
    for (int d = tid; d < D; d += 32) local_tr += sh_A[d * D + d];
    for (int o = 16; o > 0; o >>= 1) local_tr += __shfl_xor_sync(full_mask, local_tr, o);
    if (!(local_tr > 1.0e-30f)) {
        for (int mm = 0; mm < MM; ++mm) {
            for (int d = 0; d < D; ++d)
                V_out[d + mm * D] = v_ref ? v_ref[d + mm * D] : ((d == mm) ? 1.0f : 0.0f);
            normalize_vec_device(V_out + mm * D, D);
        }
        return;
    }
    const float tol = 1.0e-7f * local_tr;

    const int max_sweeps = 8;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float local_off = 0.0f;
        for (int idx = tid; idx < D * D; idx += 32) {
            int i = idx / D, j = idx - i * D;
            if (i < j) local_off += fabsf(sh_A[idx]);
        }
        for (int o = 16; o > 0; o >>= 1)
            local_off += __shfl_xor_sync(full_mask, local_off, o);
        if (local_off < tol) break;

        for (int p = 0; p < D - 1; ++p) {
            for (int q = p + 1; q < D; ++q) {
                float apq = sh_A[p * D + q];
#if defined(MANOR_M_JACOBI_READ_BARRIER) && MANOR_M_JACOBI_READ_BARRIER
                // All pivot readers, including skipped rotations, finish before
                // any lane can overwrite shared entries in this/next rotation.
                __syncwarp(full_mask);
#endif
                if (fabsf(apq) < 1.0e-12f) continue;

                float c_rot = 1.0f, s_rot = 0.0f;
                int stop = 0;
                if (tid == 0) {
                    float app = sh_A[p * D + p], aqq = sh_A[q * D + q];
                    float denom = 2.0f * apq;
                    if (fabsf(denom) < 1.0e-30f) {
                        stop = 1;
                    } else {
                        float tau = (aqq - app) / denom;
                        float t = (tau >= 0.0f)
                                ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                                : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                        c_rot = rsqrtf(1.0f + t * t);
                        s_rot = t * c_rot;
                    }
                }
                c_rot = __shfl_sync(full_mask, c_rot, 0);
                s_rot = __shfl_sync(full_mask, s_rot, 0);
                stop  = __shfl_sync(full_mask, stop, 0);
                if (stop) continue;

                /* Fused: row + diag + V updates with single sync. */
                for (int k = tid; k < D; k += 32) {
                    if (k != p && k != q) {
                        float aik = sh_A[p * D + k];
                        float aqk = sh_A[q * D + k];
                        float new_pk = c_rot * aik - s_rot * aqk;
                        float new_qk = s_rot * aik + c_rot * aqk;
                        sh_A[p * D + k] = new_pk;
                        sh_A[k * D + p] = new_pk;
                        sh_A[q * D + k] = new_qk;
                        sh_A[k * D + q] = new_qk;
                    }
                    float vip = sh_V[k * D + p], viq = sh_V[k * D + q];
                    sh_V[k * D + p] = c_rot * vip - s_rot * viq;
                    sh_V[k * D + q] = s_rot * vip + c_rot * viq;
                }
                if (tid == 0) {
                    float app = sh_A[p * D + p], aqq = sh_A[q * D + q];
                    sh_A[p * D + p] = c_rot * c_rot * app - 2.0f * s_rot * c_rot * apq + s_rot * s_rot * aqq;
                    sh_A[q * D + q] = s_rot * s_rot * app + 2.0f * s_rot * c_rot * apq + c_rot * c_rot * aqq;
                    sh_A[p * D + q] = 0.0f;
                    sh_A[q * D + p] = 0.0f;
                }
                __syncwarp();
            }
        }
    }

    /* Find top-MM eigenvalues (largest), extract eigenvectors. */
    if (tid == 0) {
        int perm[MAXD];
        for (int i = 0; i < D; ++i) perm[i] = i;
        for (int mm = 0; mm < MM; ++mm) {
            int best = mm;
            float best_val = sh_A[perm[mm] * D + perm[mm]];
            for (int j = mm + 1; j < D; ++j) {
                float val = sh_A[perm[j] * D + perm[j]];
                if (val > best_val) { best_val = val; best = j; }
            }
            if (best != mm) { int t = perm[mm]; perm[mm] = perm[best]; perm[best] = t; }
        }
        for (int mm = 0; mm < MM; ++mm) {
            int col = perm[mm];
            for (int d = 0; d < D; ++d)
                V_out[d + mm * D] = sh_V[d * D + col];
            float n2 = 0.0f;
            for (int d = 0; d < D; ++d) n2 += V_out[d + mm * D] * V_out[d + mm * D];
            float invn = (n2 > 1e-30f) ? rsqrtf(n2) : 0.0f;
            for (int d = 0; d < D; ++d) V_out[d + mm * D] *= invn;
            if (v_ref) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += V_out[d + mm * D] * v_ref[d + mm * D];
                if (dot < 0.0f) {
                    for (int d = 0; d < D; ++d) V_out[d + mm * D] = -V_out[d + mm * D];
                }
            }
        }
    }
}

#include <stdexcept>

struct OwnerMexError : std::runtime_error {
    OwnerMexError(const char* identifier, const std::string& message)
        : std::runtime_error(message), identifier_(identifier) {}
    const char* identifier() const noexcept { return identifier_; }
private:
    const char* identifier_;
};

static void owner_cuda_check(cudaError_t status, const char* where)
{
    if (status != cudaSuccess) {
        throw OwnerMexError("manor_m_phase2_mex:cuda",
            std::string(where) + ": " + cudaGetErrorString(status));
    }
}

static thread_local double owner_cuda_allocate_milliseconds = 0.0;
static thread_local double owner_cuda_free_milliseconds = 0.0;
static thread_local double owner_cuda_event_destroy_milliseconds = 0.0;
static thread_local bool owner_use_async_allocator = false;

static double owner_host_elapsed_milliseconds(
    const std::chrono::steady_clock::time_point& started)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
static double owner_phase2_core_cuda_allocate_milliseconds()
{
    return owner_cuda_allocate_milliseconds;
}

static double owner_phase2_core_cuda_free_milliseconds()
{
    return owner_cuda_free_milliseconds;
}

static double owner_phase2_core_cuda_event_destroy_milliseconds()
{
    return owner_cuda_event_destroy_milliseconds;
}
#endif

template <typename T>
class OwnerCudaBuffer {
public:
    OwnerCudaBuffer() = default;
    ~OwnerCudaBuffer()
    {
        if (!pointer_ || borrowed_) return;
        const auto started = std::chrono::steady_clock::now();
        if (async_allocator_)
            cudaFreeAsync(pointer_, nullptr);
        else
            cudaFree(pointer_);
        owner_cuda_free_milliseconds += owner_host_elapsed_milliseconds(started);
    }
    OwnerCudaBuffer(const OwnerCudaBuffer&) = delete;
    OwnerCudaBuffer& operator=(const OwnerCudaBuffer&) = delete;

    T* allocate(size_t count, const char* label)
    {
        if (pointer_)
            throw OwnerMexError("manor_m_phase2_mex:cudaOwnership",
                std::string(label) + " was allocated more than once.");
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw OwnerMexError("manor_m_phase2_mex:integerOverflow",
                std::string(label) + " allocation overflows size_t.");
        if (count == 0) return nullptr;
        const auto started = std::chrono::steady_clock::now();
        async_allocator_ = owner_use_async_allocator;
        const cudaError_t status = async_allocator_
            ? cudaMallocAsync(
                reinterpret_cast<void**>(&pointer_), sizeof(T) * count, nullptr)
            : cudaMalloc(
                reinterpret_cast<void**>(&pointer_), sizeof(T) * count);
        owner_cuda_allocate_milliseconds +=
            owner_host_elapsed_milliseconds(started);
        owner_cuda_check(status, label);
        return pointer_;
    }
    T* get() const { return pointer_; }
    T* allocate_input(size_t count, const char* label, const void* host)
    {
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
        if (pointer_ || count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw OwnerMexError("manor_m_phase2_mex:cudaOwnership", "Invalid resident input allocation.");
        pointer_ = static_cast<T*>(manor_m_resident_find(host, count * sizeof(T)));
        if (pointer_) { borrowed_ = true; return pointer_; }
#endif
        return allocate(count, label);
    }
private:
    T* pointer_ = nullptr;
    bool async_allocator_ = false;
    bool borrowed_ = false;
};

static cudaError_t owner_upload_input(void* dst, const void* src, size_t bytes)
{
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
    return manor_m_resident_upload(dst, src, bytes);
#else
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#endif
}

static mxArray* owner_named_output(const char* name, mwSize rank,
    const mwSize* dimensions, mxClassID type, bool zero_fill)
{
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
    return manor_m_minimex_named_output(name, rank, dimensions, type, zero_fill);
#else
    return type == mxLOGICAL_CLASS ? mxCreateLogicalArray(rank, dimensions) :
        mxCreateNumericArray(rank, dimensions, type, mxREAL);
#endif
}

/* Guarded cudaEvent span timers (item 23c). Inactive unless
   opts.profile_timers is set; spans only record events on the default
   stream, and elapsed times are read after the final device synchronize,
   so profiling adds no synchronization of its own. */
struct OwnerProfileTimers {
    bool enabled = false;
    std::vector<const char*> labels;
    std::vector<cudaEvent_t> starts;
    std::vector<cudaEvent_t> stops;
    ~OwnerProfileTimers()
    {
        const auto started = std::chrono::steady_clock::now();
        for (cudaEvent_t event : starts) if (event) cudaEventDestroy(event);
        for (cudaEvent_t event : stops) if (event) cudaEventDestroy(event);
        owner_cuda_event_destroy_milliseconds +=
            owner_host_elapsed_milliseconds(started);
    }
    void begin(const char* label)
    {
        if (!enabled) return;
        cudaEvent_t event = nullptr;
        owner_cuda_check(cudaEventCreate(&event), "profile span start create");
        owner_cuda_check(cudaEventRecord(event), "profile span start record");
        labels.push_back(label);
        starts.push_back(event);
        stops.push_back(nullptr);
    }
    void end()
    {
        if (!enabled) return;
        cudaEvent_t event = nullptr;
        owner_cuda_check(cudaEventCreate(&event), "profile span stop create");
        owner_cuda_check(cudaEventRecord(event), "profile span stop record");
        stops.back() = event;
    }
};

#include "../../cuda/core/manor_m_aggregation_tiles.h"
#include "../../cuda/core/manor_m_index_stream.cuh"
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
#include "../../cuda/core/manor_m_generated_phase2.cuh"
#endif

struct OwnerPhase2Opts {
    float lambda = 1.0e-5f;
    float jitter = 1.0e-6f;
    float gpu_mem_fraction = 0.92f;
    float reserve_fraction = 0.08f;
    double reserve_bytes = 0.0;
    int min_query_block = 1;
    int max_query_block = 131072;
    int regression_mode = 0;
    int iterations = 1;
    float heavy_factor = 0.10f;
    float heavy_factor_tmp = 0.20f;
    std::vector<float> hk_iter;
    bool idx_zero_based = false;
    bool idx_shared_slice = false;
    bool return_debug = false;
    bool return_histories = true;
    bool silent = false;
    bool fused_eval_use_owner_mask = true;
    bool profile_timers = false;
    bool async_allocator = false;
    int aggregation_tile_owners = 0;
    int index_tile_owners = 0;
    bool idx_compact_owner_order = false;
    bool idx_generated_response = false;
    /* Whether to run the fused evaluation at all. Previously inferred from the
       argument count, which cannot survive the evaluation arrays being
       individually optional -- omitting one silently disabled the feature. */
    bool fused_eval = false;
    /* Whether to score a second, held-out VALIDATION set alongside the test
       set in the same call. Same flag discipline as fused_eval: the flag
       decides, never the argument count. The validation set carries only X
       and Y -- it is ordinary training data whose Y was withheld from the
       model -- so it is always scored MSE-only, whatever the test block asks
       for. */
    bool fused_validation = false;
    /* Shared-core opt-in used to score an immutable checkpoint.  It executes
       only the existing K=0 fixed-order evaluator and never enters the
       regression/aggregation iteration body. */
    bool evaluation_only = false;
};

static const mxArray* owner_option(const mxArray* opts, const char* name)
{
    return opts ? mxGetField(opts, 0, name) : nullptr;
}

static double owner_scalar(const mxArray* value, const char* name)
{
    if (!value || mxIsComplex(value) || mxIsSparse(value) || !mxIsNumeric(value) ||
        mxGetNumberOfElements(value) != 1) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            std::string(name) + " must be a real numeric scalar.");
    }
    double result = mxGetScalar(value);
    if (!std::isfinite(result)) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            std::string(name) + " must be finite.");
    }
    return result;
}

static int owner_exact_int_scalar(const mxArray* value, const char* name)
{
    const double raw = owner_scalar(value, name);
    if (raw != std::floor(raw) ||
        raw < static_cast<double>(std::numeric_limits<int>::min()) ||
        raw > static_cast<double>(std::numeric_limits<int>::max())) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            std::string(name) + " must be an exact int-range integer scalar.");
    }
    return static_cast<int>(raw);
}

static bool owner_bool(const mxArray* value, const char* name)
{
    if (mxIsLogicalScalar(value)) return mxIsLogicalScalarTrue(value);
    return owner_scalar(value, name) != 0.0;
}

static OwnerPhase2Opts owner_parse_opts(const mxArray* opts)
{
    if (!opts || !mxIsStruct(opts) || mxGetNumberOfElements(opts) != 1) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            "opts must be a scalar struct.");
    }
    OwnerPhase2Opts parsed;
    auto set_float = [&](const char* name, float& destination) {
        if (const mxArray* value = owner_option(opts, name)) {
            double raw = owner_scalar(value, name);
            destination = static_cast<float>(raw);
            if (!std::isfinite(destination)) {
                throw OwnerMexError("manor_m_phase2_mex:opts",
                    std::string(name) + " is outside the supported single-precision range.");
            }
        }
    };
    set_float("lambda", parsed.lambda);
    set_float("jitter", parsed.jitter);
    set_float("gpu_mem_fraction", parsed.gpu_mem_fraction);
    set_float("reserve_fraction", parsed.reserve_fraction);
    if (const mxArray* value = owner_option(opts, "reserve_bytes"))
        parsed.reserve_bytes = owner_scalar(value, "reserve_bytes");
    if (const mxArray* value = owner_option(opts, "min_query_block"))
        parsed.min_query_block = owner_exact_int_scalar(value, "min_query_block");
    if (const mxArray* value = owner_option(opts, "max_query_block"))
        parsed.max_query_block = owner_exact_int_scalar(value, "max_query_block");
    if (const mxArray* value = owner_option(opts, "regression_mode"))
        parsed.regression_mode = owner_exact_int_scalar(value, "regression_mode");
    if (const mxArray* value = owner_option(opts, "K"))
        parsed.iterations = owner_exact_int_scalar(value, "K");
    if (const mxArray* value = owner_option(opts, "iterations"))
        parsed.iterations = owner_exact_int_scalar(value, "iterations");
    set_float("heavy_factor", parsed.heavy_factor);
    set_float("heavy_factor_tmp", parsed.heavy_factor_tmp);
    if (const mxArray* value = owner_option(opts, "hk_iter")) {
        if (mxIsComplex(value) || mxIsSparse(value) ||
            (!mxIsSingle(value) && !mxIsDouble(value)) || mxGetNumberOfElements(value) < 1) {
            throw OwnerMexError("manor_m_phase2_mex:opts",
                "hk_iter must be a nonempty real single/double vector.");
        }
        const mwSize count = mxGetNumberOfElements(value);
        parsed.hk_iter.resize((size_t)count);
        for (mwSize i = 0; i < count; ++i) {
            const float entry = mxIsSingle(value) ?
                static_cast<const float*>(mxGetData(value))[i] :
                static_cast<float>(mxGetPr(value)[i]);
            if (!std::isfinite(entry) || entry <= 0.0f)
                throw OwnerMexError("manor_m_phase2_mex:opts",
                    "hk_iter entries must be finite positive values.");
            parsed.hk_iter[(size_t)i] = entry;
        }
    }
    if (const mxArray* value = owner_option(opts, "fused_eval"))
        parsed.fused_eval = owner_bool(value, "fused_eval");
    if (const mxArray* value = owner_option(opts, "fused_validation"))
        parsed.fused_validation = owner_bool(value, "fused_validation");
    if (const mxArray* value = owner_option(opts, "idx_zero_based"))
        parsed.idx_zero_based = owner_bool(value, "idx_zero_based");
    if (const mxArray* value = owner_option(opts, "idx_shared_slice"))
        parsed.idx_shared_slice = owner_bool(value, "idx_shared_slice");
    if (const mxArray* value = owner_option(opts, "return_debug"))
        parsed.return_debug = owner_bool(value, "return_debug");
    if (const mxArray* value = owner_option(opts, "return_histories"))
        parsed.return_histories = owner_bool(value, "return_histories");
    if (const mxArray* value = owner_option(opts, "silent"))
        parsed.silent = owner_bool(value, "silent");
    if (const mxArray* value = owner_option(opts, "fused_eval_use_owner_mask"))
        parsed.fused_eval_use_owner_mask = owner_bool(value, "fused_eval_use_owner_mask");
    if (const mxArray* value = owner_option(opts, "profile_timers"))
        parsed.profile_timers = owner_bool(value, "profile_timers");
    if (const mxArray* value = owner_option(opts, "async_allocator"))
        parsed.async_allocator = owner_bool(value, "async_allocator");
    if (const mxArray* value = owner_option(opts, "aggregation_tile_owners"))
        parsed.aggregation_tile_owners = owner_exact_int_scalar(value, "aggregation_tile_owners");
    if (const mxArray* value = owner_option(opts, "index_tile_owners"))
        parsed.index_tile_owners = owner_exact_int_scalar(value, "index_tile_owners");
    if (const mxArray* value = owner_option(opts, "idx_compact_owner_order"))
        parsed.idx_compact_owner_order = owner_bool(value, "idx_compact_owner_order");
    if (const mxArray* value = owner_option(opts, "idx_generated_response"))
        parsed.idx_generated_response = owner_bool(value, "idx_generated_response");
    if (const mxArray* value = owner_option(opts, "evaluation_only"))
        parsed.evaluation_only = owner_bool(value, "evaluation_only");

    if (parsed.lambda < 0.0f || parsed.jitter < 0.0f ||
        parsed.gpu_mem_fraction <= 0.0f || parsed.gpu_mem_fraction > 1.0f ||
        parsed.reserve_fraction < 0.0f || parsed.reserve_fraction >= 1.0f ||
        parsed.reserve_bytes < 0.0 || parsed.min_query_block < 1 ||
        parsed.aggregation_tile_owners < 0 ||
        parsed.index_tile_owners < 0 ||
        parsed.max_query_block < parsed.min_query_block ||
        parsed.iterations < 1 || parsed.heavy_factor < 0.0f ||
        parsed.heavy_factor_tmp < 0.0f ||
        (parsed.regression_mode != 0 && parsed.regression_mode != 1)) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            "Invalid regression, memory, or query-block options.");
    }
    if (parsed.idx_compact_owner_order &&
        (parsed.idx_shared_slice || !parsed.idx_zero_based)) {
        throw OwnerMexError("manor_m_phase2_mex:idx",
            "A compact owner index must be explicit zero-based owner rows.");
    }
    if (parsed.idx_generated_response &&
        (parsed.idx_shared_slice || parsed.idx_compact_owner_order ||
         parsed.index_tile_owners != 0 || !parsed.idx_zero_based)) {
        throw OwnerMexError("manor_m_phase2_mex:idx",
            "Generated response indices are incompatible with shared, compact, "
            "or host-streamed indices and require zero-based mode.");
    }
#if !defined(MANOR_M_PHASE2_CORE_LIBRARY)
    if (parsed.idx_generated_response) {
        throw OwnerMexError("manor_m_phase2_mex:idx",
            "Generated response indices are available only through the shared core ABI.");
    }
#endif
    if (!parsed.hk_iter.empty() && parsed.hk_iter.size() != 1 &&
        parsed.hk_iter.size() != (size_t)parsed.iterations) {
        throw OwnerMexError("manor_m_phase2_mex:opts",
            "hk_iter must be scalar or have exactly K/iterations entries.");
    }
    if (owner_option(opts, "benchmark_fused_eval") ||
        owner_option(opts, "benchmark_fused_eval_repeats")) {
        // only launch sat behind a default-false option -- and the tiled
        // engine it was written to validate is now the only evaluator.
        // Ignoring the field would silently run a different engine than the
        // caller asked for, so refuse instead.
        throw OwnerMexError("manor_m_phase2_mex:benchmarkRemoved",
            "opts.benchmark_fused_eval and opts.benchmark_fused_eval_repeats "
            "were removed: the guarded reference kernel and its benchmark "
            "lane are gone and the tiled evaluator is the only engine. "
            "Delete the option.");
    }
    return parsed;
}

static float owner_validated_hk(double raw)
{
    const float value = static_cast<float>(raw);
    const float squared = value * value;
    const float inverse_squared = 1.0f / squared;
    if (!std::isfinite(value) || value <= 0.0f || !std::isfinite(squared) ||
        squared <= 0.0f || !std::isfinite(inverse_squared) || inverse_squared <= 0.0f) {
        throw OwnerMexError("manor_m_phase2_mex:bandwidth",
            "hk must remain finite and positive after single conversion, squaring, and inversion.");
    }
    return value;
}

static float owner_validated_rhok(double raw, bool fused_eval)
{
    const float value = static_cast<float>(raw);
    const float floored = fmaxf(value, 1.0e-12f);
    const float inverse = 1.0f / floored;
    const float beta = inverse - 1.0f;
    if (!std::isfinite(value) || value <= 0.0f || !std::isfinite(floored) ||
        floored <= 0.0f || !std::isfinite(inverse) || inverse <= 0.0f ||
        !std::isfinite(beta)) {
        throw OwnerMexError("manor_m_phase2_mex:bandwidth",
            "rhok must remain finite and positive after single conversion and device transforms.");
    }
    if (fused_eval) {
        const float squared = value * value;
        const float inverse_squared = 1.0f / squared;
        if (!std::isfinite(squared) || squared <= 0.0f ||
            !std::isfinite(inverse_squared) || inverse_squared <= 0.0f) {
            throw OwnerMexError("manor_m_phase2_mex:bandwidth",
                "rhok must remain finite and positive after fused device transforms.");
        }
    }
    return value;
}

static double owner_numeric_at(const mxArray* array, mwIndex index);

static std::vector<float> owner_validated_rhok_schedule(
    const mxArray* value, int iterations, bool fused_eval)
{
    if (!value || mxIsComplex(value) || mxIsSparse(value) || !mxIsNumeric(value) ||
        mxGetNumberOfDimensions(value) != 2 || (mxGetM(value) != 1 && mxGetN(value) != 1)) {
        throw OwnerMexError("manor_m_phase2_mex:bandwidth",
            "rhok must be a real numeric scalar or vector.");
    }
    const mwSize count = mxGetNumberOfElements(value);
    if (count != 1 && count != static_cast<mwSize>(iterations)) {
        throw OwnerMexError("manor_m_phase2_mex:bandwidth",
            "rhok must be scalar or have exactly K/iterations entries.");
    }
    std::vector<float> schedule((size_t)count);
    for (mwIndex k = 0; k < count; ++k)
        schedule[(size_t)k] = owner_validated_rhok(owner_numeric_at(value, k), fused_eval);
    return schedule;
}

static size_t owner_checked_mul(size_t left, size_t right, const char* label)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw OwnerMexError("manor_m_phase2_mex:integerOverflow",
            std::string(label) + " exceeds SIZE_MAX.");
    }
    return left * right;
}

static size_t owner_checked_add(size_t left, size_t right, const char* label)
{
    if (left > std::numeric_limits<size_t>::max() - right) {
        throw OwnerMexError("manor_m_phase2_mex:integerOverflow",
            std::string(label) + " exceeds SIZE_MAX.");
    }
    return left + right;
}

static double owner_numeric_at(const mxArray* array, mwIndex index)
{
    switch (mxGetClassID(array)) {
    case mxINT8_CLASS: return static_cast<const int8_t*>(mxGetData(array))[index];
    case mxUINT8_CLASS: return static_cast<const uint8_t*>(mxGetData(array))[index];
    case mxINT16_CLASS: return static_cast<const int16_t*>(mxGetData(array))[index];
    case mxUINT16_CLASS: return static_cast<const uint16_t*>(mxGetData(array))[index];
    case mxINT32_CLASS: return static_cast<const int32_t*>(mxGetData(array))[index];
    case mxUINT32_CLASS: return static_cast<const uint32_t*>(mxGetData(array))[index];
    case mxINT64_CLASS: return static_cast<double>(static_cast<const int64_t*>(mxGetData(array))[index]);
    case mxUINT64_CLASS: return static_cast<double>(static_cast<const uint64_t*>(mxGetData(array))[index]);
    case mxDOUBLE_CLASS: return mxGetPr(array)[index];
    case mxSINGLE_CLASS: return static_cast<const float*>(mxGetData(array))[index];
    default: break;
    }
    throw OwnerMexError("manor_m_phase2_mex:idx",
        "idx must be int32, uint32, single, or double.");
}

#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
class OwnerInputFloats {
    std::vector<float> copied_;
    const float* data_;
    size_t count_;
public:
    OwnerInputFloats(const float* source, size_t count) : data_(source), count_(count) {
        if (!manor_m_resident_find(source, count * sizeof(float))) {
            copied_.assign(source, source + count);
            data_ = copied_.data();
        }
    }
    const float* data() const { return data_; }
    size_t size() const { return count_; }
    const float* begin() const { return data_; }
    const float* end() const { return data_ + count_; }
    const float& operator[](size_t i) const { return data_[i]; }
};
#else
using OwnerInputFloats = std::vector<float>;
#endif

static OwnerInputFloats owner_float_data(const mxArray* array, const char* name)
{
    if (!mxIsSingle(array) || mxIsComplex(array) || mxIsSparse(array)) {
        throw OwnerMexError("manor_m_phase2_mex:type",
            std::string(name) + " must be a dense real single array.");
    }
    const size_t count = mxGetNumberOfElements(array);
    const float* source = static_cast<const float*>(mxGetData(array));
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
    return OwnerInputFloats(source, count);
#else
    return OwnerInputFloats(source, source + count);
#endif
}

__device__ __forceinline__ int owner_idx_at(
    const int* idx_gpu, int anchor, int rank, int kSlice,
    bool idx_shared_slice)
{
    size_t offset = idx_shared_slice ? static_cast<size_t>(rank) :
        static_cast<size_t>(anchor) * kSlice + rank;
    return idx_gpu[offset];
}

template <int MAXD>
__device__ __forceinline__ void owner_regression_barrier()
{
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    // The D10/D20 entry-distributed path launches a complete physical warp.
    // Its tail lanes write covariance entries, so they must participate in
    // every caller barrier even when the older partial-warp sync is enabled.
    if constexpr (MAXD == 12 || MAXD == 20) {
        __syncwarp(0xffffffffu);
        return;
    }
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_SYNC
    // The launch uses exactly MAXD threads, including partially populated
    // warps. Do not use __activemask() within a divergent branch as the mask.
    if constexpr (MAXD <= 32) __syncwarp(0xffffffffu >> (32 - MAXD));
    else __syncthreads();
#else
    __syncthreads();
#endif
}

template <int MAXD>
__global__ void owner_regression_kernel(
    const float* __restrict__ Xrm, const float* __restrict__ Yrm,
    const int* __restrict__ idx_gpu, const int* __restrict__ index_rows,
    const float* __restrict__ Tanrm,
    const float* __restrict__ Cenrm, const int* __restrict__ owner_index,
    int q0, int index_q0, int B, int D, int MM, int MY, int kSlice,
    bool idx_shared_slice, float hk, float rhok,
    float lambda_ridge, float jitter_scale, int regression_mode,
    uint32_t* __restrict__ neigh_mask_bits, float* __restrict__ grad_out,
    float* __restrict__ center_pre, float* __restrict__ weight_sum,
    int32_t* __restrict__ num_neigh, uint8_t* __restrict__ solver_failed,
    const int* __restrict__ stopped)
{
    const int local_q = q0 + blockIdx.x;
    const int row = threadIdx.x;
    if (blockIdx.x >= B || *stopped) return;
    const int i = owner_index[local_q];
    const int owner_local = local_q - index_q0;
    const int index_anchor = index_rows ? index_rows[owner_local] : owner_local;
    const int nsys = D + 1;

    __shared__ float sh_xi[MAXD];
    __shared__ float sh_ti[MAX_MM_MANOR * MAXD];
    __shared__ float sh_dx[MAXD];
    __shared__ float sh_z[MAXD];
    __shared__ float sh_proj[MAX_MM_MANOR];
    __shared__ float sh_yj[MAX_MY_MANOR];
    __shared__ float sh_scratch[8];

    if (row < D) sh_xi[row] = Xrm[(size_t)i * D + row];
    for (int mm = 0; mm < MM; ++mm)
        if (row < D) sh_ti[mm * MAXD + row] = Tanrm[(size_t)i * MM * D + (size_t)mm * D + row];
    owner_regression_barrier<MAXD>();

    float a01_row = 0.0f, cacc_row = 0.0f;
    float bb1_row[MAX_MY_MANOR];
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    float cov_entries[(MAXD * MAXD + 31) / 32];
    // Retained only in non-target template instantiations.  The target
    // MAXD=12/20 instantiations discard this array with the constexpr branch.
    float cov_row[MAXD];
#else
    float cov_row[MAXD];
#endif
    for (int out = 0; out < MY; ++out) bb1_row[out] = 0.0f;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    if constexpr (MAXD == 12 || MAXD == 20) {
        for (float& value : cov_entries) value = 0.0f;
    } else {
        for (int c = 0; c < MAXD; ++c) cov_row[c] = 0.0f;
    }
#else
    for (int c = 0; c < MAXD; ++c) cov_row[c] = 0.0f;
#endif
    float aa00 = 0.0f, wsum = 0.0f;
    int nnei = 0;
    float bb0[MAX_MY_MANOR];
    for (int out = 0; out < MY; ++out) bb0[out] = 0.0f;
    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

    uint32_t* mask_row = neigh_mask_bits ?
        neigh_mask_bits + (size_t)local_q * ((kSlice + 31) >> 5) : nullptr;
    const int mask_words = (kSlice + 31) >> 5;
    if (mask_row)
        for (int word = row; word < mask_words; word += blockDim.x)
            mask_row[word] = 0u;
    owner_regression_barrier<MAXD>();

#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    if constexpr (MAXD == 12 || MAXD == 20) {
        owner_regression_warp_entry_accumulate<MAXD, MAX_MY_MANOR>(
            Xrm, Yrm, idx_gpu, sh_ti, row < D ? sh_xi[row] : 0.0f,
            index_anchor, D, MM, MY, kSlice, idx_shared_slice, hk2_inv, beta,
            regression_mode, mask_row, a01_row, cacc_row, bb1_row, cov_entries,
            aa00, wsum, nnei, bb0);
    } else
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_SHUFFLE_INPUTS
    if constexpr (MAXD <= 32) {
        owner_regression_shuffle_accumulate<MAXD, MAX_MY_MANOR>(
            Xrm, Yrm, idx_gpu, sh_ti, row < D ? sh_xi[row] : 0.0f,
            index_anchor, D, MM, MY, kSlice, idx_shared_slice, hk2_inv, beta,
            regression_mode, mask_row, a01_row, cacc_row, bb1_row, cov_row,
            aa00, wsum, nnei, bb0);
    } else
#endif
    for (int s = 0; s < kSlice; ++s) {
        const int j = owner_idx_at(
            idx_gpu, index_anchor, s, kSlice, idx_shared_slice);
        // Non-owners remain valid raw observations; there is deliberately no owner[j] test.
        if (row < D) sh_dx[row] = Xrm[(size_t)j * D + row] - sh_xi[row];
        owner_regression_barrier<MAXD>();
        if (row < MM) {
            float projection = 0.0f;
            for (int d = 0; d < D; ++d) projection += sh_ti[row * MAXD + d] * sh_dx[d];
            sh_proj[row] = projection;
        }
        owner_regression_barrier<MAXD>();
        float z_row = 0.0f;
        if (row < D) {
            z_row = sh_dx[row];
            for (int mm = 0; mm < MM; ++mm) z_row += beta * sh_proj[mm] * sh_ti[mm * MAXD + row];
            sh_z[row] = z_row;
        }
        owner_regression_barrier<MAXD>();
        if (row == 0) {
            float squared = 0.0f;
            for (int d = 0; d < D; ++d) squared += sh_z[d] * sh_z[d];
            sh_scratch[0] = squared * hk2_inv;
        }
        owner_regression_barrier<MAXD>();
        const float squared = sh_scratch[0];
        if (squared >= 1.0f) continue;
        if (row == 0 && mask_row)
            mask_row[s >> 5] |= (1u << (s & 31));
        const float weight = (1.0f - squared) * (1.0f - squared);
        if (row < MY) sh_yj[row] = Yrm[(size_t)j * MY + row];
        owner_regression_barrier<MAXD>();
        if (row == 0) {
            aa00 += weight;
            wsum += weight;
            ++nnei;
            for (int out = 0; out < MY; ++out) bb0[out] += weight * sh_yj[out];
        }
        if (row < D) {
            const float xreg = regression_mode == 1 ? ((row < MM) ? sh_proj[row] : 0.0f) : z_row;
            a01_row += weight * xreg;
            cacc_row += Xrm[(size_t)j * D + row];
            for (int out = 0; out < MY; ++out) bb1_row[out] += weight * sh_yj[out] * xreg;
            for (int c = 0; c < D; ++c) {
                const float zc = regression_mode == 1 ? ((c < MM) ? sh_proj[c] : 0.0f) : sh_z[c];
                cov_row[c] += weight * xreg * zc;
            }
        }
        owner_regression_barrier<MAXD>();
    }

    extern __shared__ unsigned char regression_shared[];
    float* gathered = reinterpret_cast<float*>(regression_shared);
    if (row < D) {
        gathered[row] = a01_row;
        gathered[D + row] = cacc_row;
        for (int out = 0; out < MY; ++out) gathered[2 * D + D * D + row * MY + out] = bb1_row[out];
    }
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    if constexpr (MAXD == 12 || MAXD == 20) {
        constexpr int cov_per_lane = (MAXD * MAXD + 31) / 32;
        for (int p = 0; p < cov_per_lane; ++p) {
            const int entry = row + 32 * p;
            if (entry < D * D) gathered[2 * D + entry] = cov_entries[p];
        }
    } else
#endif
    if (row < D)
        for (int c = 0; c < D; ++c) gathered[2 * D + row * D + c] = cov_row[c];
    owner_regression_barrier<MAXD>();
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
    if constexpr (MAXD == 12 || MAXD == 20) if (row >= MAXD) return;
#endif
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_COOPERATIVE_SOLVE
    if constexpr (MAXD <= 32) {
        owner_regression_cooperative_solve<MAXD>(regression_shared, gathered,
            sh_ti, D, MM, MY, local_q, i, aa00, bb0, wsum, nnei, beta,
            lambda_ridge, jitter_scale, regression_mode, Cenrm, weight_sum,
            num_neigh, solver_failed, center_pre, grad_out);
        return;
    }
#endif
    if (row != 0) return;

    weight_sum[local_q] = wsum;
    num_neigh[local_q] = nnei;
    solver_failed[local_q] = 0u;
    for (int d = 0; d < D; ++d)
        center_pre[(size_t)local_q * D + d] = nnei > 0 ? gathered[D + d] / (float)nnei : Cenrm[(size_t)i * D + d];
    if (nnei < 2 * D) return;

#if MANOR_M_REGRESSION_SHARED_SOLVE
    /* Only lane zero consumes the small dense solve.  Keeping these arrays as
       automatic storage makes NVCC reserve the full MAXD-bounded frame for
       every thread even though nineteen of the D=20 threads return above.
       Put the same row-major double arrays after the float gather slab instead.
       Dynamic shared memory is suitably aligned; rounding the float slab to
       eight bytes also keeps the contract explicit.  The Cholesky loop and
       every arithmetic operation below remain unchanged. */
    const size_t gathered_bytes = sizeof(float) * (
        static_cast<size_t>(2) * D + static_cast<size_t>(D) * D +
        static_cast<size_t>(D) * MY);
    const size_t solve_offset = (gathered_bytes + alignof(double) - 1) &
        ~(static_cast<size_t>(alignof(double)) - 1);
    double* matrix = reinterpret_cast<double*>(
        regression_shared + solve_offset);
    double* rhs = matrix + static_cast<size_t>(nsys) * nsys;
#else
    double matrix_storage[(MAXD + 1) * (MAXD + 1)];
    double rhs_storage[(MAXD + 1) * MAX_MY_MANOR];
    double* matrix = matrix_storage;
    double* rhs = rhs_storage;
#endif
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
    const double jitter = lambda_ridge + jitter_scale * trace / fmax((double)nsys, 1.0);
    for (int t = 0; t < nsys; ++t) matrix[t * nsys + t] += jitter;
    for (int out = 0; out < MY; ++out) rhs[out] = bb0[out];
    for (int d = 0; d < D; ++d) for (int out = 0; out < MY; ++out)
        rhs[(d + 1) * MY + out] = gathered[2 * D + D * D + d * MY + out];
    bool solved = cholesky_solve_device_multi(matrix, rhs, nsys, MY);
    if (!solved) {
        const double add = 100.0 * jitter + 1e-8;
        for (int t = 0; t < nsys * nsys; ++t) matrix[t] = 0.0;
        matrix[0] = aa00;
        for (int d = 0; d < D; ++d) {
            matrix[d + 1] = gathered[d];
            matrix[(d + 1) * nsys] = gathered[d];
        }
        for (int a = 0; a < D; ++a) for (int c = 0; c < D; ++c)
            matrix[(a + 1) * nsys + c + 1] = gathered[2 * D + a * D + c];
        for (int t = 0; t < nsys; ++t) matrix[t * nsys + t] += jitter + add;
        for (int out = 0; out < MY; ++out) rhs[out] = bb0[out];
        for (int d = 0; d < D; ++d) for (int out = 0; out < MY; ++out)
            rhs[(d + 1) * MY + out] = gathered[2 * D + D * D + d * MY + out];
        solved = cholesky_solve_device_multi(matrix, rhs, nsys, MY);
    }
    if (!solved) {
        solver_failed[local_q] = 1u;
        for (int out = 0; out < MY; ++out) for (int d = 0; d < D; ++d)
            grad_out[(size_t)local_q * MY * D + (size_t)out * D + d] = 0.0f;
        return;
    }
    for (int out = 0; out < MY; ++out) {
        float gradient[MAXD];
        if (regression_mode == 1) {
            for (int d = 0; d < D; ++d) {
                float total = 0.0f;
                for (int mm = 0; mm < MM; ++mm) total += sh_ti[mm * MAXD + d] * (float)rhs[(mm + 1) * MY + out];
                gradient[d] = total;
            }
        } else {
            for (int d = 0; d < D; ++d) gradient[d] = (float)rhs[(d + 1) * MY + out];
            float coefficient[MAX_MM_MANOR];
            for (int mm = 0; mm < MM; ++mm) {
                coefficient[mm] = 0.0f;
                for (int d = 0; d < D; ++d) coefficient[mm] += sh_ti[mm * MAXD + d] * gradient[d];
            }
            for (int d = 0; d < D; ++d) {
                float correction = 0.0f;
                for (int mm = 0; mm < MM; ++mm) correction += sh_ti[mm * MAXD + d] * coefficient[mm];
                gradient[d] += beta * correction;
            }
        }
        for (int d = 0; d < D; ++d)
            grad_out[(size_t)local_q * MY * D + (size_t)out * D + d] = gradient[d];
    }
}

#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_TILED_MASK
/* A full owner-by-neighbour bit mask is the last O(M*k) Phase-2 allocation.
   Regression must still finish for every owner before the global heavy/keep
   decision, so an owner tile cannot be aggregated early.  When the requested
   aggregation capacity is smaller than M, the regression pass records the
   same gradients/counts but no mask.  This kernel then reproduces only the
   original membership predicate for one aggregation tile.  The mask buffer is
   consumed immediately and reused by the next tile.

   D10/D20 use the identical full-warp shuffle sequence as
   owner_regression_warp_entry_accumulate.  Other <=32 tiers use the identical
   partial-warp sequence as owner_regression_shuffle_accumulate, and wider tiers
   retain the shared-memory sequence from owner_regression_kernel. */
template <int MAXD>
__global__ void owner_neighbor_mask_rebuild_kernel(
    const float* __restrict__ Xrm, const int* __restrict__ idx_gpu,
    const int* __restrict__ index_rows, const float* __restrict__ Tanrm,
    const int* __restrict__ owner_index, int owner_first, int index_q0,
    int B, int D, int MM, int kSlice, bool idx_shared_slice,
    float hk, float rhok, uint32_t* __restrict__ neigh_mask_bits,
    const int* __restrict__ stopped)
{
    const int tile_q = blockIdx.x;
    const int local_q = owner_first + tile_q;
    const int row = threadIdx.x;
    if (tile_q >= B || *stopped) return;
    const int i = owner_index[local_q];
    const int owner_local = local_q - index_q0;
    const int index_anchor = index_rows ? index_rows[owner_local] : owner_local;
    const int mask_words = (kSlice + 31) >> 5;
    uint32_t* mask_row = neigh_mask_bits + (size_t)tile_q * mask_words;

    __shared__ float sh_xi[MAXD];
    __shared__ float sh_ti[MAX_MM_MANOR * MAXD];
    __shared__ float sh_dx[MAXD];
    __shared__ float sh_z[MAXD];
    __shared__ float sh_proj[MAX_MM_MANOR];
    __shared__ float sh_squared;

    if (row < D) sh_xi[row] = Xrm[(size_t)i * D + row];
    for (int mm = 0; mm < MM; ++mm)
        if (row < D)
            sh_ti[mm * MAXD + row] =
                Tanrm[(size_t)i * MM * D + (size_t)mm * D + row];
    for (int word = row; word < mask_words; word += blockDim.x)
        mask_row[word] = 0u;
    owner_regression_barrier<MAXD>();

    const float hk2_inv = 1.0f / (hk * hk);
    const float beta = 1.0f / fmaxf(rhok, 1.0e-12f) - 1.0f;

#if MANOR_M_REGRESSION_SHUFFLE_INPUTS
    if constexpr (MAXD <= 32) {
#if MANOR_M_REGRESSION_WARP_ENTRIES
        constexpr unsigned shuffle_mask =
            (MAXD == 12 || MAXD == 20) ? 0xffffffffu :
                (0xffffffffu >> (32 - MAXD));
#else
        constexpr unsigned shuffle_mask = 0xffffffffu >> (32 - MAXD);
#endif
        const float xi = row < D ? sh_xi[row] : 0.0f;
        for (int s = 0; s < kSlice; ++s) {
            const int j = owner_idx_at(
                idx_gpu, index_anchor, s, kSlice, idx_shared_slice);
            const float xj = row < D ?
                Xrm[(size_t)j * D + row] : 0.0f;
            const float dx = row < D ? xj - xi : 0.0f;
            float projection = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float source = __shfl_sync(shuffle_mask, dx, d);
                if (row < MM)
                    projection += sh_ti[row * MAXD + d] * source;
            }
            float z = dx;
            for (int mm = 0; mm < MM; ++mm) {
                const float projected =
                    __shfl_sync(shuffle_mask, projection, mm);
                if (row < D)
                    z += beta * projected * sh_ti[mm * MAXD + row];
            }
            float squared = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float source = __shfl_sync(shuffle_mask, z, d);
                if (row == 0) squared += source * source;
            }
            squared = __shfl_sync(shuffle_mask, squared * hk2_inv, 0);
            if (row == 0 && !(squared >= 1.0f))
                mask_row[s >> 5] |= (1u << (s & 31));
        }
        return;
    }
#endif

    for (int s = 0; s < kSlice; ++s) {
        const int j = owner_idx_at(
            idx_gpu, index_anchor, s, kSlice, idx_shared_slice);
        if (row < D)
            sh_dx[row] = Xrm[(size_t)j * D + row] - sh_xi[row];
        owner_regression_barrier<MAXD>();
        if (row < MM) {
            float projection = 0.0f;
            for (int d = 0; d < D; ++d)
                projection += sh_ti[row * MAXD + d] * sh_dx[d];
            sh_proj[row] = projection;
        }
        owner_regression_barrier<MAXD>();
        if (row < D) {
            float z = sh_dx[row];
            for (int mm = 0; mm < MM; ++mm)
                z += beta * sh_proj[mm] * sh_ti[mm * MAXD + row];
            sh_z[row] = z;
        }
        owner_regression_barrier<MAXD>();
        if (row == 0) {
            float squared = 0.0f;
            for (int d = 0; d < D; ++d)
                squared += sh_z[d] * sh_z[d];
            sh_squared = squared * hk2_inv;
        }
        owner_regression_barrier<MAXD>();
        if (row == 0 && !(sh_squared >= 1.0f))
            mask_row[s >> 5] |= (1u << (s & 31));
        owner_regression_barrier<MAXD>();
    }
}
#endif

template <int MAXD>
__global__ void owner_copy_state_kernel(
    const int* __restrict__ owner_index, int M, int D, int MM,
    const float* __restrict__ Tan, const float* __restrict__ Cen,
    float* __restrict__ Tan_next, float* __restrict__ Cen_next)
{
    const int q = blockIdx.x;
    const int element = threadIdx.x;
    if (q >= M) return;
    const int i = owner_index[q];
    for (int offset = element; offset < D * MM; offset += blockDim.x)
        Tan_next[(size_t)i * D * MM + offset] = Tan[(size_t)i * D * MM + offset];
    for (int d = element; d < D; d += blockDim.x)
        Cen_next[(size_t)i * D + d] = Cen[(size_t)i * D + d];
}

__global__ void owner_count_sum_kernel(
    const int32_t* __restrict__ counts, int M, unsigned long long* __restrict__ sum,
    const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || q >= M) return;
    atomicAdd(sum, static_cast<unsigned long long>(counts[q]));
}

__global__ void owner_heavy_mask_kernel(
    const int* __restrict__ owner_index, const int32_t* __restrict__ counts, int M, int two_D,
    float heavy_factor, const unsigned long long* __restrict__ sum, uint8_t* __restrict__ heavy,
    int* __restrict__ heavy_count, const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || q >= M) return;
    const double mean_count = static_cast<double>(*sum) / static_cast<double>(M > 0 ? M : 1);
    const double threshold = fmax(static_cast<double>(two_D), static_cast<double>(heavy_factor) * mean_count);
    const uint8_t selected = static_cast<double>(counts[q]) > threshold ? 1u : 0u;
    if (selected) {
        heavy[owner_index[q]] = 1u;
        atomicAdd(heavy_count, 1);
    }
}

__global__ void owner_stop_if_empty_kernel(
    const int* __restrict__ heavy_count, int* __restrict__ stopped)
{
    if (*stopped) return;
    if (*heavy_count == 0) *stopped = 1;
}

__global__ void owner_record_valid_kernel(
    const int32_t* __restrict__ counts, int32_t* __restrict__ counts_last, int M,
    int* __restrict__ valid_iterations, const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped) return;
    if (q < M) counts_last[q] = counts[q];
    if (q == 0) atomicAdd(valid_iterations, 1);
}

__global__ void owner_copy_mask_kernel(
    const uint8_t* __restrict__ source, uint8_t* __restrict__ target, int N,
    const int* __restrict__ stopped)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || i >= N) return;
    target[i] = source[i];
}

__global__ void owner_keep_stats_kernel(
    const int* __restrict__ owner_index, const int32_t* __restrict__ counts, int M,
    const uint8_t* __restrict__ heavy, unsigned long long* __restrict__ positive_sum,
    int* __restrict__ positive_count, const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || q >= M) return;
    const int count = counts[q];
    if (heavy[owner_index[q]] && count > 0) {
        atomicAdd(positive_sum, static_cast<unsigned long long>(count));
        atomicAdd(positive_count, 1);
    }
}

__global__ void owner_keep_mask_kernel(
    const int* __restrict__ owner_index, const int32_t* __restrict__ counts, int M, int two_D,
    float heavy_factor_tmp, const uint8_t* __restrict__ heavy,
    const unsigned long long* __restrict__ positive_sum, const int* __restrict__ positive_count,
    uint8_t* __restrict__ keep, int* __restrict__ keep_count, const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || q >= M) return;
    const int denominator = *positive_count > 0 ? *positive_count : 1;
    const float mean_positive = static_cast<float>(
        static_cast<double>(*positive_sum) / static_cast<double>(denominator));
    const float threshold = fmaxf(static_cast<float>(two_D), heavy_factor_tmp * mean_positive);
    const int i = owner_index[q];
    const uint8_t selected = (heavy[i] && static_cast<float>(counts[q]) > threshold) ? 1u : 0u;
    keep[i] = selected;
    if (selected) atomicAdd(keep_count, 1);
}

__global__ void owner_snapshot_counts_kernel(
    const int* __restrict__ owner_index, const int32_t* __restrict__ counts,
    int M, int N, int iteration, int32_t* __restrict__ history,
    const int* __restrict__ stopped)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || q >= M) return;
    history[(size_t)iteration * N + owner_index[q]] = counts[q];
}

__global__ void owner_snapshot_mask_kernel(
    const uint8_t* __restrict__ source, int N, int iteration,
    uint8_t* __restrict__ history, const int* __restrict__ stopped)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || i >= N) return;
    history[(size_t)iteration * N + i] = source[i];
}

__global__ void owner_scatter_final_counts_kernel(
    const int* __restrict__ owner_index, const int32_t* __restrict__ counts,
    int M, int32_t* __restrict__ full_counts)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < M) full_counts[owner_index[q]] = counts[q];
}

__global__ void owner_store_eval_history_kernel(
    const float* __restrict__ sums, size_t slots, int n_eval_h, int K, int iteration,
    float* __restrict__ mse_history, float* __restrict__ cen_history,
    float* __restrict__ cen_tan_history, float* __restrict__ tan_history,
    const int* __restrict__ stopped)
{
    const size_t slot = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || slot >= slots) return;
    const size_t rho_index = slot / (size_t)n_eval_h;
    const size_t h_index = slot - rho_index * (size_t)n_eval_h;
    const size_t n_eval_rho = slots / (size_t)n_eval_h;
    const size_t destination = (rho_index + n_eval_rho * h_index) * K + iteration;
    mse_history[destination] = sums[slot * 4];
    cen_history[destination] = sums[slot * 4 + 1];
    cen_tan_history[destination] = sums[slot * 4 + 2];
    tan_history[destination] = sums[slot * 4 + 3];
}

__global__ void owner_store_eval_coordinate_history_kernel(
    const float* __restrict__ sums, size_t eval_slots, int MY,
    int n_eval_h, int K, int iteration,
    float* __restrict__ history, const int* __restrict__ stopped)
{
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t slots = eval_slots * (size_t)MY;
    if (*stopped || index >= slots) return;
    const size_t eval_slot = index / (size_t)MY;
    const int output = static_cast<int>(index - eval_slot * (size_t)MY);
    const size_t rho_index = eval_slot / (size_t)n_eval_h;
    const size_t h_index = eval_slot - rho_index * (size_t)n_eval_h;
    const size_t n_eval_rho = eval_slots / (size_t)n_eval_h;
    const size_t destination = (rho_index + n_eval_rho *
        (h_index + (size_t)n_eval_h * output)) * K + iteration;
    history[destination] = sums[index];
}

/* Validation stores the MSE slot alone. The three geometry slots the test
   store also writes are untouched by the MSE-only tail (it returns right
   after sums[slot*4]), so routing validation through the four-destination
   kernel would mean allocating three histories that only ever receive the
   cleared zeros. Same slot-to-destination arithmetic as the test store, so
   the two histories are indexed identically. */
__global__ void owner_store_valid_history_kernel(
    const float* __restrict__ sums, size_t slots, int n_eval_h, int K,
    int iteration, float* __restrict__ mse_history,
    float* __restrict__ tangent_history, const int* __restrict__ stopped)
{
    const size_t slot = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || slot >= slots) return;
    const size_t rho_index = slot / (size_t)n_eval_h;
    const size_t h_index = slot - rho_index * (size_t)n_eval_h;
    const size_t n_eval_rho = slots / (size_t)n_eval_h;
    const size_t destination = (rho_index + n_eval_rho * h_index) * K + iteration;
    mse_history[destination] = sums[slot * 4];
    if (tangent_history) tangent_history[destination] = sums[slot * 4 + 3];
}

__global__ void owner_mark_fused_eval_executed_kernel(
    int* __restrict__ executed, const int* __restrict__ stopped)
{
    if (!*stopped) atomicAdd(executed, 1);
}

__global__ void owner_fused_eval_train_stats_guarded_kernel(
    const float* __restrict__ Xtrain, const float* __restrict__ Tan,
    int D, int MM, int Ntrain, float* __restrict__ Xtrainnorm2,
    float* __restrict__ TdotX, const int* __restrict__ stopped)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (*stopped || j >= Ntrain) return;
    float x2 = 0.0f;
    for (int d = 0; d < D; ++d) {
        const float x = Xtrain[(size_t)j * D + d];
        x2 += x * x;
    }
    Xtrainnorm2[j] = x2;
    for (int mm = 0; mm < MM; ++mm) {
        float tx = 0.0f;
        for (int d = 0; d < D; ++d) {
            tx += Tan[(size_t)j * MM * D + (size_t)mm * D + d] *
                Xtrain[(size_t)j * D + d];
        }
        TdotX[(size_t)j * MM + mm] = tx;
    }
}

__global__ void owner_scale_eval_history_kernel(
    float* __restrict__ history, size_t slots, int K, int Ntest,
    const int* __restrict__ valid_iterations)
{
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= slots * K) return;
    const int iteration = static_cast<int>(index % K);
    if (iteration >= *valid_iterations) return;
    history[index] /= static_cast<float>(Ntest > 0 ? Ntest : 1);
}

/* Tangent bases have arbitrary gauge. Raw weighted basis-column averages
   are forbidden here; preserve the weighted-projector eigenspace. */



__device__ __forceinline__ float owner_warp_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return __shfl_sync(0xffffffffu, v, 0);
}

__device__ __forceinline__ float owner_warp_min(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffffu, v, offset));
    return __shfl_sync(0xffffffffu, v, 0);
}

struct OwnerTiledShared {
    double* slot_sumW; double* slot_pred; double* center_sumW;
    float* proj; float* csum; float* tansum;
    float* minq; float* sumq; float* cnt;
    float* center_minq; float* center_sumq; float* center_cnt;
    float* xt; float* ct;
    float* tX; float* tT; float* tY; float* tC;
    float* tN2; float* tTdX; float* tKeep; float* tE; float* tP;
    float* wstage; float* pgram;
};

__host__ __device__ __forceinline__ size_t owner_tiled_shared_bytes(
    int S, int D, int MM, int MY, int JT, int warps, bool response_eval,
    bool geometry_eval, bool center_eval)
{
    const size_t doubles = (size_t)S +
        (response_eval ? (size_t)S * MY : 0) +
        (center_eval ? (size_t)S : 0);
    const size_t floats =
        (geometry_eval ? (size_t)S * D * D : 0) +
        (center_eval ? (size_t)S * D : 0) +
        (geometry_eval ? (size_t)S * MM * D : 0) +
        3 * (size_t)S + (center_eval ? 3 * (size_t)S : 0) +
        (size_t)D + (center_eval ? (size_t)D : 0) +
        (size_t)JT * ((size_t)D + (size_t)MM * D +
            (response_eval ? MY : 0) +
            (center_eval ? D : 0) + 1 + MM + 1 + 2) +
        (size_t)warps * JT +
        (geometry_eval ? (size_t)JT * ((size_t)D * (D + 1) / 2) : 0);
    return doubles * sizeof(double) + floats * sizeof(float);
}

__device__ __forceinline__ OwnerTiledShared owner_tiled_shared_map(
    char* base, int S, int D, int MM, int MY, int JT, int warps,
    bool response_eval, bool geometry_eval, bool center_eval)
{
    OwnerTiledShared m;
    double* dp = reinterpret_cast<double*>(base);
    m.slot_sumW = dp; dp += S;
    m.slot_pred = dp; if (response_eval) dp += (size_t)S * MY;
    m.center_sumW = dp; if (center_eval) dp += S;
    float* fp = reinterpret_cast<float*>(dp);
    m.proj = fp; if (geometry_eval) fp += (size_t)S * D * D;
    m.csum = fp; if (center_eval) fp += (size_t)S * D;
    m.tansum = fp; if (geometry_eval) fp += (size_t)S * MM * D;
    m.minq = fp; fp += S;
    m.sumq = fp; fp += S;
    m.cnt = fp; fp += S;
    m.center_minq = fp; if (center_eval) fp += S;
    m.center_sumq = fp; if (center_eval) fp += S;
    m.center_cnt = fp; if (center_eval) fp += S;
    m.xt = fp; fp += D;
    m.ct = fp; if (center_eval) fp += D;
    m.tX = fp; fp += (size_t)JT * D;
    m.tT = fp; fp += (size_t)JT * MM * D;
    m.tY = fp; if (response_eval) fp += (size_t)JT * MY;
    m.tC = fp; if (center_eval) fp += (size_t)JT * D;
    m.tN2 = fp; fp += JT;
    m.tTdX = fp; fp += (size_t)JT * MM;
    m.tKeep = fp; fp += JT;
    m.tE = fp; fp += JT;
    m.tP = fp; fp += JT;
    m.wstage = fp; fp += (size_t)warps * JT;
    m.pgram = fp;
    return m;
}

#if MANOR_M_DEVTAN_ORTHONORMAL_FP64
#include "manor_m_devtan_orthonormal.cuh"
#endif

/* The old guarded tail verbatim, warp-local; dims strided over lanes. */
__device__ void owner_fused_eval_tiled_tail(
    const OwnerTiledShared& m, int s_local, int ib, int lane,
    const float* __restrict__ Ytrue, const float* __restrict__ Ttrue,
    int D, int MM, int MY, int eval_slot_out, float eps_weight,
    bool response_eval, bool tangent_metric_eval, bool center_eval,
    bool center_tangent_metric_eval, float* __restrict__ sums,
    float* __restrict__ coordinate_sums
    , float* __restrict__ point_metrics, int Nquery
    )
{
    const double sumW = fmax(m.slot_sumW[s_local], (double)eps_weight);
    const double inv_sumW = 1.0 / sumW;
    float err2 = 0.0f;
    if (response_eval && lane == 0) {
        double e2 = 0.0;
        for (int my = 0; my < MY; ++my) {
            const double yhat = m.slot_pred[(size_t)s_local * MY + my] * inv_sumW;
            const double err = yhat - (double)Ytrue[(size_t)ib * MY + my];
            e2 += err * err;
            if (coordinate_sums != nullptr) {
                point_metrics[manor_eval_point_offset(eval_slot_out, 4 + my,
                    MY, Nquery, ib)] = (float)(err * err);
            }
        }
        err2 = (float)e2;
    }
    const bool geometry_eval = tangent_metric_eval || center_eval;
    if (!geometry_eval) {
        if (response_eval && lane == 0)
            point_metrics[manor_eval_point_offset(eval_slot_out, 0,
                MY, Nquery, ib)] = err2;
        return;
    }

    const float inv_sumW_f = (float)inv_sumW;
    float* proj = m.proj + (size_t)s_local * D * D;
    float* tanb = m.tansum + (size_t)s_local * MM * D;
    float* dc = center_eval ? m.csum + (size_t)s_local * D : nullptr;
    /* The tile loop filled the upper triangle only. Mirror before anything
       reads proj: this is the first read, and the power iteration below is
       the only other one. Sources are strictly upper, destinations strictly
       lower, so the sweep cannot race with itself; the warp owns this slot
       in both loops, so __syncwarp is the right fence. */
    for (int e = lane; e < D * D; e += 32) {
        const int r = e / D;
        const int c = e - r * D;
        if (r > c) proj[e] = proj[(size_t)c * D + r];
    }
    __syncwarp();
    for (int e = lane; e < D * D; e += 32) proj[e] *= inv_sumW_f;
    for (int e = lane; e < MM * D; e += 32) tanb[e] *= inv_sumW_f;
    if (center_eval) {
        const float center_inv_sumW = 1.0f /
            fmaxf((float)m.center_sumW[s_local], eps_weight);

        for (int d = lane; d < D; d += 32)
            dc[d] = dc[d] * center_inv_sumW - m.xt[d];
    }
    __syncwarp();

    /* Deflated power iteration for the top-MM eigenspace of the weighted
       projector mean. */
#ifndef MANOR_TANGENT_POWER_STEPS
#define MANOR_TANGENT_POWER_STEPS 2000
#endif
    constexpr int tangent_power_steps = MANOR_TANGENT_POWER_STEPS;
    for (int mm_hat = 0; mm_hat < MM; ++mm_hat) {
        float* q = tanb + (size_t)mm_hat * D;
        for (int prev = 0; prev < mm_hat; ++prev) {
            const float* pv = tanb + (size_t)prev * D;
            float part = 0.0f;
            for (int d = lane; d < D; d += 32) part += q[d] * pv[d];
            const float dot_prev = owner_warp_sum(part);
            for (int d = lane; d < D; d += 32) q[d] -= dot_prev * pv[d];
            __syncwarp();
        }
        float part0 = 0.0f;
        for (int d = lane; d < D; d += 32) part0 += q[d] * q[d];
        float n2 = owner_warp_sum(part0);
        if (n2 <= 1.0e-20f) {
            for (int d = lane; d < D; d += 32) q[d] = (d == mm_hat) ? 1.0f : 0.0f;
            __syncwarp();
            for (int prev = 0; prev < mm_hat; ++prev) {
                const float* pv = tanb + (size_t)prev * D;
                float p2 = 0.0f;
                for (int d = lane; d < D; d += 32) p2 += q[d] * pv[d];
                const float dot_prev = owner_warp_sum(p2);
                for (int d = lane; d < D; d += 32) q[d] -= dot_prev * pv[d];
                __syncwarp();
            }
            float pn = 0.0f;
            for (int d = lane; d < D; d += 32) pn += q[d] * q[d];
            n2 = owner_warp_sum(pn);
        }
        const float scale0 = rsqrtf(fmaxf(n2, 1.0e-30f));
        for (int d = lane; d < D; d += 32) q[d] *= scale0;
        __syncwarp();

        for (int it = 0; it < tangent_power_steps; ++it) {
            float ybuf[2] = {0.0f, 0.0f};
            for (int d = lane, r = 0; d < D; d += 32, ++r) {
                float acc = 0.0f;
                for (int c = 0; c < D; ++c)
                    acc += proj[(size_t)d * D + c] * q[c];
                ybuf[r] = acc;
            }
            __syncwarp();
            for (int d = lane, r = 0; d < D; d += 32, ++r) q[d] = ybuf[r];
            __syncwarp();
            for (int prev = 0; prev < mm_hat; ++prev) {
                const float* pv = tanb + (size_t)prev * D;
                float p3 = 0.0f;
                for (int d = lane; d < D; d += 32) p3 += q[d] * pv[d];
                const float dot_prev = owner_warp_sum(p3);
                for (int d = lane; d < D; d += 32) q[d] -= dot_prev * pv[d];
                __syncwarp();
            }
            float pn2s = 0.0f;
            for (int d = lane; d < D; d += 32) pn2s += q[d] * q[d];
            const float yn2 = owner_warp_sum(pn2s);
            if (yn2 > 1.0e-20f) {
                const float sc = rsqrtf(yn2);
                for (int d = lane; d < D; d += 32) q[d] *= sc;
            }
            __syncwarp();
        }
    }

#if MANOR_M_DEVTAN_ORTHONORMAL_FP64
    const float stable_devtan = tangent_metric_eval ?
        manor_devtan_orthonormal_metric(Ttrue + (size_t)ib * MM * D, tanb, D, MM) : 0.0f;
#else
    float frob2 = 0.0f;
    if (tangent_metric_eval) {
        for (int mm_true = 0; mm_true < MM; ++mm_true) {
            const float* tt = Ttrue + (size_t)ib * MM * D + (size_t)mm_true * D;
            for (int mm_hat = 0; mm_hat < MM; ++mm_hat) {
                const float* q = tanb + (size_t)mm_hat * D;
                float part = 0.0f;
                for (int d = lane; d < D; d += 32) part += tt[d] * q[d];
                const float dot = owner_warp_sum(part);
                frob2 += dot * dot;
            }
        }
    }
#endif
    float cen2 = 0.0f;
    float cen_tan2 = 0.0f;
    if (center_eval) {
        /* C_hat = X_query + (I - T_hat T_hat') (C_wide - X_query). */
        for (int mm = 0; mm < MM; ++mm) {
            const float* q = tanb + (size_t)mm * D;
            float p = 0.0f;
            for (int d = lane; d < D; d += 32) p += dc[d] * q[d];
            const float dot = owner_warp_sum(p);
            for (int d = lane; d < D; d += 32) dc[d] -= dot * q[d];
            __syncwarp();
        }
        for (int d = lane; d < D; d += 32)
            dc[d] += m.xt[d] - m.ct[d];
        __syncwarp();
        float cpart = 0.0f;
        for (int d = lane; d < D; d += 32) cpart += dc[d] * dc[d];
        cen2 = owner_warp_sum(cpart);
        if (center_tangent_metric_eval) {
            for (int mm = 0; mm < MM; ++mm) {
                const float* tt = Ttrue + (size_t)ib * MM * D + (size_t)mm * D;
                float p4 = 0.0f;
                for (int d = lane; d < D; d += 32) p4 += dc[d] * tt[d];
                const float dot = owner_warp_sum(p4);
                cen_tan2 += dot * dot;
            }
        }
    }
    if (lane == 0) {
        if (response_eval)
            point_metrics[manor_eval_point_offset(eval_slot_out, 0,
                MY, Nquery, ib)] = err2;
        if (center_eval) {
            point_metrics[manor_eval_point_offset(eval_slot_out, 1,
                MY, Nquery, ib)] = sqrtf(cen2);
            if (center_tangent_metric_eval)
                point_metrics[manor_eval_point_offset(eval_slot_out, 2,
                    MY, Nquery, ib)] = sqrtf(fmaxf(0.0f, cen_tan2));
        }
        if (tangent_metric_eval)
            point_metrics[manor_eval_point_offset(eval_slot_out, 3,
                MY, Nquery, ib)] =
#if MANOR_M_DEVTAN_ORTHONORMAL_FP64
                stable_devtan;
#else
                sqrtf(fmaxf(0.0f, (float)MM - frob2));
#endif
    }
}

#if MANOR_M_EVAL_MIN_BLOCKS_PER_SM > 0
__global__ __launch_bounds__(256, MANOR_M_EVAL_MIN_BLOCKS_PER_SM)
#else
__global__
#endif
void owner_fused_eval_tiled_kernel(
    const float* __restrict__ Xtest, const float* __restrict__ Ytrue,
    const float* __restrict__ Ttrue, const float* __restrict__ Ctrue,
    const float* __restrict__ Xtrain, const float* __restrict__ Ytrain,
    const float* __restrict__ Cen, const float* __restrict__ Tan,
    const float* __restrict__ Xtrainnorm2, const float* __restrict__ TdotX,
    const uint8_t* __restrict__ keep_mask,
    int D, int MM, int MY, int Ntrain, int Ntest,
    float h_base, float rhok_base,
    const float* __restrict__ eval_h_scale,
    const float* __restrict__ eval_rho_scale,
    int n_eval_h, int n_slots, int JT,
    float sum_sparse_threshold, int count_sparse_threshold,
    float fallback_exp_scale, float eps_weight,
    float center_eval_h_scale, float center_eval_rho_scale,
    bool response_eval, bool tangent_metric_eval, bool center_eval,
    bool center_tangent_metric_eval,
    float* __restrict__ sums, float* __restrict__ coordinate_sums,
    const int* __restrict__ pair_ib, const int* __restrict__ pair_offsets,
    const int* __restrict__ pair_slots, const float* __restrict__ pair_minq,
    int* __restrict__ out_pair_count, int* __restrict__ out_pairs,
    float* __restrict__ out_pair_minq,
    const int* __restrict__ stopped
    , float* __restrict__ point_metrics
    )
{
    if (*stopped) return;
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bool pair_mode = pair_ib != nullptr;
    int ib, slot_lo, slot_hi;
    if (pair_mode) {
        ib = pair_ib[blockIdx.x];
        slot_lo = pair_offsets[blockIdx.x];
        slot_hi = pair_offsets[blockIdx.x + 1];
    } else {
        ib = blockIdx.x;
        slot_lo = 0;
        slot_hi = n_slots;
    }
    if (ib >= Ntest) return;
    /* Shared accumulators are laid out for n_slots in both modes; a pair
       group holds at most n_slots slots of one test point. */
    const int S = n_slots;

    extern __shared__ char owner_tiled_smem[];
    /* One float per warp, staging the block-wide min of the tile's tE for the
       pair-mode tile skip below. Static so owner_tiled_shared_bytes and hence
       the dynamic footprint and occupancy are untouched; 32 covers the largest
       warp count a 1024-thread block can have, i.e. 128 bytes. */
    __shared__ float owner_tile_min_warp[32];
    OwnerTiledShared m = owner_tiled_shared_map(
        owner_tiled_smem, S, D, MM, MY, JT, warps, response_eval,
        tangent_metric_eval || center_eval, center_eval);

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        m.slot_sumW[i] = 0.0;
        m.minq[i] = 1.0e30f;
        m.sumq[i] = 0.0f;
        m.cnt[i] = 0.0f;
        if (center_eval) {
            m.center_sumW[i] = 0.0;
            m.center_minq[i] = 1.0e30f;
            m.center_sumq[i] = 0.0f;
            m.center_cnt[i] = 0.0f;
        }
    }
    if (response_eval) {
        for (size_t i = threadIdx.x; i < (size_t)S * MY; i += blockDim.x)
            m.slot_pred[i] = 0.0;
    }
    if (tangent_metric_eval || center_eval) {
        for (size_t i = threadIdx.x; i < (size_t)S * D * D; i += blockDim.x)
            m.proj[i] = 0.0f;
        if (center_eval) {
            for (size_t i = threadIdx.x; i < (size_t)S * D; i += blockDim.x)
                m.csum[i] = 0.0f;
        }
        for (size_t i = threadIdx.x; i < (size_t)S * MM * D; i += blockDim.x)
            m.tansum[i] = 0.0f;
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        m.xt[d] = Xtest[(size_t)ib * D + d];
        m.ct[d] = center_eval ? Ctrue[(size_t)ib * D + d] : 0.0f;
    }
    __syncthreads();

    float xt2 = 0.0f;
    for (int d = 0; d < D; ++d) xt2 += m.xt[d] * m.xt[d];

    for (int j0 = 0; j0 < Ntrain; j0 += JT) {
        const int jt_count = min(JT, Ntrain - j0);
        for (size_t i = threadIdx.x; i < (size_t)jt_count * D; i += blockDim.x)
            m.tX[i] = Xtrain[(size_t)j0 * D + i];
        for (size_t i = threadIdx.x; i < (size_t)jt_count * MM * D; i += blockDim.x)
            m.tT[i] = Tan[(size_t)j0 * MM * D + i];
        if (response_eval) {
            for (size_t i = threadIdx.x; i < (size_t)jt_count * MY; i += blockDim.x)
                m.tY[i] = Ytrain[(size_t)j0 * MY + i];
        }
        if (center_eval) {
            for (size_t i = threadIdx.x; i < (size_t)jt_count * D; i += blockDim.x)
                m.tC[i] = Cen[(size_t)j0 * D + i];
        }
        for (size_t i = threadIdx.x; i < (size_t)jt_count * MM; i += blockDim.x)
            m.tTdX[i] = TdotX[(size_t)j0 * MM + i];
        for (int i = threadIdx.x; i < jt_count; i += blockDim.x) {
            m.tN2[i] = Xtrainnorm2[j0 + i];
            m.tKeep[i] = keep_mask[j0 + i] ? 1.0f : 0.0f;
        }
        __syncthreads();

        float thread_min_tE = 1.0e30f;
        for (int jt = threadIdx.x; jt < jt_count; jt += blockDim.x) {
            float xtxj = 0.0f;
            for (int d = 0; d < D; ++d)
                xtxj += m.xt[d] * m.tX[(size_t)jt * D + d];
            float pn2 = 0.0f;
            for (int mm = 0; mm < MM; ++mm) {
                float xtv = 0.0f;
                for (int d = 0; d < D; ++d)
                    xtv += m.xt[d] *
                        m.tT[(size_t)jt * MM * D + (size_t)mm * D + d];
                const float p = xtv - m.tTdX[(size_t)jt * MM + mm];
                pn2 += p * p;
            }
            const float tE_jt = fmaxf(xt2 + m.tN2[jt] - 2.0f * xtxj, 0.0f);
            m.tE[jt] = tE_jt;
            m.tP[jt] = pn2;
            thread_min_tE = fminf(thread_min_tE, tE_jt);
        }
        /* The tangent Gram p(jt,r,c) = sum_mm T[jt][mm][r] * T[jt][mm][c] does
           not depend on the slot, so it is built once per tile here instead of
           being recomputed inside each of the S slot passes below. It is the
           same expression in the same mm order that the projector loop used to
           evaluate inline, so every accumulation downstream is bitwise what it
           was. Only the packed upper triangle is stored, matching the triangle
           the projector accumulates. The [jt][t] layout is what makes both
           sides bank-conflict-free: the consumer strides over t with jt fixed
           and so reads consecutive words, while the producer strides over t
           within a lane and writes consecutive words. The (r,c) decode sits in
           the t loop rather than the jt loop for the same reason -- one warp
           owns a whole jt, so the decode is paid ntri/32 times per jt instead
           of once per element. The existing __syncthreads() below publishes
           the cache before any slot reads it. */
        if (tangent_metric_eval || center_eval) {
            const int ntri = D * (D + 1) / 2;
            for (int jt = warp; jt < jt_count; jt += warps) {
                const float* tj = m.tT + (size_t)jt * MM * D;
                float* pj = m.pgram + (size_t)jt * ntri;
                for (int t = lane; t < ntri; t += 32) {
                    int r = 0, base = 0;
                    while (base + (D - r) <= t) { base += (D - r); ++r; }
                    const int c = r + (t - base);
                    float p = 0.0f;
                    for (int mm = 0; mm < MM; ++mm)
                        p += tj[(size_t)mm * D + r] * tj[(size_t)mm * D + c];
                    pj[t] = p;
                }
            }
        }
        /* Slot-independent, so it is reduced once per tile rather than once
           per slot-tile. The 1.0e30f identity only ever makes the result
           smaller than the true min, which keeps it a valid lower bound. */
        if (pair_mode) {
            const float warp_min_tE = owner_warp_min(thread_min_tE);
            if (lane == 0) owner_tile_min_warp[warp] = warp_min_tE;
        }
        __syncthreads();
        float tile_min_tE = 1.0e30f;
        if (pair_mode) {
            for (int w = 0; w < warps; ++w)
                tile_min_tE = fminf(tile_min_tE, owner_tile_min_warp[w]);
        }

        for (int si = slot_lo + warp; si < slot_hi; si += warps) {
            const int s_local = si - slot_lo;
            const int s = pair_mode ? pair_slots[si] : si;
            const int ir = s / n_eval_h;
            const int ih = s - ir * n_eval_h;
            const float h = h_base * eval_h_scale[ih];
            const float rhok = rhok_base * eval_rho_scale[ir];
            const float rho_eval = fmaxf(0.0f,
                1.0f / fmaxf(rhok * rhok, 1.0e-30f) - 1.0f);
            const float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);
            const float center_h = h_base * center_eval_h_scale;
            const float center_rhok = rhok_base * center_eval_rho_scale;
            const float center_rho_eval = fmaxf(0.0f,
                1.0f / fmaxf(center_rhok * center_rhok, 1.0e-30f) - 1.0f);
            const float center_inv_h2 = 1.0f /
                fmaxf(center_h * center_h, 1.0e-30f);

            /* The pair-mode counterpart of the tile_w > 0.0f guard below.
               That guard never fires here because expf() is not compactly
               supported, so the whole dense block runs for every tile.
               tE >= 0, tP >= 0, rho_eval >= 0 and inv_h2 > 0 give
               q >= tE*inv_h2 >= tile_min_tE*inv_h2 for every point of this
               tile, and fallback_exp_scale >= 0 preserves that ordering, so
               the value below is an upper bound on the exponent argument of
               every weight in the tile. The directed roundings keep the
               bound valid whatever FMA contraction the compiler picks for
               either expression. An exhaustive scan of all 2^31 negative
               floats under this build's --use_fast_math (ftz=true, so expf
               becomes __expf) puts the expf-underflows-to-zero crossover at
               -87.3365479, hence at -104 every weight is exactly +0.0f:
               wst is written and read only inside this iteration, and
               slot_sumW starts at +0.0 and only ever grows, so the skipped
               += (double)+0.0f leaves it bitwise unchanged. */
            if (pair_mode) {
                const float tile_q_lb = __fmul_rd(tile_min_tE, inv_h2);
                const float tile_exp_ub = __fmul_ru(fallback_exp_scale,
                    __fsub_ru(pair_minq[2 * (size_t)si], tile_q_lb));
                float center_tile_exp_ub = -1.0e30f;
                if (center_eval) {
                    const float center_tile_q_lb =
                        __fmul_rd(tile_min_tE, center_inv_h2);
                    center_tile_exp_ub = __fmul_ru(fallback_exp_scale,
                        __fsub_ru(pair_minq[2 * (size_t)si + 1],
                            center_tile_q_lb));
                }
                if (tile_exp_ub <= -104.0f &&
                        (!center_eval || center_tile_exp_ub <= -104.0f)) continue;
            }

            float* wst = m.wstage + (size_t)warp * JT;

            float w_sum = 0.0f, q_sum = 0.0f, q_min = 1.0e30f, n_pos = 0.0f;
            for (int jt = lane; jt < jt_count; jt += 32) {
                const float q = (m.tE[jt] + rho_eval * m.tP[jt]) * inv_h2;
                const float keep = m.tKeep[jt];
                float w;
                if (pair_mode) {
                    w = keep * expf(fallback_exp_scale *
                        (pair_minq[2 * (size_t)si] - q));
                } else {
                    const float rt = 1.0f - q;
                    w = (rt > 0.0f) ? keep * rt * rt : 0.0f;
                    q_min = fminf(q_min, keep > 0.0f ? q : 1.0e30f);
                    q_sum += w;
                    n_pos += (keep > 0.0f && rt > 0.0f) ? 1.0f : 0.0f;
                }
                wst[jt] = w;
                w_sum += w;
            }
            for (int jt = jt_count + lane; jt < JT; jt += 32) wst[jt] = 0.0f;
            const float tile_w = owner_warp_sum(w_sum);
            if (!pair_mode) {
                const float tile_q = owner_warp_sum(q_sum);
                const float tile_min = owner_warp_min(q_min);
                const float tile_n = owner_warp_sum(n_pos);
                if (lane == 0) {
                    m.sumq[s_local] += tile_q;
                    m.cnt[s_local] += tile_n;
                    m.minq[s_local] = fminf(m.minq[s_local], tile_min);
                }
            }
            if (lane == 0) m.slot_sumW[s_local] += (double)tile_w;
            __syncwarp();

            /* k << Ntrain: most tiles hold only zero weights for this slot.
               Accumulators never hold -0.0, so skipping the +0.0 adds is
               bitwise identical and removes the dense per-slot work for
               empty tiles (item 23b). tile_w is warp-uniform. */
            if (tile_w > 0.0f) {
                /* One lane sweep over the concatenation of all four
                   accumulations instead of one sweep each. MY=6, MM*D=20 and
                   The component loops cover the active dimensions across the warp, so the four loops
                   cost 1+1+1+2 = 5 sweeps to cover 91 elements where
                   ceil(91/32) = 3 suffice. The SASS emits two LDS per FFMA --
                   the warp-uniform wst[jt] is reloaded in every loop, not
                   amortised -- so the wst side falls from 5 to 3 shared
                   transactions per jt, and the FFMA and loop overhead with it.

                   The source side does not fall. Lanes of a fused sweep read
                   different arrays whose per-jt strides (55, 6, 10, 20) are
                   incongruent mod 32, so the group base banks drift apart as
                   jt advances and collide. Counted exactly over the layout
                   owner_tiled_shared_map produces at JT=32, the order below
                   (pgram, tY, tC, tT) costs 5.000 source transactions per jt:
                   the same 5 the four separate sweeps cost, and the best of
                   all 24 orders -- the naive (tY, tT, tC, pgram) costs 5.219.
                   Padding the bases cannot improve on it, because jt sweeps a
                   whole period of every pairwise drift and a constant shift
                   only permutes which jt pays. Only congruent strides remove
                   the conflicts, and making all four congruent costs 4224 B.
                   cudaOccupancyMaxActiveBlocksPerMultiprocessor puts the
                   largest dynamic allocation that still holds three 256-thread
                   blocks on this device at 33024 B, so this block has 3524 B
                   to spend and that door is shut at JT=32.

                   Each lane resolves source, stride, offset and destination
                   once, outside the jt loop. Every element still accumulates
                   the same jt sequence of the same values in the same order,
                   so each per-element acc, and hence every shared accumulator
                   this block writes, is bitwise what the four loops produced.
                   The mandatory query-indexed fixed-order reducer also keeps
                   the reported metrics reproducible across launches; no
                   unordered floating-point metric accumulation remains.

                   proj is a sum of outer products, and (r,c) and (c,r) run
                   the identical mm and jt accumulation orders over commuted
                   products, so the two are equal bitwise, not merely
                   mathematically. Only the upper triangle is accumulated
                   here; the tail mirrors it into the full D*D before the
                   first read. */
                const bool geometry_eval = tangent_metric_eval || center_eval;
                const int ntri = D * (D + 1) / 2;
                const int u_pgram = geometry_eval ? ntri : 0;
                const int u_ty = u_pgram + (response_eval ? MY : 0);
                const int u_end = u_ty + (geometry_eval ? MM * D : 0);
                for (int u = lane; u < u_end; u += 32) {
                    const float* src;
                    int stride, off;
                    float* dst = nullptr;
                    double* dst_pred = nullptr;
                    if (u < u_pgram) {
                        int r = 0, base = 0;
                        while (base + (D - r) <= u) { base += (D - r); ++r; }
                        const int c = r + (u - base);
                        src = m.pgram; stride = ntri; off = u;
                        dst = m.proj + (size_t)s_local * D * D +
                            (size_t)r * D + c;
                    } else if (u < u_ty) {
                        off = u - u_pgram;
                        src = m.tY; stride = MY;
                        dst_pred = m.slot_pred + (size_t)s_local * MY + off;
                    } else {
                        off = u - u_ty;
                        src = m.tT; stride = MM * D;
                        dst = m.tansum + (size_t)s_local * MM * D + off;
                    }
                    float acc = 0.0f;
                    for (int jt = 0; jt < jt_count; ++jt)
                        acc += wst[jt] * src[(size_t)jt * stride + off];
                    if (dst_pred != nullptr) *dst_pred += (double)acc;
                    else *dst += acc;
                }
            }
            __syncwarp();

            /* Center has a separately validated wide readout scale.  It is
               accumulated after the ordinary slot because wstage is shared;
               the result is intentionally duplicated per slot since each
               slot projects the same wide displacement through its own
               predicted tangent basis. */
            if (center_eval) {
                float center_w_sum = 0.0f, center_q_sum = 0.0f;
                float center_q_min = 1.0e30f, center_n_pos = 0.0f;
                for (int jt = lane; jt < jt_count; jt += 32) {
                    const float q = (m.tE[jt] + center_rho_eval * m.tP[jt]) *
                        center_inv_h2;
                    const float keep = m.tKeep[jt];
                    float w;
                    if (pair_mode) {
                        w = keep * expf(fallback_exp_scale *
                            (pair_minq[2 * (size_t)si + 1] - q));
                    } else {
                        const float rt = 1.0f - q;
                        w = (rt > 0.0f) ? keep * rt * rt : 0.0f;
                        center_q_min = fminf(center_q_min,
                            keep > 0.0f ? q : 1.0e30f);
                        center_q_sum += w;
                        center_n_pos += (keep > 0.0f && rt > 0.0f) ? 1.0f : 0.0f;
                    }
                    wst[jt] = w;
                    center_w_sum += w;
                }
                for (int jt = jt_count + lane; jt < JT; jt += 32)
                    wst[jt] = 0.0f;
                const float center_tile_w = owner_warp_sum(center_w_sum);
                if (!pair_mode) {
                    const float tile_q = owner_warp_sum(center_q_sum);
                    const float tile_min = owner_warp_min(center_q_min);
                    const float tile_n = owner_warp_sum(center_n_pos);
                    if (lane == 0) {
                        m.center_sumq[s_local] += tile_q;
                        m.center_cnt[s_local] += tile_n;
                        m.center_minq[s_local] =
                            fminf(m.center_minq[s_local], tile_min);
                    }
                }
                if (lane == 0)
                    m.center_sumW[s_local] += (double)center_tile_w;
                __syncwarp();
                if (center_tile_w > 0.0f) {
                    for (int d = lane; d < D; d += 32) {
                        float acc = 0.0f;
                        for (int jt = 0; jt < jt_count; ++jt)
                            acc += wst[jt] * m.tC[(size_t)jt * D + d];
                        m.csum[(size_t)s_local * D + d] += acc;
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    for (int si = slot_lo + warp; si < slot_hi; si += warps) {
        const int s_local = si - slot_lo;
        const int s = pair_mode ? pair_slots[si] : si;
        if (!pair_mode) {
            const bool ordinary_sparse =
                (m.sumq[s_local] < sum_sparse_threshold) ||
                (m.cnt[s_local] < (float)count_sparse_threshold);
            const bool center_sparse = center_eval &&
                ((m.center_sumq[s_local] < sum_sparse_threshold) ||
                 (m.center_cnt[s_local] < (float)count_sparse_threshold));
            const bool sparse = ordinary_sparse || center_sparse;
            if (sparse) {
                if (lane == 0 && out_pair_count != nullptr) {
                    const int idx = atomicAdd(out_pair_count, 1);
                    out_pairs[2 * (size_t)idx] = ib;
                    out_pairs[2 * (size_t)idx + 1] = s;
                    out_pair_minq[2 * (size_t)idx] = m.minq[s_local];
                    out_pair_minq[2 * (size_t)idx + 1] = center_eval ?
                        m.center_minq[s_local] : 0.0f;
                }
                continue;
            }
        }
        owner_fused_eval_tiled_tail(m, s_local, ib, lane, Ytrue, Ttrue,
            D, MM, MY, s, eps_weight, response_eval, tangent_metric_eval,
            center_eval, center_tangent_metric_eval, sums, coordinate_sums
            , point_metrics, Ntest
            );
    }
}

template <int MAXD>
__global__ void owner_aggregation_accumulate_kernel(
    const float* __restrict__ grad, const float* __restrict__ center_pre,
    const int* __restrict__ owner_index, const int* __restrict__ owner_lookup,
    const uint8_t* __restrict__ heavy, const uint32_t* __restrict__ neigh_mask_bits,
    const int* __restrict__ idx_gpu, const int* __restrict__ index_rows,
    int M, int D, int MY, int kSlice,
    int mask_words, bool idx_shared_slice, int* __restrict__ agg_count,
    float* __restrict__ agg_cacc,
    float* __restrict__ agg_cov, const int* __restrict__ stopped)
{
    const int q = blockIdx.x;
    const int row = threadIdx.x;
    if (*stopped || q >= M) return;
    const int index_anchor = index_rows ? index_rows[q] : q;
    int count = 0;
    float cacc = 0.0f;
    float covariance[MAXD];
    for (int c = 0; c < MAXD; ++c) covariance[c] = 0.0f;
    const uint32_t* mask = neigh_mask_bits + (size_t)q * mask_words;
    for (int word_index = 0; word_index < mask_words; ++word_index) {
        uint32_t word = mask[word_index];
        while (word) {
            const int bit = __ffs(static_cast<int>(word)) - 1;
            const int s = (word_index << 5) + bit;
            word &= word - 1u;
            if (s >= kSlice) continue;
            const int j = owner_idx_at(
                idx_gpu, index_anchor, s, kSlice, idx_shared_slice);
            if (!heavy[j]) continue;
            const int source_q = owner_lookup[j];
            if (source_q < 0) continue;
            if (row == 0) ++count;
            if (row < D) cacc += center_pre[(size_t)source_q * D + row];
            /* item 23c: read the gradient row straight from L2 instead of
               staging it through shared memory; this removes both block
               barriers per (neighbor, output). Same values in the same
               per-thread order, so the accumulation is bitwise identical. */
            if (row < D) {
                for (int out = 0; out < MY; ++out) {
                    const float* gradient_row =
                        grad + (size_t)source_q * MY * D + (size_t)out * D;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_WARP_GATHER
                    owner_accumulate_gradient_row<MAXD, true>(
                        covariance, gradient_row, row, D);
#else
                    const float value = gradient_row[row];
                    for (int c = 0; c < D; ++c)
                        covariance[c] += value * gradient_row[c];
#endif
                }
            }
        }
    }
    if (row < D) {
        agg_cacc[(size_t)q * D + row] = cacc;
        for (int c = 0; c < D; ++c)
            agg_cov[((size_t)q * D + row) * D + c] = covariance[c];
    }
    if (row == 0) agg_count[q] = count;
}

template <int MAXD>
__global__ void owner_aggregation_finalize_kernel(
    const float* __restrict__ X, const float* __restrict__ Tan, const int* __restrict__ owner_index,
    int M, int D, int MM, const int* __restrict__ agg_count, const float* __restrict__ agg_cacc,
    const float* __restrict__ agg_cov, float* __restrict__ Tan_next, float* __restrict__ Cen_next,
    const int* __restrict__ stopped)
{
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    if (*stopped || q >= M || agg_count[q] <= 0) return;
    const int i = owner_index[q];
    __shared__ float sh_A[MAXD * MAXD];
    __shared__ float sh_V[MAXD * MAXD];
    __shared__ float sh_old[MAXD * MAX_MM_MANOR];
    __shared__ float sh_new[MAXD * MAX_MM_MANOR];
    __shared__ float sh_center[MAXD];
    for (int index = tid; index < D * D; index += 32) {
        const int a = index / D, c = index - a * D;
        sh_A[a * D + c] = agg_cov[((size_t)q * D + a) * D + c];
    }
    for (int index = tid; index < D * MM; index += 32) {
        const int mm = index / D, d = index - mm * D;
        sh_old[d + mm * D] = Tan[(size_t)i * MM * D + (size_t)mm * D + d];
    }
    __syncwarp();
    shared_jacobi_top_mm_warp<MAXD>(sh_A, sh_V, D, MM, sh_old, sh_new);
    __syncwarp();
    const float inverse_count = 1.0f / static_cast<float>(agg_count[q]);
    for (int d = tid; d < D; d += 32) sh_center[d] = agg_cacc[(size_t)q * D + d] * inverse_count;
    __syncwarp();
    for (int mm = 0; mm < MM; ++mm) {
        float local_dot = 0.0f;
        for (int d = tid; d < D; d += 32)
            local_dot += sh_new[d + mm * D] * (sh_center[d] - X[(size_t)i * D + d]);
        for (int offset = 16; offset > 0; offset >>= 1)
            local_dot += __shfl_xor_sync(0xffffffffu, local_dot, offset);
        const float dot = __shfl_sync(0xffffffffu, local_dot, 0);
        for (int d = tid; d < D; d += 32) sh_center[d] -= dot * sh_new[d + mm * D];
        __syncwarp();
    }
    for (int index = tid; index < D * MM; index += 32) {
        const int mm = index / D, d = index - mm * D;
        Tan_next[(size_t)i * MM * D + (size_t)mm * D + d] = sh_new[d + mm * D];
    }
    for (int d = tid; d < D; d += 32) Cen_next[(size_t)i * D + d] = sh_center[d];
}

static int owner_validate_fused_inputs(
    bool fused_eval, int nrhs, const mxArray* const prhs[], int D, int MY, int MM,
    FusedEvalOpts& eval_opts)
{
    if (!fused_eval) return 0;
    const mxArray* x_test = prhs[10];
    const mxArray* y_test = prhs[11];
    const mxArray* t_test = prhs[12];
    const mxArray* c_test = prhs[13];
    if (!mxIsSingle(x_test) || !mxIsSingle(y_test) || !mxIsSingle(t_test) ||
            !mxIsSingle(c_test) || mxIsComplex(x_test) || mxIsComplex(y_test) ||
            mxIsComplex(t_test) || mxIsComplex(c_test) || mxIsSparse(x_test) ||
            mxIsSparse(y_test) || mxIsSparse(t_test) || mxIsSparse(c_test)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "All testing inputs must be dense real single arrays; omitted truth is an empty single array.");
    }
    if (mxGetNumberOfDimensions(x_test) != 2) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "X_Test must be a 2-D [D,Ntest] single matrix.");
    }
    if (mxGetN(x_test) > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "X_Test sample count exceeds INT_MAX.");
    }
    const int Ntest = static_cast<int>(mxGetN(x_test));
    if (mxGetNumberOfElements(x_test) != 0 && mxGetM(x_test) != (mwSize)D) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "Nonempty X_Test must have shape [D,Ntest].");
    }
    const bool has_y = mxGetNumberOfElements(y_test) != 0;
    const bool has_t = mxGetNumberOfElements(t_test) != 0;
    const bool has_c = mxGetNumberOfElements(c_test) != 0;
    if ((has_y || has_t || has_c) && Ntest < 1) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "Testing truth was supplied but X_Test has no points.");
    }
    if (has_y && (mxGetNumberOfDimensions(y_test) != 2 ||
            mxGetM(y_test) != (mwSize)MY || mxGetN(y_test) != (mwSize)Ntest)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "Nonempty Y_Test must have shape [MY,Ntest].");
    }
    if (has_t) {
        if (mxGetNumberOfDimensions(t_test) != 3) {
            throw OwnerMexError("manor_m_phase2_mex:fusedEval",
                "Nonempty T_Test must have shape [D,MM,Ntest].");
        }
        const mwSize* dimensions = mxGetDimensions(t_test);
        if (dimensions[0] != (mwSize)D || dimensions[1] != (mwSize)MM ||
                dimensions[2] != (mwSize)Ntest) {
            throw OwnerMexError("manor_m_phase2_mex:fusedEval",
                "T_Test must have shape [D, MM, Ntest].");
        }
    }
    if (has_c && (mxGetNumberOfDimensions(c_test) != 2 ||
            mxGetM(c_test) != (mwSize)D || mxGetN(c_test) != (mwSize)Ntest)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "Nonempty C_Test must have shape [D,Ntest].");
    }
    if (eval_opts.mse_only_eval && (has_t || has_c)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "mse_only_eval is a legacy restriction and requires empty T_Test and C_Test.");
    }
    if (eval_opts.tangent_only_eval && has_c) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "tangent_only_eval is a legacy restriction and requires empty C_Test.");
    }
    const mxArray* arrays[] = {x_test, y_test, t_test, c_test};
    const char* names[] = {"X_Test", "Y_Test", "T_Test", "C_Test"};
    for (int array_index = 0; array_index < 4; ++array_index) {
        const float* values = static_cast<const float*>(mxGetData(arrays[array_index]));
        const size_t count = (size_t)mxGetNumberOfElements(arrays[array_index]);
        if (owner_resident_finite_f32(values, count * sizeof(float))) continue;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) {
                throw OwnerMexError("manor_m_phase2_mex:fusedEval",
                    std::string(names[array_index]) + " must be finite.");
            }
        }
    }
    eval_opts.response_eval = has_y;
    eval_opts.tangent_metric_eval = has_t;
    eval_opts.center_eval = has_c;
    eval_opts.center_tangent_metric_eval = has_c && has_t;
    /* Guarded on nrhs: the evaluation arrays are required from argument 14,
       but the options struct is argument 15 and stays optional, so reading
       prhs[14] unconditionally was an out-of-bounds read on the 14-argument
       call that the arity check above explicitly permits. */
    if (nrhs >= 15 && (!mxIsStruct(prhs[14]) ||
            mxGetNumberOfElements(prhs[14]) != 1)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedEval",
            "eval_opts must be a scalar struct.");
    }
    return Ntest;
}

/* The validation block carries X and Y only, exactly like real training data:
   the model needs no truth geometry to predict Y, and the held-out points
   have none recorded. It is therefore always scored MSE-only and never reads
   a Tan or Xclean argument. Its point count is independent of the test set's. */
static int owner_validate_validation_inputs(
    bool fused_validation, const mxArray* const prhs[], int D, int MY)
{
    if (!fused_validation) return 0;
    const mxArray* x_valid = prhs[15];
    const mxArray* y_valid = prhs[16];
    if (!mxIsSingle(x_valid) || !mxIsSingle(y_valid) ||
        mxIsComplex(x_valid) || mxIsComplex(y_valid) ||
        mxIsSparse(x_valid) || mxIsSparse(y_valid)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
            "X_Valid and Y_Valid must be dense real single arrays.");
    }
    if (mxGetNumberOfDimensions(x_valid) != 2 ||
        mxGetNumberOfDimensions(y_valid) != 2) {
        throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
            "X_Valid and Y_Valid must be 2-D matrices.");
    }
    if (mxGetN(x_valid) > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
            "X_Valid sample count exceeds INT_MAX.");
    }
    const int Nvalid = static_cast<int>(mxGetN(x_valid));
    if (Nvalid < 1 || mxGetM(x_valid) != static_cast<mwSize>(D) ||
        mxGetM(y_valid) != static_cast<mwSize>(MY) ||
        mxGetN(y_valid) != static_cast<mwSize>(Nvalid)) {
        throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
            "X_Valid must be [D x Nvalid] and Y_Valid [MY x Nvalid].");
    }
    const float* xv = static_cast<const float*>(mxGetData(x_valid));
    const size_t xv_count = (size_t)Nvalid * D;
    if (!owner_resident_finite_f32(xv, xv_count * sizeof(float)))
        for (size_t i = 0; i < xv_count; ++i) {
            if (!std::isfinite(xv[i]))
                throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                    "X_Valid must be finite.");
        }
    const float* yv = static_cast<const float*>(mxGetData(y_valid));
    const size_t yv_count = (size_t)Nvalid * MY;
    if (!owner_resident_finite_f32(yv, yv_count * sizeof(float)))
        for (size_t i = 0; i < yv_count; ++i) {
            if (!std::isfinite(yv[i]))
                throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                    "Y_Valid must be finite.");
        }
    return Nvalid;
}

extern "C" void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nlhs != 1)
            throw OwnerMexError("manor_m_phase2_mex:nlhs", "Exactly one struct output is required.");
        if (nrhs < 10 || nrhs > 18)
            throw OwnerMexError("manor_m_phase2_mex:nrhs",
                "Expected X, Y, idx, Tan, Cen, owner_mask0, hk, rhok, MM, opts, "
                "then optionally X_Test, Y_Test, Tan_Test, Xclean_Test, an "
                "evaluation options struct, and finally X_Valid and Y_Valid. "
                "opts.fused_eval and opts.fused_validation decide whether the "
                "test and validation evaluations run; the argument count no "
                "longer does.");

        const mxArray* mxX = prhs[0];
        const mxArray* mxY = prhs[1];
        const mxArray* mxIdx = prhs[2];
        const mxArray* mxTan = prhs[3];
        const mxArray* mxCen = prhs[4];
        const mxArray* mxOwner = prhs[5];
        const mxArray* mxHk = prhs[6];
        const mxArray* mxRhok = prhs[7];
        const mxArray* mxMM = prhs[8];
        OwnerPhase2Opts opts = owner_parse_opts(prhs[9]);
        OwnerProfileTimers profile_span;
        profile_span.enabled = opts.profile_timers;
        const auto host_ms_since = [](
                const std::chrono::steady_clock::time_point& reference) {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - reference).count();
        };
        const auto host_entry_time = std::chrono::steady_clock::now();
        owner_cuda_allocate_milliseconds = 0.0;
        owner_cuda_free_milliseconds = 0.0;
        owner_cuda_event_destroy_milliseconds = 0.0;
        owner_use_async_allocator = opts.async_allocator;

        if (!mxIsSingle(mxX) || !mxIsSingle(mxY) || !mxIsSingle(mxTan) || !mxIsSingle(mxCen) ||
            mxIsComplex(mxX) || mxIsComplex(mxY) || mxIsComplex(mxTan) || mxIsComplex(mxCen) ||
            mxIsSparse(mxX) || mxIsSparse(mxY) || mxIsSparse(mxTan) || mxIsSparse(mxCen)) {
            throw OwnerMexError("manor_m_phase2_mex:type",
                "X, Y, Tan, and Cen must be dense real single arrays.");
        }
        if (mxGetNumberOfDimensions(mxX) != 2) {
            throw OwnerMexError("manor_m_phase2_mex:X",
                "X must be a dense real single [D x N] matrix.");
        }
        if (mxGetNumberOfDimensions(mxY) != 2) {
            throw OwnerMexError("manor_m_phase2_mex:Y",
                "Y must be a dense real single [MY x N] matrix.");
        }
        if (mxGetNumberOfDimensions(mxCen) != 2) {
            throw OwnerMexError("manor_m_phase2_mex:Cen",
                "Cen must be a dense real single [D x N] matrix.");
        }
        if (!mxIsLogical(mxOwner) || mxIsSparse(mxOwner) || mxIsComplex(mxOwner) || mxGetM(mxOwner) != 1) {
            throw OwnerMexError("manor_m_phase2_mex:ownerMask",
                "owner_mask0 must be a dense logical 1 x N row vector.");
        }
        if (!mxIsNumeric(mxMM) || mxIsComplex(mxMM) || mxGetNumberOfElements(mxMM) != 1) {
            throw OwnerMexError("manor_m_phase2_mex:MM", "MM must be a real numeric scalar.");
        }
        const mwSize Dmw = mxGetM(mxX), Nmw = mxGetN(mxX);
        if (Dmw < 1 || Dmw > MAX_D_MANOR || Nmw < 1 ||
            Dmw > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
            Nmw > static_cast<mwSize>(std::numeric_limits<int>::max())) {
            throw OwnerMexError("manor_m_phase2_mex:dimensions", "X dimensions are unsupported.");
        }
        const int D = static_cast<int>(Dmw), N = static_cast<int>(Nmw);
        if (mxGetN(mxOwner) != Nmw) {
            throw OwnerMexError("manor_m_phase2_mex:ownerMask",
                "owner_mask0 must be a logical 1 x N row vector matching X.");
        }
        const double mm_raw = owner_scalar(mxMM, "MM");
        if (mm_raw != std::floor(mm_raw) || mm_raw < 1.0 || mm_raw >= D || mm_raw > MAX_MM_MANOR) {
            throw OwnerMexError("manor_m_phase2_mex:MM", "MM must satisfy 1 <= MM < D and MM <= MAX_MM_MANOR.");
        }
        const mwSize MYmw = mxGetM(mxY);
        if (MYmw > static_cast<mwSize>(std::numeric_limits<int>::max())) {
            throw OwnerMexError("manor_m_phase2_mex:dimensions", "Y row count exceeds INT_MAX.");
        }
        const int MM = static_cast<int>(mm_raw), MY = static_cast<int>(MYmw);
        if (MY < 1 || MY > MAX_MY_MANOR || mxGetN(mxY) != Nmw || mxGetM(mxCen) != Dmw || mxGetN(mxCen) != Nmw ||
            mxGetNumberOfDimensions(mxTan) != 3) {
            throw OwnerMexError("manor_m_phase2_mex:dimensions", "Y, Tan, and Cen dimensions are inconsistent with X.");
        }
        const mwSize* tan_dims = mxGetDimensions(mxTan);
        if (tan_dims[0] != Dmw || tan_dims[1] != MM || tan_dims[2] != Nmw) {
            throw OwnerMexError("manor_m_phase2_mex:dimensions", "Tan dimensions are inconsistent with X.");
        }
        if (mxIsComplex(mxIdx) || mxIsSparse(mxIdx) || !mxIsNumeric(mxIdx) || mxGetNumberOfDimensions(mxIdx) != 2) {
            throw OwnerMexError("manor_m_phase2_mex:idx", "idx must be a dense real numeric matrix.");
        }
        const mwSize idxM = mxGetM(mxIdx), idxN = mxGetN(mxIdx);
        if (idxM > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
            idxN > static_cast<mwSize>(std::numeric_limits<int>::max())) {
            throw OwnerMexError("manor_m_phase2_mex:idx", "idx dimensions exceed INT_MAX.");
        }
        const int kSlice = (opts.idx_generated_response ||
            opts.idx_shared_slice || opts.idx_zero_based) ?
            static_cast<int>(idxM) : static_cast<int>(idxN);
        const bool invalid_generated = opts.idx_generated_response &&
            (!mxIsInt32(mxIdx) || idxN != 1);
        const bool invalid_shared = opts.idx_shared_slice &&
            (!opts.idx_zero_based || !mxIsInt32(mxIdx) || idxN != 1);
        const bool invalid_explicit = !opts.idx_generated_response &&
            !opts.idx_shared_slice &&
            ((opts.idx_zero_based &&
                ((!opts.idx_compact_owner_order && idxN != Nmw) ||
                 (opts.idx_compact_owner_order &&
                  (idxN < 1 || idxN > Nmw || !mxIsInt32(mxIdx))) ||
                 (!mxIsInt32(mxIdx) && !mxIsUint32(mxIdx)))) ||
             (!opts.idx_zero_based && idxM != Nmw));
        if (invalid_generated || invalid_shared || invalid_explicit ||
                kSlice < 1 || kSlice > N) {
            throw OwnerMexError("manor_m_phase2_mex:idx",
                "idx must be one-based [N,kSlice], zero-based int32/uint32 "
                "[kSlice,N], compact owner-order int32 [kSlice,M], shared "
                "zero-based int32 [kSlice,1], or the generated-mode int32 "
                "shape token [kSlice,1].");
        }
        const double hk_raw = owner_scalar(mxHk, "hk");
        if (hk_raw <= 0.0)
            throw OwnerMexError("manor_m_phase2_mex:bandwidth", "hk must be a finite positive scalar.");
        const float hk = owner_validated_hk(hk_raw);
        const bool fused_eval = opts.fused_eval;
        if (fused_eval && nrhs < 14)
            throw OwnerMexError("manor_m_phase2_mex:fusedEval",
                "opts.fused_eval is set but the evaluation inputs are missing: "
                "X_Test, Y_Test, Tan_Test and Xclean_Test are required.");
        if (!fused_eval && nrhs >= 14)
            throw OwnerMexError("manor_m_phase2_mex:fusedEval",
                "Evaluation inputs were supplied but opts.fused_eval is not "
                "set. The flag decides, not the argument count; ignoring the "
                "inputs would return a result that evaluated nothing.");
        const bool fused_validation = opts.fused_validation;
        if (fused_validation && !fused_eval)
            throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                "opts.fused_validation requires opts.fused_eval: the validation "
                "block reuses the test block's evaluation grid and schedule, so "
                "there is nothing to score it against on its own.");
        if (fused_validation && nrhs < 17)
            throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                "opts.fused_validation is set but the validation inputs are "
                "missing: X_Valid and Y_Valid are required.");
        if (!fused_validation && nrhs >= 16)
            throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                "Validation inputs were supplied but opts.fused_validation is "
                "not set. The flag decides, not the argument count; ignoring "
                "the inputs would return a result that scored no validation.");
        const std::vector<float> rhok_iter = owner_validated_rhok_schedule(mxRhok, opts.iterations, fused_eval);
        FusedEvalOpts eval_opts;
        int Ntest = 0;
        if (fused_eval) {
            eval_opts = parse_fused_eval_opts(nrhs >= 15 ? prhs[14] : nullptr);
            if (eval_opts.count_sparse_threshold <= 0) eval_opts.count_sparse_threshold = std::max(1, 2 * D);
        }
        if (opts.evaluation_only && (!fused_eval || !eval_opts.eval_k0 ||
                !eval_opts.eval_k_arr.empty() || opts.return_histories ||
                opts.return_debug || !opts.fused_eval_use_owner_mask)) {
            throw OwnerMexError("manor_m_phase2_mex:evaluationOnly",
                "evaluation_only requires fused evaluation of exactly K=0, "
                "owner-mask scoring, and disabled histories/debug output.");
        }
        Ntest = owner_validate_fused_inputs(
            fused_eval, nrhs, prhs, D, MY, MM, eval_opts);
        const int Nvalid = owner_validate_validation_inputs(
            fused_validation, prhs, D, MY);
        const bool valid_tangent = fused_validation && nrhs == 18 &&
            mxGetNumberOfElements(prhs[17]) != 0;
        if (valid_tangent) {
            const mxArray* a = prhs[17];
            const mwSize* dims = mxGetDimensions(a);
            if (!mxIsSingle(a) || mxIsComplex(a) || mxIsSparse(a) ||
                    mxGetNumberOfDimensions(a) != 3 || dims[0] != (mwSize)D ||
                    dims[1] != (mwSize)MM || dims[2] != (mwSize)Nvalid)
                throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                    "T_Valid must have shape [D,MM,Nvalid].");
            const float* values = static_cast<const float*>(mxGetData(a));
            const size_t count = mxGetNumberOfElements(a);
            if (!owner_resident_finite_f32(values, count * sizeof(float)))
                for (size_t i = 0; i < count; ++i)
                    if (!std::isfinite(values[i]))
                        throw OwnerMexError("manor_m_phase2_mex:fusedValidation",
                            "T_Valid must be finite.");
        }

        auto X = owner_float_data(mxX, "X");
        auto Y = owner_float_data(mxY, "Y");
        auto Tan = owner_float_data(mxTan, "Tan");
        auto Cen = owner_float_data(mxCen, "Cen");
        if (!owner_resident_finite_f32(X.data(), X.size() * sizeof(float)))
            for (float value : X) if (!std::isfinite(value))
                throw OwnerMexError("manor_m_phase2_mex:X", "X must be finite.");
        if (!owner_resident_finite_f32(Y.data(), Y.size() * sizeof(float)))
            for (float value : Y) if (!std::isfinite(value))
                throw OwnerMexError("manor_m_phase2_mex:Y", "Y must be finite.");
        const mxLogical* owner_logical = mxGetLogicals(mxOwner);
        std::vector<uint8_t> owner_flags(N, 0);
        std::vector<int> owner_host;
        owner_host.reserve(N);
        for (int i = 0; i < N; ++i) if (owner_logical[i]) {
            owner_flags[i] = 1;
            owner_host.push_back(i);
            for (int d = 0; d < D * MM; ++d) if (!std::isfinite(Tan[(size_t)i * D * MM + d]))
                throw OwnerMexError("manor_m_phase2_mex:ownerEstimator", "Owner Tan entries must be finite.");
            for (int d = 0; d < D; ++d) if (!std::isfinite(Cen[(size_t)i * D + d]))
                throw OwnerMexError("manor_m_phase2_mex:ownerEstimator", "Owner Cen entries must be finite.");
        }
        const int M = static_cast<int>(owner_host.size());
        const uint8_t* owner_upload_host = owner_flags.data();
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
        // Borrow only a normalized immutable logical mask. Other C callers
        // retain the original normalization/validation route.
        if (std::memcmp(owner_flags.data(), owner_logical, static_cast<size_t>(N)) == 0 &&
            manor_m_resident_find(owner_logical, static_cast<size_t>(N)))
            owner_upload_host = reinterpret_cast<const uint8_t*>(owner_logical);
#endif
        if (M == 0) {
            throw OwnerMexError("manor_m_phase2_mex:ownerCount",
                "owner_mask0 must retain at least one owner when Phase 2 is invoked.");
        }
        if (opts.idx_compact_owner_order && idxN != static_cast<mwSize>(M)) {
            throw OwnerMexError("manor_m_phase2_mex:idx",
                "Compact owner-index row count does not match owner_mask0.");
        }
        std::vector<int> owner_lookup_host(opts.evaluation_only ? 0 : N, -1);
        if (!opts.evaluation_only)
            for (int q = 0; q < M; ++q) owner_lookup_host[owner_host[q]] = q;

        const size_t index_count = opts.idx_generated_response ? 0 :
            opts.idx_shared_slice ? static_cast<size_t>(kSlice) :
            owner_checked_mul((size_t)M, (size_t)kSlice, "compact owner idx");
        const OwnerIndexTiles index_tiles(M, kSlice, opts.index_tile_owners,
            opts.idx_shared_slice, opts.iterations,
            opts.idx_generated_response);
        bool idx_borrow = false;
        bool idx_owner_gather = false;
        size_t idx_validation_count = index_count;
        const int32_t* cached_compact_index = nullptr;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
        if (!opts.idx_generated_response && !opts.idx_shared_slice &&
            !opts.idx_compact_owner_order &&
            opts.idx_zero_based && mxIsInt32(mxIdx))
            cached_compact_index = manor_m_resident_compact_index(
                static_cast<const int32_t*>(mxGetData(mxIdx)),
                reinterpret_cast<const uint8_t*>(mxGetLogicals(mxOwner)), index_count);
        const size_t source_index_bytes = owner_checked_mul(
            static_cast<size_t>(mxGetNumberOfElements(mxIdx)), sizeof(int),
            "source idx bytes");
        if (index_tiles.streamed && (cached_compact_index ||
                manor_m_resident_find(mxGetData(mxIdx), source_index_bytes)))
            throw OwnerMexError("manor_m_phase2_mex:indexTile",
                "Release phase1.response_index and phase1.compact_index explicitly before streaming.");
        idx_borrow = !opts.idx_generated_response && !opts.idx_shared_slice &&
            opts.idx_zero_based &&
            mxIsInt32(mxIdx) && (opts.idx_compact_owner_order || (M == N &&
            (index_tiles.streamed || manor_m_resident_find(
                mxGetData(mxIdx), index_count * sizeof(int)))) || cached_compact_index);
#if defined(MANOR_M_STREAM_OWNER_ROWS) && MANOR_M_STREAM_OWNER_ROWS
        // Gather only the current owner tile from the immutable full index.
        idx_owner_gather = index_tiles.streamed && !opts.idx_shared_slice &&
            !opts.idx_compact_owner_order && opts.idx_zero_based &&
            mxIsInt32(mxIdx) && M < N;
        if (idx_owner_gather) {
            idx_borrow = true;
            idx_validation_count = mxGetNumberOfElements(mxIdx);
        }
#endif
#endif
        std::vector<int> idx_host(idx_borrow ? 0 : index_count);
        const int* idx_upload_host = cached_compact_index ? cached_compact_index : idx_borrow ?
            static_cast<const int*>(mxGetData(mxIdx)) : idx_host.data();
        std::vector<uint8_t> shared_seen(opts.idx_shared_slice ? N : 0, 0u);
        /* Typed fast path (item 6): int32/double idx skip the per-element
           class switch; validation is byte-for-byte the generic loop's. */
        const bool idx_typed_fast = !opts.idx_generated_response &&
            !opts.idx_shared_slice &&
            !mxIsComplex(mxIdx) && (mxIsInt32(mxIdx) || mxIsDouble(mxIdx));
        if (opts.idx_generated_response && !opts.evaluation_only) {
            // The exact index is produced in device tiles immediately before
            // each consumer pass. There are no host values to normalize or
            // validate here.
        } else if (idx_borrow) {
            // Registration owns an immutable input. Retain full range checking
            // but amortize it once for an exact resident-session generation.
            const size_t idx_bytes = idx_validation_count * sizeof(int32_t);
            if (!owner_resident_index_validated(idx_upload_host, idx_bytes, N)) {
                const auto range_valid = [&](const int* values, size_t count) {
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_HOST_INDEX_VECTOR_CHECK
            // Scan every index, including negatives and tail elements. Only
            // the host validity reduction changes; no data/metrics are changed.
                    return manor_m_index_range_valid(values, count, N);
#else
                    for (size_t s = 0; s < count; ++s)
                        if (values[s] < 0 || values[s] >= N) return false;
                    return true;
#endif
                };
                if (idx_owner_gather) {
                    // Preserve the old validation domain: only selected rows.
                    for (int q = 0; q < M; ++q)
                        if (!range_valid(idx_upload_host + size_t(owner_host[q]) * kSlice, kSlice))
                            throw OwnerMexError("manor_m_phase2_mex:idx", "idx contains an invalid index.");
                } else if (!range_valid(idx_upload_host, index_count)) {
                    throw OwnerMexError("manor_m_phase2_mex:idx", "idx contains an invalid index.");
                }
            }
        } else if (idx_typed_fast) {
            const bool idx_from_int32 = mxIsInt32(mxIdx);
            const int32_t* idx_raw32 = idx_from_int32 ?
                static_cast<const int32_t*>(mxGetData(mxIdx)) : nullptr;
            const double* idx_raw64 = idx_from_int32 ? nullptr :
                static_cast<const double*>(mxGetData(mxIdx));
            for (int q = 0; q < M; ++q) {
                const int i = opts.idx_compact_owner_order ? q : owner_host[q];
                for (int s = 0; s < kSlice; ++s) {
                    const size_t source = opts.idx_zero_based ?
                        (size_t)s + (size_t)kSlice * i : (size_t)i + (size_t)N * s;
                    const double raw = idx_from_int32 ?
                        (double)idx_raw32[source] : idx_raw64[source];
                    const double zero = opts.idx_zero_based ? raw : raw - 1.0;
                    if (!std::isfinite(raw) || raw != std::floor(raw) ||
                            zero < 0.0 || zero >= N) {
                        throw OwnerMexError("manor_m_phase2_mex:idx",
                            "idx contains an invalid index.");
                    }
                    idx_host[(size_t)q * kSlice + s] = static_cast<int>(zero);
                }
            }
        }
        const int idx_anchor_count = opts.idx_generated_response ? 0 :
            (opts.idx_shared_slice ? 1 : (idx_typed_fast ? 0 : M));
        for (int q = 0; q < idx_anchor_count; ++q) for (int s = 0; s < kSlice; ++s) {
            const int i = opts.idx_shared_slice ? 0 :
                (opts.idx_compact_owner_order ? q : owner_host[q]);
            const mwIndex source = opts.idx_shared_slice ? s : opts.idx_zero_based ?
                s + (size_t)kSlice * i : i + (size_t)N * s;
            const double raw = owner_numeric_at(mxIdx, source);
            const double zero = opts.idx_zero_based ? raw : raw - 1.0;
            if (!std::isfinite(raw) || raw != std::floor(raw) || zero < 0.0 || zero >= N) {
                throw OwnerMexError("manor_m_phase2_mex:idx", "idx contains an invalid index.");
            }
            const int value = static_cast<int>(zero);
            if (opts.idx_shared_slice) {
                if (shared_seen[static_cast<size_t>(value)])
                    throw OwnerMexError("manor_m_phase2_mex:idx",
                        "Shared idx must contain unique candidate indices.");
                shared_seen[static_cast<size_t>(value)] = 1u;
                idx_host[static_cast<size_t>(s)] = value;
            } else {
                idx_host[(size_t)q * kSlice + s] = value;
            }
        }

        const size_t mask_words = (static_cast<size_t>(kSlice) + 31) >> 5;
        const size_t full_tan = owner_checked_mul(owner_checked_mul((size_t)N, (size_t)MM, "N*MM"), (size_t)D, "Tan");
        const size_t full_center = owner_checked_mul((size_t)N, (size_t)D, "Cen");
        const size_t compact_gradient = owner_checked_mul(owner_checked_mul((size_t)M, (size_t)MY, "M*MY"), (size_t)D, "owner gradients");
        const size_t compact_center = owner_checked_mul((size_t)M, (size_t)D, "owner centers");
        const OwnerAggregationTiles aggregation_tiles(M, opts.aggregation_tile_owners);
        const size_t aggregation_center = owner_checked_mul(
            (size_t)aggregation_tiles.capacity, (size_t)D, "aggregation tile centers");
        const size_t aggregation_cov = owner_checked_mul(
            aggregation_center, (size_t)D, "aggregation tile covariance");
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_TILED_MASK
        const bool neighbor_mask_tiled = aggregation_tiles.capacity < M;
#else
        const bool neighbor_mask_tiled = false;
#endif
        const size_t mask_owner_capacity = neighbor_mask_tiled ?
            (size_t)aggregation_tiles.capacity : (size_t)M;
        const size_t compact_mask = owner_checked_mul(
            mask_owner_capacity, mask_words, "owner masks");
        const size_t full_y = owner_checked_mul((size_t)N, (size_t)MY, "N*MY");
        size_t generated_knn_scratch = 0;
        if (opts.idx_generated_response && !opts.evaluation_only) {
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
            if (!manor_m_cuda_active_generated_phase2 ||
                    !manor_m_cuda_active_generated_phase2->context ||
                    !manor_m_cuda_active_generated_phase2->response_point_major ||
                    manor_m_cuda_active_generated_phase2->scratch_memory_bytes == 0 ||
                    manor_m_cuda_active_generated_phase2->scratch_memory_bytes >
                        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                throw OwnerMexError("manor_m_phase2_mex:generatedIndex",
                    "Generated Phase-2 requires a live response source and a "
                    "nonzero size_t-range exact-kNN scratch budget.");
            }
            generated_knn_scratch = static_cast<size_t>(
                manor_m_cuda_active_generated_phase2->scratch_memory_bytes);
#else
            throw OwnerMexError("manor_m_phase2_mex:generatedIndex",
                "Generated Phase-2 is unavailable in the native MEX entry.");
#endif
        }
        const int K = opts.iterations;
        const size_t history_nk = owner_checked_mul((size_t)N, (size_t)K, "N*K history");
        /* Iteration histories are outputs, not live model state.  Keeping all
           K slices on the GPU made a diagnostic output consume 6*N*K bytes
           even when the shared/Python caller explicitly did not request it.
           Requested histories now use one N-row device staging slice and are
           copied to their final host offsets after each global iteration.
           The default-stream ordering preserves the iteration barrier and the
           exact bytes; unrequested histories consume no device or host space. */
        const size_t device_history_n = opts.return_histories ? (size_t)N : 0;
        const size_t eval_slots = fused_eval ? owner_checked_mul(eval_opts.eval_rho_scale.size(),
            eval_opts.eval_h_scale.size(), "evaluation slots") : 0;
        const size_t eval_history_slots = fused_eval ? owner_checked_mul((size_t)K, eval_slots, "K*evaluation slots") : 0;
        const size_t eval_coordinate_history_slots = fused_eval ? owner_checked_mul(
            eval_history_slots, (size_t)MY, "K*evaluation slots*MY") : 0;
        const size_t history_bytes = owner_checked_add(
            owner_checked_add(owner_checked_mul(opts.return_histories ? history_nk : 0,
                    sizeof(int32_t), "count history bytes"),
                owner_checked_mul(owner_checked_mul(opts.return_histories ? history_nk : 0,
                    (size_t)2, "mask history"), sizeof(uint8_t), "mask history bytes"),
                "history bytes"),
            owner_checked_mul(owner_checked_add(
                owner_checked_mul(eval_history_slots, (size_t)4, "evaluation history"),
                eval_coordinate_history_slots, "evaluation plus coordinate history"),
                sizeof(float), "evaluation history bytes"),
            "history bytes");
        size_t required = 0;
        auto add_bytes = [&](size_t count, size_t width, const char* label) {
            required = owner_checked_add(required, owner_checked_mul(count, width, label), "required bytes");
        };
        const size_t eval_point_channels = owner_checked_mul(eval_slots,
            (size_t)MY + 4, "fixed-order evaluation channels");
        if (eval_point_channels > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw OwnerMexError("manor_m_phase2_mex:fused_eval",
                "Fixed-order evaluation has too many channels for one launch.");
        const size_t eval_point_count = owner_checked_mul(eval_point_channels,
            (size_t)std::max(Ntest, fused_validation ? Nvalid : 0),
            "fixed-order evaluation points");
        add_bytes(eval_point_count, sizeof(float), "fixed-order evaluation scratch");
        add_bytes(full_center, sizeof(float), "X");
        add_bytes(full_y, sizeof(float), "Y");
        add_bytes(full_tan, sizeof(float), "Tan state");
        add_bytes(full_center, sizeof(float), "Cen state");
        if (!opts.evaluation_only) {
            add_bytes(full_tan, sizeof(float), "Tan output");
            add_bytes(full_center, sizeof(float), "Cen output");
            add_bytes((size_t)M, sizeof(int), "owner indices");
        }
        add_bytes((size_t)N, sizeof(uint8_t), "owner mask");
        add_bytes(3, sizeof(int), "evaluation control scalars");
        if (!opts.evaluation_only) {
            add_bytes(index_tiles.device_bytes, 1, "idx");
            if (opts.idx_generated_response) {
                add_bytes(generated_knn_scratch, 1, "generated exact-kNN scratch");
                add_bytes(full_y, sizeof(float), "generated response SoA");
                add_bytes((size_t)M, sizeof(int32_t), "generated owner row map");
            }
            add_bytes(compact_gradient, sizeof(float), "owner gradients");
            add_bytes(compact_center, sizeof(float), "owner center pre");
            add_bytes(compact_mask, sizeof(uint32_t), "neighbor masks");
            add_bytes((size_t)M, sizeof(int32_t), "owner counts");
            add_bytes((size_t)M, sizeof(int32_t), "last owner counts");
            add_bytes((size_t)M, sizeof(float), "owner weights");
            add_bytes((size_t)N, sizeof(int), "owner lookup");
            add_bytes((size_t)M, sizeof(uint8_t), "solver status");
            add_bytes(owner_checked_mul((size_t)N, (size_t)4,
                "heavy/keep masks"), sizeof(uint8_t), "heavy/keep masks");
            add_bytes((size_t)aggregation_tiles.capacity, sizeof(int), "aggregation counts");
            add_bytes(aggregation_center, sizeof(float), "aggregation centers");
            add_bytes(aggregation_cov, sizeof(float), "aggregation covariance");
            add_bytes(2, sizeof(unsigned long long), "count reductions");
            add_bytes(3, sizeof(int), "iteration scalars");
            add_bytes(device_history_n, sizeof(int32_t), "device count history staging");
            add_bytes(owner_checked_mul(device_history_n, (size_t)2,
                "device mask history staging"), sizeof(uint8_t), "device mask history staging");
        }
        if (fused_eval) {
            add_bytes(owner_checked_mul((size_t)Ntest, (size_t)D, "X_Test"), sizeof(float), "X_Test");
            if (eval_opts.response_eval) {
                add_bytes(owner_checked_mul((size_t)Ntest, (size_t)MY, "Y_Test"), sizeof(float), "Y_Test");
            }
            if (eval_opts.tangent_metric_eval) {
                add_bytes(owner_checked_mul(owner_checked_mul((size_t)Ntest, (size_t)MM, "T_Test"), (size_t)D, "T_Test"), sizeof(float), "T_Test");
            }
            if (eval_opts.center_eval) {
                add_bytes(owner_checked_mul((size_t)Ntest, (size_t)D, "C_Test"), sizeof(float), "C_Test");
            }
            add_bytes(eval_opts.eval_h_scale.size(), sizeof(float), "evaluation h scales"); add_bytes(eval_opts.eval_rho_scale.size(), sizeof(float), "evaluation rho scales");
            add_bytes((size_t)N, sizeof(float), "evaluation train norms");
            add_bytes(owner_checked_mul((size_t)N, (size_t)MM, "evaluation tangent dot X"), sizeof(float), "evaluation tangent dot X");
            add_bytes(owner_checked_mul(eval_slots, (size_t)4, "evaluation sums"), sizeof(float), "evaluation sums");
            add_bytes(owner_checked_mul(eval_slots, (size_t)MY, "coordinate evaluation sums"),
                sizeof(float), "coordinate evaluation sums");
            add_bytes(owner_checked_mul(eval_history_slots, (size_t)4, "device evaluation history"), sizeof(float), "device evaluation history");
            add_bytes(eval_coordinate_history_slots, sizeof(float), "device coordinate evaluation history");
        }
        if (fused_validation) {
            /* The validation block adds its own points and its own two
               histories. It reuses the test block's sums, scale grids and
               train-stat buffers, which are per-slot or per-train-point and
               carry no dependence on which query set is being scored. */
            add_bytes(owner_checked_mul((size_t)Nvalid, (size_t)D, "X_Valid"), sizeof(float), "X_Valid");
            add_bytes(owner_checked_mul((size_t)Nvalid, (size_t)MY, "Y_Valid"), sizeof(float), "Y_Valid");
            add_bytes(eval_history_slots, sizeof(float), "device validation history");
            if (valid_tangent) {
                add_bytes(owner_checked_mul((size_t)Nvalid, (size_t)D * MM,
                    "T_Valid"), sizeof(float), "T_Valid");
                add_bytes(eval_history_slots, sizeof(float), "validation tangent history");
            }
            add_bytes(eval_coordinate_history_slots, sizeof(float), "device coordinate validation history");
        }
        size_t free_bytes = 0, total_bytes = 0;
        owner_cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        uint64_t reusable_pool_bytes = 0;
        if (opts.async_allocator) {
            int pools_supported = 0;
            int active_device = 0;
            owner_cuda_check(cudaGetDevice(&active_device),
                "query active device for async allocator");
            owner_cuda_check(cudaDeviceGetAttribute(
                &pools_supported, cudaDevAttrMemoryPoolsSupported, active_device),
                "query async allocator support");
            if (!pools_supported) {
                throw OwnerMexError("manor_m_phase2_mex:memoryPool",
                    "async_allocator requires CUDA memory-pool support.");
            }
            cudaMemPool_t pool = nullptr;
            uint64_t reserved_current = 0, used_current = 0;
            owner_cuda_check(cudaDeviceGetDefaultMemPool(&pool, active_device),
                "get default memory pool");
            owner_cuda_check(cudaMemPoolGetAttribute(pool,
                cudaMemPoolAttrReservedMemCurrent, &reserved_current),
                "query reserved pool memory");
            owner_cuda_check(cudaMemPoolGetAttribute(pool,
                cudaMemPoolAttrUsedMemCurrent, &used_current),
                "query used pool memory");
            reusable_pool_bytes = reserved_current > used_current
                ? reserved_current - used_current : 0;
        }
        const double reserve = std::max(opts.reserve_bytes, (double)total_bytes * opts.reserve_fraction);
        const double effective_free = (double)free_bytes
            + (double)reusable_pool_bytes;
        const double usable = std::max(
            0.0, (effective_free - reserve) * opts.gpu_mem_fraction);
        if ((double)required > usable) throw OwnerMexError("manor_m_phase2_mex:memory",
            "Insufficient available GPU memory for compact-owner iteration buffers.");

        OwnerCudaBuffer<float> d_X, d_Y, d_Tan_a, d_Cen_a, d_Tan_b, d_Cen_b, d_center_pre, d_weight, d_grad, d_agg_cacc, d_agg_cov;
        OwnerCudaBuffer<float> d_Xtest, d_Ytest, d_Ttest, d_Ctest, d_eval_h_scale, d_eval_rho_scale, d_eval_sums, d_eval_coordinate_sums, d_eval_xnorm, d_eval_tdotx;
        OwnerCudaBuffer<float> d_Xvalid, d_Yvalid, d_Tvalid, d_valid_tan_history;
        OwnerCudaBuffer<float> d_eval_points;
        if (eval_point_count > 0)
            d_eval_points.allocate(eval_point_count, "fixed-order evaluation scratch");
        OwnerCudaBuffer<int> d_pair_count, d_pair_list;
        OwnerCudaBuffer<float> d_pair_minq;
        int tiled_JT = 0;
        size_t tiled_main_shmem = 0;
        /* The validation pass is MSE-only whatever the test pass is, so it can
           carry a tile size of its own: the MSE-only layout drops the per-tile
           Xclean row and the per-tile Gram triangle, which frees shared memory
           the tile loop can spend on a wider tile. Both are launched with the
           same kernel, so the opt-in maximum has to cover whichever is larger. */
        int valid_JT = 0;
        size_t valid_shmem = 0;
        OwnerCudaBuffer<float> d_eval_mse_history, d_eval_coordinate_history, d_eval_cen_history, d_eval_cen_tan_history, d_eval_tan_history;
        OwnerCudaBuffer<float> d_valid_mse_history, d_valid_coordinate_history;
        OwnerCudaBuffer<int> d_idx, d_owner_index, d_owner_lookup, d_agg_count, d_heavy_count, d_keep_count, d_stopped, d_valid_iterations, d_keep_positive_count, d_fused_eval_executed;
        OwnerCudaBuffer<int32_t> d_num_neigh, d_num_last, d_num_history;
        OwnerCudaBuffer<uint8_t> d_solver_failed, d_owner_mask, d_heavy, d_heavy_last, d_keep, d_keep_last, d_heavy_history, d_keep_history;
        OwnerCudaBuffer<uint32_t> d_neigh_mask;
        OwnerCudaBuffer<unsigned long long> d_count_sum, d_keep_positive_sum;
        OwnerIndexStream index_stream(index_tiles);
        if (!opts.evaluation_only && index_tiles.streamed) index_stream.allocate();
        d_X.allocate_input(full_center, "X", X.data());
        d_Y.allocate_input(full_y, "Y", Y.data());
        if (!opts.evaluation_only && !index_tiles.streamed && !index_tiles.generated)
            d_idx.allocate_input(index_count, "idx", idx_upload_host);
        d_Tan_a.allocate(full_tan, "Tan state"); d_Cen_a.allocate(full_center, "Cen state");
        if (!opts.evaluation_only) {
            d_Tan_b.allocate(full_tan, "Tan next");
            d_Cen_b.allocate(full_center, "Cen next");
            d_owner_index.allocate(M, "owner indices");
        }
        d_owner_mask.allocate_input(N, "owner mask", owner_upload_host);
        d_stopped.allocate(1, "stopped");
        d_valid_iterations.allocate(1, "valid iterations");
        d_fused_eval_executed.allocate(1, "fused evaluation executed");
        if (!opts.evaluation_only) {
            d_center_pre.allocate(compact_center, "owner center pre");
            d_weight.allocate(M, "owner weights");
            d_grad.allocate(compact_gradient, "owner gradients");
            d_owner_lookup.allocate(N, "owner lookup");
            d_neigh_mask.allocate(compact_mask, "neighbor masks");
            d_num_neigh.allocate(M, "owner counts");
            d_num_last.allocate(M, "last owner counts");
            d_solver_failed.allocate(M, "solver status");
            d_heavy.allocate(N, "heavy mask");
            d_heavy_last.allocate(N, "last heavy mask");
            d_keep.allocate(N, "keep mask");
            d_keep_last.allocate(N, "last keep mask");
            d_num_history.allocate(device_history_n, "device count history staging");
            d_heavy_history.allocate(device_history_n, "device heavy history staging");
            d_keep_history.allocate(device_history_n, "device keep history staging");
            d_agg_count.allocate(aggregation_tiles.capacity, "aggregation counts");
            d_agg_cacc.allocate(aggregation_center, "aggregation centers");
            d_agg_cov.allocate(aggregation_cov, "aggregation covariance");
            d_heavy_count.allocate(1, "heavy count");
            d_keep_count.allocate(1, "keep count");
            d_count_sum.allocate(1, "owner count sum");
            d_keep_positive_sum.allocate(1, "keep positive sum");
            d_keep_positive_count.allocate(1, "keep positive count");
        }
        if (fused_eval) {
            if (Ntest > 0) d_Xtest.allocate_input((size_t)Ntest * D, "X_Test", mxGetData(prhs[10]));
            if (eval_opts.response_eval) {
                d_Ytest.allocate_input((size_t)Ntest * MY, "Y_Test", mxGetData(prhs[11]));
            }
            if (eval_opts.tangent_metric_eval) {
                d_Ttest.allocate_input((size_t)Ntest * MM * D, "T_Test", mxGetData(prhs[12]));
            }
            if (eval_opts.center_eval) {
                d_Ctest.allocate_input((size_t)Ntest * D, "C_Test", mxGetData(prhs[13]));
            }
            d_eval_h_scale.allocate(eval_opts.eval_h_scale.size(), "evaluation h scales"); d_eval_rho_scale.allocate(eval_opts.eval_rho_scale.size(), "evaluation rho scales");
            d_eval_xnorm.allocate((size_t)N, "evaluation train norms");
            d_eval_tdotx.allocate((size_t)N * MM, "evaluation tangent dot X");
            d_eval_sums.allocate(eval_slots * 4, "evaluation sums");
            d_pair_count.allocate(1, "sparse pair count");
            {
                /* One sparse-pair arena serves both passes, so it is sized for
                   whichever query set is larger. Validation is capped well
                   above the test count in the actual-N sweep, so sizing this
                   on Ntest alone would overflow it on the validation pass. */
                const size_t pair_points = (size_t)std::max(Ntest,
                    fused_validation ? Nvalid : 0);
                const size_t pair_max = owner_checked_mul(pair_points,
                    eval_slots, "sparse pair capacity");
                const size_t append_ints = owner_checked_mul(pair_max,
                    (size_t)2, "sparse pair list");
                const size_t grouped_ints = owner_checked_add(
                    owner_checked_add(pair_max,
                        owner_checked_mul(pair_points, (size_t)2,
                            "sparse pair group headers"),
                        "sparse pair grouped layout"),
                    (size_t)1, "sparse pair grouped layout");
                d_pair_list.allocate(std::max(append_ints, grouped_ints),
                    "sparse pair list");
                d_pair_minq.allocate(owner_checked_mul(pair_max, (size_t)2,
                    "sparse pair minq"), "sparse pair minq");
            }
            {
                int tiled_device = 0;
                owner_cuda_check(cudaGetDevice(&tiled_device), "tiled get device");
                cudaDeviceProp tiled_prop;
                owner_cuda_check(cudaGetDeviceProperties(&tiled_prop, tiled_device),
                    "tiled device properties");
                /* Select a tile size that fits the available shared-memory tiers. */
                const size_t shared_tiers[] = {
                    std::min((size_t)tiled_prop.sharedMemPerBlockOptin,
                             (size_t)tiled_prop.sharedMemPerMultiprocessor / 3),
                    std::min((size_t)tiled_prop.sharedMemPerBlockOptin,
                             (size_t)tiled_prop.sharedMemPerMultiprocessor / 2),
                    (size_t)tiled_prop.sharedMemPerBlockOptin};
                const int jt_candidates[] = {128, 96, 64, 48, 32, 16};
                const bool has_test_metrics = eval_opts.response_eval ||
                    eval_opts.tangent_metric_eval || eval_opts.center_eval;
                if (has_test_metrics) {
                    for (size_t shared_limit : shared_tiers) {
                        for (int cand : jt_candidates) {
                            const size_t need = owner_tiled_shared_bytes(
                                (int)eval_slots, D, MM, MY, cand, 8,
                                eval_opts.response_eval,
                                eval_opts.tangent_metric_eval || eval_opts.center_eval,
                                eval_opts.center_eval);
                            if (need <= shared_limit) {
                                tiled_JT = cand; tiled_main_shmem = need; break;
                            }
                        }
                        if (tiled_JT != 0) break;
                    }
                }
                if (has_test_metrics && tiled_JT == 0) {
                    throw OwnerMexError("manor_m_phase2_mex:tiledShared",
                        "The evaluation grid needs more shared memory than this "
                        "device offers; reduce the eval grid.");
                }
                /* Same ladder for the validation pass, at its own MSE-only
                   layout. When the test pass is already MSE-only the two agree
                   and this costs one extra pass over six candidates. */
                if (fused_validation) {
                    for (size_t shared_limit : shared_tiers) {
                        for (int cand : jt_candidates) {
                            const size_t need = owner_tiled_shared_bytes(
                                (int)eval_slots, D, MM, MY, cand, 8,
                                true, valid_tangent, false);
                            if (need <= shared_limit) { valid_JT = cand; valid_shmem = need; break; }
                        }
                        if (valid_JT != 0) break;
                    }
                    if (valid_JT == 0) {
                        throw OwnerMexError("manor_m_phase2_mex:tiledShared",
                            "The validation grid needs more shared memory than "
                            "this device offers; reduce the eval grid.");
                    }
                }
                /* One kernel, two launch footprints: the opt-in maximum is a
                   property of the function, so it has to admit the larger of
                   the two or the wider launch is refused at launch time. */
                owner_cuda_check(cudaFuncSetAttribute(owner_fused_eval_tiled_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)std::max(tiled_main_shmem, valid_shmem)),
                    "tiled shared attribute");
            }
            d_eval_coordinate_sums.allocate(eval_slots * MY, "coordinate evaluation sums");
            d_eval_mse_history.allocate(eval_history_slots, "evaluation MSE history"); d_eval_cen_history.allocate(eval_history_slots, "evaluation center history");
            d_eval_coordinate_history.allocate(eval_coordinate_history_slots, "coordinate evaluation MSE history");
            d_eval_cen_tan_history.allocate(eval_history_slots, "evaluation tangential center history"); d_eval_tan_history.allocate(eval_history_slots, "evaluation tangent history");
        }
        if (fused_validation) {
            d_Xvalid.allocate_input((size_t)Nvalid * D, "X_Valid", mxGetData(prhs[15]));
            if (valid_tangent) {
                d_Tvalid.allocate_input((size_t)Nvalid * D * MM, "T_Valid", mxGetData(prhs[17]));
                d_valid_tan_history.allocate(eval_history_slots, "validation tangent history");
            }
            d_Yvalid.allocate_input((size_t)Nvalid * MY, "Y_Valid", mxGetData(prhs[16]));
            d_valid_mse_history.allocate(eval_history_slots, "validation MSE history");
            d_valid_coordinate_history.allocate(eval_coordinate_history_slots,
                "coordinate validation MSE history");
        }
        profile_span.begin("upload_init");
        owner_cuda_check(owner_upload_input(d_X.get(), X.data(), sizeof(float) * full_center), "upload X");
        owner_cuda_check(owner_upload_input(d_Y.get(), Y.data(), sizeof(float) * full_y), "upload Y");
        if (!opts.evaluation_only && !index_tiles.streamed && !index_tiles.generated)
            owner_cuda_check(owner_upload_input(d_idx.get(), idx_upload_host, sizeof(int) * index_count), "upload idx");
        owner_cuda_check(owner_upload_input(d_Tan_a.get(), Tan.data(), sizeof(float) * full_tan), "upload Tan");
        owner_cuda_check(owner_upload_input(d_Cen_a.get(), Cen.data(), sizeof(float) * full_center), "upload Cen");
        if (!opts.evaluation_only)
            owner_cuda_check(cudaMemcpy(d_owner_index.get(), owner_host.data(), sizeof(int) * M, cudaMemcpyHostToDevice), "upload owner indices");
        if (!opts.evaluation_only)
            owner_cuda_check(cudaMemcpy(d_owner_lookup.get(), owner_lookup_host.data(), sizeof(int) * N, cudaMemcpyHostToDevice), "upload owner lookup");
        owner_cuda_check(owner_upload_input(d_owner_mask.get(), owner_upload_host, sizeof(uint8_t) * N), "upload owner mask");
        if (fused_eval) {
            if (Ntest > 0) {
                owner_cuda_check(owner_upload_input(d_Xtest.get(), mxGetData(prhs[10]),
                    sizeof(float) * (size_t)Ntest * D),
                    "upload X_Test");
            }
            if (eval_opts.response_eval) {
                owner_cuda_check(owner_upload_input(d_Ytest.get(), mxGetData(prhs[11]), sizeof(float) * (size_t)Ntest * MY), "upload Y_Test");
            }
            if (eval_opts.tangent_metric_eval) {
                owner_cuda_check(owner_upload_input(d_Ttest.get(), mxGetData(prhs[12]), sizeof(float) * (size_t)Ntest * MM * D), "upload T_Test");
            }
            if (eval_opts.center_eval) {
                owner_cuda_check(owner_upload_input(d_Ctest.get(), mxGetData(prhs[13]), sizeof(float) * (size_t)Ntest * D), "upload C_Test");
            }
            owner_cuda_check(cudaMemcpy(d_eval_h_scale.get(), eval_opts.eval_h_scale.data(), sizeof(float) * eval_opts.eval_h_scale.size(), cudaMemcpyHostToDevice), "upload evaluation h scales");
            owner_cuda_check(cudaMemcpy(d_eval_rho_scale.get(), eval_opts.eval_rho_scale.data(), sizeof(float) * eval_opts.eval_rho_scale.size(), cudaMemcpyHostToDevice), "upload evaluation rho scales");
        }
        if (fused_validation) {
            owner_cuda_check(owner_upload_input(d_Xvalid.get(), mxGetData(prhs[15]),
                sizeof(float) * (size_t)Nvalid * D), "upload X_Valid");
            owner_cuda_check(owner_upload_input(d_Yvalid.get(), mxGetData(prhs[16]),
                sizeof(float) * (size_t)Nvalid * MY), "upload Y_Valid");
            if (valid_tangent)
                owner_cuda_check(owner_upload_input(d_Tvalid.get(), mxGetData(prhs[17]),
                    sizeof(float) * (size_t)Nvalid * D * MM), "upload T_Valid");
        }
        if (!opts.evaluation_only) {
            owner_cuda_check(cudaMemset(d_Tan_b.get(), 0, sizeof(float) * full_tan), "clear Tan next");
            owner_cuda_check(cudaMemset(d_Cen_b.get(), 0, sizeof(float) * full_center), "clear Cen next");
            owner_cuda_check(cudaMemset(d_heavy_last.get(), 0, sizeof(uint8_t) * N), "clear last heavy");
            owner_cuda_check(cudaMemset(d_keep_last.get(), 0, sizeof(uint8_t) * N), "clear last keep");
            owner_cuda_check(cudaMemset(d_num_last.get(), 0, sizeof(int32_t) * M), "clear last counts");
        }
        std::vector<int32_t> num_history_host(
            opts.return_histories ? history_nk : 0, 0);
        std::vector<uint8_t> heavy_history_host(
            opts.return_histories ? history_nk : 0, 0);
        std::vector<uint8_t> keep_history_host(
            opts.return_histories ? history_nk : 0, 0);
        owner_cuda_check(cudaMemset(d_stopped.get(), 0, sizeof(int)), "clear stopped");
        owner_cuda_check(cudaMemset(d_valid_iterations.get(), 0, sizeof(int)), "clear valid iterations");
        owner_cuda_check(cudaMemset(d_fused_eval_executed.get(), 0, sizeof(int)), "clear fused evaluation executed");
        if (fused_eval) {
            owner_cuda_check(cudaMemset(d_eval_mse_history.get(), 0, sizeof(float) * eval_history_slots), "clear evaluation MSE history");
            owner_cuda_check(cudaMemset(d_eval_coordinate_history.get(), 0,
                sizeof(float) * eval_coordinate_history_slots), "clear coordinate evaluation MSE history");
            owner_cuda_check(cudaMemset(d_eval_cen_history.get(), 0, sizeof(float) * eval_history_slots), "clear evaluation center history");
            owner_cuda_check(cudaMemset(d_eval_cen_tan_history.get(), 0, sizeof(float) * eval_history_slots), "clear evaluation tangential center history");
            owner_cuda_check(cudaMemset(d_eval_tan_history.get(), 0, sizeof(float) * eval_history_slots), "clear evaluation tangent history");
        }
        if (fused_validation) {
            if (valid_tangent)
                owner_cuda_check(cudaMemset(d_valid_tan_history.get(), 0,
                    sizeof(float) * eval_history_slots), "clear validation tangent history");
            owner_cuda_check(cudaMemset(d_valid_mse_history.get(), 0,
                sizeof(float) * eval_history_slots), "clear validation MSE history");
            owner_cuda_check(cudaMemset(d_valid_coordinate_history.get(), 0,
                sizeof(float) * eval_coordinate_history_slots),
                "clear coordinate validation MSE history");
        }
        profile_span.end();
        const double host_prep_ms = host_ms_since(host_entry_time);
        const auto host_loop_time = std::chrono::steady_clock::now();

        size_t shared_bytes = sizeof(float) * (
            (size_t)2 * D + (size_t)D * D + (size_t)D * MY);
#if MANOR_M_REGRESSION_SHARED_SOLVE
        shared_bytes = (shared_bytes + alignof(double) - 1) &
            ~(static_cast<size_t>(alignof(double)) - 1);
        const size_t regression_system = static_cast<size_t>(D + 1);
        shared_bytes = owner_checked_add(
            shared_bytes,
            owner_checked_mul(
                owner_checked_add(
                    owner_checked_mul(regression_system, regression_system,
                        "regression shared matrix"),
                    owner_checked_mul(regression_system, (size_t)MY,
                        "regression shared RHS"),
                    "regression shared solve"),
                sizeof(double), "regression shared solve bytes"),
            "regression dynamic shared bytes");
#endif
        size_t free_after = 0, total_after = 0;
        owner_cuda_check(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after allocation");
        const size_t query_bytes = std::max((size_t)1, shared_bytes + mask_words * sizeof(uint32_t));
        const int memory_cap = static_cast<int>(std::min<size_t>((size_t)M, free_after / query_bytes));
        const int BQ = std::max(1, std::min({M, opts.max_query_block, std::max(opts.min_query_block, memory_cap)}));
        const int threads = 256, blocks_m = (M + threads - 1) / threads, blocks_n = (N + threads - 1) / threads;
        long long profile_sparse_pairs = 0;
        float* d_tan_current = d_Tan_a.get(); float* d_cen_current = d_Cen_a.get();
        float* d_tan_next = d_Tan_b.get(); float* d_cen_next = d_Cen_b.get();
        const bool has_test_metrics = eval_opts.response_eval ||
            eval_opts.tangent_metric_eval || eval_opts.center_eval;
        /* Which query set a pass scores, and how. Everything else -- the
           training tiles, the slot grid, the schedule, the sparse fallback --
           is shared, so the test and validation passes differ only in this
           block and run through the identical kernel and the identical host
           regrouping below. */
        struct OwnerEvalTarget {
            const float* x; const float* y; const float* t; const float* c;
            int n; bool response_eval; bool tangent_metric_eval;
            bool center_eval; bool center_tangent_metric_eval;
            int jt; size_t shmem;
            const char* main_label; const char* pair_label;
            const char* reduce_label;
        };
        auto owner_run_single_metric_eval = [&](const OwnerEvalTarget& target,
                const uint8_t* mask,
                const float* cen_state, const float* tan_state,
                float hk_value, float rhok_value,
                float* sums_ptr, float* coordinate_ptr) {
            const int Nquery = target.n;
            if (Nquery <= 0 || !(target.response_eval ||
                    target.tangent_metric_eval || target.center_eval)) return;
            profile_span.begin(target.main_label);
            // Reused scratch is packed at the current query count, not its
            // capacity. Clear also the nullable/disabled metric channels.
            owner_cuda_check(cudaMemset(d_eval_points.get(), 0,
                sizeof(float) * eval_point_channels * (size_t)Nquery),
                "clear fixed-order evaluation points");
            owner_cuda_check(cudaMemset(d_pair_count.get(), 0, sizeof(int)),
                "clear sparse pair count");
            owner_fused_eval_tiled_kernel<<<(unsigned int)Nquery, 256, target.shmem>>>(
                target.x, target.y, target.t, target.c,
                d_X.get(), d_Y.get(), cen_state, tan_state,
                d_eval_xnorm.get(), d_eval_tdotx.get(), mask,
                D, MM, MY, N, Nquery, hk_value, rhok_value,
                d_eval_h_scale.get(), d_eval_rho_scale.get(),
                (int)eval_opts.eval_h_scale.size(), (int)eval_slots, target.jt,
                eval_opts.sum_sparse_threshold, eval_opts.count_sparse_threshold,
                eval_opts.fallback_exp_scale, eval_opts.eps_weight,
                eval_opts.center_eval_h_scale, eval_opts.center_eval_rho_scale,
                target.response_eval, target.tangent_metric_eval,
                target.center_eval, target.center_tangent_metric_eval,
                sums_ptr, coordinate_ptr,
                nullptr, nullptr, nullptr, nullptr,
                d_pair_count.get(), d_pair_list.get(), d_pair_minq.get(),
                d_stopped.get()
                , d_eval_points.get()
                );
            owner_cuda_check(cudaGetLastError(), "tiled eval launch");
            int sparse_pairs = 0;
            owner_cuda_check(cudaMemcpy(&sparse_pairs, d_pair_count.get(),
                sizeof(int), cudaMemcpyDeviceToHost), "download sparse pair count");
            profile_span.end();
            profile_sparse_pairs += sparse_pairs;
            if (sparse_pairs > 0) {
                profile_span.begin(target.pair_label);
                std::vector<int> pair_host((size_t)sparse_pairs * 2);
                std::vector<float> pair_minq_host((size_t)sparse_pairs * 2);
                owner_cuda_check(cudaMemcpy(pair_host.data(), d_pair_list.get(),
                    sizeof(int) * pair_host.size(), cudaMemcpyDeviceToHost),
                    "download sparse pair list");
                owner_cuda_check(cudaMemcpy(pair_minq_host.data(), d_pair_minq.get(),
                    sizeof(float) * pair_minq_host.size(), cudaMemcpyDeviceToHost),
                    "download sparse pair minq");
                /* Group pairs by test point (ascending; within a group the
                   append order is kept). Per-(test,slot) results do not
                   depend on the grouping, only the streaming reuse does. */
                std::vector<int> order((size_t)sparse_pairs);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                    [&](int left, int right) {
                        return pair_host[2 * (size_t)left] <
                            pair_host[2 * (size_t)right];
                    });
                std::vector<int> group_ib;
                std::vector<int> group_offsets;
                std::vector<int> slots_flat((size_t)sparse_pairs);
                std::vector<float> minq_flat((size_t)sparse_pairs * 2);
                for (int position = 0; position < sparse_pairs; ++position) {
                    const int pair_index = order[(size_t)position];
                    const int pair_test = pair_host[2 * (size_t)pair_index];
                    if (group_ib.empty() || group_ib.back() != pair_test) {
                        group_ib.push_back(pair_test);
                        group_offsets.push_back(position);
                    }
                    slots_flat[(size_t)position] =
                        pair_host[2 * (size_t)pair_index + 1];
                    minq_flat[2 * (size_t)position] =
                        pair_minq_host[2 * (size_t)pair_index];
                    minq_flat[2 * (size_t)position + 1] =
                        pair_minq_host[2 * (size_t)pair_index + 1];
                }
                group_offsets.push_back(sparse_pairs);
                const int group_count = (int)group_ib.size();
                std::vector<int> packed;
                packed.reserve(group_ib.size() + group_offsets.size() +
                    slots_flat.size());
                packed.insert(packed.end(), group_ib.begin(), group_ib.end());
                packed.insert(packed.end(), group_offsets.begin(),
                    group_offsets.end());
                packed.insert(packed.end(), slots_flat.begin(), slots_flat.end());
                owner_cuda_check(cudaMemcpy(d_pair_list.get(), packed.data(),
                    sizeof(int) * packed.size(), cudaMemcpyHostToDevice),
                    "upload grouped sparse pairs");
                owner_cuda_check(cudaMemcpy(d_pair_minq.get(), minq_flat.data(),
                    sizeof(float) * minq_flat.size(), cudaMemcpyHostToDevice),
                    "upload grouped sparse minq");
                const int* pair_ib_ptr = d_pair_list.get();
                const int* pair_offsets_ptr = pair_ib_ptr + group_count;
                const int* pair_slots_ptr = pair_offsets_ptr + group_count + 1;
                owner_fused_eval_tiled_kernel<<<(unsigned int)group_count, 256, target.shmem>>>(
                    target.x, target.y, target.t, target.c,
                    d_X.get(), d_Y.get(), cen_state, tan_state,
                    d_eval_xnorm.get(), d_eval_tdotx.get(), mask,
                    D, MM, MY, N, Nquery, hk_value, rhok_value,
                    d_eval_h_scale.get(), d_eval_rho_scale.get(),
                    (int)eval_opts.eval_h_scale.size(), (int)eval_slots, target.jt,
                    eval_opts.sum_sparse_threshold, eval_opts.count_sparse_threshold,
                    eval_opts.fallback_exp_scale, eval_opts.eps_weight,
                    eval_opts.center_eval_h_scale, eval_opts.center_eval_rho_scale,
                    target.response_eval, target.tangent_metric_eval,
                    target.center_eval, target.center_tangent_metric_eval,
                    sums_ptr, coordinate_ptr,
                    pair_ib_ptr, pair_offsets_ptr, pair_slots_ptr,
                    d_pair_minq.get(),
                    nullptr, nullptr, nullptr,
                    d_stopped.get()
                    , d_eval_points.get()
                    );
                owner_cuda_check(cudaGetLastError(), "tiled sparse eval launch");
                profile_span.end();
            }
            // Both sparse and ordinary writers have finished in this stream.
            // The tree and query assignment do not depend on their scheduling.
            profile_span.begin(target.reduce_label);
            manor_eval_fixed_order_sum_kernel<<<(unsigned int)eval_point_channels,
                manor_eval_sum_threads>>>(d_eval_points.get(), (size_t)Nquery,
                MY, sums_ptr, coordinate_ptr, d_stopped.get(),
                    !valid_tangent ? ~0u :
                    ((target.response_eval ? (1u | (((1u << MY) - 1u) << 4)) : 0u) |
                     (target.tangent_metric_eval ? 8u : 0u) |
                     (target.center_eval ? 2u : 0u) |
                     (target.center_tangent_metric_eval ? 4u : 0u)));
            owner_cuda_check(cudaGetLastError(), "fixed-order metric sum launch");
            profile_span.end();
        };
        const OwnerEvalTarget test_target{
            d_Xtest.get(), d_Ytest.get(), d_Ttest.get(), d_Ctest.get(),
            Ntest, eval_opts.response_eval, eval_opts.tangent_metric_eval,
            eval_opts.center_eval, eval_opts.center_tangent_metric_eval,
            tiled_JT, tiled_main_shmem,
            "eval_main", "eval_pair"
            , "eval_reduce"
            };
        /* Validation never reads T or C: it is MSE-only by construction, and
           the MSE-only tail returns before either is touched. */
        const OwnerEvalTarget valid_target{
            d_Xvalid.get(), d_Yvalid.get(), d_Tvalid.get(), nullptr,
            Nvalid, true, valid_tangent, false, false, valid_JT, valid_shmem,
            "valid_main", "valid_pair"
            , "valid_reduce"
            };
        // Preserve the separate response-only and tangent-only arithmetic
        // layouts used by the existing scientific records. Only model updates
        // are shared; each metric family keeps its own original tile width.
        OwnerEvalTarget test_response = test_target, test_geometry = test_target;
        OwnerEvalTarget valid_response = valid_target, valid_geometry = valid_target;
        if (valid_tangent) {
            int device = 0; cudaDeviceProp prop;
            owner_cuda_check(cudaGetDevice(&device), "shared readout device");
            owner_cuda_check(cudaGetDeviceProperties(&prop, device), "shared readout properties");
            auto specialize = [&](OwnerEvalTarget& target, bool response) {
                target.response_eval = response;
                target.tangent_metric_eval = !response && target.tangent_metric_eval;
                target.center_eval = !response && target.center_eval;
                target.center_tangent_metric_eval = !response && target.center_tangent_metric_eval;
                target.jt = 0;
                const size_t tiers[] = {
                    std::min((size_t)prop.sharedMemPerBlockOptin, (size_t)prop.sharedMemPerMultiprocessor / 3),
                    std::min((size_t)prop.sharedMemPerBlockOptin, (size_t)prop.sharedMemPerMultiprocessor / 2),
                    (size_t)prop.sharedMemPerBlockOptin};
                const int candidates[] = {128, 96, 64, 48, 32, 16};
                for (size_t budget : tiers) {
                    for (int jt : candidates) {
                        size_t bytes = owner_tiled_shared_bytes((int)eval_slots,
                            D, MM, MY, jt, 8, response,
                            target.tangent_metric_eval || target.center_eval, target.center_eval);
                        if (bytes <= budget) { target.jt = jt; target.shmem = bytes; break; }
                    }
                    if (target.jt) break;
                }
                if (!target.jt) throw OwnerMexError("manor_m_phase2_mex:tiledShared",
                    "Separate metric readout does not fit shared memory.");
            };
            specialize(test_response, true); specialize(test_geometry, false);
            specialize(valid_response, true); specialize(valid_geometry, false);
            owner_cuda_check(cudaFuncSetAttribute(owner_fused_eval_tiled_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)std::max({test_response.shmem, test_geometry.shmem,
                    valid_response.shmem, valid_geometry.shmem})), "separate readout shared attribute");
        }
        auto owner_run_tiled_eval = [&](const OwnerEvalTarget& target,
                const uint8_t* mask, const float* center, const float* tangent,
                float h, float rho, float* sums, float* coordinates) {
            if (valid_tangent && target.response_eval && target.tangent_metric_eval) {
                const bool validation = &target == &valid_target;
                owner_run_single_metric_eval(validation ? valid_response : test_response,
                    mask, center, tangent, h, rho, sums, coordinates);
                owner_run_single_metric_eval(validation ? valid_geometry : test_geometry,
                    mask, center, tangent, h, rho, sums, coordinates);
            } else owner_run_single_metric_eval(target, mask, center, tangent,
                h, rho, sums, coordinates);
        };
        /* K=0: the fused eval of the pre-iteration (Phase-1) state at the
           first-iteration bandwidths. Same kernel, same eval set, same
           metric as every K>=1 slot; owner mask because no keep mask
           exists yet. Raw sums go straight to the host (<=175 floats);
           d_eval_sums is reused and re-cleared by the loop evals. */
        std::vector<float> k0_sums, k0_coordinate_sums;
        std::vector<float> k0_valid_sums, k0_valid_coordinate_sums;
        if (fused_eval && eval_opts.eval_k0) {
            profile_span.begin("eval_stats");
            owner_cuda_check(cudaMemset(d_eval_sums.get(), 0, sizeof(float) * eval_slots * 4), "clear K0 evaluation sums");
            owner_cuda_check(cudaMemset(d_eval_coordinate_sums.get(), 0,
                sizeof(float) * eval_slots * MY), "clear K0 coordinate evaluation sums");
            owner_mark_fused_eval_executed_kernel<<<1, 1>>>(d_fused_eval_executed.get(), d_stopped.get());
            owner_fused_eval_train_stats_guarded_kernel<<<blocks_n, threads>>>(
                d_X.get(), d_tan_current, D, MM, N, d_eval_xnorm.get(), d_eval_tdotx.get(), d_stopped.get());
            profile_span.end();
            const float hk_0 = opts.hk_iter.empty() ? hk : opts.hk_iter[0];
            const float rhok_0 = rhok_iter[0];
            if (has_test_metrics) {
                owner_run_tiled_eval(test_target, d_owner_mask.get(), d_cen_current,
                    d_tan_current, hk_0, rhok_0, d_eval_sums.get(),
                    d_eval_coordinate_sums.get());
                k0_sums.resize(eval_slots * 4);
                k0_coordinate_sums.resize(eval_slots * (size_t)MY);
                owner_cuda_check(cudaMemcpy(k0_sums.data(), d_eval_sums.get(),
                    sizeof(float) * eval_slots * 4, cudaMemcpyDeviceToHost),
                    "download K0 evaluation sums");
                owner_cuda_check(cudaMemcpy(k0_coordinate_sums.data(),
                    d_eval_coordinate_sums.get(), sizeof(float) * eval_slots * MY,
                    cudaMemcpyDeviceToHost), "download K0 coordinate evaluation sums");
            }
            if (fused_validation) {
                /* Second pass over the same Phase-1 state and the same
                   train stats, which the first pass already wrote. */
                owner_cuda_check(cudaMemset(d_eval_sums.get(), 0, sizeof(float) * eval_slots * 4), "clear K0 validation sums");
                owner_cuda_check(cudaMemset(d_eval_coordinate_sums.get(), 0,
                    sizeof(float) * eval_slots * MY), "clear K0 coordinate validation sums");
                owner_run_tiled_eval(valid_target, d_owner_mask.get(), d_cen_current, d_tan_current,
                    hk_0, rhok_0, d_eval_sums.get(), d_eval_coordinate_sums.get());
                k0_valid_sums.resize(eval_slots * 4);
                k0_valid_coordinate_sums.resize(eval_slots * (size_t)MY);
                owner_cuda_check(cudaMemcpy(k0_valid_sums.data(), d_eval_sums.get(),
                    sizeof(float) * eval_slots * 4, cudaMemcpyDeviceToHost), "download K0 validation sums");
                owner_cuda_check(cudaMemcpy(k0_valid_coordinate_sums.data(), d_eval_coordinate_sums.get(),
                    sizeof(float) * eval_slots * MY, cudaMemcpyDeviceToHost), "download K0 coordinate validation sums");
            }
        }
        if (!opts.evaluation_only) for (int k = 0; k < K; ++k) {
            const float hk_k = opts.hk_iter.empty() ? hk : opts.hk_iter[opts.hk_iter.size() == 1 ? 0 : (size_t)k];
            const float rhok_k = rhok_iter[rhok_iter.size() == 1 ? 0 : (size_t)k];
            profile_span.begin("iter_setup");
            owner_cuda_check(cudaMemset(d_num_neigh.get(), 0, sizeof(int32_t) * M), "clear owner counts");
            owner_cuda_check(cudaMemset(d_grad.get(), 0, sizeof(float) * compact_gradient), "clear owner gradients");
            owner_cuda_check(cudaMemset(d_heavy.get(), 0, sizeof(uint8_t) * N), "clear heavy mask");
            owner_cuda_check(cudaMemset(d_keep.get(), 0, sizeof(uint8_t) * N), "clear keep mask");
            owner_cuda_check(cudaMemset(d_heavy_count.get(), 0, sizeof(int)), "clear heavy count");
            owner_cuda_check(cudaMemset(d_keep_count.get(), 0, sizeof(int)), "clear keep count");
            owner_dispatch_maxd(D, [&](auto tier) {
                constexpr int MAXD_T = decltype(tier)::value;
                owner_copy_state_kernel<MAXD_T><<<M, MAXD_T>>>(d_owner_index.get(), M, D, MM,
                    d_tan_current, d_cen_current, d_tan_next, d_cen_next);
            });
            profile_span.end();
            profile_span.begin("regression");
            owner_dispatch_maxd(D, [&](auto tier) {
                constexpr int MAXD_T = decltype(tier)::value;
                auto launch_regression = [&](const int* tile_idx,
                        const int* index_rows, int owner_first,
                        int owner_count) {
                    for (int q0 = owner_first;
                            q0 < owner_first + owner_count;) {
                        const int block_count = std::min(
                            BQ, owner_first + owner_count - q0);
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_REGRESSION_WARP_ENTRIES
                        constexpr int regression_threads =
                            (MAXD_T == 12 || MAXD_T == 20) ? 32 : MAXD_T;
#else
                        constexpr int regression_threads = MAXD_T;
#endif
                        owner_regression_kernel<MAXD_T><<<block_count, regression_threads, shared_bytes>>>(
                            d_X.get(), d_Y.get(), tile_idx, index_rows,
                            d_tan_current, d_cen_current, d_owner_index.get(),
                            q0, owner_first, block_count, D, MM, MY, kSlice,
                            opts.idx_shared_slice,
                            hk_k, rhok_k, opts.lambda, opts.jitter, opts.regression_mode,
                            neighbor_mask_tiled ? nullptr : d_neigh_mask.get(),
                            d_grad.get(), d_center_pre.get(), d_weight.get(), d_num_neigh.get(),
                            d_solver_failed.get(), d_stopped.get());
                        owner_cuda_check(cudaGetLastError(), "owner regression kernel launch");
                        q0 += block_count;
                    }
                };
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
                if (index_tiles.generated) {
                    owner_generated_phase2_for_each(
                        d_Y.get(), owner_host.data(), M, N, MY, kSlice,
                        [&](const int* generated_idx,
                                const int* generated_rows, int owner_first,
                                int owner_count) {
                            launch_regression(generated_idx, generated_rows,
                                owner_first, owner_count);
                        });
                } else
#endif
                for (int tile_q0 = 0; tile_q0 < M;) {
                    const int tile_M = index_tiles.streamed ?
                        index_tiles.size_at(tile_q0) : M;
                    const int* tile_idx = index_tiles.streamed ?
                        index_stream.prepare(idx_upload_host, tile_q0,
                            idx_owner_gather ? owner_host.data() : nullptr) :
                        d_idx.get();
                    launch_regression(tile_idx, nullptr, tile_q0, tile_M);
                    if (index_tiles.streamed) index_stream.consumed();
                    tile_q0 += tile_M;
                }
            });
            profile_span.end();
            profile_span.begin("mask_chain");
            owner_cuda_check(cudaMemset(d_count_sum.get(), 0, sizeof(unsigned long long)), "clear owner count sum");
            owner_count_sum_kernel<<<blocks_m, threads>>>(d_num_neigh.get(), M, d_count_sum.get(), d_stopped.get());
            owner_heavy_mask_kernel<<<blocks_m, threads>>>(d_owner_index.get(), d_num_neigh.get(), M, 2 * D,
                opts.heavy_factor, d_count_sum.get(), d_heavy.get(), d_heavy_count.get(), d_stopped.get());
            owner_stop_if_empty_kernel<<<1, 1>>>(d_heavy_count.get(), d_stopped.get());
            owner_record_valid_kernel<<<blocks_m, threads>>>(d_num_neigh.get(), d_num_last.get(), M,
                d_valid_iterations.get(), d_stopped.get());
            owner_copy_mask_kernel<<<blocks_n, threads>>>(d_heavy.get(), d_heavy_last.get(), N, d_stopped.get());
            owner_cuda_check(cudaMemset(d_keep_positive_sum.get(), 0, sizeof(unsigned long long)), "clear keep sum");
            owner_cuda_check(cudaMemset(d_keep_positive_count.get(), 0, sizeof(int)), "clear keep count");
            owner_keep_stats_kernel<<<blocks_m, threads>>>(d_owner_index.get(), d_num_neigh.get(), M, d_heavy.get(),
                d_keep_positive_sum.get(), d_keep_positive_count.get(), d_stopped.get());
            owner_keep_mask_kernel<<<blocks_m, threads>>>(d_owner_index.get(), d_num_neigh.get(), M, 2 * D,
                opts.heavy_factor_tmp, d_heavy.get(), d_keep_positive_sum.get(), d_keep_positive_count.get(),
                d_keep.get(), d_keep_count.get(), d_stopped.get());
            owner_copy_mask_kernel<<<blocks_n, threads>>>(d_keep.get(), d_keep_last.get(), N, d_stopped.get());
            profile_span.end();
            // Regression and the mask/stop chain completed for every owner.
            // Reuse only local scratch; each tile still reads global gradients
            // and the immutable old state and writes disjoint next-state rows.
            auto launch_aggregation = [&](const int* index_tile,
                    const int* index_rows, int owner_first,
                    int owner_count) {
            for (int q0 = owner_first; q0 < owner_first + owner_count;) {
                const int tile_M = std::min(aggregation_tiles.size_at(q0),
                    owner_first + owner_count - q0);
                const int* tile_owners = d_owner_index.get() + q0;
                const uint32_t* tile_mask = d_neigh_mask.get() +
                    (neighbor_mask_tiled ? 0 : (size_t)q0 * mask_words);
                const int* tile_idx = index_rows ? index_tile :
                    index_tile + OwnerAggregationTiles::index_offset(
                        q0 - owner_first, kSlice, opts.idx_shared_slice);
                const int* tile_rows = index_rows ?
                    index_rows + (q0 - owner_first) : nullptr;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_TILED_MASK
                if (neighbor_mask_tiled) {
                    profile_span.begin("mask_chain");
                    owner_dispatch_maxd(D, [&](auto tier) {
                        constexpr int MAXD_T = decltype(tier)::value;
#if MANOR_M_REGRESSION_WARP_ENTRIES
                        constexpr int mask_threads =
                            (MAXD_T == 12 || MAXD_T == 20) ? 32 : MAXD_T;
#else
                        constexpr int mask_threads = MAXD_T;
#endif
                        owner_neighbor_mask_rebuild_kernel<MAXD_T>
                            <<<tile_M, mask_threads>>>(
                                d_X.get(), tile_idx, tile_rows,
                                d_tan_current, d_owner_index.get(), q0, q0,
                                tile_M, D, MM, kSlice, opts.idx_shared_slice,
                                hk_k, rhok_k, d_neigh_mask.get(),
                                d_stopped.get());
                    });
                    owner_cuda_check(cudaGetLastError(),
                        "owner neighbor-mask rebuild kernel launch");
                    profile_span.end();
                }
#endif
                profile_span.begin("agg_accumulate");
#if defined(MANOR_M_PHASE2_CORE_LIBRARY) && MANOR_M_AGGREGATION_WARP_ENTRIES
            if (D == 10 || D == 20) {
                auto launch = [&](auto dimension) {
                    constexpr int ACTUAL_D = decltype(dimension)::value;
                    owner_aggregation_warp_accumulate_kernel<ACTUAL_D, 4><<<(static_cast<unsigned int>(tile_M) + 3u) / 4u, 128>>>(
                        d_grad.get(), d_center_pre.get(), tile_owners, d_owner_lookup.get(),
                        d_heavy.get(), tile_mask, tile_idx, tile_rows,
                        tile_M, D, MY, kSlice,
                        (int)mask_words, opts.idx_shared_slice, d_agg_count.get(),
                        d_agg_cacc.get(), d_agg_cov.get(), d_stopped.get());
                };
                if (D == 10) launch(std::integral_constant<int, 10>{});
                else launch(std::integral_constant<int, 20>{});
            } else
#endif
            owner_dispatch_maxd(D, [&](auto tier) {
                constexpr int MAXD_T = decltype(tier)::value;
                owner_aggregation_accumulate_kernel<MAXD_T><<<tile_M, MAXD_T>>>(d_grad.get(), d_center_pre.get(),
                    tile_owners, d_owner_lookup.get(), d_heavy.get(), tile_mask,
                    tile_idx, tile_rows, tile_M, D, MY,
                    kSlice, (int)mask_words, opts.idx_shared_slice, d_agg_count.get(),
                    d_agg_cacc.get(), d_agg_cov.get(), d_stopped.get());
            });
            profile_span.end();
            profile_span.begin("agg_finalize");
            owner_dispatch_maxd(D, [&](auto tier) {
                constexpr int MAXD_T = decltype(tier)::value;
                owner_aggregation_finalize_kernel<MAXD_T><<<tile_M, 32>>>(d_X.get(), d_tan_current, tile_owners, tile_M,
                    D, MM, d_agg_count.get(), d_agg_cacc.get(), d_agg_cov.get(), d_tan_next, d_cen_next, d_stopped.get());
            });
            owner_cuda_check(cudaGetLastError(), "owner aggregation kernel launch");
            profile_span.end();
                q0 += tile_M;
            }
            };
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
            if (index_tiles.generated) {
                owner_generated_phase2_for_each(
                    d_Y.get(), owner_host.data(), M, N, MY, kSlice,
                    [&](const int* generated_idx,
                            const int* generated_rows, int owner_first,
                            int owner_count) {
                        launch_aggregation(generated_idx, generated_rows,
                            owner_first, owner_count);
                    });
            } else
#endif
            for (int index_q0 = 0; index_q0 < M;) {
                const int index_M = index_tiles.streamed ?
                    index_tiles.size_at(index_q0) : M;
                const int* index_tile = index_tiles.streamed ?
                    index_stream.prepare(idx_upload_host, index_q0,
                        idx_owner_gather ? owner_host.data() : nullptr) : d_idx.get();
                launch_aggregation(index_tile, nullptr, index_q0, index_M);
                if (index_tiles.streamed) index_stream.consumed();
                index_q0 += index_M;
            }
            profile_span.begin("snapshots");
            if (opts.return_histories) {
                owner_cuda_check(cudaMemset(d_num_history.get(), 0,
                    sizeof(int32_t) * (size_t)N), "clear count history staging");
                owner_cuda_check(cudaMemset(d_heavy_history.get(), 0,
                    sizeof(uint8_t) * (size_t)N), "clear heavy history staging");
                owner_cuda_check(cudaMemset(d_keep_history.get(), 0,
                    sizeof(uint8_t) * (size_t)N), "clear keep history staging");
                owner_snapshot_counts_kernel<<<blocks_m, threads>>>(
                    d_owner_index.get(), d_num_neigh.get(), M, N, 0,
                    d_num_history.get(), d_stopped.get());
                owner_snapshot_mask_kernel<<<blocks_n, threads>>>(
                    d_heavy.get(), N, 0, d_heavy_history.get(), d_stopped.get());
                owner_snapshot_mask_kernel<<<blocks_n, threads>>>(
                    d_keep.get(), N, 0, d_keep_history.get(), d_stopped.get());
                const size_t offset = (size_t)k * (size_t)N;
                owner_cuda_check(cudaMemcpyAsync(num_history_host.data() + offset,
                    d_num_history.get(), sizeof(int32_t) * (size_t)N,
                    cudaMemcpyDeviceToHost), "stream count history to host");
                owner_cuda_check(cudaMemcpyAsync(heavy_history_host.data() + offset,
                    d_heavy_history.get(), sizeof(uint8_t) * (size_t)N,
                    cudaMemcpyDeviceToHost), "stream heavy history to host");
                owner_cuda_check(cudaMemcpyAsync(keep_history_host.data() + offset,
                    d_keep_history.get(), sizeof(uint8_t) * (size_t)N,
                    cudaMemcpyDeviceToHost), "stream keep history to host");
            }
            profile_span.end();
            if (fused_eval && fused_eval_should_run_k(eval_opts, k + 1)) {
                const uint8_t* eval_mask = opts.fused_eval_use_owner_mask ?
                    d_owner_mask.get() : d_keep.get();
                profile_span.begin("eval_stats");
                owner_cuda_check(cudaMemset(d_eval_sums.get(), 0, sizeof(float) * eval_slots * 4), "clear evaluation sums");
                owner_cuda_check(cudaMemset(d_eval_coordinate_sums.get(), 0,
                    sizeof(float) * eval_slots * MY), "clear coordinate evaluation sums");
                owner_mark_fused_eval_executed_kernel<<<1, 1>>>(d_fused_eval_executed.get(), d_stopped.get());
                owner_fused_eval_train_stats_guarded_kernel<<<blocks_n, threads>>>(
                    d_X.get(), d_tan_next, D, MM, N, d_eval_xnorm.get(), d_eval_tdotx.get(), d_stopped.get());
                profile_span.end();
                if (has_test_metrics) {
                    owner_run_tiled_eval(test_target, eval_mask, d_cen_next,
                        d_tan_next, hk_k, rhok_k, d_eval_sums.get(),
                        d_eval_coordinate_sums.get());
                    profile_span.begin("eval_store");
                    owner_store_eval_history_kernel<<<
                        (eval_slots + threads - 1) / threads, threads>>>(
                        d_eval_sums.get(), eval_slots,
                        (int)eval_opts.eval_h_scale.size(), K, k,
                        d_eval_mse_history.get(), d_eval_cen_history.get(),
                        d_eval_cen_tan_history.get(), d_eval_tan_history.get(),
                        d_stopped.get());
                    owner_store_eval_coordinate_history_kernel<<<
                        (eval_slots * MY + threads - 1) / threads, threads>>>(
                        d_eval_coordinate_sums.get(), eval_slots, MY,
                        (int)eval_opts.eval_h_scale.size(), K, k,
                        d_eval_coordinate_history.get(), d_stopped.get());
                    profile_span.end();
                }
                if (fused_validation) {
                    /* Same state, same mask, same bandwidths, different query
                       set. The sums buffers are reused because the test store
                       above has already consumed them. */
                    owner_cuda_check(cudaMemset(d_eval_sums.get(), 0, sizeof(float) * eval_slots * 4), "clear validation sums");
                    owner_cuda_check(cudaMemset(d_eval_coordinate_sums.get(), 0,
                        sizeof(float) * eval_slots * MY), "clear coordinate validation sums");
                    owner_run_tiled_eval(valid_target, eval_mask, d_cen_next, d_tan_next,
                        hk_k, rhok_k, d_eval_sums.get(), d_eval_coordinate_sums.get());
                    profile_span.begin("valid_store");
                    owner_store_valid_history_kernel<<<(eval_slots + threads - 1) / threads, threads>>>(
                        d_eval_sums.get(), eval_slots,
                        (int)eval_opts.eval_h_scale.size(), K, k,
                        d_valid_mse_history.get(), d_valid_tan_history.get(), d_stopped.get());
                    owner_store_eval_coordinate_history_kernel<<<
                        (eval_slots * MY + threads - 1) / threads, threads>>>(
                        d_eval_coordinate_sums.get(), eval_slots, MY,
                        (int)eval_opts.eval_h_scale.size(), K, k,
                        d_valid_coordinate_history.get(), d_stopped.get());
                    profile_span.end();
                }
            }
            std::swap(d_tan_current, d_tan_next); std::swap(d_cen_current, d_cen_next);
        }
        owner_cuda_check(cudaDeviceSynchronize(), "owner iteration synchronize");
        if (!opts.evaluation_only) index_stream.finish();
        const double device_loop_ms = host_ms_since(host_loop_time);
        if (fused_eval && eval_history_slots > 0) {
            const int history_blocks = static_cast<int>((eval_history_slots + threads - 1) / threads);
            const int coordinate_history_blocks = static_cast<int>(
                (eval_coordinate_history_slots + threads - 1) / threads);
            if (eval_opts.response_eval) {
                owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                    d_eval_mse_history.get(), eval_slots, K, Ntest,
                    d_valid_iterations.get());
                owner_scale_eval_history_kernel<<<coordinate_history_blocks, threads>>>(
                    d_eval_coordinate_history.get(), eval_slots * (size_t)MY,
                    K, Ntest, d_valid_iterations.get());
            }
            if (eval_opts.center_eval) {
                owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                    d_eval_cen_history.get(), eval_slots, K, Ntest,
                    d_valid_iterations.get());
            }
            if (eval_opts.center_tangent_metric_eval) {
                owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                    d_eval_cen_tan_history.get(), eval_slots, K, Ntest,
                    d_valid_iterations.get());
            }
            if (eval_opts.tangent_metric_eval) {
                owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                    d_eval_tan_history.get(), eval_slots, K, Ntest,
                    d_valid_iterations.get());
            }
            owner_cuda_check(cudaGetLastError(), "owner scale evaluation history launch");
            if (fused_validation) {
                /* Nvalid, not Ntest: the histories hold summed squared error
                   over their own query set, and the two sets differ in size. */
                owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                    d_valid_mse_history.get(), eval_slots, K, Nvalid, d_valid_iterations.get());
                owner_scale_eval_history_kernel<<<coordinate_history_blocks, threads>>>(
                    d_valid_coordinate_history.get(), eval_slots * (size_t)MY, K, Nvalid,
                    d_valid_iterations.get());
                if (valid_tangent)
                    owner_scale_eval_history_kernel<<<history_blocks, threads>>>(
                        d_valid_tan_history.get(), eval_slots, K, Nvalid,
                        d_valid_iterations.get());
                owner_cuda_check(cudaGetLastError(), "owner scale validation history launch");
            }
        }
        owner_cuda_check(cudaDeviceSynchronize(), "owner history synchronize");
        std::vector<std::pair<const char*, float>> profile_span_totals;
        if (profile_span.enabled) {
            for (size_t index = 0; index < profile_span.labels.size(); ++index) {
                float elapsed = 0.0f;
                owner_cuda_check(cudaEventElapsedTime(&elapsed,
                    profile_span.starts[index], profile_span.stops[index]),
                    "profile span elapsed");
                bool found = false;
                for (auto& entry : profile_span_totals) {
                    if (std::strcmp(entry.first, profile_span.labels[index]) == 0) {
                        entry.second += elapsed;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    profile_span_totals.emplace_back(
                        profile_span.labels[index], elapsed);
                }
            }
        }
        const auto host_output_time = std::chrono::steady_clock::now();
        int valid_iterations = 0, stopped_no_heavy = 0;
        std::vector<int32_t> compact_counts(opts.evaluation_only ? 0 : M);
        std::vector<uint8_t> compact_solver_failed(opts.evaluation_only ? 0 : M);
        bool export_gradient = !opts.evaluation_only;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
        export_gradient = !opts.evaluation_only &&
            (opts.return_debug || manor_m_minimex_output_bindings.empty());
#endif
        std::vector<uint8_t> heavy_last(opts.evaluation_only ? 0 : N);
        std::vector<uint8_t> keep_last(opts.evaluation_only ? 0 : N);
        std::vector<float> owner_grad(export_gradient ? compact_gradient : 0);
        owner_cuda_check(cudaMemcpy(&valid_iterations, d_valid_iterations.get(), sizeof(int), cudaMemcpyDeviceToHost), "download valid iterations");
        owner_cuda_check(cudaMemcpy(&stopped_no_heavy, d_stopped.get(), sizeof(int), cudaMemcpyDeviceToHost), "download stopped");
        int fused_eval_executed = 0;
        owner_cuda_check(cudaMemcpy(&fused_eval_executed, d_fused_eval_executed.get(), sizeof(int), cudaMemcpyDeviceToHost), "download fused evaluation executed");
        if (!opts.evaluation_only) {
            owner_cuda_check(cudaMemcpy(compact_counts.data(), d_num_last.get(), sizeof(int32_t) * M, cudaMemcpyDeviceToHost), "download last counts");
            owner_cuda_check(cudaMemcpy(heavy_last.data(), d_heavy_last.get(), sizeof(uint8_t) * N, cudaMemcpyDeviceToHost), "download last heavy");
            owner_cuda_check(cudaMemcpy(keep_last.data(), d_keep_last.get(), sizeof(uint8_t) * N, cudaMemcpyDeviceToHost), "download last keep");
            owner_cuda_check(cudaMemcpy(compact_solver_failed.data(), d_solver_failed.get(), sizeof(uint8_t) * M, cudaMemcpyDeviceToHost), "download solver status");
        }
        if (export_gradient)
            owner_cuda_check(cudaMemcpy(owner_grad.data(), d_grad.get(), sizeof(float) * compact_gradient, cudaMemcpyDeviceToHost), "download owner gradients");
        if (!opts.evaluation_only) {
            owner_cuda_check(cudaMemset(d_tan_next, 0, sizeof(float) * full_tan), "clear output tangent");
            owner_cuda_check(cudaMemset(d_cen_next, 0, sizeof(float) * full_center), "clear output center");
            owner_dispatch_maxd(D, [&](auto tier) {
                constexpr int MAXD_T = decltype(tier)::value;
                owner_copy_state_kernel<MAXD_T><<<M, MAXD_T>>>(d_owner_index.get(), M, D, MM,
                    d_tan_current, d_cen_current, d_tan_next, d_cen_next);
            });
            owner_cuda_check(cudaDeviceSynchronize(), "owner output synchronize");
        }

        const char* fields[] = {"Tan_next", "Cen_next", "num_neigh", "heavy_mask", "keep_mask", "heavy_mask_iter",
            "keep_mask_iter", "num_neigh_iter", "valid_iterations", "stopped_no_heavy", "fused_eval_executed", "owner_count", "solver_failed", "profile", "grad_debug",
            "EvalMSE", "EvalMSECoordinate", "EvalDevCen", "EvalDevCenTan", "EvalDevTan",
            "EvalMSE0", "EvalMSECoordinate0", "EvalDevCen0", "EvalDevCenTan0", "EvalDevTan0",
            "EvalValidMSE", "EvalValidMSECoordinate",
            "EvalValidMSE0", "EvalValidMSECoordinate0",
            "fused_validation_count", "EvalValidDevTan", "EvalValidDevTan0"};
        plhs[0] = mxCreateStructMatrix(1, 1, 32, fields);
        const mwSize state_output_n = opts.evaluation_only ? 0 : (mwSize)N;
        mwSize tan_output_dims[3] = {(mwSize)D, (mwSize)MM, state_output_n};
        mwSize cen_output_dims[2] = {(mwSize)D, state_output_n};
        mwSize vector_output_dims[2] = {1, state_output_n};
        mxArray* tan_out = owner_named_output("Tan_next", 3, tan_output_dims, mxSINGLE_CLASS, false);
        mxArray* cen_out = owner_named_output("Cen_next", 2, cen_output_dims, mxSINGLE_CLASS, false);
        mxArray* count_out = owner_named_output("num_neigh", 2, vector_output_dims, mxINT32_CLASS, true);
        mxArray* heavy_out = owner_named_output("heavy_mask", 2, vector_output_dims, mxLOGICAL_CLASS, false);
        mxArray* keep_out = owner_named_output("keep_mask", 2, vector_output_dims, mxLOGICAL_CLASS, false);
        mxArray* solver_out = owner_named_output("solver_failed", 2, vector_output_dims, mxLOGICAL_CLASS, true);
        if (!opts.evaluation_only) {
            owner_cuda_check(cudaMemcpy(mxGetData(tan_out), d_tan_next, sizeof(float) * full_tan, cudaMemcpyDeviceToHost), "download Tan next");
            owner_cuda_check(cudaMemcpy(mxGetData(cen_out), d_cen_next, sizeof(float) * full_center, cudaMemcpyDeviceToHost), "download Cen next");
            int32_t* count_data = static_cast<int32_t*>(mxGetData(count_out));
            for (int q = 0; q < M; ++q) { const int i = owner_host[q]; count_data[i] = compact_counts[q]; mxGetLogicals(solver_out)[i] = compact_solver_failed[q] != 0; }
            for (int i = 0; i < N; ++i) { mxGetLogicals(heavy_out)[i] = heavy_last[i] != 0; mxGetLogicals(keep_out)[i] = keep_last[i] != 0; }
        }
        mwSize iter_dims[2] = {
            opts.return_histories ? (mwSize)N : 0,
            opts.return_histories ? (mwSize)K : 0};
        mxArray* heavy_iter_out = owner_named_output("heavy_mask_iter", 2, iter_dims, mxLOGICAL_CLASS, false);
        mxArray* keep_iter_out = owner_named_output("keep_mask_iter", 2, iter_dims, mxLOGICAL_CLASS, false);
        mxArray* count_iter_out = owner_named_output("num_neigh_iter", 2, iter_dims, mxINT32_CLASS, false);
        if (opts.return_histories) {
            std::memcpy(mxGetData(count_iter_out), num_history_host.data(),
                sizeof(int32_t) * history_nk);
            std::memcpy(mxGetLogicals(heavy_iter_out), heavy_history_host.data(),
                sizeof(uint8_t) * history_nk);
            std::memcpy(mxGetLogicals(keep_iter_out), keep_history_host.data(),
                sizeof(uint8_t) * history_nk);
        }
        mxSetField(plhs[0], 0, "Tan_next", tan_out); mxSetField(plhs[0], 0, "Cen_next", cen_out);
        mxSetField(plhs[0], 0, "num_neigh", count_out); mxSetField(plhs[0], 0, "heavy_mask", heavy_out); mxSetField(plhs[0], 0, "keep_mask", keep_out);
        mxSetField(plhs[0], 0, "heavy_mask_iter", heavy_iter_out); mxSetField(plhs[0], 0, "keep_mask_iter", keep_iter_out); mxSetField(plhs[0], 0, "num_neigh_iter", count_iter_out);
        mxSetField(plhs[0], 0, "valid_iterations", mxCreateDoubleScalar((double)valid_iterations));
        mxSetField(plhs[0], 0, "stopped_no_heavy", mxCreateLogicalScalar(stopped_no_heavy != 0));
        mxSetField(plhs[0], 0, "fused_eval_executed", mxCreateDoubleScalar((double)fused_eval_executed));
        mxSetField(plhs[0], 0, "owner_count", mxCreateDoubleScalar((double)M)); mxSetField(plhs[0], 0, "solver_failed", solver_out);
        const char* profile_fields[] = {"BQ", "owner_count", "memory_required_bytes", "memory_usable_bytes", "cub_temp_bytes", "K", "history_bytes", "eval_block_threads",
            "rhok_schedule_length",
            "idx_shared_slice", "idx_compact_owner_order", "memory_index_bytes", "fused_eval_use_owner_mask",
            "mse_only_eval", "tangent_only_eval", "response_eval",
            "tangent_metric_eval", "center_eval", "center_tangent_metric_eval",
            "center_eval_h_scale", "center_eval_rho_scale", "async_allocator",
            "timer_labels", "timer_ms", "host_prep_ms",
            "device_loop_ms", "output_pack_ms", "sparse_pairs_total",
            "aggregation_tile_owners", "aggregation_tiles_per_iteration",
            "aggregation_scratch_bytes", "index_streamed", "index_generated",
            "index_logical_bytes",
            "index_pinned_bytes", "index_tile_owners", "index_tiles_per_pass",
            "index_buffer_count", "index_upload_calls", "index_h2d_bytes",
            "index_host_fill_ms", "index_host_reuse_wait_ms", "index_copy_ms"};
        mxArray* profile = mxCreateStructMatrix(1, 1, 43, profile_fields);
        mxSetField(profile, 0, "BQ", mxCreateDoubleScalar((double)BQ)); mxSetField(profile, 0, "owner_count", mxCreateDoubleScalar((double)M));
        mxSetField(profile, 0, "memory_required_bytes", mxCreateDoubleScalar((double)required)); mxSetField(profile, 0, "memory_usable_bytes", mxCreateDoubleScalar(usable));
        mxSetField(profile, 0, "cub_temp_bytes", mxCreateDoubleScalar(0)); mxSetField(profile, 0, "K", mxCreateDoubleScalar((double)K));
        mxSetField(profile, 0, "history_bytes", mxCreateDoubleScalar((double)history_bytes));
        mxSetField(profile, 0, "eval_block_threads", mxCreateDoubleScalar(fused_eval ? 64.0 : 0.0));
        mxSetField(profile, 0, "rhok_schedule_length", mxCreateDoubleScalar((double)rhok_iter.size()));
        mxSetField(profile, 0, "idx_shared_slice", mxCreateLogicalScalar(opts.idx_shared_slice));
        mxSetField(profile, 0, "idx_compact_owner_order",
            mxCreateLogicalScalar(opts.idx_compact_owner_order));
        mxSetField(profile, 0, "memory_index_bytes", mxCreateDoubleScalar((double)index_tiles.device_bytes));
        mxSetField(profile, 0, "index_streamed", mxCreateLogicalScalar(index_tiles.streamed));
        mxSetField(profile, 0, "index_generated", mxCreateLogicalScalar(index_tiles.generated));
        mxSetField(profile, 0, "index_logical_bytes", mxCreateDoubleScalar((double)index_tiles.logical_bytes));
        mxSetField(profile, 0, "index_pinned_bytes", mxCreateDoubleScalar(
            index_tiles.streamed ? (double)index_tiles.device_bytes : 0.0));
        mxSetField(profile, 0, "index_tile_owners", mxCreateDoubleScalar((double)index_tiles.capacity));
        mxSetField(profile, 0, "index_tiles_per_pass", mxCreateDoubleScalar((double)index_tiles.count()));
        mxSetField(profile, 0, "index_buffer_count", mxCreateDoubleScalar((double)index_tiles.buffers));
        mxSetField(profile, 0, "index_upload_calls", mxCreateDoubleScalar((double)index_stream.upload_calls));
        mxSetField(profile, 0, "index_h2d_bytes", mxCreateDoubleScalar((double)index_stream.host_to_device_bytes));
        mxSetField(profile, 0, "index_host_fill_ms", mxCreateDoubleScalar(index_stream.host_fill_milliseconds));
        mxSetField(profile, 0, "index_host_reuse_wait_ms", mxCreateDoubleScalar(index_stream.host_reuse_wait_milliseconds));
        mxSetField(profile, 0, "index_copy_ms", mxCreateDoubleScalar(index_stream.copy_milliseconds));
        mxSetField(profile, 0, "fused_eval_use_owner_mask",
            mxCreateLogicalScalar(opts.fused_eval_use_owner_mask));
        mxSetField(profile, 0, "mse_only_eval",
            mxCreateLogicalScalar(eval_opts.mse_only_eval));
        mxSetField(profile, 0, "tangent_only_eval",
            mxCreateLogicalScalar(eval_opts.tangent_only_eval));
        mxSetField(profile, 0, "response_eval",
            mxCreateLogicalScalar(eval_opts.response_eval));
        mxSetField(profile, 0, "tangent_metric_eval",
            mxCreateLogicalScalar(eval_opts.tangent_metric_eval));
        mxSetField(profile, 0, "center_eval",
            mxCreateLogicalScalar(eval_opts.center_eval));
        mxSetField(profile, 0, "center_tangent_metric_eval",
            mxCreateLogicalScalar(eval_opts.center_tangent_metric_eval));
        mxSetField(profile, 0, "center_eval_h_scale",
            mxCreateDoubleScalar((double)eval_opts.center_eval_h_scale));
        mxSetField(profile, 0, "center_eval_rho_scale",
            mxCreateDoubleScalar((double)eval_opts.center_eval_rho_scale));
        mxSetField(profile, 0, "async_allocator",
            mxCreateLogicalScalar(opts.async_allocator));
        mxSetField(profile, 0, "aggregation_tile_owners",
            mxCreateDoubleScalar((double)aggregation_tiles.capacity));
        mxSetField(profile, 0, "aggregation_tiles_per_iteration",
            mxCreateDoubleScalar((double)aggregation_tiles.count()));
        mxSetField(profile, 0, "aggregation_scratch_bytes", mxCreateDoubleScalar(
            (double)aggregation_tiles.capacity * sizeof(int) +
            (double)(aggregation_center + aggregation_cov) * sizeof(float)));
        {
            const mwSize span_count = (mwSize)profile_span_totals.size();
            mxArray* timer_labels = mxCreateCellMatrix(1, span_count);
            mxArray* timer_ms = mxCreateDoubleMatrix(1, span_count, mxREAL);
            for (mwSize index = 0; index < span_count; ++index) {
                mxSetCell(timer_labels, index,
                    mxCreateString(profile_span_totals[(size_t)index].first));
                mxGetPr(timer_ms)[index] =
                    (double)profile_span_totals[(size_t)index].second;
            }
            mxSetField(profile, 0, "timer_labels", timer_labels);
            mxSetField(profile, 0, "timer_ms", timer_ms);
            mxSetField(profile, 0, "host_prep_ms", mxCreateDoubleScalar(host_prep_ms));
            mxSetField(profile, 0, "device_loop_ms", mxCreateDoubleScalar(device_loop_ms));
            mxSetField(profile, 0, "output_pack_ms",
                mxCreateDoubleScalar(host_ms_since(host_output_time)));
            mxSetField(profile, 0, "sparse_pairs_total",
                mxCreateDoubleScalar((double)profile_sparse_pairs));
        }
        mxSetField(plhs[0], 0, "profile", profile);
        if (fused_eval) {
            const mwSize n_rho = (mwSize)eval_opts.eval_rho_scale.size(), n_h = (mwSize)eval_opts.eval_h_scale.size();
            mwSize eval_dims[3] = {(mwSize)K, n_rho, n_h};
            mwSize eval_coordinate_dims[4] = {(mwSize)K, n_rho, n_h, (mwSize)MY};
            mxArray* eval_mse = mxCreateNumericArray(3, eval_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* eval_coordinate = mxCreateNumericArray(4, eval_coordinate_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* eval_cen = mxCreateNumericArray(3, eval_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* eval_cen_tan = mxCreateNumericArray(3, eval_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* eval_tan = mxCreateNumericArray(3, eval_dims, mxSINGLE_CLASS, mxREAL);
            const float missing = std::numeric_limits<float>::quiet_NaN();
            if (eval_opts.response_eval) {
                owner_cuda_check(cudaMemcpy(mxGetData(eval_mse),
                    d_eval_mse_history.get(), sizeof(float) * eval_history_slots,
                    cudaMemcpyDeviceToHost), "download evaluation MSE history");
                owner_cuda_check(cudaMemcpy(mxGetData(eval_coordinate),
                    d_eval_coordinate_history.get(),
                    sizeof(float) * eval_coordinate_history_slots,
                    cudaMemcpyDeviceToHost),
                    "download coordinate evaluation MSE history");
            } else {
                std::fill_n(static_cast<float*>(mxGetData(eval_mse)),
                    eval_history_slots, missing);
                std::fill_n(static_cast<float*>(mxGetData(eval_coordinate)),
                    eval_coordinate_history_slots, missing);
            }
            if (eval_opts.center_eval)
                owner_cuda_check(cudaMemcpy(mxGetData(eval_cen),
                    d_eval_cen_history.get(), sizeof(float) * eval_history_slots,
                    cudaMemcpyDeviceToHost), "download evaluation center history");
            else
                std::fill_n(static_cast<float*>(mxGetData(eval_cen)),
                    eval_history_slots, missing);
            if (eval_opts.center_tangent_metric_eval)
                owner_cuda_check(cudaMemcpy(mxGetData(eval_cen_tan),
                    d_eval_cen_tan_history.get(), sizeof(float) * eval_history_slots,
                    cudaMemcpyDeviceToHost),
                    "download evaluation tangential center history");
            else
                std::fill_n(static_cast<float*>(mxGetData(eval_cen_tan)),
                    eval_history_slots, missing);
            if (eval_opts.tangent_metric_eval)
                owner_cuda_check(cudaMemcpy(mxGetData(eval_tan),
                    d_eval_tan_history.get(), sizeof(float) * eval_history_slots,
                    cudaMemcpyDeviceToHost), "download evaluation tangent history");
            else
                std::fill_n(static_cast<float*>(mxGetData(eval_tan)),
                    eval_history_slots, missing);
            mxSetField(plhs[0], 0, "EvalMSE", eval_mse);
            mxSetField(plhs[0], 0, "EvalMSECoordinate", eval_coordinate);
            mxSetField(plhs[0], 0, "EvalDevCen", eval_cen);
            mxSetField(plhs[0], 0, "EvalDevCenTan", eval_cen_tan); mxSetField(plhs[0], 0, "EvalDevTan", eval_tan);
        } else {
            mxSetField(plhs[0], 0, "EvalMSE", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalMSECoordinate", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevCen", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevCenTan", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevTan", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        }
        if (fused_eval && eval_opts.eval_k0 && !k0_sums.empty()) {
            const mwSize n_rho0 = (mwSize)eval_opts.eval_rho_scale.size();
            const mwSize n_h0 = (mwSize)eval_opts.eval_h_scale.size();
            const size_t n_h_size = (size_t)n_h0;
            mwSize k0_dims[3] = {1, n_rho0, n_h0};
            mwSize k0_coordinate_dims[4] = {1, n_rho0, n_h0, (mwSize)MY};
            mxArray* k0_mse = mxCreateNumericArray(3, k0_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* k0_coordinate = mxCreateNumericArray(4, k0_coordinate_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* k0_cen = mxCreateNumericArray(3, k0_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* k0_cen_tan = mxCreateNumericArray(3, k0_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* k0_tan = mxCreateNumericArray(3, k0_dims, mxSINGLE_CLASS, mxREAL);
            float* k0_mse_data = static_cast<float*>(mxGetData(k0_mse));
            float* k0_cen_data = static_cast<float*>(mxGetData(k0_cen));
            float* k0_cen_tan_data = static_cast<float*>(mxGetData(k0_cen_tan));
            float* k0_tan_data = static_cast<float*>(mxGetData(k0_tan));
            float* k0_coordinate_data = static_cast<float*>(mxGetData(k0_coordinate));
            const float k0_denominator = (float)std::max(Ntest, 1);
            const float k0_missing = std::numeric_limits<float>::quiet_NaN();
            for (size_t slot = 0; slot < eval_slots; ++slot) {
                const size_t rho_index = slot / n_h_size;
                const size_t h_index = slot - rho_index * n_h_size;
                const size_t dst = rho_index + (size_t)n_rho0 * h_index;
                k0_mse_data[dst] = eval_opts.response_eval ?
                    k0_sums[slot * 4] / k0_denominator : k0_missing;
                k0_cen_data[dst] = eval_opts.center_eval ?
                    k0_sums[slot * 4 + 1] / k0_denominator : k0_missing;
                k0_cen_tan_data[dst] = eval_opts.center_tangent_metric_eval ?
                    k0_sums[slot * 4 + 2] / k0_denominator : k0_missing;
                k0_tan_data[dst] = eval_opts.tangent_metric_eval ?
                    k0_sums[slot * 4 + 3] / k0_denominator : k0_missing;
            }
            for (size_t index = 0; index < eval_slots * (size_t)MY; ++index) {
                const size_t slot = index / (size_t)MY;
                const size_t output = index - slot * (size_t)MY;
                const size_t rho_index = slot / n_h_size;
                const size_t h_index = slot - rho_index * n_h_size;
                const size_t dst = rho_index + (size_t)n_rho0 *
                    (h_index + n_h_size * output);
                k0_coordinate_data[dst] = eval_opts.response_eval ?
                    k0_coordinate_sums[index] / k0_denominator : k0_missing;
            }
            mxSetField(plhs[0], 0, "EvalMSE0", k0_mse);
            mxSetField(plhs[0], 0, "EvalMSECoordinate0", k0_coordinate);
            mxSetField(plhs[0], 0, "EvalDevCen0", k0_cen);
            mxSetField(plhs[0], 0, "EvalDevCenTan0", k0_cen_tan);
            mxSetField(plhs[0], 0, "EvalDevTan0", k0_tan);
        } else {
            mxSetField(plhs[0], 0, "EvalMSE0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalMSECoordinate0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevCen0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevCenTan0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalDevTan0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        }
        /* Validation outputs. Only the two MSE families exist: the held-out
           points carry no truth geometry, so devCen/devTan have no meaning
           here and are not emitted rather than being emitted as NaN. */
        if (fused_validation) {
            const mwSize n_rho = (mwSize)eval_opts.eval_rho_scale.size();
            const mwSize n_h = (mwSize)eval_opts.eval_h_scale.size();
            mwSize valid_dims[3] = {(mwSize)K, n_rho, n_h};
            mwSize valid_coordinate_dims[4] = {(mwSize)K, n_rho, n_h, (mwSize)MY};
            mxArray* valid_mse = mxCreateNumericArray(3, valid_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* valid_coordinate = mxCreateNumericArray(4, valid_coordinate_dims, mxSINGLE_CLASS, mxREAL);
            owner_cuda_check(cudaMemcpy(mxGetData(valid_mse), d_valid_mse_history.get(),
                sizeof(float) * eval_history_slots, cudaMemcpyDeviceToHost),
                "download validation MSE history");
            owner_cuda_check(cudaMemcpy(mxGetData(valid_coordinate), d_valid_coordinate_history.get(),
                sizeof(float) * eval_coordinate_history_slots, cudaMemcpyDeviceToHost),
                "download coordinate validation MSE history");
            mxSetField(plhs[0], 0, "EvalValidMSE", valid_mse);
            mxSetField(plhs[0], 0, "EvalValidMSECoordinate", valid_coordinate);
        } else {
            mxSetField(plhs[0], 0, "EvalValidMSE", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalValidMSECoordinate", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        }
        if (fused_validation && !k0_valid_sums.empty()) {
            const mwSize n_rho0 = (mwSize)eval_opts.eval_rho_scale.size();
            const mwSize n_h0 = (mwSize)eval_opts.eval_h_scale.size();
            const size_t n_h_size = (size_t)n_h0;
            mwSize k0_dims[3] = {1, n_rho0, n_h0};
            mwSize k0_coordinate_dims[4] = {1, n_rho0, n_h0, (mwSize)MY};
            mxArray* k0_valid_mse = mxCreateNumericArray(3, k0_dims, mxSINGLE_CLASS, mxREAL);
            mxArray* k0_valid_coordinate = mxCreateNumericArray(4, k0_coordinate_dims, mxSINGLE_CLASS, mxREAL);
            float* k0_valid_mse_data = static_cast<float*>(mxGetData(k0_valid_mse));
            float* k0_valid_coordinate_data = static_cast<float*>(mxGetData(k0_valid_coordinate));
            const float k0_valid_denominator = (float)std::max(Nvalid, 1);
            for (size_t slot = 0; slot < eval_slots; ++slot) {
                const size_t rho_index = slot / n_h_size;
                const size_t h_index = slot - rho_index * n_h_size;
                const size_t dst = rho_index + (size_t)n_rho0 * h_index;
                k0_valid_mse_data[dst] = k0_valid_sums[slot * 4] / k0_valid_denominator;
            }
            for (size_t index = 0; index < eval_slots * (size_t)MY; ++index) {
                const size_t slot = index / (size_t)MY;
                const size_t output = index - slot * (size_t)MY;
                const size_t rho_index = slot / n_h_size;
                const size_t h_index = slot - rho_index * n_h_size;
                const size_t dst = rho_index + (size_t)n_rho0 *
                    (h_index + n_h_size * output);
                k0_valid_coordinate_data[dst] =
                    k0_valid_coordinate_sums[index] / k0_valid_denominator;
            }
            mxSetField(plhs[0], 0, "EvalValidMSE0", k0_valid_mse);
            mxSetField(plhs[0], 0, "EvalValidMSECoordinate0", k0_valid_coordinate);
        } else {
            mxSetField(plhs[0], 0, "EvalValidMSE0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalValidMSECoordinate0", mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        }
        if (valid_tangent) {
            mwSize dims[3] = {(mwSize)K, (mwSize)eval_opts.eval_rho_scale.size(),
                (mwSize)eval_opts.eval_h_scale.size()};
            mxArray* history = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
            owner_cuda_check(cudaMemcpy(mxGetData(history), d_valid_tan_history.get(),
                sizeof(float) * eval_history_slots, cudaMemcpyDeviceToHost),
                "download validation tangent history");
            mxSetField(plhs[0], 0, "EvalValidDevTan", history);
            if (!k0_valid_sums.empty()) {
                dims[0] = 1;
                mxArray* initial = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
                float* values = static_cast<float*>(mxGetData(initial));
                const size_t nh = eval_opts.eval_h_scale.size();
                const size_t nr = eval_opts.eval_rho_scale.size();
                for (size_t slot = 0; slot < eval_slots; ++slot)
                    values[slot / nh + nr * (slot % nh)] =
                        k0_valid_sums[slot * 4 + 3] / (float)Nvalid;
                mxSetField(plhs[0], 0, "EvalValidDevTan0", initial);
            } else mxSetField(plhs[0], 0, "EvalValidDevTan0",
                mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        } else {
            mxSetField(plhs[0], 0, "EvalValidDevTan",
                mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
            mxSetField(plhs[0], 0, "EvalValidDevTan0",
                mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL));
        }
        /* Point count, not an execution counter: fused_eval_executed already
           counts the iterations, and the validation pass runs in lockstep with
           it. What a caller cannot otherwise recover is how many points the
           reported validation MSE was averaged over. */
        mxSetField(plhs[0], 0, "fused_validation_count",
            mxCreateDoubleScalar(fused_validation ? (double)Nvalid : 0.0));
        if (opts.return_debug) {
            mwSize grad_dims[3] = {(mwSize)D, (mwSize)MY, (mwSize)N}; mxArray* grad_out = mxCreateNumericArray(3, grad_dims, mxSINGLE_CLASS, mxREAL);
            float* grad_data = static_cast<float*>(mxGetData(grad_out));
            for (int q = 0; q < M; ++q) for (int out = 0; out < MY; ++out) for (int d = 0; d < D; ++d)
                grad_data[(size_t)owner_host[q] * MY * D + (size_t)out * D + d] = owner_grad[(size_t)q * MY * D + (size_t)out * D + d];
            mxSetField(plhs[0], 0, "grad_debug", grad_out);
        }
        if (!opts.silent) mexPrintf("[phase2 owner] owners=%d/%d valid=%d stopped=%d BQ=%d rhoSchedule=%zu\n",
            M, N, valid_iterations, stopped_no_heavy, BQ, rhok_iter.size());
    } catch (const OwnerMexError& error) {
        mexErrMsgIdAndTxt(error.identifier(), "%s", error.what());
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("manor_m_phase2_mex:exception", "%s", error.what());
    }
}
