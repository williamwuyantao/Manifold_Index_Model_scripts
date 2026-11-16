/* MANOR CUDA adapter for the multiscale response-neighborhood and local-regression pipeline. */

#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
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
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MAX_D_MANOR
#define MAX_D_MANOR 48
#endif
#ifndef MAX_MM_MANOR
#define MAX_MM_MANOR 8
#endif
#ifndef MAX_R_MANOR
#define MAX_R_MANOR 256
#endif

struct Phase1Opts {
    float gpu_mem_fraction = 0.92f;
    float reserve_fraction = 0.08f;
    double reserve_bytes   = 6.0 * 1024.0 * 1024.0 * 1024.0;
    float heavy_factor     = 0.10f;
    float jitter           = 1e-6f;
    int   min_query_block  = 128;
    int   max_query_block  = 4096;
    bool  return_debug     = false;
    bool  gpu_pass2        = true;
    bool  profile_timers   = false;
    bool  verbose          = false;
    bool  silent           = false;
};

#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
static Phase1Opts parse_opts(const mxArray* opts)
{
    Phase1Opts o;
    if (!opts || !mxIsStruct(opts)) return o;
    auto getf = [&](const char* name) -> const mxArray* { return mxGetField(opts, 0, name); };
    if (auto a = getf("gpu_mem_fraction")) o.gpu_mem_fraction = (float)mxGetScalar(a);
    if (auto a = getf("reserve_fraction")) o.reserve_fraction = (float)mxGetScalar(a);
    if (auto a = getf("reserve_bytes"))    o.reserve_bytes    = mxGetScalar(a);
    if (auto a = getf("heavy_factor"))     o.heavy_factor     = (float)mxGetScalar(a);
    if (auto a = getf("jitter"))           o.jitter           = (float)mxGetScalar(a);
    if (auto a = getf("min_query_block"))  o.min_query_block  = (int)mxGetScalar(a);
    if (auto a = getf("max_query_block"))  o.max_query_block  = (int)mxGetScalar(a);
    if (auto a = getf("return_debug"))     o.return_debug = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (auto a = getf("gpu_pass2"))        o.gpu_pass2    = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (auto a = getf("profile_timers"))   o.profile_timers = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (auto a = getf("verbose"))          o.verbose = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    if (auto a = getf("silent"))           o.silent  = mxIsLogicalScalarTrue(a) || (mxGetScalar(a) != 0.0);
    return o;
}

inline bool is_single(const mxArray* a) { return mxIsSingle(a) && !mxIsComplex(a); }
inline bool is_double_real(const mxArray* a) { return mxIsDouble(a) && !mxIsComplex(a); }

static std::vector<float> to_float_vec(const mxArray* a) {
    mwSize n = mxGetNumberOfElements(a);
    std::vector<float> out(n);
    if (is_single(a)) {
        auto p = (const float*)mxGetData(a);
        std::copy(p, p + n, out.begin());
    } else if (is_double_real(a)) {
        auto p = mxGetPr(a);
        for (mwSize i = 0; i < n; ++i) out[i] = (float)p[i];
    } else {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:type", "Expected real single/double.");
    }
    return out;
}
#endif

__device__ __forceinline__ bool bitset_get_u32(const uint32_t* row, int pos)
{
    return ((row[pos >> 5] >> (pos & 31)) & uint32_t(1)) != 0u;
}

__device__ static inline float geom_score_from_evals_device(const float* evals, int D, int MM)
{
    int lo_den = min(max(MM - 1, 0), D - 1);
    int lo_num = min(max(MM,     0), D - 1);
    int hi_num = min(max(D - MM - 1, 0), D - 1);
    int hi_den = min(max(D - MM,     0), D - 1);

    float num = fmaxf(evals[lo_num] * evals[hi_num], 1.0e-20f);
    float den = fmaxf(evals[lo_den] * evals[hi_den], 1.0e-20f);
    return logf(fmaxf(num / den, 1.0e-20f));
}

/* Cyclic Jacobi sweep eigensolver: each sweep does D*(D-1)/2 sequential
   rotations, parallelized across 32 lanes within each rotation update.
   Convergence in ~10 sweeps for typical D=40 matrices, vs ~D^2 single
   rotations for the find-max strategy. */
__device__ static void shared_jacobi_eigen_device_warp(float* A, int D,
                                                       float* eigvals, float* eigvecs)
{
    const unsigned full_mask = 0xffffffffu;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < D * D; idx += 32) eigvecs[idx] = 0.0f;
    __syncwarp();
    for (int d = tid; d < D; d += 32) eigvecs[d * D + d] = 1.0f;
    __syncwarp();

    const int max_sweeps = 8;
    /* Trace-scaled stopping test for the Jacobi sweeps. */
    float local_tr = 0.0f;
    for (int d = tid; d < D; d += 32) local_tr += A[d * D + d];
    for (int o = 16; o > 0; o >>= 1)
        local_tr += __shfl_xor_sync(full_mask, local_tr, o);
    const float tol = 1.0e-7f * local_tr;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        /* off-diagonal L1 norm — cooperative reduction. */
        float local_off = 0.0f;
        for (int idx = tid; idx < D * D; idx += 32) {
            int i = idx / D;
            int j = idx - i * D;
            if (i < j) local_off += fabsf(A[idx]);
        }
        for (int o = 16; o > 0; o >>= 1)
            local_off += __shfl_xor_sync(full_mask, local_off, o);
        if (local_off < tol) break;

        /* Cyclic sweep: iterate (p,q) in row-major order. */
        for (int p = 0; p < D - 1; ++p) {
            for (int q = p + 1; q < D; ++q) {
                float apq = A[p * D + q];
                float app = A[p * D + p];
                float aqq = A[q * D + q];
                /* Volta and newer GPUs independently schedule warp lanes.
                   Complete every lane's reads before any lane can advance to
                   the skip path or overwrite these matrix entries. */
                __syncwarp(full_mask);
                if (fabsf(apq) < 1.0e-12f) continue;

                /* Every lane recomputes the rotation rather than lane 0
                   computing it and broadcasting.  Predicating 31 lanes off does
                   not make the instructions cheaper, so the redundancy is free,
                   and it removes three __shfl_sync from the innermost loop.
                   Same three shared reads, same instructions, every lane --
                   so the values are exactly what lane 0 used to broadcast.
                   The skip test is warp-uniform (denom comes from apq, which
                   all lanes read), so the whole warp continues together. */
                float denom = 2.0f * apq;
                if (fabsf(denom) < 1.0e-30f) continue;
                float tau = (aqq - app) / denom;
                float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                float c_rot = rsqrtf(1.0f + t * t);
                float s_rot = t * c_rot;

                /* Fused: row update + diag update + V update with single sync at end.
                   The diag update writes A[p,p],A[q,q],A[p,q] which are NOT touched
                   by the row update (which skips k=p,q).  The V update only reads V.
                   So all three operations are independent and need only one sync. */
                for (int k = tid; k < D; k += 32) {
                    if (k != p && k != q) {
                        float aik = A[p * D + k];
                        float aqk = A[q * D + k];
                        float new_pk = c_rot * aik - s_rot * aqk;
                        float new_qk = s_rot * aik + c_rot * aqk;
                        A[p * D + k] = new_pk;
                        A[k * D + p] = new_pk;
                        A[q * D + k] = new_qk;
                        A[k * D + q] = new_qk;
                    }
                    /* V update for column k. */
                    float vip = eigvecs[k * D + p];
                    float viq = eigvecs[k * D + q];
                    eigvecs[k * D + p] = c_rot * vip - s_rot * viq;
                    eigvecs[k * D + q] = s_rot * vip + c_rot * viq;
                }
                if (tid == 0) {
                    /* app and aqq are the values hoisted above: the row update
                       writes A[p][k] and A[q][k] only for k != p,q. */
                    A[p * D + p] = c_rot * c_rot * app - 2.0f * s_rot * c_rot * apq + s_rot * s_rot * aqq;
                    A[q * D + q] = s_rot * s_rot * app + 2.0f * s_rot * c_rot * apq + c_rot * c_rot * aqq;
                    A[p * D + q] = 0.0f;
                    A[q * D + p] = 0.0f;
                }
                __syncwarp();
            }
        }
    }

    for (int d = tid; d < D; d += 32) eigvals[d] = A[d * D + d];
    __syncwarp();

    if (tid == 0) {
        for (int i = 0; i < D - 1; ++i) {
            int best = i;
            for (int j = i + 1; j < D; ++j)
                if (eigvals[j] < eigvals[best]) best = j;
            if (best != i) {
                float tv = eigvals[i];
                eigvals[i] = eigvals[best];
                eigvals[best] = tv;
                for (int k = 0; k < D; ++k) {
                    float tmp = eigvecs[k * D + i];
                    eigvecs[k * D + i] = eigvecs[k * D + best];
                    eigvecs[k * D + best] = tmp;
                }
            }
        }
    }
    __syncwarp();
}

/* Brent-Luk round-robin (tournament) Jacobi, used only by stable_h_select_kernel.
   Generic in D: an odd D is padded with one dummy index (the largest), whose
   pair is the identity rotation, so the schedule is D rounds of (D-1)/2 real
   pairs; an even D gives D-1 rounds of D/2 pairs.  Within a round the pairs are
   index-disjoint, so their rotations commute and the whole round is applied as
   one two-sided 2x2 block update per (row-pair, column-pair).  Serial depth per
   sweep is D-1 rounds instead of D(D-1)/2 rotations, and the div/sqrt chain
   that sets the angle runs once per round instead of once per rotation. */
__device__ __forceinline__ void rr_pair_device(int r, int i, int n, int* p, int* q)
{
    int m = n - 1;
    int a, b;
    if (i == 0) { a = m; b = r; }
    else {
        a = r + i;     if (a >= m) a -= m;
        b = r - i + m; if (b >= m) b -= m;
    }
    *p = a < b ? a : b;
    *q = a < b ? b : a;
}

__device__ static void shared_jacobi_eigen_rr_device_warp(
    float* A, int D, float* eigvals, float* eigvecs, float* rot)
{
    const unsigned full_mask = 0xffffffffu;
    const int tid = threadIdx.x;
    const int n = D + (D & 1);
    const int P = n >> 1;
    const int rounds = n - 1;
    float* rot_c = rot;
    float* rot_s = rot + P;

    for (int idx = tid; idx < D * D; idx += 32) eigvecs[idx] = 0.0f;
    __syncwarp();
    for (int d = tid; d < D; d += 32) eigvecs[d * D + d] = 1.0f;
    __syncwarp();

    const int max_sweeps = 8;
    /* Trace-scaled stopping test; see shared_jacobi_eigen_device_warp above
       for the measurement.  local_off is an absolute sum of |A[i][j]|, so a
       fixed constant makes the exit depend on the units of A. */
    float local_tr = 0.0f;
    for (int d = tid; d < D; d += 32) local_tr += A[d * D + d];
    for (int o = 16; o > 0; o >>= 1)
        local_tr += __shfl_xor_sync(full_mask, local_tr, o);
    const float tol = 1.0e-7f * local_tr;
    const int a_tasks = P * P;
    const int total_tasks = a_tasks + D * P;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        float local_off = 0.0f;
        for (int idx = tid; idx < D * D; idx += 32) {
            int i = idx / D;
            int j = idx - i * D;
            if (i < j) local_off += fabsf(A[idx]);
        }
        for (int o = 16; o > 0; o >>= 1)
            local_off += __shfl_xor_sync(full_mask, local_off, o);
        if (local_off < tol) break;

        for (int r = 0; r < rounds; ++r) {
            for (int i = tid; i < P; i += 32) {
                int p, q;
                rr_pair_device(r, i, n, &p, &q);
                float c_rot = 1.0f;
                float s_rot = 0.0f;
                if (q < D) {
                    float apq = A[p * D + q];
                    float denom = 2.0f * apq;
                    if (fabsf(apq) >= 1.0e-12f && fabsf(denom) >= 1.0e-30f) {
                        float app = A[p * D + p];
                        float aqq = A[q * D + q];
                        float tau = (aqq - app) / denom;
                        float t = (tau >= 0.0f)
                                ? 1.0f / (tau + sqrtf(1.0f + tau * tau))
                                : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                        c_rot = rsqrtf(1.0f + t * t);
                        s_rot = t * c_rot;
                    }
                }
                rot_c[i] = c_rot;
                rot_s[i] = s_rot;
            }
            __syncwarp();

            /* The P*P blocks partition A exactly, so a lane's four elements are
               touched by no other lane and the round needs no interior sync.
               Only the upper block triangle computes; it mirrors its result into
               the lower one, which keeps A bitwise symmetric the way the cyclic
               version's paired writes did. */
            for (int tsk = tid; tsk < total_tasks; tsk += 32) {
                if (tsk < a_tasks) {
                    int rp = tsk / P;
                    int cp = tsk - rp * P;
                    if (rp > cp) continue;
                    int p, q, p2, q2;
                    rr_pair_device(r, rp, n, &p, &q);
                    rr_pair_device(r, cp, n, &p2, &q2);
                    float c1 = rot_c[rp];
                    float s1 = rot_s[rp];
                    bool qok = (q < D);
                    bool q2ok = (q2 < D);
                    if (rp == cp) {
                        if (!qok) continue;
                        float app = A[p * D + p];
                        float aqq = A[q * D + q];
                        float apq = A[p * D + q];
                        A[p * D + p] = c1 * c1 * app - 2.0f * s1 * c1 * apq + s1 * s1 * aqq;
                        A[q * D + q] = s1 * s1 * app + 2.0f * s1 * c1 * apq + c1 * c1 * aqq;
                        A[p * D + q] = 0.0f;
                        A[q * D + p] = 0.0f;
                    } else {
                        float c2 = rot_c[cp];
                        float s2 = rot_s[cp];
                        float b00 = A[p * D + p2];
                        float b01 = q2ok ? A[p * D + q2] : 0.0f;
                        float b10 = qok ? A[q * D + p2] : 0.0f;
                        float b11 = (qok && q2ok) ? A[q * D + q2] : 0.0f;
                        float m00 = c1 * b00 - s1 * b10;
                        float m01 = c1 * b01 - s1 * b11;
                        float m10 = s1 * b00 + c1 * b10;
                        float m11 = s1 * b01 + c1 * b11;
                        float w00 = c2 * m00 - s2 * m01;
                        float w01 = s2 * m00 + c2 * m01;
                        float w10 = c2 * m10 - s2 * m11;
                        float w11 = s2 * m10 + c2 * m11;
                        A[p * D + p2] = w00;
                        A[p2 * D + p] = w00;
                        if (q2ok) { A[p * D + q2] = w01; A[q2 * D + p] = w01; }
                        if (qok)  { A[q * D + p2] = w10; A[p2 * D + q] = w10; }
                        if (qok && q2ok) { A[q * D + q2] = w11; A[q2 * D + q] = w11; }
                    }
                } else {
                    int v = tsk - a_tasks;
                    int k = v / P;
                    int pr = v - k * P;
                    int p, q;
                    rr_pair_device(r, pr, n, &p, &q);
                    if (q >= D) continue;
                    float c1 = rot_c[pr];
                    float s1 = rot_s[pr];
                    float vip = eigvecs[k * D + p];
                    float viq = eigvecs[k * D + q];
                    eigvecs[k * D + p] = c1 * vip - s1 * viq;
                    eigvecs[k * D + q] = s1 * vip + c1 * viq;
                }
            }
            __syncwarp();
        }
    }

    for (int d = tid; d < D; d += 32) eigvals[d] = A[d * D + d];
    __syncwarp();

    if (tid == 0) {
        for (int i = 0; i < D - 1; ++i) {
            int best = i;
            for (int j = i + 1; j < D; ++j)
                if (eigvals[j] < eigvals[best]) best = j;
            if (best != i) {
                float tv = eigvals[i];
                eigvals[i] = eigvals[best];
                eigvals[best] = tv;
                for (int k = 0; k < D; ++k) {
                    float tmp = eigvecs[k * D + i];
                    eigvecs[k * D + i] = eigvecs[k * D + best];
                    eigvecs[k * D + best] = tmp;
                }
            }
        }
    }
    __syncwarp();
}

/* ================================================================
   KERNEL 2: Compact top-tmax sorted entries from full sort output.
   ================================================================ */
template <typename Rank>
__global__ void compact_topk_kernel(
    const float*    __restrict__ sorted_keys,
    const Rank* __restrict__ sorted_vals,
    int B, int kSlice, int tmax,
    float*    __restrict__ top_keys,
    Rank* __restrict__ top_vals)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * tmax;
    if (tid >= total) return;
    int b = tid / tmax;
    int p = tid - b * tmax;
    top_keys[tid] = sorted_keys[(size_t)b * kSlice + p];
    top_vals[tid] = sorted_vals[(size_t)b * kSlice + p];
}

#include "manor_m_h_null_table.h"

#include <stdexcept>

class StableMexError : public std::runtime_error {
public:
    StableMexError(const char* identifier, const std::string& message)
        : std::runtime_error(message), identifier_(identifier) {}

    const char* identifier() const noexcept { return identifier_.c_str(); }

private:
    std::string identifier_;
};

static void cuda_check(cudaError_t status, const char* file, int line)
{
    if (status == cudaSuccess) return;
    std::string message = cudaGetErrorString(status);
    message += " (";
    message += file;
    message += ":";
    message += std::to_string(line);
    message += ")";
    throw StableMexError("manor_m_phase1_mex:cuda", message);
}

#define STABLE_CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

template <typename T>
class StableCudaBuffer {
public:
    StableCudaBuffer() = default;
    ~StableCudaBuffer() { if (pointer_) cudaFree(pointer_); }

    StableCudaBuffer(const StableCudaBuffer&) = delete;
    StableCudaBuffer& operator=(const StableCudaBuffer&) = delete;

    T* allocate(size_t count)
    {
        if (pointer_)
            throw StableMexError("manor_m_phase1_mex:cudaOwnership",
                "CUDA buffer was allocated more than once.");
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw StableMexError("manor_m_phase1_mex:integerOverflow",
                "CUDA allocation size overflow.");
        if (count == 0) return nullptr;
        STABLE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_),
            sizeof(T) * count));
        return pointer_;
    }

private:
    T* pointer_ = nullptr;
};

struct StablePhase1Opts {
    Phase1Opts memory;
    double heavy_factor = 0.10;
    float jitter = 1.0e-6f;
    int window = 7;
    float neighbor_scaled_max = 1.0f;
    float split_scaled_max = 6.0f;
    uint64_t split_seed = 730021ULL;
    bool idx_zero_based = false;
    bool idx_shared_slice = false;

    /* Kept opaque here so the unchanged kernels can also be compiled into the
       language-neutral core without importing MATLAB headers.  Only the MEX
       parser/entry point interprets this as an mxArray. */
    const void* response_Y = nullptr;
    int response_kslice = 0;
    bool return_response_index = false;
    double cut_max_fraction = 0.8;
    int selection_mode = 0;
    int active_anchor_count = -1;
    bool h_null_correction = false;
};

/* Stable-H null-offset correction.
   The correction is applied to cross-cut score comparisons; the H-sign
   branch and eligibility gate use the raw score. */

static const HNullCurve* h_null_find_curve(int D, int MM)
{
    for (int index = 0; index < H_NULL_TABLE_COUNT; ++index) {
        if (H_NULL_TABLE[index].D == D && H_NULL_TABLE[index].MM == MM)
            return &H_NULL_TABLE[index];
    }
    return nullptr;
}

/* A = q * H_null interpolated linearly in log q, clamped at both ends.
   Clamping the amplitude rather than the offset keeps the out-of-range decay
   at the correct A/q rate instead of freezing the offset at a finite value. */
static double h_null_offset(const HNullCurve& curve, double q)
{
    if (!(q > 0.0)) return 0.0;
    double log_q = std::log(q);
    int last = curve.knot_count - 1;
    double amplitude;
    if (log_q <= curve.log_q[0]) {
        amplitude = curve.amplitude[0];
    } else if (log_q >= curve.log_q[last]) {
        amplitude = curve.amplitude[last];
    } else {
        int low = 0;
        int high = last;
        while (high - low > 1) {
            int middle = (low + high) / 2;
            if (curve.log_q[middle] <= log_q) low = middle; else high = middle;
        }
        double span = curve.log_q[high] - curve.log_q[low];
        double t = span > 0.0 ? (log_q - curve.log_q[low]) / span : 0.0;
        amplitude = curve.amplitude[low] +
            t * (curve.amplitude[high] - curve.amplitude[low]);
    }
    return amplitude / q;
}

/* Require a calibration curve for the requested (D,MM). */
static void h_null_fill_offsets(
    int D, int MM, const int32_t* cuts, int count, float* out,
    const char* identifier)
{
    const HNullCurve* curve = h_null_find_curve(D, MM);
    if (!curve) {
        std::string known;
        for (int index = 0; index < H_NULL_TABLE_COUNT; ++index) {
            known += (index ? ", (" : "(");
            known += std::to_string(H_NULL_TABLE[index].D);
            known += ",";
            known += std::to_string(H_NULL_TABLE[index].MM);
            known += ")";
        }
#if defined(MANOR_M_PHASE1_CORE_LIBRARY)
        std::string message =
            "opts.h_null_correction is on but no Stable-H null-offset "
            "calibration exists for (D,MM)=(" + std::to_string(D) + "," +
            std::to_string(MM) + "). Calibrated (D,MM): " + known +
            ". Extend the calibration table or leave opts.h_null_correction off.";
        throw StableMexError(identifier, message);
#else
        mexErrMsgIdAndTxt(identifier,
            "opts.h_null_correction is on but no Stable-H null-offset "
            "calibration exists for (D,MM)=(%d,%d). Calibrated (D,MM): %s. "
            "Extend the calibration table or leave opts.h_null_correction off.",
            D, MM, known.c_str());
#endif
    }
    for (int index = 0; index < count; ++index)
        out[index] = static_cast<float>(
            h_null_offset(*curve, static_cast<double>(cuts[index])));
}


enum StableSelectionMode {
    STABLE_POSITIVE_H = 0,
    TOP5_ABS_H_MEDIAN_CUT = 2,
    /* Internal shared-core mode.  It is deliberately not accepted by the
       public MATLAB option parser: the hybrid C ABI uses it to emit the
       primary and fallback decisions from one pass over the same cut ladder. */
    HYBRID_STABLE_AND_TOP5 = 3
};

static int checked_product_int(int left, int right, const char* label)
{
    if (left < 0 || right < 0 ||
        (right != 0 && left > std::numeric_limits<int>::max() / right)) {
        throw StableMexError("manor_m_phase1_mex:integerOverflow",
            std::string(label) + " exceeds INT_MAX.");
    }
    return left * right;
}

static size_t checked_add_size(
    size_t left, size_t right, const char* label)
{
    if (left > std::numeric_limits<size_t>::max() - right) {
        throw StableMexError("manor_m_phase1_mex:integerOverflow",
            std::string(label) + " byte sum exceeds SIZE_MAX.");
    }
    return left + right;
}

static size_t checked_mul_size(
    size_t left, size_t right, const char* label)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw StableMexError("manor_m_phase1_mex:integerOverflow",
            std::string(label) + " byte product exceeds SIZE_MAX.");
    }
    return left * right;
}

static size_t stable_mask_words(int kSlice)
{
    return (static_cast<size_t>(kSlice) + 31u) / 32u;
}

struct StableQueryMemory {
    size_t sort_payload_bytes = 0;
    size_t sort_temp_bytes = 0;
    size_t aggregate_scratch_bytes = 0;
    size_t total_bytes = 0;
};

struct StableMemoryBudget {
    int BQ = 0;
    int upper_bound = 0;
    int grid_y_limit = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    double reserve_bytes = 0.0;
    size_t usable_bytes = 0;
    size_t fixed_input_bytes = 0;
    size_t index_bytes = 0;
    size_t fixed_estimator_bytes = 0;
    size_t fixed_mask_bytes = 0;
    size_t fixed_diagnostic_bytes = 0;
    size_t fixed_bytes = 0;
    StableQueryMemory selected_query;
    size_t required_bytes = 0;
    size_t next_required_bytes = 0;
};

template <typename Rank>
static size_t compute_sort_temp_bytes_for_rank(int BQ, int kSlice)
{
    int item_count = checked_product_int(BQ, kSlice, "BQ*kSlice");
    size_t bytes = 0;
    const float* keys_in = nullptr;
    float* keys_out = nullptr;
    const Rank* values_in = nullptr;
    Rank* values_out = nullptr;
    const int* offsets = nullptr;
    STABLE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, bytes, keys_in, keys_out, values_in, values_out,
        item_count, BQ, offsets, offsets));
    return bytes;
}

static size_t compute_sort_temp_bytes(int BQ, int kSlice)
{
    return kSlice <= static_cast<int>(std::numeric_limits<uint16_t>::max())
        ? compute_sort_temp_bytes_for_rank<uint16_t>(BQ, kSlice)
        : compute_sort_temp_bytes_for_rank<uint32_t>(BQ, kSlice);
}

static StableQueryMemory stable_query_memory(
    int BQ, int D, int kSlice, int tmax)
{
    StableQueryMemory result;
    size_t block = static_cast<size_t>(BQ);
    size_t item_count = checked_mul_size(
        block, static_cast<size_t>(kSlice), "sort item count");
    size_t top_count = checked_mul_size(
        block, static_cast<size_t>(tmax), "top item count");
    size_t key_bytes = checked_mul_size(
        checked_mul_size(item_count, 2, "sort key buffers"),
        sizeof(float), "sort key bytes");
    const size_t rank_bytes = kSlice <=
        static_cast<int>(std::numeric_limits<uint16_t>::max())
        ? sizeof(uint16_t) : sizeof(uint32_t);
    size_t value_bytes = checked_mul_size(
        checked_mul_size(item_count, 2, "sort value buffers"),
        rank_bytes, "sort value bytes");
    size_t top_key_bytes = checked_mul_size(
        top_count, sizeof(float), "top key bytes");
    size_t top_value_bytes = checked_mul_size(
        top_count, rank_bytes, "top value bytes");
    size_t offset_bytes = checked_mul_size(
        checked_add_size(block, 1, "segment offset count"),
        sizeof(int), "segment offset bytes");
    result.sort_payload_bytes = checked_add_size(
        checked_add_size(key_bytes, value_bytes, "sort key/value bytes"),
        checked_add_size(
            checked_add_size(top_key_bytes, top_value_bytes,
                "top key/value bytes"),
            offset_bytes, "top/offset bytes"),
        "sort payload bytes");
    result.sort_temp_bytes = compute_sort_temp_bytes(BQ, kSlice);

    size_t covariance_count = checked_mul_size(
        block,
        checked_mul_size(static_cast<size_t>(D), D,
            "aggregate covariance elements per query"),
        "aggregate covariance elements");
    size_t center_count = checked_mul_size(
        block, static_cast<size_t>(D), "aggregate center elements");
    result.aggregate_scratch_bytes = checked_add_size(
        checked_add_size(
            checked_mul_size(covariance_count, sizeof(float),
                "aggregate covariance bytes"),
            checked_mul_size(center_count, sizeof(float),
                "aggregate center bytes"),
            "aggregate covariance/center bytes"),
        checked_mul_size(block, sizeof(int32_t),
            "aggregate count bytes"),
        "aggregate scratch bytes");
    result.total_bytes = checked_add_size(
        checked_add_size(result.sort_payload_bytes,
            result.sort_temp_bytes, "sort payload/temp bytes"),
        result.aggregate_scratch_bytes, "query workspace bytes");
    return result;
}

static StableMemoryBudget stable_choose_query_block(
    int D, int MM, int N, int active_anchor_count, int kSlice, int tmax, int R,
    bool idx_shared_slice, const Phase1Opts& opts)
{
    StableMemoryBudget budget;
    int active_device = 0;
    STABLE_CUDA_CHECK(cudaGetDevice(&active_device));
    cudaDeviceProp properties{};
    STABLE_CUDA_CHECK(cudaGetDeviceProperties(&properties, active_device));
    budget.grid_y_limit = properties.maxGridSize[1];
    if (budget.grid_y_limit < 1)
        throw StableMexError("manor_m_phase1_mex:queryBlock",
            "The active CUDA device reports an invalid grid-y limit.");

    STABLE_CUDA_CHECK(cudaMemGetInfo(&budget.free_bytes, &budget.total_bytes));

    int integer_cap = std::numeric_limits<int>::max() / kSlice;
    budget.upper_bound = std::min(active_anchor_count, std::min(opts.max_query_block,
        std::min(integer_cap, budget.grid_y_limit)));
    if (budget.upper_bound < 1)
        throw StableMexError("manor_m_phase1_mex:queryBlock",
            "No valid query block satisfies active_anchor_count, max_query_block, "
            "INT_MAX/kSlice, and the CUDA grid-y limit.");

    size_t n = static_cast<size_t>(N);
    size_t d = static_cast<size_t>(D);
    size_t mm = static_cast<size_t>(MM);
    size_t slice = static_cast<size_t>(kSlice);
    size_t mask_words = stable_mask_words(kSlice);
    size_t point_count = checked_mul_size(d, n, "X element count");
    size_t index_count = idx_shared_slice ? slice :
        checked_mul_size(static_cast<size_t>(active_anchor_count), slice,
            "active index element count");
    budget.index_bytes = checked_mul_size(
        index_count, sizeof(int), "index bytes");
    budget.fixed_input_bytes = checked_add_size(
        checked_add_size(
            checked_mul_size(point_count, sizeof(float), "X bytes"),
            budget.index_bytes,
            "X/index bytes"),
        checked_mul_size(static_cast<size_t>(R), sizeof(int), "cut bytes"),
        "fixed input bytes");

    size_t tangent_per_point = checked_mul_size(d, mm,
        "tangent elements per point");
    size_t estimator_per_point = checked_add_size(
        tangent_per_point, d, "tangent/center elements per point");
    budget.fixed_estimator_bytes = checked_mul_size(
        checked_mul_size(
            checked_mul_size(n, estimator_per_point,
                "raw estimator elements"),
            2, "raw/aggregated estimator elements"),
        sizeof(float), "raw/aggregated estimator bytes");
    budget.fixed_mask_bytes = checked_add_size(
        checked_mul_size(n, sizeof(uint8_t), "owner mask bytes"),
        checked_mul_size(
            checked_mul_size(static_cast<size_t>(active_anchor_count),
                mask_words, "active neighborhood mask words"),
            sizeof(uint32_t), "neighborhood mask bytes"),
        "fixed mask bytes");

    const size_t diagnostic_bytes_per_point =
        2 * sizeof(int32_t) + sizeof(uint8_t) +
        8 * sizeof(float) + sizeof(uint32_t);
    budget.fixed_diagnostic_bytes = checked_mul_size(
        n, diagnostic_bytes_per_point, "fixed diagnostic bytes");
    budget.fixed_bytes = checked_add_size(
        checked_add_size(budget.fixed_input_bytes,
            budget.fixed_estimator_bytes, "fixed input/estimator bytes"),
        checked_add_size(budget.fixed_mask_bytes,
            budget.fixed_diagnostic_bytes, "fixed mask/diagnostic bytes"),
        "all fixed bytes");

    budget.reserve_bytes = std::max(opts.reserve_bytes,
        static_cast<double>(budget.total_bytes) * opts.reserve_fraction);
    double after_reserve = std::max(0.0,
        static_cast<double>(budget.free_bytes) - budget.reserve_bytes);
    double usable = std::min(after_reserve,
        after_reserve * static_cast<double>(opts.gpu_mem_fraction));
    if (usable >= static_cast<double>(std::numeric_limits<size_t>::max()))
        budget.usable_bytes = std::numeric_limits<size_t>::max();
    else
        budget.usable_bytes = static_cast<size_t>(std::floor(usable));

    int low = 1;
    int high = budget.upper_bound;
    int best = 0;
    while (low <= high) {
        int candidate = low + (high - low) / 2;
        StableQueryMemory query = stable_query_memory(
            candidate, D, kSlice, tmax);
        size_t required = checked_add_size(
            budget.fixed_bytes, query.total_bytes, "candidate required bytes");
        if (required <= budget.usable_bytes) {
            best = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    if (best == 0) {
        throw StableMexError("manor_m_phase1_mex:queryBlock",
            "GPU budget cannot fit fixed Phase-1 resources plus one query.");
    }
    int minimum = std::min(budget.upper_bound,
        std::max(1, opts.min_query_block));
    if (best < minimum) {
        throw StableMexError("manor_m_phase1_mex:queryBlock",
            "GPU budget cannot satisfy opts.min_query_block without OOM risk.");
    }

    budget.BQ = best;
    budget.selected_query = stable_query_memory(best, D, kSlice, tmax);
    budget.required_bytes = checked_add_size(
        budget.fixed_bytes, budget.selected_query.total_bytes,
        "selected required bytes");
    if (best < budget.upper_bound) {
        StableQueryMemory next = stable_query_memory(best + 1, D, kSlice, tmax);
        budget.next_required_bytes = checked_add_size(
            budget.fixed_bytes, next.total_bytes, "next required bytes");
    }
    return budget;
}

#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
static double numeric_at(const mxArray* array, mwIndex index)
{
    switch (mxGetClassID(array)) {
        case mxDOUBLE_CLASS: return static_cast<const double*>(mxGetData(array))[index];
        case mxSINGLE_CLASS: return static_cast<const float*>(mxGetData(array))[index];
        case mxINT32_CLASS: return static_cast<const int32_T*>(mxGetData(array))[index];
        case mxUINT32_CLASS: return static_cast<const uint32_T*>(mxGetData(array))[index];
        case mxINT64_CLASS: return static_cast<double>(static_cast<const int64_T*>(mxGetData(array))[index]);
        case mxUINT64_CLASS: return static_cast<double>(static_cast<const uint64_T*>(mxGetData(array))[index]);
        case mxINT16_CLASS: return static_cast<const int16_T*>(mxGetData(array))[index];
        case mxUINT16_CLASS: return static_cast<const uint16_T*>(mxGetData(array))[index];
        case mxINT8_CLASS: return static_cast<const int8_T*>(mxGetData(array))[index];
        case mxUINT8_CLASS: return static_cast<const uint8_T*>(mxGetData(array))[index];
        default:
            mexErrMsgIdAndTxt("manor_m_phase1_mex:type",
                "Expected a real numeric array.");
    }
    return 0.0;
}

static double scalar_field(const mxArray* array, const char* name)
{
    if (!array || mxIsComplex(array) || mxIsSparse(array) ||
        !mxIsNumeric(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:scalar",
            "%s must be a real numeric scalar.", name);
    }
    double value = numeric_at(array, 0);
    if (!std::isfinite(value)) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:scalar",
            "%s must be finite.", name);
    }
    return value;
}

static const mxArray* option_field(const mxArray* opts, const char* name)
{
    return opts ? mxGetField(opts, 0, name) : nullptr;
}

static uint64_t seed_scalar(const mxArray* array)
{
    if (!array || mxIsComplex(array) || mxIsSparse(array) ||
        !mxIsNumeric(array) || mxGetNumberOfElements(array) != 1) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:splitSeed",
            "opts.split_seed must be a real integer scalar.");
    }
    if (mxIsUint64(array)) {
        return static_cast<const uint64_T*>(mxGetData(array))[0];
    }
    if (mxIsInt64(array)) {
        int64_T value = static_cast<const int64_T*>(mxGetData(array))[0];
        if (value < 0) {
            mexErrMsgIdAndTxt("manor_m_phase1_mex:splitSeed",
                "opts.split_seed must be nonnegative.");
        }
        return static_cast<uint64_t>(value);
    }
    double value = scalar_field(array, "opts.split_seed");
    if (value < 0.0 || value != std::floor(value) || value > 9007199254740992.0) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:splitSeed",
            "Non-uint64 opts.split_seed must be an exact nonnegative integer <= 2^53.");
    }
    return static_cast<uint64_t>(value);
}

static bool bool_scalar(const mxArray* array, const char* name)
{
    if (mxIsLogicalScalar(array)) return mxIsLogicalScalarTrue(array);
    return scalar_field(array, name) != 0.0;
}

static StablePhase1Opts stable_parse_opts(const mxArray* opts)
{
    if (opts && (!mxIsStruct(opts) || mxGetNumberOfElements(opts) != 1)) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:opts",
            "opts must be a scalar struct.");
    }
    StablePhase1Opts parsed;
    parsed.memory = parse_opts(opts);
    parsed.heavy_factor = static_cast<double>(parsed.memory.heavy_factor);
    parsed.jitter = parsed.memory.jitter;

    if (const mxArray* value = option_field(opts, "heavy_factor"))
        parsed.heavy_factor = scalar_field(value, "opts.heavy_factor");
    if (const mxArray* value = option_field(opts, "jitter")) {
        double raw = scalar_field(value, "opts.jitter");
        parsed.jitter = static_cast<float>(raw);
        if (!std::isfinite(parsed.jitter)) {
            mexErrMsgIdAndTxt("manor_m_phase1_mex:jitter",
                "opts.jitter is outside the supported single-precision range.");
        }
    }
    if (const mxArray* value = option_field(opts, "window")) {
        double raw = scalar_field(value, "opts.window");
        if (raw != std::floor(raw) || raw < 1.0 || raw > MAX_R_MANOR)
            mexErrMsgIdAndTxt("manor_m_phase1_mex:window",
                "opts.window must be an integer in [1, MAX_R_MANOR].");
        parsed.window = static_cast<int>(raw);
    }
    if (const mxArray* value = option_field(opts, "neighbor_scaled_max")) {
        double raw = scalar_field(value, "opts.neighbor_scaled_max");
        parsed.neighbor_scaled_max = static_cast<float>(raw);
        if (!std::isfinite(parsed.neighbor_scaled_max))
            mexErrMsgIdAndTxt("manor_m_phase1_mex:threshold",
                "opts.neighbor_scaled_max is outside the supported range.");
    }
    if (const mxArray* value = option_field(opts, "split_scaled_max")) {
        double raw = scalar_field(value, "opts.split_scaled_max");
        parsed.split_scaled_max = static_cast<float>(raw);
        if (!std::isfinite(parsed.split_scaled_max))
            mexErrMsgIdAndTxt("manor_m_phase1_mex:threshold",
                "opts.split_scaled_max is outside the supported range.");
    }
    if (const mxArray* value = option_field(opts, "split_seed"))
        parsed.split_seed = seed_scalar(value);
    if (const mxArray* value = option_field(opts, "idx_zero_based"))
        parsed.idx_zero_based = bool_scalar(value, "opts.idx_zero_based");
    if (const mxArray* value = option_field(opts, "idx_shared_slice"))
        parsed.idx_shared_slice = bool_scalar(value, "opts.idx_shared_slice");
    if (const mxArray* value = option_field(opts, "response_Y"))
        parsed.response_Y = value;
    if (const mxArray* value = option_field(opts, "response_kslice")) {
        double raw = scalar_field(value, "opts.response_kslice");
        if (raw != std::floor(raw) || raw < 1.0 ||
            raw > static_cast<double>(std::numeric_limits<int>::max())) {
            throw StableMexError("manor_m_phase1_mex:responseIndex",
                "opts.response_kslice must be a positive integer.");
        }
        parsed.response_kslice = static_cast<int>(raw);
    }
    if (const mxArray* value = option_field(opts, "return_response_index"))
        parsed.return_response_index =
            bool_scalar(value, "opts.return_response_index");
    if (const mxArray* value = option_field(opts, "cut_max_fraction")) {
        double raw = scalar_field(value, "opts.cut_max_fraction");
        if (!(raw > 0.0 && raw <= 1.0)) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:cutMaxFraction",
                "opts.cut_max_fraction must lie in (0,1].");
        }
        parsed.cut_max_fraction = raw;
    }
    if (const mxArray* value = option_field(opts, "selection_mode")) {
        double raw = scalar_field(value, "opts.selection_mode");
        /* Selection modes are an explicit enumeration. */
        if (raw != STABLE_POSITIVE_H && raw != TOP5_ABS_H_MEDIAN_CUT) {
            throw StableMexError("manor_m_phase1_mex:selectionMode",
                "opts.selection_mode must be 0 (Stable-H) or 2 (top-5-|H| "
                "median cut).");
        }
        parsed.selection_mode = static_cast<int>(raw);
    }
    if (const mxArray* value = option_field(opts, "active_anchor_count")) {
        double raw = scalar_field(value, "opts.active_anchor_count");
        if (raw != std::floor(raw) || raw < 1.0 ||
            raw > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:activeAnchors",
                "opts.active_anchor_count must be a positive integer.");
        }
        parsed.active_anchor_count = static_cast<int>(raw);
    }
    if (const mxArray* value = option_field(opts, "h_null_correction"))
        parsed.h_null_correction =
            bool_scalar(value, "opts.h_null_correction");

    if (!std::isfinite(parsed.heavy_factor) || parsed.heavy_factor < 0.0)
        mexErrMsgIdAndTxt("manor_m_phase1_mex:heavyFactor",
            "opts.heavy_factor must be finite and nonnegative.");
    if (!std::isfinite(parsed.jitter) || parsed.jitter < 0.0f)
        mexErrMsgIdAndTxt("manor_m_phase1_mex:jitter",
            "opts.jitter must be finite and nonnegative.");
    if ((parsed.window & 1) == 0)
        mexErrMsgIdAndTxt("manor_m_phase1_mex:window",
            "opts.window must be odd.");
    if (!std::isfinite(parsed.neighbor_scaled_max) || parsed.neighbor_scaled_max < 0.0f ||
        !std::isfinite(parsed.split_scaled_max) || parsed.split_scaled_max < 0.0f)
        mexErrMsgIdAndTxt("manor_m_phase1_mex:threshold",
            "Stable-H thresholds must be finite and nonnegative.");
    if (!std::isfinite(parsed.memory.gpu_mem_fraction) || parsed.memory.gpu_mem_fraction <= 0.0f ||
        !std::isfinite(parsed.memory.reserve_fraction) || parsed.memory.reserve_fraction < 0.0f ||
        !std::isfinite(parsed.memory.reserve_bytes) || parsed.memory.reserve_bytes < 0.0 ||
        parsed.memory.min_query_block < 1 ||
        parsed.memory.max_query_block < parsed.memory.min_query_block) {
        mexErrMsgIdAndTxt("manor_m_phase1_mex:memoryOpts",
            "Invalid GPU memory or query-block options.");
    }
    return parsed;
}
#endif

__device__ __forceinline__ uint64_t splitmix64_device(uint64_t value)
{
    uint64_t z = value + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ int split_label_device(
    int anchor, int rank, uint64_t seed)
{
    uint64_t pair_key = seed ^
        (static_cast<uint64_t>(anchor + 1) * 0xD1B54A32D192ED03ULL);
    int orientation = static_cast<int>(
        splitmix64_device(pair_key ^ static_cast<uint64_t>(rank >> 1)) & 1ULL);
    return (rank & 1) ? 1 - orientation : orientation;
}

__device__ float basis_distance_device(
    const float* left, const float* right, int D, int MM)
{
    /* Single precision is sufficient for this thresholded basis-distance calculation. */
    float left_norm2 = 0.0f;
    float right_norm2 = 0.0f;
    float cross_norm2 = 0.0f;
    for (int first = 0; first < MM; ++first) {
        for (int second = 0; second < MM; ++second) {
            float left_dot = 0.0f;
            float right_dot = 0.0f;
            float cross_dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                left_dot += left[d + D * first] * left[d + D * second];
                right_dot += right[d + D * first] * right[d + D * second];
                cross_dot += left[d + D * first] * right[d + D * second];
            }
            left_norm2 += left_dot * left_dot;
            right_norm2 += right_dot * right_dot;
            cross_norm2 += cross_dot * cross_dot;
        }
    }
    float distance2 = (left_norm2 + right_norm2 - 2.0f * cross_norm2) /
        (2.0f * static_cast<float>(MM));
    return sqrtf(fmaxf(0.0f, distance2));
}

__device__ void normalize_basis_device(float* basis, int D, int MM)
{
    const unsigned full_mask = 0xffffffffu;
    int tid = threadIdx.x;
    for (int column = 0; column < MM; ++column) {
        float norm2 = 0.0f;
        for (int d = tid; d < D; d += 32) {
            float value = basis[d + D * column];
            norm2 += value * value;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            norm2 += __shfl_down_sync(full_mask, norm2, offset);
        float inverse_norm = rsqrtf(fmaxf(
            __shfl_sync(full_mask, norm2, 0), 1.0e-30f));
        for (int d = tid; d < D; d += 32)
            basis[d + D * column] *= inverse_norm;
        __syncwarp();
    }
}

__device__ __forceinline__ float isotropic_radius(
    const float* eigenvalues, int D)
{
    /* Rad0 as the RMS radius of an ISOTROPIC cloud of the same volume:
     *
     *     Rad0 = ( prod_d sqrt(lambda_d) )^(1/D) * sqrt(D)
     *          = det(C)^(1/(2D)) * sqrt(D)
     *
     * over the eigenvalues of the selected cut's local covariance. The old
     * definition -- the largest neighbour distance -- is set by whichever
     * single direction stretches furthest, so a wide Phase-1 cut drags it up
     * and no "h = f * Rad0" rule survives across response widths. The
     * geometric mean keeps the units of a radius and lets a blow-up confined
     * to the MM tangential directions enter only at power MM/D.
     *
     * Summed in logs rather than as a product so the D-fold product cannot
     * underflow single precision. Called by one thread; the eigenvalues are
     * the ones the extract kernels already have in shared memory. */
    float log_sum = 0.0f;
    for (int d = 0; d < D; ++d)
        log_sum += logf(fmaxf(eigenvalues[d], 1.0e-30f));
    return expf(log_sum / (2.0f * static_cast<float>(D))) *
        sqrtf(static_cast<float>(D));
}

__device__ __forceinline__ int idx_at(
    const int* idx_gpu, int anchor, int rank, int kSlice,
    bool idx_shared_slice, int index_anchor_base = 0)
{
    size_t offset = idx_shared_slice ? static_cast<size_t>(rank) :
        static_cast<size_t>(anchor - index_anchor_base) * kSlice + rank;
    return idx_gpu[offset];
}

template <typename Rank>
__global__ __launch_bounds__(256, 2)
void phase1_dist_idx_kernel(
    const float* __restrict__ Xcm,
    const int* __restrict__ idx_gpu,
    int D, int N, int q0, int B, int kSlice, bool idx_shared_slice,
    float* __restrict__ keys_out,
    Rank* __restrict__ vals_out, int index_anchor_base = 0)
{
    int b = blockIdx.y;
    if (b >= B) return;
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    int anchor = q0 + b;

    extern __shared__ float xi_shared[];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        xi_shared[d] = Xcm[d + static_cast<size_t>(anchor) * D];
    __syncthreads();
    if (rank >= kSlice) return;

    int neighbor = idx_at(
        idx_gpu, anchor, rank, kSlice, idx_shared_slice, index_anchor_base);
    float distance2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < MAX_D_MANOR; ++d) {
        if (d < D) {
            float difference = Xcm[d + static_cast<size_t>(neighbor) * D] -
                xi_shared[d];
            distance2 += difference * difference;
        }
    }
    size_t output = static_cast<size_t>(b) * kSlice + rank;
    keys_out[output] = distance2;
    vals_out[output] = static_cast<Rank>(rank);
}

/* The launch-bound value is part of the CUDA execution configuration. */
template <typename Rank>
__global__ __launch_bounds__(32, MANOR_M_PHASE1_SELECT_MIN_BLOCKS_PER_SM)
void stable_h_select_kernel(
    const float* __restrict__ Xcm,
    const int* __restrict__ idx_gpu,
    const Rank* __restrict__ top_off,
    const int* __restrict__ cuts,
    int D, int MM, int kSlice, int tmax, int R, bool idx_shared_slice,
    int q0, int B, float jitter, int window,
    float neighbor_scaled_max, float split_scaled_max, uint64_t split_seed,
    int selection_mode,
    /* Per-cut Stable-H null offset, or nullptr when opts.h_null_correction is
     * off.  Null means "subtract nothing", and the off path then executes the
     * unmodified arithmetic, so it is bitwise identical to the old kernel. */
    const float* __restrict__ h_null,
    int32_t* __restrict__ selected_index0,
    uint32_t* __restrict__ selected_cut0,
    uint8_t* __restrict__ covered0,
    float* __restrict__ geom0,
    float* __restrict__ stable_h0,
    float* __restrict__ neighbor_scaled0,
    float* __restrict__ split_scaled0,
    float* __restrict__ split_h_a0,
    float* __restrict__ split_h_b0,
    int32_t* __restrict__ fallback_selected_index0,
    uint32_t* __restrict__ fallback_selected_cut0,
    uint8_t* __restrict__ fallback_covered0,
    float* __restrict__ fallback_geom0,
    float* __restrict__ fallback_stable_h0,
    float* __restrict__ fallback_neighbor_scaled0,
    float* __restrict__ fallback_split_scaled0,
    float* __restrict__ fallback_split_h_a0,
    float* __restrict__ fallback_split_h_b0,
    int index_anchor_base = 0,
    int temporary_anchor_base = 0)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;
    int anchor = q0 + b;
    int temporary_anchor = anchor - temporary_anchor_base;
    const float* xi = Xcm + static_cast<size_t>(anchor) * D;
    const Rank* offsets = top_off + static_cast<size_t>(b) * tmax;

    extern __shared__ float dynamic_shared[];
    float* sum_a = dynamic_shared;
    float* sum_b = sum_a + D;
    float* outer_a = sum_b + D;
    float* outer_b = outer_a + D * D;
    float* covariance = outer_b + D * D;
    float* eigenvectors = covariance + D * D;
    float* eigenvalues = eigenvectors + D * D;
    float* split_basis = eigenvalues + D;
    float* ring_basis = split_basis + D * MM;
    float* ring_h = ring_basis + window * D * MM;
    float* ring_h_a = ring_h + window;
    float* ring_h_b = ring_h_a + window;
    float* ring_split_distance = ring_h_b + window;
    float* median_scratch = ring_split_distance + window;
    float* jacobi_rot = median_scratch + window;

    for (int d = tid; d < 2 * D; d += 32)
        dynamic_shared[d] = 0.0f;
    for (int index = tid; index < 2 * D * D; index += 32)
        outer_a[index] = 0.0f;
    __syncwarp();

    int count_a = 0;
    int count_b = 0;
    int best_index = -1;
    int best_cut = 0;
    float best_stable_h = -3.402823466e+38F;
    float best_h = nanf("");
    float best_neighbor = nanf("");
    float best_split = nanf("");
    float best_h_a = nanf("");
    float best_h_b = nanf("");
    int half_window = window / 2;
    int top_abs_index[5];
    float top_abs_h[5];
    float top_abs_score[5];
    int top_abs_count = 0;
    const bool stable_requested = selection_mode == STABLE_POSITIVE_H ||
        selection_mode == HYBRID_STABLE_AND_TOP5;
    const bool fallback_requested = selection_mode == TOP5_ABS_H_MEDIAN_CUT ||
        selection_mode == HYBRID_STABLE_AND_TOP5;

    for (int cut_index = 0; cut_index < R; ++cut_index) {
        int start = cut_index == 0 ? 0 : cuts[cut_index - 1];
        int stop = cuts[cut_index];
        for (int rank = start; rank < stop; ++rank) {
            int local_offset = static_cast<int>(offsets[rank]);
            int neighbor = idx_at(
                idx_gpu, anchor, local_offset, kSlice, idx_shared_slice, index_anchor_base);
            int label = split_label_device(anchor, rank, split_seed);
            for (int d = tid; d < D; d += 32)
                covariance[d] = Xcm[d + static_cast<size_t>(neighbor) * D] - xi[d];
            __syncwarp();
            float* target_sum = label == 0 ? sum_a : sum_b;
            float* target_outer = label == 0 ? outer_a : outer_b;
            for (int d = tid; d < D; d += 32)
                target_sum[d] += covariance[d];
            for (int index = tid; index < D * D; index += 32) {
                int row = index / D;
                int column = index - row * D;
                target_outer[index] += covariance[row] * covariance[column];
            }
            if (label == 0) ++count_a; else ++count_b;
            __syncwarp();
        }

        float inverse = 1.0f / static_cast<float>(stop);
        for (int index = tid; index < D * D; index += 32) {
            int row = index / D;
            int column = index - row * D;
            float mean_row = (sum_a[row] + sum_b[row]) * inverse;
            float mean_column = (sum_a[column] + sum_b[column]) * inverse;
            float moment = (outer_a[index] + outer_b[index]) * inverse;
            covariance[index] = moment - mean_row * mean_column +
                (row == column ? jitter : 0.0f);
        }
        __syncwarp();
        shared_jacobi_eigen_rr_device_warp(covariance, D, eigenvalues, eigenvectors, jacobi_rot);
        for (int d = tid; d < D; d += 32)
            eigenvalues[d] = fmaxf(eigenvalues[d], 1.0e-12f);
        __syncwarp();
        int ring_slot = cut_index % window;
        if (tid == 0)
            ring_h[ring_slot] = geom_score_from_evals_device(eigenvalues, D, MM);
        /* Only STABLE_POSITIVE_H reads any of this: ring_basis feeds its
           window rotation test, and ring_h_a, ring_h_b and ring_split_distance
           feed its split-half eligibility gates.  TOP5_ABS_H_MEDIAN_CUT ranks
           on ring_h alone, so for it the two split eigendecompositions, the
           basis copies and the subspace distance are computed and discarded --
           two of the three decompositions per cut, on the hybrid fallback pass
           that is half of Phase 1.  The branch is warp-uniform. */
        if (stable_requested) {
            for (int index = tid; index < D * MM; index += 32) {
                int column = index / D;
                int row = index - column * D;
                ring_basis[ring_slot * D * MM + index] = eigenvectors[row * D + column];
            }
            __syncwarp();
            normalize_basis_device(
                ring_basis + ring_slot * D * MM, D, MM);

            float inverse_a = 1.0f / static_cast<float>(count_a);
            for (int index = tid; index < D * D; index += 32) {
                int row = index / D;
                int column = index - row * D;
                covariance[index] = outer_a[index] * inverse_a -
                    (sum_a[row] * inverse_a) * (sum_a[column] * inverse_a) +
                    (row == column ? jitter : 0.0f);
            }
            __syncwarp();
            shared_jacobi_eigen_rr_device_warp(covariance, D, eigenvalues, eigenvectors, jacobi_rot);
            for (int d = tid; d < D; d += 32)
                eigenvalues[d] = fmaxf(eigenvalues[d], 1.0e-12f);
            __syncwarp();
            if (tid == 0)
                ring_h_a[ring_slot] = geom_score_from_evals_device(eigenvalues, D, MM);
            for (int index = tid; index < D * MM; index += 32) {
                int column = index / D;
                int row = index - column * D;
                split_basis[index] = eigenvectors[row * D + column];
            }
            __syncwarp();
            normalize_basis_device(split_basis, D, MM);

            float inverse_b = 1.0f / static_cast<float>(count_b);
            for (int index = tid; index < D * D; index += 32) {
                int row = index / D;
                int column = index - row * D;
                covariance[index] = outer_b[index] * inverse_b -
                    (sum_b[row] * inverse_b) * (sum_b[column] * inverse_b) +
                    (row == column ? jitter : 0.0f);
            }
            __syncwarp();
            shared_jacobi_eigen_rr_device_warp(covariance, D, eigenvalues, eigenvectors, jacobi_rot);
            for (int d = tid; d < D; d += 32)
                eigenvalues[d] = fmaxf(eigenvalues[d], 1.0e-12f);
            __syncwarp();
            if (tid == 0) {
                ring_h_b[ring_slot] = geom_score_from_evals_device(eigenvalues, D, MM);
            }
            for (int index = tid; index < D * MM; index += 32) {
                int column = index / D;
                int row = index - column * D;
                covariance[index] = eigenvectors[row * D + column];
            }
            __syncwarp();
            normalize_basis_device(covariance, D, MM);
            if (tid == 0)
                ring_split_distance[ring_slot] = basis_distance_device(
                    split_basis, covariance, D, MM);
            __syncwarp();
        }

        if (tid == 0 && fallback_requested) {
            float current_h = ring_h[ring_slot];
            /* Cross-cut ranking: debias.  top_abs_h keeps the RAW H, so the
             * chosen cut changes but the reported H at it does not. */
            float current_score = fabsf(
                h_null ? current_h - h_null[cut_index] : current_h);
            int slot = top_abs_count;
            for (int position = 0; position < top_abs_count; ++position) {
                if (current_score > top_abs_score[position]) {
                    slot = position;
                    break;
                }
            }
            if (slot < 5) {
                int limit = top_abs_count < 5 ? top_abs_count : 4;
                for (int position = limit; position > slot; --position) {
                    top_abs_index[position] = top_abs_index[position - 1];
                    top_abs_h[position] = top_abs_h[position - 1];
                    top_abs_score[position] = top_abs_score[position - 1];
                }
                top_abs_index[slot] = cut_index;
                top_abs_h[slot] = current_h;
                top_abs_score[slot] = current_score;
                if (top_abs_count < 5) ++top_abs_count;
            }
        }

        if (tid == 0 && stable_requested &&
            cut_index >= window - 1) {
            int center = cut_index - half_window;
            int center_slot = center % window;
            bool all_positive = true;
            for (int slot = 0; slot < window; ++slot) {
                median_scratch[slot] = ring_h[slot];
                /* Eligibility gate stays on RAW H -- deliberately not debiased,
                 * see the scope note above. */
                all_positive = all_positive && ring_h[slot] > 0.0f;
            }
            /* Cross-cut ranking: debias each cut against its own zero before
             * the median.  Ring slot s holds the cut whose index is congruent
             * to s modulo window, so walking the window's cut indices hits
             * every slot exactly once.  Reordering is irrelevant: the sort
             * below only needs the multiset, and all_positive is an AND. */
            if (h_null) {
                for (int position = cut_index - window + 1;
                     position <= cut_index; ++position) {
                    median_scratch[position % window] -= h_null[position];
                }
            }
            for (int position = 1; position < window; ++position) {
                float key = median_scratch[position];
                int previous = position - 1;
                while (previous >= 0 && median_scratch[previous] > key) {
                    median_scratch[previous + 1] = median_scratch[previous];
                    --previous;
                }
                median_scratch[previous + 1] = key;
            }
            float median_h = median_scratch[half_window];
            const float* center_basis = ring_basis + center_slot * D * MM;
            float maximum_distance = 0.0f;
            for (int position = center - half_window;
                 position <= center + half_window; ++position) {
                const float* other_basis = ring_basis + (position % window) * D * MM;
                maximum_distance = fmaxf(maximum_distance,
                    basis_distance_device(center_basis, other_basis, D, MM));
            }
            float root_cut = sqrtf(static_cast<float>(cuts[center]));
            float neighbor_scaled = root_cut * maximum_distance;
            float split_scaled = root_cut * ring_split_distance[center_slot];
            bool eligible = all_positive &&
                neighbor_scaled <= neighbor_scaled_max &&
                ring_h_a[center_slot] > 0.0f &&
                ring_h_b[center_slot] > 0.0f &&
                split_scaled <= split_scaled_max;
            if (eligible && (best_index < 0 || median_h > best_stable_h ||
                (median_h == best_stable_h && cuts[center] < best_cut))) {
                best_index = center;
                best_cut = cuts[center];
                best_stable_h = median_h;
                best_h = ring_h[center_slot];
                best_neighbor = neighbor_scaled;
                best_split = split_scaled;
                best_h_a = ring_h_a[center_slot];
                best_h_b = ring_h_b[center_slot];
            }
        }
        __syncwarp();
    }

    if (tid == 0) {
        int fallback_index = -1;
        int fallback_cut = 0;
        float fallback_h = nanf("");
        float fallback_score = nanf("");
        if (fallback_requested && top_abs_count > 0) {
            int sorted_index[5];
            float sorted_h[5];
            for (int position = 0; position < top_abs_count; ++position) {
                sorted_index[position] = top_abs_index[position];
                sorted_h[position] = top_abs_h[position];
            }
            for (int position = 1; position < top_abs_count; ++position) {
                int index_key = sorted_index[position];
                float h_key = sorted_h[position];
                int previous = position - 1;
                while (previous >= 0 && sorted_index[previous] > index_key) {
                    sorted_index[previous + 1] = sorted_index[previous];
                    sorted_h[previous + 1] = sorted_h[previous];
                    --previous;
                }
                sorted_index[previous + 1] = index_key;
                sorted_h[previous + 1] = h_key;
            }
            int middle = top_abs_count / 2;
            fallback_index = sorted_index[middle];
            fallback_cut = cuts[fallback_index];
            fallback_h = sorted_h[middle];
            fallback_score = fabsf(fallback_h);
        }
        if (selection_mode == TOP5_ABS_H_MEDIAN_CUT) {
            best_index = fallback_index;
            best_cut = fallback_cut;
            best_h = fallback_h;
            best_stable_h = fallback_score;
        }
        bool covered = best_index >= 0;
        selected_index0[temporary_anchor] = best_index;
        selected_cut0[temporary_anchor] = covered ? static_cast<uint32_t>(best_cut) : 0u;
        covered0[temporary_anchor] = covered ? 1u : 0u;
        geom0[temporary_anchor] = covered ? best_h : nanf("");
        stable_h0[temporary_anchor] = covered ? best_stable_h : nanf("");
        neighbor_scaled0[temporary_anchor] = covered ? best_neighbor : nanf("");
        split_scaled0[temporary_anchor] = covered ? best_split : nanf("");
        split_h_a0[temporary_anchor] = covered ? best_h_a : nanf("");
        split_h_b0[temporary_anchor] = covered ? best_h_b : nanf("");
        if (selection_mode == HYBRID_STABLE_AND_TOP5) {
            bool fallback_covered = fallback_index >= 0;
            fallback_selected_index0[temporary_anchor] = fallback_index;
            fallback_selected_cut0[temporary_anchor] = fallback_covered ?
                static_cast<uint32_t>(fallback_cut) : 0u;
            fallback_covered0[temporary_anchor] = fallback_covered ? 1u : 0u;
            fallback_geom0[temporary_anchor] = fallback_covered ? fallback_h : nanf("");
            fallback_stable_h0[temporary_anchor] = fallback_covered ?
                fallback_score : nanf("");
            /* The fallback output does not compute split diagnostics. */
            fallback_neighbor_scaled0[temporary_anchor] = nanf("");
            fallback_split_scaled0[temporary_anchor] = nanf("");
            fallback_split_h_a0[temporary_anchor] = nanf("");
            fallback_split_h_b0[temporary_anchor] = nanf("");
        }
    }
}


__global__ __launch_bounds__(32, 4)
void owner_aggregate_kernel_mm(
    const uint32_t* __restrict__ neigh_mask0_bits,
    const uint8_t* __restrict__ owner_mask0,
    const float* __restrict__ tangent_raw,
    const float* __restrict__ center_raw,
    const int* __restrict__ idx_gpu,
    int D, int MM, int N, int kSlice, int mask_words_per_row,
    bool idx_shared_slice,
    int q0, int B,
    float* __restrict__ covariance_out,
    float* __restrict__ center_sum_out,
    int32_t* __restrict__ count_out, int index_anchor_base = 0)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int anchor = q0 + b;
    float* covariance = covariance_out + static_cast<size_t>(b) * D * D;
    float* center_sum = center_sum_out + static_cast<size_t>(b) * D;

    if (!owner_mask0[anchor]) {
        for (int index = tid; index < D * D; index += 32)
            covariance[index] = 0.0f;
        for (int d = tid; d < D; d += 32) center_sum[d] = 0.0f;
        if (tid == 0) count_out[b] = 0;
        return;
    }

    extern __shared__ float dynamic_shared[];
    float* shared_covariance = dynamic_shared;
    float* shared_center = shared_covariance + D * D;
    __shared__ int shared_count;
    for (int index = tid; index < D * D; index += 32)
        shared_covariance[index] = 0.0f;
    for (int d = tid; d < D; d += 32) shared_center[d] = 0.0f;
    if (tid == 0) shared_count = 0;
    __syncwarp();

    const uint32_t* neighborhood = neigh_mask0_bits +
        static_cast<size_t>(anchor) * mask_words_per_row;
    int local_count = 0;
    for (int word = 0; word < mask_words_per_row; ++word) {
        uint32_t bits = neighborhood[word];
        while (bits) {
            int bit = __ffs(static_cast<int>(bits)) - 1;
            bits &= bits - 1u;
            int position = (word << 5) + bit;
            if (position >= kSlice) continue;
            int source = idx_at(
                idx_gpu, anchor, position, kSlice, idx_shared_slice, index_anchor_base);
            if (static_cast<unsigned>(source) >= static_cast<unsigned>(N) ||
                !owner_mask0[source]) {
                continue;
            }
            ++local_count;
            for (int d = tid; d < D; d += 32)
                shared_center[d] += center_raw[d + static_cast<size_t>(source) * D];
            for (int column = 0; column < MM; ++column) {
                const float* source_tangent = tangent_raw +
                    static_cast<size_t>(D) * (column + static_cast<size_t>(MM) * source);
                for (int index = tid; index < D * D; index += 32) {
                    int row = index / D;
                    int other = index - row * D;
                    shared_covariance[index] +=
                        source_tangent[row] * source_tangent[other];
                }
            }
        }
    }
    if (tid == 0) shared_count = local_count;
    __syncwarp();

    for (int index = tid; index < D * D; index += 32)
        covariance[index] = shared_covariance[index];
    for (int d = tid; d < D; d += 32) center_sum[d] = shared_center[d];
    if (tid == 0) count_out[b] = shared_count;
}


__global__ void mask_nonowners_kernel(
    const uint8_t* __restrict__ owner_mask0,
    int D, int MM, int N,
    float* __restrict__ tangent0,
    float* __restrict__ center0)
{
    int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= N || owner_mask0[anchor]) return;
    float* tangent = tangent0 + static_cast<size_t>(D) * MM * anchor;
    float* center = center0 + static_cast<size_t>(D) * anchor;
    for (int index = 0; index < D * MM; ++index) tangent[index] = 0.0f;
    for (int d = 0; d < D; ++d) center[d] = 0.0f;
}

static size_t compute_selection_shared_bytes(int D, int MM, int window)
{
    size_t float_count = static_cast<size_t>(3) * D +
        static_cast<size_t>(4) * D * D +
        static_cast<size_t>(window + 1) * D * MM +
        static_cast<size_t>(5) * window +
        static_cast<size_t>(2) * ((D + (D & 1)) / 2);
    return float_count * sizeof(float);
}

/* Owner-aggregate / final-extract / owner-finalize used to size their scratch
 * by MAX_D_MANOR, which reserved 9424, 29760 and 20560 static bytes and capped
 * residency at 10, 3 and 4 blocks/SM against the 24-block cap.  The carve below
 * is the actual D (and MM), matching compute_selection_shared_bytes above.
 * Worst case is MAX_D_MANOR=48, MAX_MM_MANOR=8 -> 29760 bytes, still under the
 * 48 KB default dynamic limit, so no cudaFuncAttributeMaxDynamicSharedMemorySize
 * opt-in is required for any supported (D, MM). */
static size_t compute_owner_aggregate_shared_bytes(int D)
{
    size_t float_count = static_cast<size_t>(D) * D + static_cast<size_t>(D);
    return float_count * sizeof(float);
}

static size_t compute_final_extract_shared_bytes(int D, int MM)
{
    size_t float_count = static_cast<size_t>(3) * D * D +
        static_cast<size_t>(3) * D +
        static_cast<size_t>(D) * MM;
    return float_count * sizeof(float);
}

static size_t compute_owner_finalize_shared_bytes(int D, int MM)
{
    size_t float_count = static_cast<size_t>(2) * D * D +
        static_cast<size_t>(3) * D +
        static_cast<size_t>(D) * MM;
    return float_count * sizeof(float);
}

#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
static mxArray* single_vector(const float* device, int N)
{
    mxArray* output = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(output), device, sizeof(float) * N,
        cudaMemcpyDeviceToHost));
    return output;
}

static mxArray* uint64_scalar(uint64_t value)
{
    mxArray* output = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    static_cast<uint64_T*>(mxGetData(output))[0] = static_cast<uint64_T>(value);
    return output;
}
#endif



namespace {

constexpr int ADAPTIVE_MAX_SCALES = 16;

struct AdaptivePhase1Opts {
    StablePhase1Opts stable;
    int active_anchor_count = -1;
};

struct AdaptiveMemoryBudget {
    StableMemoryBudget stable;
    size_t extra_fixed_bytes = 0;
};

/* Guarded span timers (opts.profile_timers, default off).  Duplicated from
 * OwnerProfileTimers in manor_m_phase2_mex.cu rather than shared
 * through a header, because the two MEX files compile independently; the
 * surfaced shape is deliberately the same, profile.timer_labels and
 * profile.timer_ms, so one reader serves both phases.
 *
 * Device spans record cudaEvents on the default stream and are read only
 * after the final device synchronize, so profiling adds no synchronization of
 * its own.  Host spans take the steady clock instead: a stage that issues no
 * GPU work has no GPU timeline to measure, and Phase 1 spends most of its
 * wall time in exactly such stages.  Totals accumulate by label, so a label
 * reused across loop iterations reports the sum.
 *
 * No event is created before the memory budget is chosen, so cudaMemGetInfo
 * -- and therefore BQ -- sees an identical device whether or not profiling is
 * on. */
struct StableProfileSpans {
    struct Span {
        const char* label = nullptr;
        cudaEvent_t start = nullptr;   /* non-null marks a device span */
        cudaEvent_t stop = nullptr;
        std::chrono::steady_clock::time_point host_start{};
        double host_ms = 0.0;
    };
    struct SpanTotal {
        const char* label = nullptr;
        double ms = 0.0;
    };

    bool enabled = false;
    std::vector<Span> spans;

    StableProfileSpans() = default;
    StableProfileSpans(const StableProfileSpans&) = delete;
    StableProfileSpans& operator=(const StableProfileSpans&) = delete;
    ~StableProfileSpans()
    {
        for (Span& span : spans) {
            if (span.start) cudaEventDestroy(span.start);
            if (span.stop) cudaEventDestroy(span.stop);
        }
    }

    void begin(const char* label)
    {
        if (!enabled) return;
        Span span;
        span.label = label;
        STABLE_CUDA_CHECK(cudaEventCreate(&span.start));
        STABLE_CUDA_CHECK(cudaEventRecord(span.start));
        spans.push_back(span);
    }

    void begin_host(const char* label)
    {
        if (!enabled) return;
        Span span;
        span.label = label;
        span.host_start = std::chrono::steady_clock::now();
        spans.push_back(span);
    }

    void end()
    {
        if (!enabled) return;
        Span& span = spans.back();
        if (span.start) {
            STABLE_CUDA_CHECK(cudaEventCreate(&span.stop));
            STABLE_CUDA_CHECK(cudaEventRecord(span.stop));
        } else {
            span.host_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - span.host_start).count();
        }
    }

    /* Totals by label in first-appearance order.  Device spans are only
     * readable once the device has synchronized; an unbalanced begin leaves a
     * null stop event and cudaEventElapsedTime raises rather than hiding it. */
    std::vector<SpanTotal> totals() const
    {
        std::vector<SpanTotal> result;
        for (const Span& span : spans) {
            double elapsed = span.host_ms;
            if (span.start) {
                float device_ms = 0.0f;
                STABLE_CUDA_CHECK(cudaEventElapsedTime(
                    &device_ms, span.start, span.stop));
                elapsed = static_cast<double>(device_ms);
            }
            bool found = false;
            for (SpanTotal& entry : result) {
                if (std::strcmp(entry.label, span.label) == 0) {
                    entry.ms += elapsed;
                    found = true;
                    break;
                }
            }
            if (!found) {
                SpanTotal entry;
                entry.label = span.label;
                entry.ms = elapsed;
                result.push_back(entry);
            }
        }
        return result;
    }
};

/* One-based idx staging: one branch on the class, then a typed loop.
 * numeric_at pays mxGetClassID and mxGetData on EVERY element, and
 * both are loop-invariant.  This layout is [N x kSlice] while the device
 * wants [kSlice x anchor], so it is a transpose and still has to stage
 * through a host vector.  The arithmetic, the four validity checks and the
 * error identifier are the same expressions in the same order as the loop
 * this replaces. */
#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
template <typename T>
void adaptive_stage_one_based(
    const T* source, int anchor_count, int kSlice, int N, int32_t* destination)
{
    for (int anchor = 0; anchor < anchor_count; ++anchor) {
        for (int rank = 0; rank < kSlice; ++rank) {
            double raw = static_cast<double>(
                source[static_cast<size_t>(anchor) +
                       static_cast<size_t>(N) * rank]);
            double zero_based = raw - 1.0;
            if (!std::isfinite(raw) || raw != std::floor(raw) ||
                zero_based < 0.0 || zero_based >= N) {
                mexErrMsgIdAndTxt(
                    "manor_m_phase1_mex:idx",
                    "idx_nested contains an invalid index.");
            }
            destination[static_cast<size_t>(anchor) * kSlice + rank] =
                static_cast<int32_t>(zero_based);
        }
    }
}

void adaptive_stage_one_based_any(
    const mxArray* array, int anchor_count, int kSlice, int N,
    int32_t* destination)
{
    const void* data = mxGetData(array);
#define ADAPTIVE_STAGE_CASE(class_id, type) \
    case class_id: \
        adaptive_stage_one_based(static_cast<const type*>(data), \
            anchor_count, kSlice, N, destination); \
        break
    switch (mxGetClassID(array)) {
        ADAPTIVE_STAGE_CASE(mxDOUBLE_CLASS, double);
        ADAPTIVE_STAGE_CASE(mxSINGLE_CLASS, float);
        ADAPTIVE_STAGE_CASE(mxINT32_CLASS, int32_T);
        ADAPTIVE_STAGE_CASE(mxUINT32_CLASS, uint32_T);
        ADAPTIVE_STAGE_CASE(mxINT64_CLASS, int64_T);
        ADAPTIVE_STAGE_CASE(mxUINT64_CLASS, uint64_T);
        ADAPTIVE_STAGE_CASE(mxINT16_CLASS, int16_T);
        ADAPTIVE_STAGE_CASE(mxUINT16_CLASS, uint16_T);
        ADAPTIVE_STAGE_CASE(mxINT8_CLASS, int8_T);
        ADAPTIVE_STAGE_CASE(mxUINT8_CLASS, uint8_T);
        default:
            /* numeric_at's own rejection, kept verbatim. */
            mexErrMsgIdAndTxt("manor_m_phase1_mex:type",
                "Expected a real numeric array.");
    }
#undef ADAPTIVE_STAGE_CASE
}

AdaptivePhase1Opts adaptive_parse_opts(const mxArray* opts)
{
    AdaptivePhase1Opts parsed;
    parsed.stable = stable_parse_opts(opts);
    if (!option_field(opts, "max_query_block")) {
        parsed.stable.memory.max_query_block =
            std::numeric_limits<int>::max();
    }
    if (const mxArray* value = option_field(opts, "active_anchor_count")) {
        double raw = scalar_field(value, "opts.active_anchor_count");
        if (raw != std::floor(raw) || raw < 1.0 ||
            raw > static_cast<double>(std::numeric_limits<int>::max())) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:activeAnchors",
                "opts.active_anchor_count must be a positive integer.");
        }
        parsed.active_anchor_count = static_cast<int>(raw);
    }
    if (option_field(opts, "cross_scale_projector_max")) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:crossScaleRemoved",
            "opts.cross_scale_projector_max was removed: Y performs no "
            "cross-scale coupling and the plateau rule no longer exists. "
            "Delete the option; every Y scale runs its own selection and the "
            "best Stable-H wins.");
    }
    return parsed;
}
#endif

size_t adaptive_extra_fixed_bytes(int N)
{
    const size_t n = static_cast<size_t>(N);
    const size_t selector_record =
        2 * sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint8_t) +
        6 * sizeof(float);
    // The best-Stable-H rule keeps only the temporary and selected records.
    return checked_mul_size(
        n, 2 * selector_record, "adaptive selector-state bytes");
}

AdaptiveMemoryBudget adaptive_choose_query_block(
    int D, int MM, int N, int active_anchor_count, int kSlice, int tmax, int R,
    bool idx_shared_slice, const Phase1Opts& opts)
{
    AdaptiveMemoryBudget result;
    result.stable = stable_choose_query_block(
        D, MM, N, active_anchor_count, kSlice, tmax, R,
        idx_shared_slice, opts);
    result.extra_fixed_bytes = adaptive_extra_fixed_bytes(N);
    result.stable.fixed_bytes = checked_add_size(
        result.stable.fixed_bytes, result.extra_fixed_bytes,
        "adaptive fixed bytes");

    int low = 1;
    int high = result.stable.upper_bound;
    int best = 0;
    while (low <= high) {
        int candidate = low + (high - low) / 2;
        StableQueryMemory query = stable_query_memory(
            candidate, D, kSlice, tmax);
        size_t required = checked_add_size(
            result.stable.fixed_bytes, query.total_bytes,
            "adaptive candidate bytes");
        if (required <= result.stable.usable_bytes) {
            best = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    int minimum = std::min(result.stable.upper_bound,
        std::max(1, opts.min_query_block));
    if (best < minimum) {
        throw StableMexError(
            "manor_m_phase1_mex:queryBlock",
            "GPU budget cannot fit adaptive selector state and min_query_block.");
    }
    result.stable.BQ = best;
    result.stable.selected_query = stable_query_memory(
        best, D, kSlice, tmax);
    result.stable.required_bytes = checked_add_size(
        result.stable.fixed_bytes, result.stable.selected_query.total_bytes,
        "adaptive selected bytes");
    result.stable.next_required_bytes = 0;
    if (best < result.stable.upper_bound) {
        StableQueryMemory next = stable_query_memory(
            best + 1, D, kSlice, tmax);
        result.stable.next_required_bytes = checked_add_size(
            result.stable.fixed_bytes, next.total_bytes,
            "adaptive next-query bytes");
    }
    return result;
}

template <typename Rank>
__global__ __launch_bounds__(32, 4)
void adaptive_compact_nested_topk_kernel(
    const float* __restrict__ sorted_keys,
    const Rank* __restrict__ sorted_ranks,
    int B, int kSlice, int response_count, int tmax,
    int output_stride,
    float* __restrict__ top_keys,
    Rank* __restrict__ top_ranks)
{
    int b = blockIdx.x;
    int lane = threadIdx.x;
    if (b >= B) return;
    /* Every lane sees the same ballots, so the prefix base is warp-uniform.
       Keeping it lane-local avoids a read/write race on a shared scalar under
       independent thread scheduling without changing any destination. */
    int selected = 0;
    const float* key_row = sorted_keys + static_cast<size_t>(b) * kSlice;
    const Rank* rank_row = sorted_ranks + static_cast<size_t>(b) * kSlice;
    float* output_key = top_keys + static_cast<size_t>(b) * output_stride;
    Rank* output_rank = top_ranks + static_cast<size_t>(b) * output_stride;
    for (int base = 0; base < kSlice; base += 32) {
        int position = base + lane;
        bool include = position < kSlice &&
            static_cast<int>(rank_row[position]) < response_count;
        uint32_t ballot = __ballot_sync(0xffffffffu, include);
        int before = lane == 0 ? 0 :
            __popc(ballot & ((uint32_t(1) << lane) - 1u));
        int current = selected;
        int destination = current + before;
        if (include && destination < tmax) {
            output_key[destination] = key_row[position];
            output_rank[destination] = rank_row[position];
        }
        selected = current + __popc(ballot);
        __syncwarp();
        if (selected >= tmax) break;
    }
}

/* The cross-scale rule: every Y scale runs its own single-scale Stable-H
 * selection and the running best wins. There is no cross-scale constraint of
 * any kind, so no per-scale selector basis is ever built.
 *
 * Mirrors the cross-scale reduction of phase1_stable_torch in
 * pytorch/torch_manor_m/manor_m_torch.py:1109-1121: maximise stable_h, break
 * ties on the smaller cut, then on the smaller (narrower) scale position.
 * Scales are streamed in ascending order, so testing for a STRICT improvement
 * reproduces that lexicographic order while keeping O(1) state per anchor: the
 * running best is itself the answer, and no second pass is needed.
 *
 * One thread per anchor -- there is no basis to compare or copy, so there is
 * nothing for a warp to cooperate on. */
__global__ void adaptive_stream_best_kernel(
    int scale, int q0, int B,
    const int32_t* __restrict__ temporary_index,
    const uint32_t* __restrict__ temporary_cut,
    const uint8_t* __restrict__ temporary_covered,
    const float* __restrict__ temporary_geom,
    const float* __restrict__ temporary_stable_h,
    const float* __restrict__ temporary_neighbor,
    const float* __restrict__ temporary_split,
    const float* __restrict__ temporary_h_a,
    const float* __restrict__ temporary_h_b,
    int32_t* __restrict__ selected_scale,
    int32_t* __restrict__ selected_index,
    uint32_t* __restrict__ selected_cut,
    uint8_t* __restrict__ selected_covered,
    float* __restrict__ selected_geom,
    float* __restrict__ selected_stable_h,
    float* __restrict__ selected_neighbor,
    float* __restrict__ selected_split,
    float* __restrict__ selected_h_a,
    float* __restrict__ selected_h_b,
    int temporary_anchor_base = 0)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int anchor = q0 + b;
    int temporary_anchor = anchor - temporary_anchor_base;
    if (scale == 0) {
        selected_scale[anchor] = -1;
        selected_index[anchor] = -1;
        selected_cut[anchor] = 0u;
        selected_covered[anchor] = 0u;
    }
    /* An uncovered scale scores -inf in the reference and can never win; the
     * per-scale kernel writes NaN into every record of such a scale, so it
     * must not be read either. */
    if (!temporary_covered[temporary_anchor]) return;
    float stable_h = temporary_stable_h[temporary_anchor];
    uint32_t cut = temporary_cut[temporary_anchor];
    bool better = !selected_covered[anchor] ||
        stable_h > selected_stable_h[anchor] ||
        (stable_h == selected_stable_h[anchor] && cut < selected_cut[anchor]);
    if (!better) return;
    selected_scale[anchor] = scale;
    selected_index[anchor] = temporary_index[temporary_anchor];
    selected_cut[anchor] = cut;
    selected_covered[anchor] = 1u;
    selected_geom[anchor] = temporary_geom[temporary_anchor];
    selected_stable_h[anchor] = stable_h;
    selected_neighbor[anchor] = temporary_neighbor[temporary_anchor];
    selected_split[anchor] = temporary_split[temporary_anchor];
    selected_h_a[anchor] = temporary_h_a[temporary_anchor];
    selected_h_b[anchor] = temporary_h_b[temporary_anchor];
}

template <typename Rank>
__global__ __launch_bounds__(32, 4)
void adaptive_compact_selected_topk_kernel(
    const float* __restrict__ sorted_keys,
    const Rank* __restrict__ sorted_ranks,
    const int32_t* __restrict__ selected_scale,
    const int32_t* __restrict__ response_counts,
    const int32_t* __restrict__ tmax_by_scale,
    int q0, int B, int kSlice, int output_stride,
    float* __restrict__ top_keys,
    Rank* __restrict__ top_ranks)
{
    int b = blockIdx.x;
    int lane = threadIdx.x;
    if (b >= B) return;
    int anchor = q0 + b;
    int scale = selected_scale[anchor];
    if (scale < 0) return;
    int response_count = response_counts[scale];
    int tmax = tmax_by_scale[scale];
    /* As above, the ballot count is warp-uniform; no shared counter is
       required and therefore no shared read/write race is possible. */
    int selected = 0;
    const float* key_row = sorted_keys + static_cast<size_t>(b) * kSlice;
    const Rank* rank_row = sorted_ranks + static_cast<size_t>(b) * kSlice;
    float* output_key = top_keys + static_cast<size_t>(b) * output_stride;
    Rank* output_rank = top_ranks + static_cast<size_t>(b) * output_stride;
    for (int base = 0; base < kSlice; base += 32) {
        int position = base + lane;
        bool include = position < kSlice &&
            static_cast<int>(rank_row[position]) < response_count;
        uint32_t ballot = __ballot_sync(0xffffffffu, include);
        int before = lane == 0 ? 0 :
            __popc(ballot & ((uint32_t(1) << lane) - 1u));
        int current = selected;
        int destination = current + before;
        if (include && destination < tmax) {
            output_key[destination] = key_row[position];
            output_rank[destination] = rank_row[position];
        }
        selected = current + __popc(ballot);
        __syncwarp();
        if (selected >= tmax) break;
    }
}

/* 24 is the hardware block ceiling for 32-thread blocks here, and asking for
   it pins the register budget at 65536/(24*32) = 85.  Before the dynamic-shared
   conversion this kernel was capped at 3 blocks/SM by 29760 bytes of static
   shared, so registers never mattered; now they are the only thing between it
   and the cap, and (…, 1) lets ptxas take up to 255.  At 80 registers it clears
   the cap by one block -- an accident, and one that would quietly hand back
   part of the 5.51x if a later edit pushed it to 88. */
template <typename Rank>
__global__ __launch_bounds__(32, 24)
void adaptive_final_extract_kernel(
    const float* __restrict__ Xcm,
    const int* __restrict__ idx_gpu,
    const float* __restrict__ top_dist2,
    const Rank* __restrict__ top_off,
    const int* __restrict__ cuts,
    const int32_t* __restrict__ response_counts,
    const int32_t* __restrict__ selected_scale,
    const int32_t* __restrict__ selected_index0,
    const float* __restrict__ selected_geom0,
    int R, int D, int MM, int kSlice, int tmax, int q0, int B,
    bool idx_shared_slice, float jitter,
    float* __restrict__ tangent0,
    float* __restrict__ center0,
    float* __restrict__ radius0,
    float* __restrict__ axial_tangent0,
    int32_t* __restrict__ num_neigh0,
    uint32_t* __restrict__ neigh_mask0_bits,
    int mask_words_per_row, int index_anchor_base = 0,
    int mask_anchor_base = 0)
{
    const unsigned full_mask = 0xffffffffu;
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B) return;
    int anchor = q0 + b;
    const float* xi = Xcm + static_cast<size_t>(anchor) * D;
    int scale = selected_scale[anchor];
    int selected_index = selected_index0[anchor];
    uint32_t* mask_i = neigh_mask0_bits +
        static_cast<size_t>(anchor - mask_anchor_base) * mask_words_per_row;

    extern __shared__ float dynamic_shared[];
    float* outer = dynamic_shared;
    float* covariance = outer + D * D;
    float* eigenvectors = covariance + D * D;
    float* sum = eigenvectors + D * D;
    float* eigenvalues = sum + D;
    float* center = eigenvalues + D;
    float* basis = center + D;

    for (int index = tid; index < D * MM; index += 32) basis[index] = 0.0f;
    for (int d = tid; d < D; d += 32) center[d] = xi[d];
    for (int word = tid; word < mask_words_per_row; word += 32)
        mask_i[word] = 0u;
    __syncwarp();

    if (scale < 0 || selected_index < 0) {
        for (int index = tid; index < D * MM; index += 32)
            tangent0[index + static_cast<size_t>(anchor) * D * MM] = 0.0f;
        for (int d = tid; d < D; d += 32)
            center0[d + static_cast<size_t>(anchor) * D] = xi[d];
        if (tid == 0) {
            radius0[anchor] = 0.0f;
            axial_tangent0[anchor] = 0.0f;
            num_neigh0[anchor] = 0;
        }
        return;
    }

    int cut = cuts[selected_index + scale * R];
    int response_count = response_counts[scale];
    const Rank* offsets = top_off + static_cast<size_t>(b) * tmax;
    for (int d = tid; d < D; d += 32) sum[d] = 0.0f;
    for (int index = tid; index < D * D; index += 32) outer[index] = 0.0f;
    __syncwarp();
    for (int rank = 0; rank < cut; ++rank) {
        int response_rank = static_cast<int>(offsets[rank]);
        int neighbor = idx_at(
            idx_gpu, anchor, response_rank, kSlice, idx_shared_slice, index_anchor_base);
        for (int d = tid; d < D; d += 32)
            covariance[d] = Xcm[d + static_cast<size_t>(neighbor) * D] - xi[d];
        __syncwarp();
        for (int d = tid; d < D; d += 32) sum[d] += covariance[d];
        for (int index = tid; index < D * D; index += 32) {
            int row = index / D;
            int column = index - row * D;
            outer[index] += covariance[row] * covariance[column];
        }
        __syncwarp();
    }
    float inverse = 1.0f / static_cast<float>(cut);
    for (int index = tid; index < D * D; index += 32) {
        int row = index / D;
        int column = index - row * D;
        covariance[index] = outer[index] * inverse -
            (sum[row] * inverse) * (sum[column] * inverse) +
            (row == column ? jitter : 0.0f);
    }
    __syncwarp();
    shared_jacobi_eigen_device_warp(covariance, D, eigenvalues, eigenvectors);

    int eigenvector_start = selected_geom0[anchor] > 0.0f ? 0 : D - MM;
    for (int index = tid; index < D * MM; index += 32) {
        int column = index / D;
        int row = index - column * D;
        basis[index] = eigenvectors[row * D + eigenvector_start + column];
    }
    __syncwarp();
    normalize_basis_device(basis, D, MM);
    for (int index = tid; index < D * MM; index += 32)
        tangent0[index + static_cast<size_t>(anchor) * D * MM] = basis[index];

    float radius2 = fmaxf(
        top_dist2[static_cast<size_t>(b) * tmax + cut - 1], 0.0f);
    float radius_limit = radius2 + 1.0e-6f * fmaxf(1.0f, radius2);
    for (int candidate = tid; candidate < response_count; candidate += 32) {
        int neighbor = idx_at(
            idx_gpu, anchor, candidate, kSlice, idx_shared_slice, index_anchor_base);
        float distance2 = 0.0f;
        for (int d = 0; d < D; ++d) {
            float difference = Xcm[d + static_cast<size_t>(neighbor) * D] - xi[d];
            distance2 += difference * difference;
        }
        if (distance2 <= radius_limit) {
            atomicOr(reinterpret_cast<unsigned int*>(mask_i + (candidate >> 5)),
                uint32_t(1) << (candidate & 31));
        }
    }
    __syncwarp();

    int local_count = 0;
    for (int candidate = tid; candidate < response_count; candidate += 32)
        if (bitset_get_u32(mask_i, candidate)) ++local_count;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(full_mask, local_count, offset);
    int selected_count = __shfl_sync(full_mask, local_count, 0);
    float inverse_count = selected_count > 0 ?
        1.0f / static_cast<float>(selected_count) : 0.0f;

    for (int d = 0; d < D; ++d) {
        float local_sum = 0.0f;
        for (int candidate = tid; candidate < response_count; candidate += 32) {
            if (!bitset_get_u32(mask_i, candidate)) continue;
            int neighbor = idx_at(
                idx_gpu, anchor, candidate, kSlice, idx_shared_slice, index_anchor_base);
            local_sum += Xcm[d + static_cast<size_t>(neighbor) * D] - xi[d];
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(full_mask, local_sum, offset);
        if (tid == 0) center[d] = xi[d] + local_sum * inverse_count;
    }
    __syncwarp();
    for (int d = tid; d < D; d += 32)
        center0[d + static_cast<size_t>(anchor) * D] = center[d];
    __syncwarp();

    float local_axial = 0.0f;
    for (int candidate = tid; candidate < response_count; candidate += 32) {
        if (!bitset_get_u32(mask_i, candidate)) continue;
        int neighbor = idx_at(
            idx_gpu, anchor, candidate, kSlice, idx_shared_slice, index_anchor_base);
        float projection2 = 0.0f;
        for (int column = 0; column < MM; ++column) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d)
                dot += basis[d + D * column] *
                    (Xcm[d + static_cast<size_t>(neighbor) * D] - center[d]);
            projection2 += dot * dot;
        }
        local_axial = fmaxf(local_axial, sqrtf(projection2));
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_axial = fmaxf(local_axial,
            __shfl_down_sync(full_mask, local_axial, offset));
    if (tid == 0) {
        /* radius2 keeps defining the neighbourhood (radius_limit above); only
         * the REPORTED Rad0 changes. Helper is shared with the fixed-scale
         * extract kernel (manor_m_phase1_mex.cu). */
        radius0[anchor] = isotropic_radius(eigenvalues, D);
        axial_tangent0[anchor] = local_axial;
        num_neigh0[anchor] = selected_count;
    }
}

__global__ void adaptive_owner_stats_kernel(
    const uint8_t* __restrict__ covered,
    const int32_t* __restrict__ num_neigh,
    int N,
    unsigned long long* __restrict__ stats)
{
    int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= N || !covered[anchor]) return;
    atomicAdd(stats, 1ULL);
    atomicAdd(stats + 1, static_cast<unsigned long long>(num_neigh[anchor]));
}

__global__ void adaptive_owner_threshold_kernel(
    const unsigned long long* __restrict__ stats,
    int D, double heavy_factor,
    double* __restrict__ threshold)
{
    if (blockIdx.x || threadIdx.x) return;
    double mean = stats[0] > 0ULL ?
        static_cast<double>(stats[1]) / static_cast<double>(stats[0]) : 0.0;
    threshold[0] = fmax(2.0 * static_cast<double>(D), heavy_factor * mean);
}

__global__ void adaptive_owner_mask_kernel(
    const uint8_t* __restrict__ covered,
    const int32_t* __restrict__ num_neigh,
    const double* __restrict__ threshold,
    int N,
    uint8_t* __restrict__ owner,
    uint8_t* __restrict__ rejection,
    unsigned long long* __restrict__ stats)
{
    int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= N) return;
    bool retained = covered[anchor] &&
        static_cast<double>(num_neigh[anchor]) > threshold[0];
    owner[anchor] = retained ? 1u : 0u;
    rejection[anchor] = retained ? 0u : (covered[anchor] ? 2u : 1u);
    if (retained) atomicAdd(stats + 2, 1ULL);
}

__global__ __launch_bounds__(32, 2)
void adaptive_owner_finalize_kernel_mm(
    const float* __restrict__ covariance_in,
    const float* __restrict__ center_sum_in,
    const int32_t* __restrict__ count_in,
    const float* __restrict__ Xcm,
    const float* __restrict__ tangent_raw,
    const float* __restrict__ center_raw,
    const uint8_t* __restrict__ owner_mask0,
    int D, int MM, int q0, int B,
    const double* __restrict__ support_threshold,
    float* __restrict__ tangent0,
    float* __restrict__ center0)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    int anchor = q0 + b;
    if (!owner_mask0[anchor]) return;

    extern __shared__ float dynamic_shared[];
    float* matrix = dynamic_shared;
    float* eigenvectors = matrix + D * D;
    float* eigenvalues = eigenvectors + D * D;
    float* center = eigenvalues + D;
    float* anchor_point = center + D;
    float* tangent = anchor_point + D;
    __shared__ int use_eigenvectors;
    __shared__ int refined;
    for (int index = tid; index < D * MM; index += 32) {
        int column = index / D;
        int d = index - column * D;
        tangent[index] = tangent_raw[
            d + static_cast<size_t>(D) * (column + static_cast<size_t>(MM) * anchor)];
    }
    for (int d = tid; d < D; d += 32) {
        center[d] = center_raw[d + static_cast<size_t>(anchor) * D];
        anchor_point[d] = Xcm[d + static_cast<size_t>(anchor) * D];
    }
    if (tid == 0) { use_eigenvectors = 0; refined = 0; }
    __syncwarp();

    int count = count_in[b];
    if (static_cast<double>(count) > support_threshold[0]) {
        const float* covariance = covariance_in + static_cast<size_t>(b) * D * D;
        for (int index = tid; index < D * D; index += 32)
            matrix[index] = covariance[index];
        __syncwarp();
        shared_jacobi_eigen_device_warp(matrix, D, eigenvalues, eigenvectors);
        __syncwarp();
        if (tid == 0) use_eigenvectors = eigenvalues[D - 1] > 0.0f ? 1 : 0;
        __syncwarp();
        if (use_eigenvectors) {
            for (int column = 0; column < MM; ++column) {
                int eigen_column = D - MM + column;
                float local_sign = 0.0f;
                for (int d = tid; d < D; d += 32)
                    local_sign += eigenvectors[d * D + eigen_column] *
                        tangent[d + column * D];
                for (int offset = 16; offset > 0; offset >>= 1)
                    local_sign += __shfl_xor_sync(0xffffffffu, local_sign, offset);
                float sign = local_sign >= 0.0f ? 1.0f : -1.0f;
                float local_norm2 = 0.0f;
                for (int d = tid; d < D; d += 32) {
                    float value = eigenvectors[d * D + eigen_column];
                    local_norm2 += value * value;
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    local_norm2 += __shfl_xor_sync(
                        0xffffffffu, local_norm2, offset);
                float inverse_norm = local_norm2 > 1.0e-30f ?
                    rsqrtf(local_norm2) : 0.0f;
                for (int d = tid; d < D; d += 32)
                    tangent[d + column * D] =
                        eigenvectors[d * D + eigen_column] * inverse_norm * sign;
                __syncwarp();
            }
            const float* center_sum = center_sum_in + static_cast<size_t>(b) * D;
            float inverse_count = 1.0f / static_cast<float>(count);
            for (int d = tid; d < D; d += 32)
                center[d] = center_sum[d] * inverse_count;
            if (tid == 0) refined = 1;
            __syncwarp();
        }
    }
    if (refined) {
        for (int column = 0; column < MM; ++column) {
            float local_dot = 0.0f;
            for (int d = tid; d < D; d += 32)
                local_dot += tangent[d + column * D] *
                    (center[d] - anchor_point[d]);
            for (int offset = 16; offset > 0; offset >>= 1)
                local_dot += __shfl_xor_sync(0xffffffffu, local_dot, offset);
            float dot = __shfl_sync(0xffffffffu, local_dot, 0);
            for (int d = tid; d < D; d += 32)
                center[d] -= dot * tangent[d + column * D];
            __syncwarp();
        }
    }
    float* tangent_out = tangent0 + static_cast<size_t>(D) * MM * anchor;
    float* center_out = center0 + static_cast<size_t>(D) * anchor;
    for (int index = tid; index < D * MM; index += 32)
        tangent_out[index] = tangent[index];
    for (int d = tid; d < D; d += 32) center_out[d] = center[d];
}

} // namespace



#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
namespace manor_knn {

/* ------------------------------------------------------------------ limits */

/* Y is the response; Phase 2 already caps its width at MAX_MY_MANOR = 8. */
static const int kMaxDy = 8;

/* Ranks are uint16 downstream (manor_m_phase1_mex.cu rejects kSlice > 65535),
 * but a GPU-resident [kSlice x N] index runs out of memory long before that:
 * 2.4 GB at N = 1e6, 38.4 GB at N = 4e6.  Fail at the real ceiling, loudly. */
static const int kMaxK = 65535;

static const int kHistBuckets = 256;   /* also the radix-select digit width */
/* Scan launch configuration. */
static const int kScanWarps = 8;       /* warps per scan block              */
static const int kScanTile = 256;      /* reference points staged in shared */
static const int kScanQueryBlock = 8;  /* queries held in registers, per warp */
static const int kHistWarps = 8;
static const int kHistTile = 512;

/* Fine threshold-histogram configuration. */
static const int kFineShift = 18;
static const int kOctaveShift = 23;    /* bootstrap: one bucket = one octave */

/* Bootstrap threshold target. */
static const double kTauTargetFactor = 2.29;
static const int kSubStride = 32;

/* Candidate-storage multiplier; overflow triggers a retry. */
static const int kCapFactor = 6;

/* Extra candidates carried into the float64 re-rank. */
static const int kRefineBand = 32;

/* Slack above k+band in the compacted buffer, for exact float32 key ties. */
static const int kCompactSlack = 64;

/* Scratch space for the segmented radix sort. */
static const size_t kSortTempSlack = 4096;

/* ------------------------------------------------------------------- errors */

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& text) : std::runtime_error(text) {}
};

#define MANOR_KNN_CHECK(expression)                                            \
    do {                                                                       \
        cudaError_t knn_status_ = (expression);                                \
        if (knn_status_ != cudaSuccess) {                                      \
            throw ::manor_knn::Error(                                          \
                std::string("response kNN CUDA failure: ") +                   \
                cudaGetErrorString(knn_status_) + " (" #expression ")");       \
        }                                                                      \
    } while (0)

/* -------------------------------------------------------------- diagnostics */

/* Slot layout of the device diagnostic counters. */
enum DiagSlot {
    kDiagOwnerMisplaced = 0,  /* rank 0 was not the query itself             */
    kDiagExactTie = 1,        /* adjacent emitted ranks at equal float64 d2  */
    kDiagCompactOverflow = 2, /* compacted survivors exceeded their capacity */
    kDiagSlots = 3
};

struct Stats {
    double bootstrap_ms = 0.0;
    double threshold_ms = 0.0;
    double scan_ms = 0.0;
    double retry_ms = 0.0;
    double select_ms = 0.0;
    double refine_ms = 0.0;
    double sort_ms = 0.0;
    double emit_ms = 0.0;
    double total_ms = 0.0;
    long long retry_rounds = 0;
    long long retry_queries = 0;
    unsigned long long owner_misplaced = 0;
    unsigned long long exact_ties = 0;
    unsigned long long compact_overflow = 0;
    double mean_candidates = 0.0;
    double max_candidates = 0.0;
};

/* ------------------------------------------------------------ small helpers */

__host__ __device__ __forceinline__ unsigned int knn_bucket_base(
    float high, int shift)
{
    unsigned int bits = 0u;
#if defined(__CUDA_ARCH__)
    bits = __float_as_uint(high);
#else
    memcpy(&bits, &high, sizeof(bits));
#endif
    return (bits >> shift) - static_cast<unsigned int>(kHistBuckets);
}

__device__ __forceinline__ float knn_bucket_edge(
    unsigned int base, int bucket, int shift)
{
    return __uint_as_float((base + static_cast<unsigned int>(bucket))
                           << shift);
}

/* ------------------------------------------------------------ Ysub / Yaos */

/* Compacted strided subsample, so the threshold pass reads a contiguous SoA
 * block instead of a stride-32 gather (which would cost one sector per point
 * per query block). */
__global__ void knn_subsample_kernel(
    const float* __restrict__ Y, int N, int DY, int stride, int Ns,
    float* __restrict__ Ysub)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Ns * DY) return;
    int d = index / Ns;
    int s = index - d * Ns;
    Ysub[static_cast<size_t>(d) * Ns + s] =
        Y[static_cast<size_t>(d) * N + static_cast<size_t>(s) * stride];
}

/* Padded array-of-structures copy, 8 floats (32 B, one sector) per point.  The
 * float64 re-rank gathers scattered points; SoA would cost DY sectors each. */
__global__ void knn_pack_aos_kernel(
    const float* __restrict__ Y, int N, int DY, float* __restrict__ Yaos)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;
    for (int d = 0; d < kMaxDy; ++d) {
        Yaos[static_cast<size_t>(index) * kMaxDy + d] =
            (d < DY) ? Y[static_cast<size_t>(d) * N + index] : 0.0f;
    }
}

/* ------------------------------------------------------------- pass A: tau */

/* One warp per query.  Histograms the subsampled squared distances into 256
 * power-of-two buckets and returns the bucket edge at which the cumulative
 * count reaches target_rank.  Buckets below the window are folded into bucket
 * 0 (conservative: the returned edge is then an over-estimate); candidates
 * above the window are dropped, and a query whose count never reaches the
 * target gets the top edge and is corrected by the scan's retry loop. */
template <int DY, int W, int R>
__global__ __launch_bounds__(W * 32) void knn_threshold_kernel(
    const float* __restrict__ Ysub, int Ns,
    const float* __restrict__ Y, int N,
    int query_first, int query_stride, int query_count,
    unsigned int bucket_base, int shift, int target_rank,
    float* __restrict__ tau_out)
{
    extern __shared__ float knn_threshold_shared[];
    float* tile = knn_threshold_shared;                    /* R * DY */
    unsigned int* hist =
        reinterpret_cast<unsigned int*>(tile + R * DY);    /* W * 256 */

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x * W + warp;
    const bool active = (slot < query_count);
    const int query = query_first + slot * query_stride;

    unsigned int* my_hist = hist + static_cast<size_t>(warp) * kHistBuckets;
    for (int b = lane; b < kHistBuckets; b += 32) my_hist[b] = 0u;

    float q[DY];
#pragma unroll
    for (int d = 0; d < DY; ++d)
        q[d] = active ? Y[static_cast<size_t>(d) * N + query] : 0.0f;
    __syncwarp();

    for (int base = 0; base < Ns; base += R) {
        const int len = min(R, Ns - base);
        __syncthreads();
        for (int d = 0; d < DY; ++d)
            for (int t = threadIdx.x; t < len; t += W * 32)
                tile[d * R + t] = Ysub[static_cast<size_t>(d) * Ns + base + t];
        __syncthreads();
        if (!active) continue;
        for (int t = lane; t < len; t += 32) {
            float d2 = 0.0f;
#pragma unroll
            for (int d = 0; d < DY; ++d) {
                float diff = tile[d * R + t] - q[d];
                d2 = fmaf(diff, diff, d2);
            }
            int bucket = static_cast<int>((__float_as_uint(d2) >> shift) -
                                          bucket_base);
            if (bucket < kHistBuckets) {
                if (bucket < 0) bucket = 0;
                atomicAdd(&my_hist[bucket], 1u);
            }
        }
    }
    __syncthreads();
    if (!active || lane != 0) return;

    unsigned int cumulative = 0u;
    int chosen = kHistBuckets;
    for (int b = 0; b < kHistBuckets; ++b) {
        cumulative += my_hist[b];
        if (cumulative >= static_cast<unsigned int>(target_rank)) {
            chosen = b;
            break;
        }
    }
    tau_out[slot] = knn_bucket_edge(bucket_base, chosen + 1, shift);
}

/* -------------------------------------------------------------- pass B: scan */

/* Register-blocked over queries: each warp owns QB queries and each lane owns
 * one reference point of the shared tile, so a tile point is loaded once and
 * reused QB times.  Per-query shared state is one counter. */
template <int DY, int W, int R, int QB>
__global__ __launch_bounds__(W * 32) void knn_scan_kernel(
    const float* __restrict__ Y, int N, int query_first,
    const int* __restrict__ slot_list, int slot_count,
    const float* __restrict__ tau, int cap,
    unsigned int* __restrict__ count,
    unsigned int* __restrict__ candidate_key,
    int* __restrict__ candidate_value)
{
    extern __shared__ float knn_scan_shared[];
    float* tile = knn_scan_shared;                              /* R * DY  */
    unsigned int* local_count =
        reinterpret_cast<unsigned int*>(tile + R * DY);         /* W * QB  */

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int list_base = (blockIdx.x * W + warp) * QB;
    unsigned int* my_count = local_count + warp * QB;

    /* The slot ids are deliberately NOT held in registers.  They are needed
     * only inside the append branch, which fires for about one candidate in
     * 1600, and once at the end; recomputing them there instead of carrying
     * QB of them through the scan is worth two extra blocks per SM. */
    float q[QB][DY];
    float threshold[QB];
#pragma unroll
    for (int u = 0; u < QB; ++u) {
        int position = list_base + u;
        bool valid = (position < slot_count);
        int slot = valid ? (slot_list ? slot_list[position] : position) : -1;
        /* An invalid lane keeps a negative threshold, so d2 < threshold is
         * false for every non-negative d2 and it never appends. */
        threshold[u] = valid ? tau[slot] : -1.0f;
        int query = valid ? (query_first + slot) : 0;
#pragma unroll
        for (int d = 0; d < DY; ++d)
            q[u][d] = Y[static_cast<size_t>(d) * N + query];
        if (lane == 0) my_count[u] = 0u;
    }
    __syncwarp();

    for (int base = 0; base < N; base += R) {
        const int len = min(R, N - base);
        __syncthreads();
        for (int d = 0; d < DY; ++d)
            for (int t = threadIdx.x; t < len; t += W * 32)
                tile[d * R + t] = Y[static_cast<size_t>(d) * N + base + t];
        __syncthreads();
        for (int t = lane; t < len; t += 32) {
            float reference[DY];
#pragma unroll
            for (int d = 0; d < DY; ++d) reference[d] = tile[d * R + t];
#pragma unroll
            for (int u = 0; u < QB; ++u) {
                float d2 = 0.0f;
#pragma unroll
                for (int d = 0; d < DY; ++d) {
                    float diff = reference[d] - q[u][d];
                    d2 = fmaf(diff, diff, d2);
                }
                if (d2 < threshold[u]) {
                    unsigned int position = atomicAdd(&my_count[u], 1u);
                    if (position < static_cast<unsigned int>(cap)) {
                        int slot = slot_list ? slot_list[list_base + u]
                                             : (list_base + u);
                        size_t offset =
                            static_cast<size_t>(slot) * cap + position;
                        candidate_key[offset] = __float_as_uint(d2);
                        candidate_value[offset] = base + t;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (lane != 0) return;
#pragma unroll
    for (int u = 0; u < QB; ++u) {
        int position = list_base + u;
        if (position >= slot_count) continue;
        int slot = slot_list ? slot_list[position] : position;
        count[slot] = my_count[u];
    }
}

/* --------------------------------------------------------- retry bookkeeping */

/* Geometric bracketing on tau.  A shortfall raises the lower bound, an
 * overflow lowers the upper bound, and once both are known the next threshold
 * is their geometric mean, so the loop converges without oscillating. */
__global__ void knn_retry_kernel(
    const unsigned int* __restrict__ count, int slot_count,
    int want, int cap,
    float* __restrict__ tau, float* __restrict__ tau_low,
    float* __restrict__ tau_high,
    int* __restrict__ retry_list, unsigned int* __restrict__ retry_count)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) return;
    unsigned int have = count[slot];
    if (have >= static_cast<unsigned int>(want) &&
        have <= static_cast<unsigned int>(cap)) {
        return;
    }
    float current = tau[slot];
    float low = tau_low[slot];
    float high = tau_high[slot];
    if (have < static_cast<unsigned int>(want)) {
        low = current;
        tau[slot] = (high > 0.0f) ? sqrtf(low * high) : (current * 4.0f);
    } else {
        high = current;
        tau[slot] = (low > 0.0f) ? sqrtf(low * high) : (current * 0.5f);
    }
    tau_low[slot] = low;
    tau_high[slot] = high;
    retry_list[atomicAdd(retry_count, 1u)] = slot;
}

/* ------------------------------------------------- pass C1: select + compact */

/* Exact radix-select of the want-th smallest key, four 8-bit digits, then a
 * compaction of everything at or below it.  One block per query. */
__global__ void knn_select_kernel(
    const unsigned int* __restrict__ candidate_key,
    const int* __restrict__ candidate_value,
    const unsigned int* __restrict__ count, int cap, int want,
    int compact_cap, int* __restrict__ compact_value,
    int* __restrict__ compact_length,
    unsigned long long* __restrict__ diagnostics)
{
    __shared__ unsigned int histogram[kHistBuckets];
    __shared__ unsigned int written;
    __shared__ unsigned int selected;
    __shared__ unsigned int below;

    const int slot = blockIdx.x;
    const size_t base = static_cast<size_t>(slot) * cap;
    const unsigned int have = count[slot];

    unsigned int prefix = 0u;
    unsigned int remaining = static_cast<unsigned int>(want);
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < kHistBuckets; b += blockDim.x)
            histogram[b] = 0u;
        __syncthreads();
        const unsigned int mask =
            (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (unsigned int j = threadIdx.x; j < have; j += blockDim.x) {
            unsigned int key = candidate_key[base + j];
            if ((key & mask) == (prefix & mask))
                atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int cumulative = 0u;
            int digit = kHistBuckets - 1;
            for (int b = 0; b < kHistBuckets; ++b) {
                if (cumulative + histogram[b] >= remaining) {
                    digit = b;
                    break;
                }
                cumulative += histogram[b];
            }
            selected = static_cast<unsigned int>(digit);
            below = cumulative;
        }
        __syncthreads();
        prefix |= selected << shift;
        remaining -= below;
        __syncthreads();
    }

    if (threadIdx.x == 0) written = 0u;
    __syncthreads();
    for (unsigned int j = threadIdx.x; j < have; j += blockDim.x) {
        if (candidate_key[base + j] <= prefix) {
            unsigned int position = atomicAdd(&written, 1u);
            if (position < static_cast<unsigned int>(compact_cap)) {
                compact_value[static_cast<size_t>(slot) * compact_cap +
                              position] = candidate_value[base + j];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (written > static_cast<unsigned int>(compact_cap)) {
            atomicAdd(&diagnostics[kDiagCompactOverflow], 1ull);
            written = static_cast<unsigned int>(compact_cap);
        }
        compact_length[slot] = static_cast<int>(written);
    }
}

__global__ void knn_offsets_kernel(
    const int* __restrict__ compact_length, int slot_count, int compact_cap,
    int* __restrict__ offset_end)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) return;
    offset_end[slot] = slot * compact_cap + compact_length[slot];
}

/* ------------------------------------------------ pass C2: float64 re-rank */

/* Survivors are recomputed in float64 before final ordering. */
__global__ void knn_refine_kernel(
    const float* __restrict__ Yaos, int DY, int query_first,
    const int* __restrict__ compact_value,
    const int* __restrict__ compact_length,
    int slot_count, int compact_cap,
    unsigned long long* __restrict__ refined_key)
{
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(slot_count) * compact_cap;
    if (index >= total) return;
    int slot = static_cast<int>(index / compact_cap);
    int rank = static_cast<int>(index - static_cast<size_t>(slot) * compact_cap);
    if (rank >= compact_length[slot]) {
        refined_key[index] = ~0ull;
        return;
    }
    const float* query = Yaos + static_cast<size_t>(query_first + slot) * kMaxDy;
    const float* point =
        Yaos + static_cast<size_t>(compact_value[index]) * kMaxDy;
    double d2 = 0.0;
    for (int d = 0; d < DY; ++d) {
        double diff = static_cast<double>(point[d]) - static_cast<double>(query[d]);
        d2 += diff * diff;
    }
    /* d2 >= 0, so the IEEE bit pattern is already monotone as an unsigned. */
    refined_key[index] = static_cast<unsigned long long>(__double_as_longlong(d2));
}

/* --------------------------------------------------------- pass C3: emit */

#include "../../cuda/core/manor_m_knn_ties.cuh"

__global__ void knn_emit_kernel(
    const unsigned long long* __restrict__ sorted_key,
    const int* __restrict__ sorted_value,
    const int* __restrict__ compact_length,
    int slot_count, int compact_cap, int query_first, int K,
    int* __restrict__ index_out,
    unsigned long long* __restrict__ diagnostics)
{
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(slot_count) * compact_cap;
    if (index >= total) return;
    int slot = static_cast<int>(index / compact_cap);
    int rank = static_cast<int>(index - static_cast<size_t>(slot) * compact_cap);
    if (rank >= compact_length[slot]) return;
    const size_t base = static_cast<size_t>(slot) * compact_cap;
    const int target = manor_knn_tie_rank(sorted_key + base, sorted_value + base,
        compact_length[slot], rank, query_first + slot, K);
    if (target >= K) return;
    size_t source = static_cast<size_t>(slot) * compact_cap + rank;
    int value = sorted_value[source];
    index_out[static_cast<size_t>(query_first + slot) * K + target] = value;
    if (target == 0) {
        if (value != query_first + slot)
            atomicAdd(&diagnostics[kDiagOwnerMisplaced], 1ull);
    } else if (sorted_key[source] == sorted_key[base + target - 1]) {
        atomicAdd(&diagnostics[kDiagExactTie], 1ull);
    }
}

/* ============================================================ host driver */

namespace detail {

template <typename T>
class Buffer {
public:
    Buffer() : pointer_(nullptr) {}
    ~Buffer() { if (pointer_) cudaFree(pointer_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* allocate(size_t count)
    {
        if (pointer_) { cudaFree(pointer_); pointer_ = nullptr; }
        if (count == 0) return nullptr;
        MANOR_KNN_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
        return pointer_;
    }
    T* get() const { return pointer_; }
private:
    T* pointer_;
};

/* A CUDA event pair that reports elapsed milliseconds, or does nothing when
 * timing is off.  There is exactly one timing path; nothing switches engines. */
class Timer {
public:
    explicit Timer(bool enabled) : enabled_(enabled)
    {
        if (!enabled_) return;
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~Timer()
    {
        if (!enabled_) return;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void begin() { if (enabled_) cudaEventRecord(start_); }
    void end(double* accumulator)
    {
        if (!enabled_) return;
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        *accumulator += milliseconds;
    }
private:
    bool enabled_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

template <int DY>
void launch_threshold(const float* Ysub, int Ns, const float* Y, int N,
                      int query_first, int query_stride, int query_count,
                      unsigned int bucket_base, int shift, int target_rank,
                      float* tau_out)
{
    const int W = kHistWarps;
    const int R = kHistTile;
    size_t shared = static_cast<size_t>(R) * DY * sizeof(float) +
                    static_cast<size_t>(W) * kHistBuckets * sizeof(unsigned int);
    int blocks = (query_count + W - 1) / W;
    knn_threshold_kernel<DY, W, R><<<blocks, W * 32, shared>>>(
        Ysub, Ns, Y, N, query_first, query_stride, query_count,
        bucket_base, shift, target_rank, tau_out);
}

template <int DY>
void launch_scan(const float* Y, int N, int query_first,
                 const int* slot_list, int slot_count, const float* tau,
                 int cap, unsigned int* count, unsigned int* candidate_key,
                 int* candidate_value)
{
    const int W = kScanWarps;
    const int R = kScanTile;
    const int QB = kScanQueryBlock;
    size_t shared = static_cast<size_t>(R) * DY * sizeof(float) +
                    static_cast<size_t>(W) * QB * sizeof(unsigned int);
    int per_block = W * QB;
    int blocks = (slot_count + per_block - 1) / per_block;
    knn_scan_kernel<DY, W, R, QB><<<blocks, W * 32, shared>>>(
        Y, N, query_first, slot_list, slot_count, tau, cap, count,
        candidate_key, candidate_value);
}

typedef void (*ThresholdLauncher)(const float*, int, const float*, int, int,
                                  int, int, unsigned int, int, int, float*);
typedef void (*ScanLauncher)(const float*, int, int, const int*, int,
                             const float*, int, unsigned int*, unsigned int*,
                             int*);

inline ThresholdLauncher threshold_launcher(int DY)
{
    switch (DY) {
        case 1: return &launch_threshold<1>;
        case 2: return &launch_threshold<2>;
        case 3: return &launch_threshold<3>;
        case 4: return &launch_threshold<4>;
        case 5: return &launch_threshold<5>;
        case 6: return &launch_threshold<6>;
        case 7: return &launch_threshold<7>;
        case 8: return &launch_threshold<8>;
        default: return nullptr;
    }
}

inline ScanLauncher scan_launcher(int DY)
{
    switch (DY) {
        case 1: return &launch_scan<1>;
        case 2: return &launch_scan<2>;
        case 3: return &launch_scan<3>;
        case 4: return &launch_scan<4>;
        case 5: return &launch_scan<5>;
        case 6: return &launch_scan<6>;
        case 7: return &launch_scan<7>;
        case 8: return &launch_scan<8>;
        default: return nullptr;
    }
}

}  // namespace detail

struct Options {

    size_t scratch_bytes = static_cast<size_t>(7) << 30;
    bool collect_stats = false;
    int max_retry_rounds = 12;
};

/* Build the exact [K x query_count] zero-based response-neighbour index.
 *
 * Y             device, [DY x N] float32, SoA (Y[d * N + i]).
 * index_out     device, K * query_count int32, written as
 *               index_out[anchor * K + rank] -- the layout idx_at reads.
 * query_first   anchors are the first query_count reference points, matching
 *               Phase 1's active_anchor_count prefix.
 */
inline void build_index(const float* Y, int N, int DY, int query_count, int K,
                        int* index_out, const Options& options, Stats* stats)
{
    if (N < 1) throw Error("response kNN: N must be positive.");
    if (DY < 1 || DY > kMaxDy)
        throw Error("response kNN: the response width must lie in [1," +
                    std::to_string(kMaxDy) + "].");
    if (query_count < 1 || query_count > N)
        throw Error("response kNN: the anchor count must lie in [1,N].");
    if (K < 1 || K > N)
        throw Error("response kNN: kSlice must lie in [1,N].");
    if (K > kMaxK)
        throw Error("response kNN: kSlice exceeds the uint16 rank limit of " +
                    std::to_string(kMaxK) + ".");

    detail::ThresholdLauncher threshold = detail::threshold_launcher(DY);
    detail::ScanLauncher scan = detail::scan_launcher(DY);
    if (!threshold || !scan)
        throw Error("response kNN: unsupported response width.");

    Stats local;
    const bool timing = options.collect_stats;
    detail::Timer timer(timing);

    const int stride = std::min(kSubStride, std::max(1, N / 1024));
    const int Ns = (N + stride - 1) / stride;
    const int cap = std::min(N, std::max(K + kRefineBand, K * kCapFactor));
    const int want = std::min(N, K + kRefineBand);
    const int compact_cap = std::min(cap, want + kCompactSlack);
    int target_rank = static_cast<int>(
        (static_cast<double>(K) * kTauTargetFactor) / stride + 0.5);
    target_rank = std::max(1, std::min(Ns, target_rank));

    /* Allocations that do not scale with the chunk: the strided subsample, the
     * padded AoS copy of Y that the float64 re-rank gathers from, the
     * diagnostic counters, and CUB's alignment constant. */
    const size_t fixed_bytes =
        static_cast<size_t>(Ns) * DY * sizeof(float) +
        static_cast<size_t>(N) * kMaxDy * sizeof(float) +
        kDiagSlots * sizeof(unsigned long long) +
        kSortTempSlack;

    /* Per slot: the candidate store, the survivors and their sorted output,
     * the six one-per-slot bookkeeping arrays with their two segment offsets,
     * and the segmented sort's own workspace.  That last term is the one that
     * used to be missing -- CUB allocates it on top of everything above, so a
     * budget without it is an 18.9 % undercount at k=600. */
    const size_t sort_temp_per_slot = static_cast<size_t>(compact_cap) *
        (sizeof(unsigned long long) + sizeof(int));
    const size_t bytes_per_slot =
        static_cast<size_t>(cap) * (sizeof(unsigned int) + sizeof(int)) +
        static_cast<size_t>(compact_cap) *
            (2 * sizeof(unsigned long long) + 2 * sizeof(int)) +
        sort_temp_per_slot +
        8 * sizeof(int);
    if (options.scratch_bytes < fixed_bytes + bytes_per_slot) {
        throw Error("response kNN: a budget of " +
                    std::to_string(options.scratch_bytes) +
                    " bytes cannot hold one anchor; " +
                    std::to_string(fixed_bytes + bytes_per_slot) +
                    " is the minimum at this N and kSlice.");
    }
    const size_t affordable =
        (options.scratch_bytes - fixed_bytes) / bytes_per_slot;
    int chunk = static_cast<int>(std::min<size_t>(
        affordable, static_cast<size_t>(query_count)));

    detail::Buffer<float> subsample_buffer, aos_buffer;
    detail::Buffer<float> tau_buffer, tau_low_buffer, tau_high_buffer;
    detail::Buffer<unsigned int> count_buffer, candidate_key_buffer;
    detail::Buffer<int> candidate_value_buffer, compact_value_buffer;
    detail::Buffer<int> compact_length_buffer, retry_list_buffer;
    detail::Buffer<int> offset_begin_buffer, offset_end_buffer;
    detail::Buffer<int> sorted_value_buffer;
    detail::Buffer<unsigned long long> refined_key_buffer, sorted_key_buffer;
    detail::Buffer<unsigned int> retry_count_buffer;
    detail::Buffer<unsigned long long> diagnostics_buffer;
    detail::Buffer<unsigned char> sort_temp_buffer;

    float* Ysub = subsample_buffer.allocate(static_cast<size_t>(Ns) * DY);
    float* Yaos = aos_buffer.allocate(static_cast<size_t>(N) * kMaxDy);
    float* tau = tau_buffer.allocate(chunk);
    float* tau_low = tau_low_buffer.allocate(chunk);
    float* tau_high = tau_high_buffer.allocate(chunk);
    unsigned int* count = count_buffer.allocate(chunk);
    unsigned int* candidate_key =
        candidate_key_buffer.allocate(static_cast<size_t>(chunk) * cap);
    int* candidate_value =
        candidate_value_buffer.allocate(static_cast<size_t>(chunk) * cap);
    int* compact_value =
        compact_value_buffer.allocate(static_cast<size_t>(chunk) * compact_cap);
    int* compact_length = compact_length_buffer.allocate(chunk);
    int* retry_list = retry_list_buffer.allocate(chunk);
    unsigned int* retry_count = retry_count_buffer.allocate(1);
    unsigned long long* refined_key =
        refined_key_buffer.allocate(static_cast<size_t>(chunk) * compact_cap);
    unsigned long long* sorted_key =
        sorted_key_buffer.allocate(static_cast<size_t>(chunk) * compact_cap);
    int* sorted_value =
        sorted_value_buffer.allocate(static_cast<size_t>(chunk) * compact_cap);
    int* offset_begin = offset_begin_buffer.allocate(chunk + 1);
    int* offset_end = offset_end_buffer.allocate(chunk + 1);
    unsigned long long* diagnostics = diagnostics_buffer.allocate(kDiagSlots);
    MANOR_KNN_CHECK(cudaMemset(diagnostics, 0,
                               kDiagSlots * sizeof(unsigned long long)));

    {
        int threads = 256;
        int blocks = (Ns * DY + threads - 1) / threads;
        knn_subsample_kernel<<<blocks, threads>>>(Y, N, DY, stride, Ns, Ysub);
        blocks = (N + threads - 1) / threads;
        knn_pack_aos_kernel<<<blocks, threads>>>(Y, N, DY, Yaos);
        MANOR_KNN_CHECK(cudaPeekAtLastError());
    }

    /* Segment offsets are fixed: segment s starts at s * compact_cap and ends
     * at s * compact_cap + compact_length[s]. */
    std::vector<int> begin_host(chunk + 1);
    for (int s = 0; s <= chunk; ++s) begin_host[s] = s * compact_cap;
    MANOR_KNN_CHECK(cudaMemcpy(offset_begin, begin_host.data(),
                               (chunk + 1) * sizeof(int),
                               cudaMemcpyHostToDevice));

    size_t sort_temp_bytes = 0;
    MANOR_KNN_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, sort_temp_bytes, refined_key, sorted_key, compact_value,
        sorted_value, chunk * compact_cap, chunk, offset_begin, offset_end));
    /* The chunk above was chosen with sort_temp_per_slot reserved for exactly
     * this allocation.  A CUB that asks for more would put the search back
     * over the budget it was given, silently; say so instead. */
    if (sort_temp_bytes >
        static_cast<size_t>(chunk) * sort_temp_per_slot + kSortTempSlack) {
        throw Error("response kNN: the segmented sort wants " +
                    std::to_string(sort_temp_bytes) +
                    " workspace bytes, above the " +
                    std::to_string(static_cast<size_t>(chunk) *
                                       sort_temp_per_slot + kSortTempSlack) +
                    " the budget reserved.");
    }
    unsigned char* sort_temp = sort_temp_buffer.allocate(sort_temp_bytes);

    /* Bootstrap the histogram window: one octave per bucket over a strided
     * query sample gives the global scale of the k-th squared radius. */
    timer.begin();
    const int sample_stride = std::max(1, query_count / 4096);
    const int sample_count = (query_count + sample_stride - 1) / sample_stride;
    threshold(Ysub, Ns, Y, N, 0, sample_stride, std::min(sample_count, chunk),
              0u, kOctaveShift, target_rank, tau);
    MANOR_KNN_CHECK(cudaPeekAtLastError());
    std::vector<float> sample_tau(std::min(sample_count, chunk));
    MANOR_KNN_CHECK(cudaMemcpy(sample_tau.data(), tau,
                               sample_tau.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    float window_high = 0.0f;
    for (float value : sample_tau) window_high = std::max(window_high, value);
    if (!(window_high > 0.0f))
        throw Error("response kNN: the threshold bootstrap found no scale.");
    /* Four octaves of headroom above the sampled maximum: the sample is 4096
     * queries out of N, so the true maximum sits above it. */
    window_high *= 4.0f;
    const unsigned int bucket_base = knn_bucket_base(window_high, kFineShift);
    timer.end(&local.bootstrap_ms);

    std::vector<unsigned int> count_host;
    if (timing) count_host.resize(chunk);

    for (int q0 = 0; q0 < query_count; q0 += chunk) {
        const int slots = std::min(chunk, query_count - q0);

        timer.begin();
        threshold(Ysub, Ns, Y, N, q0, 1, slots, bucket_base, kFineShift,
                  target_rank, tau);
        MANOR_KNN_CHECK(cudaPeekAtLastError());
        MANOR_KNN_CHECK(cudaMemset(tau_low, 0, slots * sizeof(float)));
        MANOR_KNN_CHECK(cudaMemset(tau_high, 0, slots * sizeof(float)));
        timer.end(&local.threshold_ms);

        timer.begin();
        scan(Y, N, q0, nullptr, slots, tau, cap, count, candidate_key,
             candidate_value);
        MANOR_KNN_CHECK(cudaPeekAtLastError());
        timer.end(&local.scan_ms);

        if (timing) {
            MANOR_KNN_CHECK(cudaMemcpy(count_host.data(), count,
                                       slots * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost));
            double total = 0.0;
            for (int s = 0; s < slots; ++s) {
                total += count_host[s];
                local.max_candidates =
                    std::max(local.max_candidates,
                             static_cast<double>(count_host[s]));
            }
            local.mean_candidates += total;
        }

        timer.begin();
        for (int round = 0; round < options.max_retry_rounds; ++round) {
            MANOR_KNN_CHECK(cudaMemset(retry_count, 0, sizeof(unsigned int)));
            int threads = 256;
            /* The selection needs want = k + band candidates, not just k. */
            knn_retry_kernel<<<(slots + threads - 1) / threads, threads>>>(
                count, slots, want, cap, tau, tau_low, tau_high, retry_list,
                retry_count);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
            unsigned int pending = 0u;
            MANOR_KNN_CHECK(cudaMemcpy(&pending, retry_count,
                                       sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost));
            if (pending == 0u) break;
            if (round + 1 == options.max_retry_rounds) {
                throw Error("response kNN: " + std::to_string(pending) +
                            " anchors did not reach a workable threshold in " +
                            std::to_string(options.max_retry_rounds) +
                            " rounds.");
            }
            local.retry_rounds += 1;
            local.retry_queries += pending;
            scan(Y, N, q0, retry_list, static_cast<int>(pending), tau, cap,
                 count, candidate_key, candidate_value);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
        }
        timer.end(&local.retry_ms);

        timer.begin();
        knn_select_kernel<<<slots, 128>>>(
            candidate_key, candidate_value, count, cap, want, compact_cap,
            compact_value, compact_length, diagnostics);
        MANOR_KNN_CHECK(cudaPeekAtLastError());
        {
            /* end[s] = s * compact_cap + compact_length[s] */
            int threads = 256;
            int blocks = (slots + threads - 1) / threads;
            knn_offsets_kernel<<<blocks, threads>>>(compact_length, slots,
                                                    compact_cap, offset_end);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
        }
        timer.end(&local.select_ms);

        timer.begin();
        {
            size_t total = static_cast<size_t>(slots) * compact_cap;
            int threads = 256;
            size_t blocks = (total + threads - 1) / threads;
            knn_refine_kernel<<<static_cast<unsigned int>(blocks), threads>>>(
                Yaos, DY, q0, compact_value, compact_length, slots, compact_cap,
                refined_key);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
        }
        timer.end(&local.refine_ms);

        timer.begin();
        size_t used_temp = sort_temp_bytes;
        MANOR_KNN_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            sort_temp, used_temp, refined_key, sorted_key, compact_value,
            sorted_value, slots * compact_cap, slots, offset_begin,
            offset_end));
        timer.end(&local.sort_ms);

        timer.begin();
        {
            size_t total = static_cast<size_t>(slots) * compact_cap;
            int threads = 256;
            size_t blocks = (total + threads - 1) / threads;
            knn_emit_kernel<<<static_cast<unsigned int>(blocks), threads>>>(
                sorted_key, sorted_value, compact_length, slots, compact_cap, q0, K, index_out,
                diagnostics);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
        }
        timer.end(&local.emit_ms);
    }

    unsigned long long diagnostics_host[kDiagSlots] = {0ull, 0ull, 0ull};
    MANOR_KNN_CHECK(cudaMemcpy(diagnostics_host, diagnostics,
                               kDiagSlots * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
    MANOR_KNN_CHECK(cudaDeviceSynchronize());
    if (diagnostics_host[kDiagCompactOverflow] != 0ull) {
        throw Error("response kNN: " +
                    std::to_string(diagnostics_host[kDiagCompactOverflow]) +
                    " anchors overflowed the survivor buffer.");
    }
    if (diagnostics_host[kDiagOwnerMisplaced] != 0ull) {
        throw Error("response kNN: " +
                    std::to_string(diagnostics_host[kDiagOwnerMisplaced]) +
                    " anchors did not rank themselves first.");
    }

    if (stats) {
        local.owner_misplaced = diagnostics_host[kDiagOwnerMisplaced];
        local.exact_ties = diagnostics_host[kDiagExactTie];
        local.compact_overflow = diagnostics_host[kDiagCompactOverflow];
        if (timing && query_count > 0)
            local.mean_candidates /= static_cast<double>(query_count);
        local.total_ms = local.bootstrap_ms + local.threshold_ms +
                         local.scan_ms + local.retry_ms + local.select_ms +
                         local.refine_ms + local.sort_ms + local.emit_ms;
        *stats = local;
    }
}

}  // namespace manor_knn
#endif

#if !defined(MANOR_M_PHASE1_CORE_LIBRARY)
namespace {

/* Stage opts.response_Y into the SoA float32 layout the search wants.
 *
 * MATLAB hands over a column-major [DY x N] matrix, i.e. point-major in memory,
 * while the scan reads Y[d * N + i] so that every lane-consecutive load is
 * coalesced.  The transpose is 24 MB at N = 1e6 and happens once.
 *
 * The response coordinates are reduced to float32 before the float64 re-rank. */
static int stage_response_soa(
    const mxArray* response, int N, std::vector<float>& destination)
{
    if (mxIsComplex(response) || mxIsSparse(response) ||
        (!mxIsSingle(response) && !mxIsDouble(response)) ||
        mxGetNumberOfDimensions(response) != 2) {
        throw StableMexError("manor_m_phase1_mex:responseIndex",
            "opts.response_Y must be a dense real single/double [DY x N] matrix.");
    }
    mwSize rows = mxGetM(response);
    mwSize columns = mxGetN(response);
    if (columns != static_cast<mwSize>(N) || rows < 1 ||
        rows > static_cast<mwSize>(manor_knn::kMaxDy)) {
        throw StableMexError("manor_m_phase1_mex:responseIndex",
            "opts.response_Y must be [DY x N] with DY in [1,MAX_MY_MANOR].");
    }
    int DY = static_cast<int>(rows);
    destination.resize(static_cast<size_t>(DY) * N);
    const bool is_single = mxIsSingle(response);
    const float* source_single =
        is_single ? static_cast<const float*>(mxGetData(response)) : nullptr;
    const double* source_double =
        is_single ? nullptr : static_cast<const double*>(mxGetData(response));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < DY; ++d) {
            size_t in = static_cast<size_t>(i) * DY + d;
            float value = is_single ? source_single[in]
                                    : static_cast<float>(source_double[in]);
            if (!std::isfinite(value)) {
                throw StableMexError("manor_m_phase1_mex:responseIndex",
                    "opts.response_Y must contain finite values representable "
                    "as single precision.");
            }
            destination[static_cast<size_t>(d) * N + i] = value;
        }
    }
    return DY;
}

/* Run the search and leave the [kSlice x anchors] index in d_idx, which is
 * exactly the layout idx_at reads, so all four consumer kernels are
 * untouched. */
static void build_response_index(
    const std::vector<float>& soa, int DY, int N, int anchors, int kSlice,
    int* d_idx, const Phase1Opts& memory)
{
    StableCudaBuffer<float> response_resource;
    float* d_response =
        response_resource.allocate(static_cast<size_t>(DY) * N);
    STABLE_CUDA_CHECK(cudaMemcpy(d_response, soa.data(),
        sizeof(float) * static_cast<size_t>(DY) * N, cudaMemcpyHostToDevice));

    /* The search runs after stable_choose_query_block has sized Phase 1
     * against free memory and after host_alloc has taken all of it, and that
     * budget knows nothing about the search.  A constant here is therefore a
     * constant ON TOP of a full budget -- fine while the card has 90 GB spare,
     * the first thing to OOM when it does not.  Re-measure instead, under the
     * same reserve and fraction the query block was chosen with, and keep the
     * default as a ceiling: on a card with room the search is unchanged, and
     * on one without it takes what is actually there.
     *
     * Only a ceiling, never a floor.  Raising it is not a free speedup: the
     * chunk response is a sawtooth, not a plateau, and this default sits in a
     * trough -- see Options::scratch_bytes before touching either number. */
    manor_knn::Options knn_options;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    STABLE_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    double reserve = std::max(memory.reserve_bytes,
        static_cast<double>(total_bytes) * memory.reserve_fraction);
    double headroom =
        std::max(0.0, static_cast<double>(free_bytes) - reserve) *
        static_cast<double>(memory.gpu_mem_fraction);
    if (headroom < static_cast<double>(knn_options.scratch_bytes))
        knn_options.scratch_bytes = static_cast<size_t>(headroom);

    manor_knn::Stats knn_stats;
    try {
        manor_knn::build_index(d_response, N, DY, anchors, kSlice, d_idx,
                               knn_options, &knn_stats);
    } catch (const manor_knn::Error& error) {
        throw StableMexError("manor_m_phase1_mex:responseIndex", error.what());
    }
}

void adaptive_phase1_mex_impl(
    int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 5 || nrhs > 6) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:nrhs",
            "Usage: p1 = manor_m_phase1_mex("
            "X, idx_nested, response_counts, cuts_by_scale, MM, opts)");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:nlhs",
            "Exactly one output struct is required.");
    }
    const auto total_start = std::chrono::steady_clock::now();
    const mxArray* mxX = prhs[0];
    const mxArray* mxIdx = prhs[1];
    const mxArray* mxCounts = prhs[2];
    const mxArray* mxCuts = prhs[3];
    const mxArray* mxOpts = nrhs == 6 ? prhs[5] : nullptr;
    if (mxIsComplex(mxX) || mxIsSparse(mxX) ||
        (!mxIsSingle(mxX) && !mxIsDouble(mxX)) ||
        mxGetNumberOfDimensions(mxX) != 2) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:X",
            "X must be a dense real single/double [D x N] matrix.");
    }
    mwSize Dmw = mxGetM(mxX);
    mwSize Nmw = mxGetN(mxX);
    if (Dmw < 2 || Dmw > MAX_D_MANOR || Nmw < 1 ||
        Dmw > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
        Nmw > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:dimensions",
            "X dimensions exceed the supported D/N limits.");
    }
    int D = static_cast<int>(Dmw);
    int N = static_cast<int>(Nmw);
    double mm_raw = scalar_field(prhs[4], "MM");
    if (mm_raw != std::floor(mm_raw) || mm_raw < 1.0 || mm_raw >= D ||
        mm_raw > MAX_MM_MANOR) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:MM",
            "MM must be an integer satisfying 1 <= MM < D.");
    }
    int MM = static_cast<int>(mm_raw);
    AdaptivePhase1Opts opts = adaptive_parse_opts(mxOpts);
    StableProfileSpans profile_span;
    profile_span.enabled = opts.stable.memory.profile_timers;
    profile_span.begin_host("host_validate");
    int active_anchor_count = opts.active_anchor_count < 0 ?
        N : opts.active_anchor_count;
    if (active_anchor_count > N) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:activeAnchors",
            "opts.active_anchor_count cannot exceed N.");
    }
    opts.stable.memory.max_query_block = std::min(
        opts.stable.memory.max_query_block, active_anchor_count);
    profile_span.end();

    /* X is uploaded exactly once.  When it is already single the mxArray
     * buffer is byte-for-byte what the device wants, so the staging vector is
     * skipped and the upload reads MATLAB's memory directly -- to_float_vec
     * only ran std::copy on that case anyway.  The finite check still visits
     * every element, and on the double path it still inspects the CONVERTED
     * float, so a double that overflows single is rejected exactly as before. */
    profile_span.begin_host("host_stage_x");
    std::vector<float> X_host;
    const float* X_upload = nullptr;
    if (is_single(mxX)) {
        X_upload = static_cast<const float*>(mxGetData(mxX));
    } else {
        X_host = to_float_vec(mxX);
        X_upload = X_host.data();
    }
    const size_t x_element_count = static_cast<size_t>(D) * N;
    for (size_t element = 0; element < x_element_count; ++element) {
        if (!std::isfinite(X_upload[element])) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:X",
                "X must contain finite values representable as single precision.");
        }
    }
    profile_span.end();

    profile_span.begin_host("host_stage_idx");
    /* opts.response_Y present means the index is not an input at all: the
     * search below produces it, idx_nested is ignored, and kSlice comes from
     * opts.response_kslice. */
    const bool compute_index = (opts.stable.response_Y != nullptr);
    std::vector<float> response_host;
    int response_dim = 0;
    if (compute_index && opts.stable.idx_shared_slice) {
        throw StableMexError("manor_m_phase1_mex:responseIndex",
            "opts.response_Y and opts.idx_shared_slice are mutually exclusive: "
            "a shared slice is one vector every anchor reuses, not a search.");
    }
    if (compute_index && mxGetNumberOfElements(mxIdx) != 0) {
        /* Two sources for one array is a silent choice of engine.  Ask for the
         * search with an empty idx_nested, or supply the index and leave
         * response_Y unset. */
        throw StableMexError("manor_m_phase1_mex:responseIndex",
            "opts.response_Y was supplied together with a non-empty "
            "idx_nested; pass an empty idx_nested to request the search.");
    }
    if (!compute_index && (mxIsComplex(mxIdx) || mxIsSparse(mxIdx) ||
        !mxIsNumeric(mxIdx) || mxGetNumberOfDimensions(mxIdx) != 2)) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:idx",
            "idx_nested must be a dense real numeric matrix.");
    }
    mwSize idxM = compute_index ? 0 : mxGetM(mxIdx);
    mwSize idxN = compute_index ? 0 : mxGetN(mxIdx);
    int kSlice = 0;
    if (compute_index) {
        response_dim = stage_response_soa(
            static_cast<const mxArray*>(opts.stable.response_Y),
            N, response_host);
        kSlice = opts.stable.response_kslice;
        if (kSlice < 1 || kSlice > N) {
            throw StableMexError("manor_m_phase1_mex:responseIndex",
                "opts.response_kslice must lie in [1,N].");
        }
    } else if (opts.stable.idx_shared_slice) {
        if (!opts.stable.idx_zero_based || !mxIsInt32(mxIdx) ||
            idxN != 1 || idxM < 1) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:idx",
                "Shared idx_nested must be zero-based int32 [kSlice x 1].");
        }
        kSlice = static_cast<int>(idxM);
    } else if (opts.stable.idx_zero_based) {
        if ((!mxIsInt32(mxIdx) && !mxIsUint32(mxIdx)) || idxN != Nmw) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:idx",
                "Zero-based idx_nested must be int32/uint32 [kSlice x N].");
        }
        kSlice = static_cast<int>(idxM);
    } else {
        if (idxM != Nmw) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:idx",
                "One-based idx_nested must have shape [N x kSlice].");
        }
        kSlice = static_cast<int>(idxN);
    }
    if (kSlice < 1 || kSlice > 65535) {
        /* Response-neighborhood ranks are stored as uint16 values. */
        throw StableMexError("manor_m_phase1_mex:kSlice",
            "The largest response support must lie in [1,65535].");
    }
    size_t index_count = opts.stable.idx_shared_slice ?
        static_cast<size_t>(kSlice) :
        static_cast<size_t>(active_anchor_count) * kSlice;

    /* Validate the supplied neighbor index and upload it in the layout expected
     * by the CUDA kernels. */
    std::vector<int32_t> idx_host;
    const void* idx_upload = nullptr;
    if (compute_index) {
        /* Nothing to stage: the search writes d_idx in place, below. */
    } else if (opts.stable.idx_shared_slice) {
        idx_host.resize(index_count);
        std::vector<uint8_t> shared_seen(static_cast<size_t>(N), 0u);
        for (int rank = 0; rank < kSlice; ++rank) {
            double raw = numeric_at(mxIdx, rank);
            double zero_based = raw;
            if (!std::isfinite(raw) || raw != std::floor(raw) ||
                zero_based < 0.0 || zero_based >= N) {
                mexErrMsgIdAndTxt(
                    "manor_m_phase1_mex:idx",
                    "idx_nested contains an invalid index.");
            }
            int32_t value = static_cast<int32_t>(zero_based);
            if (shared_seen[static_cast<size_t>(value)]) {
                mexErrMsgIdAndTxt(
                    "manor_m_phase1_mex:idx",
                    "Shared idx_nested must contain unique indices.");
            }
            shared_seen[static_cast<size_t>(value)] = 1u;
            idx_host[rank] = value;
        }
        idx_upload = idx_host.data();
    } else if (opts.stable.idx_zero_based) {
        const uint32_T* source = static_cast<const uint32_T*>(mxGetData(mxIdx));
        const uint32_T limit = static_cast<uint32_T>(N);
        uint32_T invalid = 0u;
        for (size_t element = 0; element < index_count; ++element)
            invalid |= (source[element] >= limit) ? 1u : 0u;
        if (invalid) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:idx",
                "idx_nested contains an invalid index.");
        }
        idx_upload = source;
    } else {
        idx_host.resize(index_count);
        adaptive_stage_one_based_any(
            mxIdx, active_anchor_count, kSlice, N, idx_host.data());
        idx_upload = idx_host.data();
    }
    profile_span.end();

    profile_span.begin_host("host_validate");
    if (mxIsComplex(mxCounts) || mxIsSparse(mxCounts) ||
        !mxIsNumeric(mxCounts) || mxGetNumberOfElements(mxCounts) < 1 ||
        (mxGetM(mxCounts) != 1 && mxGetN(mxCounts) != 1)) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:responseCounts",
            "response_counts must be a real numeric vector with at least one scale.");
    }
    size_t scale_count = mxGetNumberOfElements(mxCounts);
    if (scale_count > ADAPTIVE_MAX_SCALES) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:responseCounts",
            "response_counts exceeds ADAPTIVE_MAX_SCALES.");
    }
    int S = static_cast<int>(scale_count);
    if (S > 1 && opts.stable.selection_mode != STABLE_POSITIVE_H) {
        /* Cross-scale reduction supports only the positive Stable-H mode. */
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:selectionMode",
            "Multi-scale response selection supports only selection_mode=0 "
            "(positive Stable-H); a single scale accepts every mode.");
    }
    std::vector<int32_t> counts_host(S);
    for (int scale = 0; scale < S; ++scale) {
        double raw = numeric_at(mxCounts, scale);
        if (!std::isfinite(raw) || raw != std::floor(raw) ||
            raw < 9.0 * D || raw > kSlice ||
            (scale > 0 && raw <= counts_host[scale - 1])) {
            mexErrMsgIdAndTxt(
                "manor_m_phase1_mex:responseCounts",
                "response_counts must be strictly increasing integers in [9D,kSlice].");
        }
        counts_host[scale] = static_cast<int32_t>(raw);
    }
    if (counts_host.back() != kSlice) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:responseCounts",
            "The final response count must equal size(idx_nested,1/support axis).");
    }

    if (mxIsComplex(mxCuts) || mxIsSparse(mxCuts) || !mxIsNumeric(mxCuts) ||
        mxGetNumberOfDimensions(mxCuts) != 2 || mxGetN(mxCuts) != scale_count ||
        mxGetM(mxCuts) < 1 || mxGetM(mxCuts) > MAX_R_MANOR) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:cuts",
            "cuts_by_scale must be a real numeric [R x num_scales] matrix.");
    }
    int R = static_cast<int>(mxGetM(mxCuts));
    if (opts.stable.window > R) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:window",
            "opts.window cannot exceed the number of cuts per scale.");
    }
    std::vector<int32_t> cuts_host(static_cast<size_t>(R) * S);
    std::vector<int32_t> tmax_host(S);
    int tmax = 0;
    for (int scale = 0; scale < S; ++scale) {
        int maximum_allowed = static_cast<int>(std::floor(
            static_cast<double>(counts_host[scale]) *
            opts.stable.cut_max_fraction));
        for (int index = 0; index < R; ++index) {
            double raw = numeric_at(
                mxCuts, index + static_cast<size_t>(R) * scale);
            if (!std::isfinite(raw) || raw != std::floor(raw) ||
                raw < 3.0 * D || raw > maximum_allowed ||
                (index > 0 && raw <= cuts_host[index - 1 + scale * R])) {
                mexErrMsgIdAndTxt(
                    "manor_m_phase1_mex:cuts",
                    "Each cut column must increase within "
                    "[3D,floor(count*opts.cut_max_fraction)].");
            }
            cuts_host[index + scale * R] = static_cast<int32_t>(raw);
        }
        tmax_host[scale] = cuts_host[R - 1 + scale * R];
        tmax = std::max(tmax, static_cast<int>(tmax_host[scale]));
    }
    profile_span.end();

    /* Host span, not a device span: every call below is a blocking driver
     * call, and no event may exist before cudaMemGetInfo picks BQ. */
    profile_span.begin_host("host_budget");
    size_t selection_shared_bytes = compute_selection_shared_bytes(
        D, MM, opts.stable.window);
    size_t final_extract_shared_bytes = compute_final_extract_shared_bytes(D, MM);
    size_t owner_aggregate_shared_bytes = compute_owner_aggregate_shared_bytes(D);
    size_t owner_finalize_shared_bytes = compute_owner_finalize_shared_bytes(D, MM);
    int active_device = 0;
    cudaDeviceProp properties{};
    STABLE_CUDA_CHECK(cudaGetDevice(&active_device));
    STABLE_CUDA_CHECK(cudaGetDeviceProperties(&properties, active_device));
    if (selection_shared_bytes >
        static_cast<size_t>(properties.sharedMemPerBlockOptin)) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:sharedMemory",
            "Stable-H selector exceeds opt-in shared memory.");
    }
    STABLE_CUDA_CHECK(cudaFuncSetAttribute(
        stable_h_select_kernel<uint16_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(selection_shared_bytes)));
    STABLE_CUDA_CHECK(cudaFuncSetCacheConfig(
        stable_h_select_kernel<uint16_t>, cudaFuncCachePreferShared));

    AdaptiveMemoryBudget memory_budget = adaptive_choose_query_block(
        D, MM, N, active_anchor_count, kSlice, tmax, R,
        opts.stable.idx_shared_slice, opts.stable.memory);
    int BQ = memory_budget.stable.BQ;
    int max_items_int = checked_product_int(BQ, kSlice, "BQ*kSlice");
    int max_top_int = checked_product_int(BQ, tmax, "BQ*tmax");
    size_t max_items = static_cast<size_t>(max_items_int);
    size_t max_top = static_cast<size_t>(max_top_int);
    size_t basis_count = static_cast<size_t>(D) * MM * N;
    int mask_words = static_cast<int>(stable_mask_words(kSlice));
    profile_span.end();

    profile_span.begin_host("host_alloc");
    StableCudaBuffer<float> X_resource, keys_in_resource, keys_out_resource;
    StableCudaBuffer<float> top_keys_resource, tangent_resource;
    StableCudaBuffer<float> center_resource, radius_resource, axial_resource;
    StableCudaBuffer<float> aggregated_tangent_resource, aggregated_center_resource;
    StableCudaBuffer<float> aggregate_covariance_resource, aggregate_center_sum_resource;
    StableCudaBuffer<int> idx_resource, cuts_resource, offsets_resource;
    StableCudaBuffer<float> h_null_resource;
    StableCudaBuffer<int32_t> response_counts_resource, tmax_resource;
    StableCudaBuffer<uint16_t> values_in_resource, values_out_resource;
    StableCudaBuffer<uint16_t> top_values_resource;
    StableCudaBuffer<uint8_t> temporary_covered_resource;
    StableCudaBuffer<uint8_t> selected_covered_resource, owner_resource;
    StableCudaBuffer<uint8_t> rejection_resource;
    StableCudaBuffer<uint32_t> temporary_cut_resource, selected_cut_resource;
    StableCudaBuffer<int32_t> temporary_index_resource;
    StableCudaBuffer<int32_t> selected_scale_resource, selected_index_resource;
    StableCudaBuffer<int32_t> num_neigh_resource, aggregate_count_resource;
    StableCudaBuffer<uint32_t> neighborhood_mask_resource;
    StableCudaBuffer<unsigned long long> owner_stats_resource;
    StableCudaBuffer<double> support_threshold_resource;
    StableCudaBuffer<unsigned char> sort_temp_resource;

    StableCudaBuffer<float> temporary_geom_resource, temporary_stable_h_resource;
    StableCudaBuffer<float> temporary_neighbor_resource, temporary_split_resource;
    StableCudaBuffer<float> temporary_h_a_resource, temporary_h_b_resource;
    StableCudaBuffer<float> selected_geom_resource, selected_stable_h_resource;
    StableCudaBuffer<float> selected_neighbor_resource, selected_split_resource;
    StableCudaBuffer<float> selected_h_a_resource, selected_h_b_resource;

    float* d_X = X_resource.allocate(static_cast<size_t>(D) * N);
    int* d_idx = idx_resource.allocate(index_count);
    int* d_cuts = cuts_resource.allocate(static_cast<size_t>(R) * S);
    int32_t* d_response_counts = response_counts_resource.allocate(S);
    int32_t* d_tmax_by_scale = tmax_resource.allocate(S);
    float* d_keys_in = keys_in_resource.allocate(max_items);
    float* d_keys_out = keys_out_resource.allocate(max_items);
    uint16_t* d_values_in = values_in_resource.allocate(max_items);
    uint16_t* d_values_out = values_out_resource.allocate(max_items);
    float* d_top_keys = top_keys_resource.allocate(max_top);
    uint16_t* d_top_values = top_values_resource.allocate(max_top);
    int* d_offsets = offsets_resource.allocate(static_cast<size_t>(BQ) + 1);

#define ALLOC_FLOAT_RECORD(prefix, rows) \
    float* d_##prefix##_geom = prefix##_geom_resource.allocate(rows); \
    float* d_##prefix##_stable_h = prefix##_stable_h_resource.allocate(rows); \
    float* d_##prefix##_neighbor = prefix##_neighbor_resource.allocate(rows); \
    float* d_##prefix##_split = prefix##_split_resource.allocate(rows); \
    float* d_##prefix##_h_a = prefix##_h_a_resource.allocate(rows); \
    float* d_##prefix##_h_b = prefix##_h_b_resource.allocate(rows)
    ALLOC_FLOAT_RECORD(temporary, N);
    ALLOC_FLOAT_RECORD(selected, N);
#undef ALLOC_FLOAT_RECORD

    int32_t* d_temporary_index = temporary_index_resource.allocate(N);
    uint32_t* d_temporary_cut = temporary_cut_resource.allocate(N);
    uint8_t* d_temporary_covered = temporary_covered_resource.allocate(N);
    int32_t* d_selected_scale = selected_scale_resource.allocate(N);
    int32_t* d_selected_index = selected_index_resource.allocate(N);
    uint32_t* d_selected_cut = selected_cut_resource.allocate(N);
    uint8_t* d_selected_covered = selected_covered_resource.allocate(N);

    float* d_tangent = tangent_resource.allocate(basis_count);
    float* d_center = center_resource.allocate(static_cast<size_t>(D) * N);
    float* d_radius = radius_resource.allocate(N);
    float* d_axial = axial_resource.allocate(N);
    int32_t* d_num_neigh = num_neigh_resource.allocate(N);
    uint32_t* d_neighborhood_mask = neighborhood_mask_resource.allocate(
        static_cast<size_t>(active_anchor_count) * mask_words);
    uint8_t* d_owner = owner_resource.allocate(N);
    uint8_t* d_rejection = rejection_resource.allocate(N);
    unsigned long long* d_owner_stats = owner_stats_resource.allocate(3);
    double* d_support_threshold = support_threshold_resource.allocate(1);
    float* d_aggregated_tangent = aggregated_tangent_resource.allocate(basis_count);
    float* d_aggregated_center = aggregated_center_resource.allocate(
        static_cast<size_t>(D) * N);
    float* d_aggregate_covariance = aggregate_covariance_resource.allocate(
        static_cast<size_t>(BQ) * D * D);
    float* d_aggregate_center_sum = aggregate_center_sum_resource.allocate(
        static_cast<size_t>(BQ) * D);
    int32_t* d_aggregate_count = aggregate_count_resource.allocate(BQ);
    profile_span.end();

    profile_span.begin("upload");
    STABLE_CUDA_CHECK(cudaMemcpy(d_X, X_upload,
        sizeof(float) * static_cast<size_t>(D) * N, cudaMemcpyHostToDevice));
    if (compute_index) {
        build_response_index(response_host, response_dim, N,
                             active_anchor_count, kSlice, d_idx,
                             opts.stable.memory);
    } else {
        STABLE_CUDA_CHECK(cudaMemcpy(d_idx, idx_upload,
            sizeof(int32_t) * index_count, cudaMemcpyHostToDevice));
    }
    STABLE_CUDA_CHECK(cudaMemcpy(d_cuts, cuts_host.data(),
        sizeof(int32_t) * static_cast<size_t>(R) * S, cudaMemcpyHostToDevice));

    /* Per-cut Stable-H null offsets, laid out [R x S] exactly like d_cuts so
     * each scale's launch takes the matching column.  Every scale has its own
     * cut ladder and therefore its own zeros.  Host side, once per call. */
    float* d_h_null = nullptr;
    if (opts.stable.h_null_correction) {
        std::vector<float> h_null_host(static_cast<size_t>(R) * S);
        h_null_fill_offsets(D, MM, cuts_host.data(), R * S, h_null_host.data(),
            "manor_m_phase1_mex:hNullCalibration");
        d_h_null = h_null_resource.allocate(static_cast<size_t>(R) * S);
        STABLE_CUDA_CHECK(cudaMemcpy(d_h_null, h_null_host.data(),
            sizeof(float) * static_cast<size_t>(R) * S,
            cudaMemcpyHostToDevice));
    }
    STABLE_CUDA_CHECK(cudaMemcpy(d_response_counts, counts_host.data(),
        sizeof(int32_t) * S, cudaMemcpyHostToDevice));
    STABLE_CUDA_CHECK(cudaMemcpy(d_tmax_by_scale, tmax_host.data(),
        sizeof(int32_t) * S, cudaMemcpyHostToDevice));
    STABLE_CUDA_CHECK(cudaMemset(d_owner_stats, 0,
        3 * sizeof(unsigned long long)));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_scale, 0xff,
        sizeof(int32_t) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_index, 0xff,
        sizeof(int32_t) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_cut, 0,
        sizeof(uint32_t) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_covered, 0,
        sizeof(uint8_t) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_geom, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_stable_h, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_neighbor, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_split, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_h_a, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_selected_h_b, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_tangent, 0,
        sizeof(float) * basis_count));
    STABLE_CUDA_CHECK(cudaMemset(d_center, 0,
        sizeof(float) * static_cast<size_t>(D) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_radius, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_axial, 0, sizeof(float) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_num_neigh, 0, sizeof(int32_t) * N));
    STABLE_CUDA_CHECK(cudaMemset(d_neighborhood_mask, 0,
        sizeof(uint32_t) * static_cast<size_t>(active_anchor_count) * mask_words));

    std::vector<int> offsets_host(BQ + 1);
    for (int block = 0; block <= BQ; ++block)
        offsets_host[block] = block * kSlice;
    STABLE_CUDA_CHECK(cudaMemcpy(d_offsets, offsets_host.data(),
        sizeof(int) * (BQ + 1), cudaMemcpyHostToDevice));
    profile_span.end();

    profile_span.begin_host("host_alloc");
    size_t sort_temp_bytes = 0;
    STABLE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, sort_temp_bytes, d_keys_in, d_keys_out,
        d_values_in, d_values_out, max_items_int, BQ,
        d_offsets, d_offsets + 1));
    void* d_sort_temp = sort_temp_resource.allocate(sort_temp_bytes);
    if (sort_temp_bytes != memory_budget.stable.selected_query.sort_temp_bytes) {
        throw StableMexError(
            "manor_m_phase1_mex:memoryBudget",
            "CUB sort workspace differs from the adaptive memory budget.");
    }

    const int threads = 256;
    profile_span.end();
    for (int q0 = 0; q0 < active_anchor_count; q0 += BQ) {
        int B = std::min(BQ, active_anchor_count - q0);
        int item_count = checked_product_int(B, kSlice, "B*kSlice");
        int selector_blocks = (B + threads - 1) / threads;
        dim3 distance_grid((kSlice + threads - 1) / threads, B);
        profile_span.begin("dist_idx");
        phase1_dist_idx_kernel<<<distance_grid, threads, sizeof(float) * D>>>(
            d_X, d_idx, D, N, q0, B, kSlice, opts.stable.idx_shared_slice,
            d_keys_in, d_values_in);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        profile_span.end();
        profile_span.begin("sort");
        STABLE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            d_sort_temp, sort_temp_bytes, d_keys_in, d_keys_out,
            d_values_in, d_values_out, item_count, B,
            d_offsets, d_offsets + 1));
        profile_span.end();

        for (int scale = 0; scale < S; ++scale) {
            int scale_tmax = tmax_host[scale];
            profile_span.begin("scale_topk");
            adaptive_compact_nested_topk_kernel<<<B, 32>>>(
                d_keys_out, d_values_out, B, kSlice, counts_host[scale],
                scale_tmax, scale_tmax, d_top_keys, d_top_values);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            profile_span.end();
            profile_span.begin("scale_select");
            stable_h_select_kernel<<<B, 32, selection_shared_bytes>>>(
                d_X, d_idx, d_top_values, d_cuts + scale * R,
                D, MM, kSlice, scale_tmax, R,
                opts.stable.idx_shared_slice, q0, B,
                opts.stable.jitter, opts.stable.window,
                opts.stable.neighbor_scaled_max, opts.stable.split_scaled_max,
                opts.stable.split_seed, opts.stable.selection_mode,
                d_h_null ? d_h_null + scale * R : nullptr,
                d_temporary_index, d_temporary_cut, d_temporary_covered,
                d_temporary_geom, d_temporary_stable_h,
                d_temporary_neighbor, d_temporary_split,
                d_temporary_h_a, d_temporary_h_b,
                nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            profile_span.end();
            profile_span.begin("scale_stream");
            adaptive_stream_best_kernel<<<selector_blocks, threads>>>(
                scale, q0, B,
                d_temporary_index, d_temporary_cut,
                d_temporary_covered, d_temporary_geom,
                d_temporary_stable_h,
                d_temporary_neighbor, d_temporary_split,
                d_temporary_h_a, d_temporary_h_b,
                d_selected_scale, d_selected_index, d_selected_cut,
                d_selected_covered, d_selected_geom, d_selected_stable_h,
                d_selected_neighbor, d_selected_split,
                d_selected_h_a, d_selected_h_b);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            profile_span.end();
        }

        profile_span.begin("final_extract");
        adaptive_compact_selected_topk_kernel<<<B, 32>>>(
            d_keys_out, d_values_out, d_selected_scale,
            d_response_counts, d_tmax_by_scale, q0, B, kSlice, tmax,
            d_top_keys, d_top_values);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        adaptive_final_extract_kernel<<<B, 32, final_extract_shared_bytes>>>(
            d_X, d_idx, d_top_keys, d_top_values, d_cuts,
            d_response_counts, d_selected_scale, d_selected_index,
            d_selected_geom,
            R, D, MM, kSlice, tmax, q0, B,
            opts.stable.idx_shared_slice, opts.stable.jitter,
            d_tangent, d_center, d_radius, d_axial, d_num_neigh,
            d_neighborhood_mask, mask_words);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        profile_span.end();
    }

    int blocks = N / threads + (N % threads != 0 ? 1 : 0);
    profile_span.begin("owner_rule");
    adaptive_owner_stats_kernel<<<blocks, threads>>>(
        d_selected_covered, d_num_neigh, N, d_owner_stats);
    adaptive_owner_threshold_kernel<<<1, 1>>>(
        d_owner_stats, D, opts.stable.heavy_factor, d_support_threshold);
    adaptive_owner_mask_kernel<<<blocks, threads>>>(
        d_selected_covered, d_num_neigh, d_support_threshold,
        N, d_owner, d_rejection, d_owner_stats);
    STABLE_CUDA_CHECK(cudaPeekAtLastError());
    profile_span.end();

    for (int q0 = 0; q0 < active_anchor_count; q0 += BQ) {
        int B = std::min(BQ, active_anchor_count - q0);
        profile_span.begin("owner_aggregate");
        owner_aggregate_kernel_mm<<<B, 32, owner_aggregate_shared_bytes>>>(
            d_neighborhood_mask, d_owner, d_tangent, d_center, d_idx,
            D, MM, N, kSlice, mask_words, opts.stable.idx_shared_slice,
            q0, B, d_aggregate_covariance, d_aggregate_center_sum,
            d_aggregate_count);
        adaptive_owner_finalize_kernel_mm<<<B, 32, owner_finalize_shared_bytes>>>(
            d_aggregate_covariance, d_aggregate_center_sum, d_aggregate_count,
            d_X, d_tangent, d_center, d_owner, D, MM, q0, B,
            d_support_threshold, d_aggregated_tangent, d_aggregated_center);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        profile_span.end();
    }
    profile_span.begin("mask_nonowners");
    mask_nonowners_kernel<<<blocks, threads>>>(
        d_owner, D, MM, N, d_aggregated_tangent, d_aggregated_center);
    STABLE_CUDA_CHECK(cudaPeekAtLastError());
    profile_span.end();
    STABLE_CUDA_CHECK(cudaDeviceSynchronize());

    profile_span.begin_host("host_download");
    std::vector<uint8_t> covered_host(N), owner_host(N), rejection_host(N);
    std::vector<int32_t> num_neigh_host(N), selected_scale_host(N);
    std::vector<int32_t> selected_index_host(N);
    std::array<unsigned long long, 3> owner_stats_host{};
    double support_threshold_host = 0.0;
    STABLE_CUDA_CHECK(cudaMemcpy(covered_host.data(), d_selected_covered,
        sizeof(uint8_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(owner_host.data(), d_owner,
        sizeof(uint8_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(rejection_host.data(), d_rejection,
        sizeof(uint8_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(num_neigh_host.data(), d_num_neigh,
        sizeof(int32_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(selected_scale_host.data(), d_selected_scale,
        sizeof(int32_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(selected_index_host.data(), d_selected_index,
        sizeof(int32_t) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(owner_stats_host.data(), d_owner_stats,
        3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(&support_threshold_host, d_support_threshold,
        sizeof(double), cudaMemcpyDeviceToHost));
    profile_span.end();

    profile_span.begin_host("host_output_pack");
    const char* basic_fields[] = {
        "Tan0", "Cen0", "Rad0", "axial_tan0", "num_neigh0",
        "owner_mask0", "stable_covered", "selected_scale0",
        "selected_response_count0", "selected_cut0", "geom0", "stable_h0",
        "neighbor_scaled0", "split_scaled0", "split_h_a0", "split_h_b0",
        "rejection_code0", "retention", "profile"};
    const char* debug_fields[] = {
        "Tan0", "Cen0", "Rad0", "axial_tan0", "num_neigh0",
        "owner_mask0", "stable_covered", "selected_scale0",
        "selected_response_count0", "selected_cut0", "geom0", "stable_h0",
        "neighbor_scaled0", "split_scaled0", "split_h_a0", "split_h_b0",
        "rejection_code0", "retention", "profile", "TanRaw0", "CenRaw0",
        "selected_index0"};
    /* Counts from the arrays themselves: a literal that disagrees with the
       list reads past its end and still compiles cleanly. */
    plhs[0] = mxCreateStructMatrix(1, 1,
        opts.stable.memory.return_debug
            ? (int)(sizeof(debug_fields) / sizeof(debug_fields[0]))
            : (int)(sizeof(basic_fields) / sizeof(basic_fields[0])),
        opts.stable.memory.return_debug ? debug_fields : basic_fields);

    /* Phase 2 reads the same index, so a computed one has to be able to leave
     * Phase 1.  mxAddField keeps the two field lists above untouched, so the
     * struct every existing caller sees is unchanged. */
    if (compute_index && opts.stable.return_response_index) {
        if (mxAddField(plhs[0], "response_index") < 0) {
            throw StableMexError("manor_m_phase1_mex:responseIndex",
                "Could not add response_index to the Phase-1 output.");
        }
        mxArray* mx_response_index = mxCreateNumericMatrix(
            kSlice, active_anchor_count, mxINT32_CLASS, mxREAL);
        STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_response_index), d_idx,
            sizeof(int32_t) * index_count, cudaMemcpyDeviceToHost));
        mxSetField(plhs[0], 0, "response_index", mx_response_index);
    }

    mwSize tangent_dimensions[3] = {
        static_cast<mwSize>(D), static_cast<mwSize>(MM), static_cast<mwSize>(N)};
    mxArray* mx_tangent = mxCreateNumericArray(
        3, tangent_dimensions, mxSINGLE_CLASS, mxREAL);
    mxArray* mx_center = mxCreateNumericMatrix(D, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mx_radius = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mx_axial = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxREAL);
    mxArray* mx_num_neigh = mxCreateNumericMatrix(1, N, mxINT32_CLASS, mxREAL);
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_tangent), d_aggregated_tangent,
        sizeof(float) * basis_count, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_center), d_aggregated_center,
        sizeof(float) * static_cast<size_t>(D) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_radius), d_radius,
        sizeof(float) * N, cudaMemcpyDeviceToHost));
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_axial), d_axial,
        sizeof(float) * N, cudaMemcpyDeviceToHost));
    std::copy(num_neigh_host.begin(), num_neigh_host.end(),
        static_cast<int32_T*>(mxGetData(mx_num_neigh)));
    mxSetField(plhs[0], 0, "Tan0", mx_tangent);
    mxSetField(plhs[0], 0, "Cen0", mx_center);
    mxSetField(plhs[0], 0, "Rad0", mx_radius);
    mxSetField(plhs[0], 0, "axial_tan0", mx_axial);
    mxSetField(plhs[0], 0, "num_neigh0", mx_num_neigh);

    mxArray* mx_owner = mxCreateLogicalMatrix(1, N);
    mxArray* mx_covered = mxCreateLogicalMatrix(1, N);
    for (int anchor = 0; anchor < N; ++anchor) {
        mxGetLogicals(mx_owner)[anchor] = owner_host[anchor] != 0;
        mxGetLogicals(mx_covered)[anchor] = covered_host[anchor] != 0;
    }
    mxSetField(plhs[0], 0, "owner_mask0", mx_owner);
    mxSetField(plhs[0], 0, "stable_covered", mx_covered);

    mxArray* mx_scale = mxCreateNumericMatrix(1, N, mxUINT32_CLASS, mxREAL);
    mxArray* mx_response = mxCreateNumericMatrix(1, N, mxUINT32_CLASS, mxREAL);
    uint32_T* scale_data = static_cast<uint32_T*>(mxGetData(mx_scale));
    uint32_T* response_data = static_cast<uint32_T*>(mxGetData(mx_response));
    for (int anchor = 0; anchor < N; ++anchor) {
        int scale = selected_scale_host[anchor];
        scale_data[anchor] = scale >= 0 ? static_cast<uint32_T>(scale + 1) : 0u;
        response_data[anchor] = scale >= 0 ?
            static_cast<uint32_T>(counts_host[scale]) : 0u;
    }
    mxSetField(plhs[0], 0, "selected_scale0", mx_scale);
    mxSetField(plhs[0], 0, "selected_response_count0", mx_response);

    mxArray* mx_selected_cut = mxCreateNumericMatrix(
        1, N, mxUINT32_CLASS, mxREAL);
    STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_selected_cut), d_selected_cut,
        sizeof(uint32_t) * N, cudaMemcpyDeviceToHost));
    mxSetField(plhs[0], 0, "selected_cut0", mx_selected_cut);
    mxSetField(plhs[0], 0, "geom0", single_vector(d_selected_geom, N));
    mxSetField(plhs[0], 0, "stable_h0",
        single_vector(d_selected_stable_h, N));
    mxSetField(plhs[0], 0, "neighbor_scaled0",
        single_vector(d_selected_neighbor, N));
    mxSetField(plhs[0], 0, "split_scaled0",
        single_vector(d_selected_split, N));
    mxSetField(plhs[0], 0, "split_h_a0",
        single_vector(d_selected_h_a, N));
    mxSetField(plhs[0], 0, "split_h_b0",
        single_vector(d_selected_h_b, N));
    mxArray* mx_rejection = mxCreateNumericMatrix(1, N, mxUINT8_CLASS, mxREAL);
    std::copy(rejection_host.begin(), rejection_host.end(),
        static_cast<uint8_T*>(mxGetData(mx_rejection)));
    mxSetField(plhs[0], 0, "rejection_code0", mx_rejection);

    const char* retention_fields[] = {
        "stable_covered_count", "stable_covered_fraction", "owner_count",
        "owner_fraction", "support_threshold", "heavy_factor", "jitter",
        "neighbor_scaled_max", "split_scaled_max", "window", "split_seed",
        "cut_max_fraction", "scale_count",
        "active_anchor_count"};
    mxArray* retention = mxCreateStructMatrix(1, 1, 14, retention_fields);
    double covered_count = static_cast<double>(owner_stats_host[0]);
    double owner_count = static_cast<double>(owner_stats_host[2]);
    mxSetField(retention, 0, "stable_covered_count", mxCreateDoubleScalar(covered_count));
    mxSetField(retention, 0, "stable_covered_fraction", mxCreateDoubleScalar(covered_count / N));
    mxSetField(retention, 0, "owner_count", mxCreateDoubleScalar(owner_count));
    mxSetField(retention, 0, "owner_fraction", mxCreateDoubleScalar(owner_count / N));
    mxSetField(retention, 0, "support_threshold", mxCreateDoubleScalar(support_threshold_host));
    mxSetField(retention, 0, "heavy_factor", mxCreateDoubleScalar(opts.stable.heavy_factor));
    mxSetField(retention, 0, "jitter", mxCreateDoubleScalar(opts.stable.jitter));
    mxSetField(retention, 0, "neighbor_scaled_max", mxCreateDoubleScalar(opts.stable.neighbor_scaled_max));
    mxSetField(retention, 0, "split_scaled_max", mxCreateDoubleScalar(opts.stable.split_scaled_max));
    mxSetField(retention, 0, "window", mxCreateDoubleScalar(opts.stable.window));
    mxSetField(retention, 0, "split_seed", uint64_scalar(opts.stable.split_seed));
    mxSetField(retention, 0, "cut_max_fraction",
        mxCreateDoubleScalar(opts.stable.cut_max_fraction));
    mxSetField(retention, 0, "scale_count", mxCreateDoubleScalar(S));
    mxSetField(retention, 0, "active_anchor_count",
        mxCreateDoubleScalar(active_anchor_count));
    mxSetField(plhs[0], 0, "retention", retention);
    profile_span.end();

    double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_start).count();
    const char* profile_fields[] = {
        "total_sec", "BQ", "BQ_upper_bound", "scale_count",
        "selection_shared_bytes", "idx_shared_slice", "memory_index_bytes",
        "memory_free_bytes", "memory_total_bytes", "memory_reserve_bytes",
        "memory_usable_bytes", "memory_fixed_bytes", "memory_adaptive_extra_bytes",
        "memory_sort_payload_bytes", "memory_sort_temp_bytes",
        "memory_aggregate_scratch_bytes", "memory_query_bytes",
        "memory_required_bytes", "memory_next_required_bytes",
        "active_anchor_count", "timer_labels", "timer_ms"};
    /* Count from the array itself: a literal that disagrees with the list
       reads past its end and still compiles cleanly. */
    mxArray* profile = mxCreateStructMatrix(1, 1,
        static_cast<int>(sizeof(profile_fields) / sizeof(profile_fields[0])),
        profile_fields);
    mxSetField(profile, 0, "total_sec", mxCreateDoubleScalar(total_seconds));
    mxSetField(profile, 0, "BQ", mxCreateDoubleScalar(BQ));
    mxSetField(profile, 0, "BQ_upper_bound", mxCreateDoubleScalar(memory_budget.stable.upper_bound));
    mxSetField(profile, 0, "scale_count", mxCreateDoubleScalar(S));
    mxSetField(profile, 0, "selection_shared_bytes", mxCreateDoubleScalar(selection_shared_bytes));
    mxSetField(profile, 0, "idx_shared_slice", mxCreateLogicalScalar(opts.stable.idx_shared_slice));
    mxSetField(profile, 0, "memory_index_bytes", mxCreateDoubleScalar(memory_budget.stable.index_bytes));
    mxSetField(profile, 0, "memory_free_bytes", mxCreateDoubleScalar(memory_budget.stable.free_bytes));
    mxSetField(profile, 0, "memory_total_bytes", mxCreateDoubleScalar(memory_budget.stable.total_bytes));
    mxSetField(profile, 0, "memory_reserve_bytes", mxCreateDoubleScalar(memory_budget.stable.reserve_bytes));
    mxSetField(profile, 0, "memory_usable_bytes", mxCreateDoubleScalar(memory_budget.stable.usable_bytes));
    mxSetField(profile, 0, "memory_fixed_bytes", mxCreateDoubleScalar(memory_budget.stable.fixed_bytes));
    mxSetField(profile, 0, "memory_adaptive_extra_bytes", mxCreateDoubleScalar(memory_budget.extra_fixed_bytes));
    mxSetField(profile, 0, "memory_sort_payload_bytes", mxCreateDoubleScalar(memory_budget.stable.selected_query.sort_payload_bytes));
    mxSetField(profile, 0, "memory_sort_temp_bytes", mxCreateDoubleScalar(memory_budget.stable.selected_query.sort_temp_bytes));
    mxSetField(profile, 0, "memory_aggregate_scratch_bytes", mxCreateDoubleScalar(memory_budget.stable.selected_query.aggregate_scratch_bytes));
    mxSetField(profile, 0, "memory_query_bytes", mxCreateDoubleScalar(memory_budget.stable.selected_query.total_bytes));
    mxSetField(profile, 0, "memory_required_bytes", mxCreateDoubleScalar(memory_budget.stable.required_bytes));
    mxSetField(profile, 0, "memory_next_required_bytes", mxCreateDoubleScalar(memory_budget.stable.next_required_bytes));
    mxSetField(profile, 0, "active_anchor_count",
        mxCreateDoubleScalar(active_anchor_count));
    {
        /* Safe here: every device span closed before the synchronize above.
           Empty 1x0 arrays when profiling is off, matching Phase 2. */
        const std::vector<StableProfileSpans::SpanTotal> span_totals =
            profile_span.totals();
        const mwSize span_count = static_cast<mwSize>(span_totals.size());
        mxArray* timer_labels = mxCreateCellMatrix(1, span_count);
        mxArray* timer_ms = mxCreateDoubleMatrix(1, span_count, mxREAL);
        for (mwSize index = 0; index < span_count; ++index) {
            const StableProfileSpans::SpanTotal& entry =
                span_totals[static_cast<size_t>(index)];
            mxSetCell(timer_labels, index, mxCreateString(entry.label));
            mxGetPr(timer_ms)[index] = entry.ms;
        }
        mxSetField(profile, 0, "timer_labels", timer_labels);
        mxSetField(profile, 0, "timer_ms", timer_ms);
    }
    mxSetField(plhs[0], 0, "profile", profile);

    if (opts.stable.memory.return_debug) {
        mxArray* mx_raw = mxCreateNumericArray(
            3, tangent_dimensions, mxSINGLE_CLASS, mxREAL);
        STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_raw), d_tangent,
            sizeof(float) * basis_count, cudaMemcpyDeviceToHost));
        mxSetField(plhs[0], 0, "TanRaw0", mx_raw);
        mxArray* mx_center_raw = mxCreateNumericMatrix(
            D, N, mxSINGLE_CLASS, mxREAL);
        STABLE_CUDA_CHECK(cudaMemcpy(mxGetData(mx_center_raw), d_center,
            sizeof(float) * static_cast<size_t>(D) * N, cudaMemcpyDeviceToHost));
        mxSetField(plhs[0], 0, "CenRaw0", mx_center_raw);
        mxArray* mx_selected_index = mxCreateNumericMatrix(
            1, N, mxUINT32_CLASS, mxREAL);
        uint32_T* data = static_cast<uint32_T*>(mxGetData(mx_selected_index));
        for (int anchor = 0; anchor < N; ++anchor)
            data[anchor] = selected_index_host[anchor] >= 0 ?
                static_cast<uint32_T>(selected_index_host[anchor] + 1) : 0u;
        mxSetField(plhs[0], 0, "selected_index0", mx_selected_index);
    }
    if (!opts.stable.memory.silent) {
        mexPrintf(
            "[adaptive-stable-h-dev] D=%d MM=%d N=%d active=%d scales=%d kMax=%d "
            "R=%d BQ=%d covered=%llu owner=%llu total=%.3fs\n",
            D, MM, N, active_anchor_count, S, kSlice, R, BQ,
            owner_stats_host[0], owner_stats_host[2], total_seconds);
    }
}

} // namespace

extern "C" void mexFunction(
    int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        adaptive_phase1_mex_impl(nlhs, plhs, nrhs, prhs);
    } catch (const StableMexError& error) {
        mexErrMsgIdAndTxt(error.identifier(), "%s", error.what());
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:exception",
            "%s", error.what());
    } catch (...) {
        mexErrMsgIdAndTxt(
            "manor_m_phase1_mex:exception",
            "Unknown C++ exception.");
    }
}
#endif
