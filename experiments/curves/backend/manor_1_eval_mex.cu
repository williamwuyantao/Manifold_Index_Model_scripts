#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cuda",
                          "%s (%s:%d)",
                          cudaGetErrorString(code), file, line);
    }
}

#ifndef MAX_D_MANOR_EVAL
#define MAX_D_MANOR_EVAL 64
#endif

#ifndef EXACT_EIG_D_THRESHOLD
#define EXACT_EIG_D_THRESHOLD 32
#endif

#ifndef EVAL_BLOCK
#define EVAL_BLOCK 256
#endif

// Exact specialized eval is no longer treated as D=20-only.
// This file keeps legacy option names for compatibility, but the active
// structured dispatch covers a supported list of small/medium D values.

#ifndef D20_TRAIN_TILE
#define D20_TRAIN_TILE 64
#endif

#ifndef D20_HYPER_TILE
#define D20_HYPER_TILE 5
#endif

#ifndef D20_HYPER_SAMPLES_PER_BLOCK
#define D20_HYPER_SAMPLES_PER_BLOCK 32
#endif

#ifndef D20_HYPER_BLOCK_THREADS
#define D20_HYPER_BLOCK_THREADS (D20_HYPER_TILE * D20_HYPER_SAMPLES_PER_BLOCK)
#endif

#define MANOR_EVAL_EXACT_SPECIALIZED_D_LIST(X) \
    X(3)  X(4)  X(5)  X(6)  X(7)  X(8)  X(9)  X(10) X(11) X(12) \
    X(15) X(18) X(20) X(21) X(24) X(27) X(30)

struct EvalOpts {
    int   power_iter_steps          = 12;   // only used for D > EXACT_EIG_D_THRESHOLD fallback
    float sum_sparse_threshold      = 1.0e1f;
    int   count_sparse_threshold    = 1;
    float fallback_exp_scale        = 2.0f;
    float eps_weight                = 1.0e-20f;

    int   manual_chunk              = 0;
    float auto_chunk_mem_fraction   = 0.80f;
    double auto_chunk_safety_bytes  = 0.0;
    int   auto_chunk_min            = 1024;
    int   auto_chunk_max            = 131072;
    double auto_chunk_target_pairs  = 8.0e9;

    int   enable_host_double_buffer = 1;
    int   exact_tangent_small_d     = 1;    // exact Jacobi eig when D <= EXACT_EIG_D_THRESHOLD
    int   enable_exact_d20_specialization = 1; // legacy option name: enables structured exact-D specialization list for supported small/medium D
    int   enable_exact_d20_train_tiling = 1;   // legacy option name: enables tiled exact-D specialized path when supported
    int   enable_exact_d20_multi_hyper_tile = 1; // legacy option name: reuse shared train tiles across up to 5 hyper values for supported specialized D
    int   profile_timers            = 0;
    int   silent                    = 0;
};

struct EvalState {
    bool estimator_initialized = false;
    bool test_cached = false;
    bool locked = false;

    int D = 0;
    int Ntrain = 0;
    int m = 0;
    int Ntest_cached = 0;
    int Dtest_cached = 0;

    float h = 1.0f;
    EvalOpts opts;

    double last_init_sec = 0.0;
    double last_cache_test_sec = 0.0;
    double last_total_sec = 0.0;
    double last_host_pack_sec = 0.0;
    double last_stream_wait_sec = 0.0;
    double last_h2d_sec = 0.0;
    double last_kernel_sec = 0.0;
    double last_d2h_sec = 0.0;
    int last_chunkB = 0;
    int last_num_chunks = 0;
    int last_used_cached_test = 0;
    int last_double_buffer = 0;

    // estimator-side persistent device data (column-major [D,N] when matrix)
    float* d_Ytrain      = nullptr;   // [Ntrain]
    float* d_Xtrain      = nullptr;   // [D*Ntrain] column-major
    float* d_Cen         = nullptr;   // [D*Ntrain] column-major
    float* d_V           = nullptr;   // [D*Ntrain] column-major
    float* d_VdotX       = nullptr;   // [Ntrain]
    float* d_Xtrainnorm2 = nullptr;   // [Ntrain]
    float* d_RhoScale    = nullptr;   // [m]

    // full cached test data on device (column-major [D,Ntest] for matrix)
    float* d_Xtest_all = nullptr;
    float* d_Ytrue_all = nullptr;
    float* d_Vtrue_all = nullptr;
    float* d_Ctrue_all = nullptr;

    // streams
    bool streams_created = false;
    cudaStream_t streams[2];

    // chunk buffers for streamed eval_full
    int chunk_cap = 0;         // max chunk size allocated
    int partial_cap_gridx = 0; // max number of x-blocks allocated

    float* d_Xq[2] = {nullptr, nullptr};     // [D*chunk_cap] column-major
    float* d_Yq[2] = {nullptr, nullptr};     // [chunk_cap]
    float* d_Vq[2] = {nullptr, nullptr};     // [D*chunk_cap] column-major
    float* d_Cq[2] = {nullptr, nullptr};     // [D*chunk_cap] column-major

    float* d_partialMSE[2]       = {nullptr, nullptr}; // [m * partial_cap_gridx]
    float* d_partialDevCen[2]    = {nullptr, nullptr};
    float* d_partialDevCenTan[2] = {nullptr, nullptr};
    float* d_partialDevVec[2]    = {nullptr, nullptr};

    float* d_sumMSE[2]       = {nullptr, nullptr}; // [m]
    float* d_sumDevCen[2]    = {nullptr, nullptr};
    float* d_sumDevCenTan[2] = {nullptr, nullptr};
    float* d_sumDevVec[2]    = {nullptr, nullptr};

    // pinned host staging for streamed eval_full
    float* h_Xq[2] = {nullptr, nullptr};
    float* h_Yq[2] = {nullptr, nullptr};
    float* h_Vq[2] = {nullptr, nullptr};
    float* h_Cq[2] = {nullptr, nullptr};

    float* h_sumMSE[2]       = {nullptr, nullptr}; // [m]
    float* h_sumDevCen[2]    = {nullptr, nullptr};
    float* h_sumDevCenTan[2] = {nullptr, nullptr};
    float* h_sumDevVec[2]    = {nullptr, nullptr};
};

static EvalState G;
static inline bool is_eval_exact_specialized_dim(int D)
{
    switch (D) {
#define MANOR_CASE_EXACT_DIM(d) case d: return true;
        MANOR_EVAL_EXACT_SPECIALIZED_D_LIST(MANOR_CASE_EXACT_DIM)
#undef MANOR_CASE_EXACT_DIM
        default: return false;
    }
}

static inline bool eval_exact_specialization_enabled()
{
    return (G.opts.enable_exact_d20_specialization != 0) &&
           (G.opts.exact_tangent_small_d != 0) &&
           is_eval_exact_specialized_dim(G.D);
}

static bool g_at_exit_registered = false;

__host__ __device__ __forceinline__ static size_t cm_index(int D, int row, int col)
{
    return (size_t)row + (size_t)col * (size_t)D;
}

static void touch_lock()
{
    if (!G.locked) {
        mexLock();
        G.locked = true;
    }
}

static void maybe_unlock()
{
    if (G.locked && !G.estimator_initialized && !G.test_cached) {
        mexUnlock();
        G.locked = false;
    }
}

static void free_streamed_buffers()
{
    for (int s = 0; s < 2; ++s) {
        if (G.d_Xq[s]) CUDA_CHECK(cudaFree(G.d_Xq[s]));
        if (G.d_Yq[s]) CUDA_CHECK(cudaFree(G.d_Yq[s]));
        if (G.d_Vq[s]) CUDA_CHECK(cudaFree(G.d_Vq[s]));
        if (G.d_Cq[s]) CUDA_CHECK(cudaFree(G.d_Cq[s]));

        if (G.d_partialMSE[s])       CUDA_CHECK(cudaFree(G.d_partialMSE[s]));
        if (G.d_partialDevCen[s])    CUDA_CHECK(cudaFree(G.d_partialDevCen[s]));
        if (G.d_partialDevCenTan[s]) CUDA_CHECK(cudaFree(G.d_partialDevCenTan[s]));
        if (G.d_partialDevVec[s])    CUDA_CHECK(cudaFree(G.d_partialDevVec[s]));

        if (G.d_sumMSE[s])       CUDA_CHECK(cudaFree(G.d_sumMSE[s]));
        if (G.d_sumDevCen[s])    CUDA_CHECK(cudaFree(G.d_sumDevCen[s]));
        if (G.d_sumDevCenTan[s]) CUDA_CHECK(cudaFree(G.d_sumDevCenTan[s]));
        if (G.d_sumDevVec[s])    CUDA_CHECK(cudaFree(G.d_sumDevVec[s]));

        if (G.h_Xq[s]) CUDA_CHECK(cudaFreeHost(G.h_Xq[s]));
        if (G.h_Yq[s]) CUDA_CHECK(cudaFreeHost(G.h_Yq[s]));
        if (G.h_Vq[s]) CUDA_CHECK(cudaFreeHost(G.h_Vq[s]));
        if (G.h_Cq[s]) CUDA_CHECK(cudaFreeHost(G.h_Cq[s]));

        if (G.h_sumMSE[s])       CUDA_CHECK(cudaFreeHost(G.h_sumMSE[s]));
        if (G.h_sumDevCen[s])    CUDA_CHECK(cudaFreeHost(G.h_sumDevCen[s]));
        if (G.h_sumDevCenTan[s]) CUDA_CHECK(cudaFreeHost(G.h_sumDevCenTan[s]));
        if (G.h_sumDevVec[s])    CUDA_CHECK(cudaFreeHost(G.h_sumDevVec[s]));

        G.d_Xq[s] = G.d_Yq[s] = G.d_Vq[s] = G.d_Cq[s] = nullptr;
        G.d_partialMSE[s] = G.d_partialDevCen[s] = G.d_partialDevCenTan[s] = G.d_partialDevVec[s] = nullptr;
        G.d_sumMSE[s] = G.d_sumDevCen[s] = G.d_sumDevCenTan[s] = G.d_sumDevVec[s] = nullptr;
        G.h_Xq[s] = G.h_Yq[s] = G.h_Vq[s] = G.h_Cq[s] = nullptr;
        G.h_sumMSE[s] = G.h_sumDevCen[s] = G.h_sumDevCenTan[s] = G.h_sumDevVec[s] = nullptr;
    }
    G.chunk_cap = 0;
    G.partial_cap_gridx = 0;
}

static void free_estimator_buffers()
{
    free_streamed_buffers();

    if (G.d_Ytrain)      CUDA_CHECK(cudaFree(G.d_Ytrain));
    if (G.d_Xtrain)      CUDA_CHECK(cudaFree(G.d_Xtrain));
    if (G.d_Cen)         CUDA_CHECK(cudaFree(G.d_Cen));
    if (G.d_V)           CUDA_CHECK(cudaFree(G.d_V));
    if (G.d_VdotX)       CUDA_CHECK(cudaFree(G.d_VdotX));
    if (G.d_Xtrainnorm2) CUDA_CHECK(cudaFree(G.d_Xtrainnorm2));
    if (G.d_RhoScale)    CUDA_CHECK(cudaFree(G.d_RhoScale));

    G.d_Ytrain = G.d_Xtrain = G.d_Cen = G.d_V = G.d_VdotX = G.d_Xtrainnorm2 = G.d_RhoScale = nullptr;

    G.estimator_initialized = false;
    G.D = 0;
    G.Ntrain = 0;
    G.m = 0;
}

static void free_test_cache()
{
    if (G.d_Xtest_all) CUDA_CHECK(cudaFree(G.d_Xtest_all));
    if (G.d_Ytrue_all) CUDA_CHECK(cudaFree(G.d_Ytrue_all));
    if (G.d_Vtrue_all) CUDA_CHECK(cudaFree(G.d_Vtrue_all));
    if (G.d_Ctrue_all) CUDA_CHECK(cudaFree(G.d_Ctrue_all));

    G.d_Xtest_all = G.d_Ytrue_all = G.d_Vtrue_all = G.d_Ctrue_all = nullptr;
    G.test_cached = false;
    G.Ntest_cached = 0;
    G.Dtest_cached = 0;
}

static void free_all_buffers()
{
    free_estimator_buffers();
    free_test_cache();

    if (G.streams_created) {
        CUDA_CHECK(cudaStreamDestroy(G.streams[0]));
        CUDA_CHECK(cudaStreamDestroy(G.streams[1]));
        G.streams_created = false;
    }

    maybe_unlock();
}

static void atExitHandler()
{
    try {
        free_all_buffers();
    } catch (...) {
    }
}

static void ensure_streams()
{
    if (!G.streams_created) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&G.streams[0], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&G.streams[1], cudaStreamNonBlocking));
        G.streams_created = true;
    }
}

static bool is_real_numeric_single_or_double(const mxArray* a)
{
    if (a == nullptr || mxIsComplex(a)) return false;
    mxClassID cid = mxGetClassID(a);
    return (cid == mxSINGLE_CLASS || cid == mxDOUBLE_CLASS);
}

static std::string get_command_string(const mxArray* a)
{
    if (!mxIsChar(a)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cmd", "First input must be a command string.");
    }
    char* s = mxArrayToString(a);
    if (!s) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cmd", "Failed to read command string.");
    }
    std::string out(s);
    mxFree(s);
    return out;
}

static const mxArray* get_required_field(const mxArray* s, const char* name)
{
    if (!mxIsStruct(s)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:struct", "Expected a struct.");
    }
    const mxArray* f = mxGetField(s, 0, name);
    if (!f) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:field", "Missing required field '%s'.", name);
    }
    return f;
}

static const mxArray* get_optional_field(const mxArray* s, const char* name)
{
    if (!mxIsStruct(s)) return nullptr;
    return mxGetField(s, 0, name);
}

static float read_scalar_float(const mxArray* a)
{
    if (!is_real_numeric_single_or_double(a) || mxGetNumberOfElements(a) != 1) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:scalar", "Expected a real numeric scalar.");
    }
    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        return ((const float*)mxGetData(a))[0];
    } else {
        return (float)((const double*)mxGetData(a))[0];
    }
}

static EvalOpts parse_opts(const mxArray* opts, int D)
{
    EvalOpts o;
    o.count_sparse_threshold = std::max(1, 2 * D);

    if (!opts) return o;
    if (!mxIsStruct(opts)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:opts", "opts must be a struct.");
    }

    auto get_if = [&](const char* name) -> const mxArray* {
        return mxGetField(opts, 0, name);
    };

    if (const mxArray* a = get_if("power_iter_steps"))          o.power_iter_steps = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("sum_sparse_threshold"))      o.sum_sparse_threshold = (float)mxGetScalar(a);
    if (const mxArray* a = get_if("count_sparse_threshold"))    o.count_sparse_threshold = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("fallback_exp_scale"))        o.fallback_exp_scale = (float)mxGetScalar(a);
    if (const mxArray* a = get_if("eps_weight"))                o.eps_weight = (float)mxGetScalar(a);

    if (const mxArray* a = get_if("manual_chunk"))              o.manual_chunk = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("auto_chunk_mem_fraction"))   o.auto_chunk_mem_fraction = (float)mxGetScalar(a);
    if (const mxArray* a = get_if("auto_chunk_safety_bytes"))   o.auto_chunk_safety_bytes = (double)mxGetScalar(a);
    if (const mxArray* a = get_if("auto_chunk_min"))            o.auto_chunk_min = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("auto_chunk_max"))            o.auto_chunk_max = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("auto_chunk_target_pairs"))   o.auto_chunk_target_pairs = (double)mxGetScalar(a);

    if (const mxArray* a = get_if("enable_host_double_buffer")) o.enable_host_double_buffer = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("enable_exact_d20_specialization")) o.enable_exact_d20_specialization = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("enable_exact_d20_train_tiling")) o.enable_exact_d20_train_tiling = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("enable_exact_d20_multi_hyper_tile")) o.enable_exact_d20_multi_hyper_tile = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("profile_timers")) o.profile_timers = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("silent")) o.silent = (int)std::llround(mxGetScalar(a));
    if (const mxArray* a = get_if("exact_tangent_small_d"))     o.exact_tangent_small_d = (int)std::llround(mxGetScalar(a));
    else if (const mxArray* a = get_if("enable_covariance_tangent")) o.exact_tangent_small_d = (int)std::llround(mxGetScalar(a)); // legacy alias

    o.power_iter_steps = std::max(1, o.power_iter_steps);
    o.count_sparse_threshold = std::max(1, o.count_sparse_threshold);
    o.sum_sparse_threshold = std::max(0.0f, o.sum_sparse_threshold);
    o.fallback_exp_scale = std::max(0.0f, o.fallback_exp_scale);
    o.eps_weight = std::max(1.0e-30f, o.eps_weight);

    o.manual_chunk = std::max(0, o.manual_chunk);
    o.auto_chunk_mem_fraction = std::min(0.95f, std::max(0.05f, o.auto_chunk_mem_fraction));
    o.auto_chunk_min = std::max(1, o.auto_chunk_min);
    o.auto_chunk_max = std::max(o.auto_chunk_min, o.auto_chunk_max);
    o.auto_chunk_target_pairs = std::max(1.0, o.auto_chunk_target_pairs);

    return o;
}

static void copy_linear_numeric_to_float_vector(const mxArray* a, std::vector<float>& out)
{
    if (!is_real_numeric_single_or_double(a)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:type", "Expected real numeric single/double.");
    }
    size_t n = mxGetNumberOfElements(a);
    out.resize(n);

    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        std::memcpy(out.data(), mxGetData(a), n * sizeof(float));
    } else {
        const double* p = (const double*)mxGetData(a);
        for (size_t i = 0; i < n; ++i) out[i] = (float)p[i];
    }
}

static void upload_linear_numeric_to_device_float(const mxArray* a, float*& d_dst)
{
    if (!is_real_numeric_single_or_double(a)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:type", "Expected real numeric single/double.");
    }
    size_t n = mxGetNumberOfElements(a);
    CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));

    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        CUDA_CHECK(cudaMemcpy(d_dst, mxGetData(a), n * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        std::vector<float> tmp;
        copy_linear_numeric_to_float_vector(a, tmp);
        CUDA_CHECK(cudaMemcpy(d_dst, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
}

static void upload_matrix_cm_numeric_to_device_float(const mxArray* a, int D, int N, float*& d_dst)
{
    if (!is_real_numeric_single_or_double(a)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:type", "Expected real numeric single/double matrix.");
    }
    if ((int)mxGetM(a) != D || (int)mxGetN(a) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:dim", "Matrix size mismatch.");
    }

    size_t nElem = (size_t)D * (size_t)N;
    CUDA_CHECK(cudaMalloc(&d_dst, nElem * sizeof(float)));

    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        CUDA_CHECK(cudaMemcpy(d_dst, mxGetData(a), nElem * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        const double* src = (const double*)mxGetData(a);
        std::vector<float> tmp(nElem);
        for (size_t i = 0; i < nElem; ++i) tmp[i] = (float)src[i];
        CUDA_CHECK(cudaMemcpy(d_dst, tmp.data(), nElem * sizeof(float), cudaMemcpyHostToDevice));
    }
}

static void copy_matrix_chunk_cm_to_pinned(const mxArray* a, int D, int q0, int B, float* dst)
{
    size_t nElem = (size_t)D * (size_t)B;
    size_t offset = (size_t)q0 * (size_t)D;

    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        const float* src = (const float*)mxGetData(a);
        std::memcpy(dst, src + offset, nElem * sizeof(float));
    } else {
        const double* src = (const double*)mxGetData(a);
        for (size_t i = 0; i < nElem; ++i) dst[i] = (float)src[offset + i];
    }
}

static void copy_vector_chunk_to_pinned(const mxArray* a, int q0, int B, float* dst)
{
    size_t nElem = (size_t)B;
    size_t offset = (size_t)q0;

    if (mxGetClassID(a) == mxSINGLE_CLASS) {
        const float* src = (const float*)mxGetData(a);
        std::memcpy(dst, src + offset, nElem * sizeof(float));
    } else {
        const double* src = (const double*)mxGetData(a);
        for (size_t i = 0; i < nElem; ++i) dst[i] = (float)src[offset + i];
    }
}

__global__ void train_stats_kernel_cm(
    const float* __restrict__ Xtrain,
    const float* __restrict__ V,
    int D, int Ntrain,
    float* __restrict__ Xtrainnorm2,
    float* __restrict__ VdotX)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Ntrain) return;

    float xx = 0.0f;
    float vx = 0.0f;
    for (int d = 0; d < D; ++d) {
        float x = Xtrain[cm_index(D, d, j)];
        float v = V[cm_index(D, d, j)];
        xx += x * x;
        vx += v * x;
    }
    Xtrainnorm2[j] = xx;
    VdotX[j] = vx;
}

__device__ __forceinline__ float safe_rsqrtf_eval(float x)
{
    return rsqrtf(fmaxf(x, 1.0e-30f));
}

__device__ void jacobi_top_eigvec_exact_small(
    const float* cov_in, int D, float* out_vec)
{
    const int S = EXACT_EIG_D_THRESHOLD;
    float A[EXACT_EIG_D_THRESHOLD * EXACT_EIG_D_THRESHOLD];
    float U[EXACT_EIG_D_THRESHOLD * EXACT_EIG_D_THRESHOLD];

    for (int i = 0; i < S * S; ++i) {
        A[i] = 0.0f;
        U[i] = 0.0f;
    }
    for (int i = 0; i < D; ++i) {
        U[i * S + i] = 1.0f;
        for (int j = 0; j < D; ++j) {
            A[i * S + j] = cov_in[i * S + j];
        }
    }

    if (D == 1) {
        out_vec[0] = 1.0f;
        return;
    }

    const int max_sweeps = 12 * S * S;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int p = 0, q = 1;
        float max_off = 0.0f;
        for (int i = 0; i < D; ++i) {
            for (int j = i + 1; j < D; ++j) {
                float aij = fabsf(A[i * S + j]);
                if (aij > max_off) {
                    max_off = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off <= 1.0e-6f) break;

        float app = A[p * S + p];
        float aqq = A[q * S + q];
        float apq = A[p * S + q];
        if (fabsf(apq) <= 1.0e-30f) continue;

        float tau = (aqq - app) / (2.0f * apq);
        float t = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
        float c = safe_rsqrtf_eval(1.0f + t * t);
        float s = t * c;

        for (int k = 0; k < D; ++k) {
            if (k == p || k == q) continue;
            float aik = A[p * S + k];
            float aqk = A[q * S + k];
            float new_aik = c * aik - s * aqk;
            float new_aqk = s * aik + c * aqk;
            A[p * S + k] = new_aik;
            A[k * S + p] = new_aik;
            A[q * S + k] = new_aqk;
            A[k * S + q] = new_aqk;
        }

        float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        A[p * S + p] = new_app;
        A[q * S + q] = new_aqq;
        A[p * S + q] = 0.0f;
        A[q * S + p] = 0.0f;

        for (int k = 0; k < D; ++k) {
            float ukp = U[k * S + p];
            float ukq = U[k * S + q];
            U[k * S + p] = c * ukp - s * ukq;
            U[k * S + q] = s * ukp + c * ukq;
        }
    }

    int best = 0;
    float best_eval = A[0];
    for (int i = 1; i < D; ++i) {
        float ev = A[i * S + i];
        if (ev > best_eval) {
            best_eval = ev;
            best = i;
        }
    }

    float n2 = 0.0f;
    for (int d = 0; d < D; ++d) {
        out_vec[d] = U[d * S + best];
        n2 += out_vec[d] * out_vec[d];
    }
    if (n2 <= 1.0e-30f) {
        for (int d = 0; d < D; ++d) out_vec[d] = 0.0f;
        out_vec[0] = 1.0f;
    } else {
        float invn = safe_rsqrtf_eval(n2);
        for (int d = 0; d < D; ++d) out_vec[d] *= invn;
    }
}

template<int DEXACT>
__device__ void jacobi_top_eigvec_exact_fixed(
    const float* cov_in, float* out_vec)
{
    float A[DEXACT * DEXACT];
    float U[DEXACT * DEXACT];

#pragma unroll
    for (int i = 0; i < DEXACT * DEXACT; ++i) {
        A[i] = cov_in[i];
        U[i] = 0.0f;
    }
#pragma unroll
    for (int i = 0; i < DEXACT; ++i) U[i * DEXACT + i] = 1.0f;

    const int max_sweeps = 12 * DEXACT * DEXACT;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int p = 0, q = 1;
        float max_off = 0.0f;
#pragma unroll
        for (int i = 0; i < DEXACT; ++i) {
#pragma unroll
            for (int j = i + 1; j < DEXACT; ++j) {
                float aij = fabsf(A[i * DEXACT + j]);
                if (aij > max_off) {
                    max_off = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off <= 1.0e-6f) break;

        float app = A[p * DEXACT + p];
        float aqq = A[q * DEXACT + q];
        float apq = A[p * DEXACT + q];
        if (fabsf(apq) <= 1.0e-30f) continue;

        float tau = (aqq - app) / (2.0f * apq);
        float t = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
        float c = safe_rsqrtf_eval(1.0f + t * t);
        float s = t * c;

#pragma unroll
        for (int k = 0; k < DEXACT; ++k) {
            if (k == p || k == q) continue;
            float aik = A[p * DEXACT + k];
            float aqk = A[q * DEXACT + k];
            float new_aik = c * aik - s * aqk;
            float new_aqk = s * aik + c * aqk;
            A[p * DEXACT + k] = new_aik;
            A[k * DEXACT + p] = new_aik;
            A[q * DEXACT + k] = new_aqk;
            A[k * DEXACT + q] = new_aqk;
        }

        float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        A[p * DEXACT + p] = new_app;
        A[q * DEXACT + q] = new_aqq;
        A[p * DEXACT + q] = 0.0f;
        A[q * DEXACT + p] = 0.0f;

#pragma unroll
        for (int k = 0; k < DEXACT; ++k) {
            float ukp = U[k * DEXACT + p];
            float ukq = U[k * DEXACT + q];
            U[k * DEXACT + p] = c * ukp - s * ukq;
            U[k * DEXACT + q] = s * ukp + c * ukq;
        }
    }

    int best = 0;
    float best_eval = A[0];
#pragma unroll
    for (int i = 1; i < DEXACT; ++i) {
        float ev = A[i * DEXACT + i];
        if (ev > best_eval) {
            best_eval = ev;
            best = i;
        }
    }

    float n2 = 0.0f;
#pragma unroll
    for (int d = 0; d < DEXACT; ++d) {
        out_vec[d] = U[d * DEXACT + best];
        n2 += out_vec[d] * out_vec[d];
    }
    if (n2 <= 1.0e-30f) {
#pragma unroll
        for (int d = 0; d < DEXACT; ++d) out_vec[d] = 0.0f;
        out_vec[0] = 1.0f;
    } else {
        float invn = safe_rsqrtf_eval(n2);
#pragma unroll
        for (int d = 0; d < DEXACT; ++d) out_vec[d] *= invn;
    }
}

__device__ __forceinline__ float warp_allreduce_sum_float(float v)
{
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(mask, v, off);
    return __shfl_sync(mask, v, 0);
}

template<int WARPS_PER_BLOCK>
__global__ __launch_bounds__(32 * WARPS_PER_BLOCK, 2) void eval_fast_warp_atomic_kernel_cm(
    const float* __restrict__ Xtest,   // [D,B] column-major
    const float* __restrict__ Ytrue,   // [B]
    const float* __restrict__ Vtrue,   // [D,B] column-major
    const float* __restrict__ Ctrue,   // [D,B] column-major

    const float* __restrict__ Ytrain,      // [Ntrain]
    const float* __restrict__ Xtrain,      // [D,Ntrain] column-major
    const float* __restrict__ Cen,         // [D,Ntrain] column-major
    const float* __restrict__ V,           // [D,Ntrain] column-major
    const float* __restrict__ VdotX,       // [Ntrain]
    const float* __restrict__ Xtrainnorm2, // [Ntrain]
    const float* __restrict__ rhoScale,    // [m]

    float h,
    int D, int B, int Ntrain, int m,
    int power_iter_steps,
    float sum_sparse_threshold,
    int count_sparse_threshold,
    float fallback_exp_scale,
    float eps_weight,
    int exact_tangent_small_d,

    float* __restrict__ sumMSE,
    float* __restrict__ sumDevCen,
    float* __restrict__ sumDevCenTan,
    float* __restrict__ sumDevVec)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int ib = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int ir = blockIdx.y;

    if (ir >= m || ib >= B) return;

    __shared__ float s_cov[WARPS_PER_BLOCK][EXACT_EIG_D_THRESHOLD * EXACT_EIG_D_THRESHOLD];
    __shared__ float s_u[WARPS_PER_BLOCK][EXACT_EIG_D_THRESHOLD];

    float xt_d = 0.0f, vt_d = 0.0f, ct_d = 0.0f;
    if (lane < D) {
        xt_d = Xtest[cm_index(D, lane, ib)];
        vt_d = Vtrue[cm_index(D, lane, ib)];
        ct_d = Ctrue[cm_index(D, lane, ib)];
    }

    float xt2 = warp_allreduce_sum_float((lane < D) ? (xt_d * xt_d) : 0.0f);

    const float rho = rhoScale[ir];
    const float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);
    const bool use_exact_cov = (exact_tangent_small_d && D <= EXACT_EIG_D_THRESHOLD);

    float min_q = 1.0e30f;
    float sum_quartic = 0.0f;
    int count_pos = 0;

    for (int j = 0; j < Ntrain; ++j) {
        float xjd = (lane < D) ? Xtrain[cm_index(D, lane, j)] : 0.0f;
        float vjd = (lane < D) ? V[cm_index(D, lane, j)] : 0.0f;

        float xtxj = warp_allreduce_sum_float(xt_d * xjd);
        float xtvj = warp_allreduce_sum_float(xt_d * vjd);

        if (lane == 0) {
            float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
            eucl2 = fmaxf(eucl2, 0.0f);
            float proj = xtvj - VdotX[j];
            float q = (eucl2 + rho * proj * proj) * inv_h2;

            min_q = fminf(min_q, q);
            float rt2 = 1.0f - q * q;
            if (rt2 > 0.0f) {
                sum_quartic += rt2 * rt2;
                count_pos += 1;
            }
        }
    }

    min_q = __shfl_sync(0xffffffffu, min_q, 0);
    sum_quartic = __shfl_sync(0xffffffffu, sum_quartic, 0);
    count_pos = __shfl_sync(0xffffffffu, count_pos, 0);
    const bool sparse = (sum_quartic < sum_sparse_threshold) || (count_pos < count_sparse_threshold);

    float sumW = 0.0f;
    float pred = 0.0f;
    float csum_d = 0.0f;
    float u_d = 0.0f;
    float cov_row[EXACT_EIG_D_THRESHOLD];

    if (use_exact_cov && lane < D) {
#pragma unroll
        for (int b = 0; b < EXACT_EIG_D_THRESHOLD; ++b) cov_row[b] = 0.0f;
    }

    for (int j = 0; j < Ntrain; ++j) {
        float xjd = (lane < D) ? Xtrain[cm_index(D, lane, j)] : 0.0f;
        float vjd = (lane < D) ? V[cm_index(D, lane, j)] : 0.0f;
        float cjd = (lane < D) ? Cen[cm_index(D, lane, j)] : 0.0f;

        float xtxj = warp_allreduce_sum_float(xt_d * xjd);
        float xtvj = warp_allreduce_sum_float(xt_d * vjd);

        float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
        eucl2 = fmaxf(eucl2, 0.0f);
        float proj = xtvj - VdotX[j];
        float q = (eucl2 + rho * proj * proj) * inv_h2;

        float w;
        if (sparse) {
            w = expf(fallback_exp_scale * (-q + min_q));
        } else {
            float rt2 = 1.0f - q * q;
            w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
        }

        if (lane == 0) {
            sumW += w;
            pred += w * Ytrain[j];
        }

        if (lane < D) {
            csum_d += w * cjd;
            if (use_exact_cov) {
                float va = vjd;
#pragma unroll
                for (int b = 0; b < EXACT_EIG_D_THRESHOLD; ++b) {
                    if (b < D) {
                        float vb = __shfl_sync(0xffffffffu, vjd, b);
                        cov_row[b] += w * va * vb;
                    }
                }
            } else {
                u_d += w * vjd;
            }
        }
    }

    sumW = __shfl_sync(0xffffffffu, sumW, 0);
    pred = __shfl_sync(0xffffffffu, pred, 0);

    sumW = fmaxf(sumW, eps_weight);
    pred /= sumW;

    if (lane < D) csum_d /= sumW;

    if (use_exact_cov) {
        if (lane < D) {
            float* cov_ptr = s_cov[warp_id];
#pragma unroll
            for (int b = 0; b < EXACT_EIG_D_THRESHOLD; ++b) {
                if (b < D) cov_ptr[lane * EXACT_EIG_D_THRESHOLD + b] = cov_row[b] / sumW;
            }
        }
        __syncwarp();
        if (lane == 0) {
            jacobi_top_eigvec_exact_small(s_cov[warp_id], D, s_u[warp_id]);
        }
        __syncwarp();
        if (lane < D) {
            u_d = s_u[warp_id][lane];
        } else {
            u_d = 0.0f;
        }
    } else {
        if (lane < D) u_d /= sumW;
        float u2 = warp_allreduce_sum_float((lane < D) ? (u_d * u_d) : 0.0f);
        if (u2 <= 1.0e-30f) {
            u_d = (lane == 0 ? 1.0f : 0.0f);
        } else {
            float invn = safe_rsqrtf_eval(u2);
            if (lane < D) u_d *= invn;
            else u_d = 0.0f;
        }

        for (int it = 0; it < power_iter_steps; ++it) {
            float unew_d = 0.0f;

            for (int j = 0; j < Ntrain; ++j) {
                float xjd = (lane < D) ? Xtrain[cm_index(D, lane, j)] : 0.0f;
                float vjd = (lane < D) ? V[cm_index(D, lane, j)] : 0.0f;

                float xtxj = warp_allreduce_sum_float(xt_d * xjd);
                float xtvj = warp_allreduce_sum_float(xt_d * vjd);

                float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
                eucl2 = fmaxf(eucl2, 0.0f);
                float proj = xtvj - VdotX[j];
                float q = (eucl2 + rho * proj * proj) * inv_h2;

                float w;
                if (sparse) {
                    w = expf(fallback_exp_scale * (-q + min_q));
                } else {
                    float rt2 = 1.0f - q * q;
                    w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
                }

                float vu = warp_allreduce_sum_float((lane < D) ? (vjd * u_d) : 0.0f);
                if (lane < D) {
                    unew_d += w * vu * vjd;
                }
            }

            if (lane < D) unew_d /= sumW;
            float n2 = warp_allreduce_sum_float((lane < D) ? (unew_d * unew_d) : 0.0f);
            if (n2 <= 1.0e-30f) break;
            float invn = safe_rsqrtf_eval(n2);
            if (lane < D) u_d = unew_d * invn;
        }
    }

    float dc = (lane < D) ? (csum_d - ct_d) : 0.0f;
    float cen2 = warp_allreduce_sum_float(dc * dc);
    float cen_tan = warp_allreduce_sum_float(dc * vt_d);

    float dm = (lane < D) ? (u_d - vt_d) : 0.0f;
    float dp = (lane < D) ? (u_d + vt_d) : 0.0f;
    float devm = warp_allreduce_sum_float(dm * dm);
    float devp = warp_allreduce_sum_float(dp * dp);

    if (lane == 0) {
        float err = pred - Ytrue[ib];
        atomicAdd(sumMSE + ir, err * err);
        atomicAdd(sumDevCen + ir, sqrtf(cen2));
        atomicAdd(sumDevCenTan + ir, fabsf(cen_tan));
        atomicAdd(sumDevVec + ir, sqrtf(fminf(devm, devp)));
    }
}




template<int DEXACT, int TILE, int HYPERS_PER_BLOCK, int SAMPLES_PER_BLOCK>
__global__ void eval_partials_kernel_cm_exact_tiled_multi_hyper(
    const float* __restrict__ Xtest,
    const float* __restrict__ Ytrue,
    const float* __restrict__ Vtrue,
    const float* __restrict__ Ctrue,

    const float* __restrict__ Ytrain,
    const float* __restrict__ Xtrain,
    const float* __restrict__ Cen,
    const float* __restrict__ V,
    const float* __restrict__ VdotX,
    const float* __restrict__ Xtrainnorm2,
    const float* __restrict__ rhoScale,

    float h,
    int B, int Ntrain, int m,
    float sum_sparse_threshold,
    int count_sparse_threshold,
    float fallback_exp_scale,
    float eps_weight,
    int gridX,

    float* __restrict__ partialMSE,
    float* __restrict__ partialDevCen,
    float* __restrict__ partialDevCenTan,
    float* __restrict__ partialDevVec)
{
    constexpr int D = DEXACT;
    constexpr int BLOCK = HYPERS_PER_BLOCK * SAMPLES_PER_BLOCK;

    const int tid = threadIdx.x;
    const int hyper_local = tid / SAMPLES_PER_BLOCK;
    const int lane = tid - hyper_local * SAMPLES_PER_BLOCK;
    const int ir_base = blockIdx.y * HYPERS_PER_BLOCK;
    const int ir = ir_base + hyper_local;
    const int ib = blockIdx.x * SAMPLES_PER_BLOCK + lane;
    const bool active = (hyper_local < HYPERS_PER_BLOCK) && (ir < m) && (ib < B);

    __shared__ float sX[TILE * D];
    __shared__ float sV[TILE * D];
    __shared__ float sC[TILE * D];
    __shared__ float sY[TILE];
    __shared__ float sVdotX[TILE];
    __shared__ float sXnorm2[TILE];

    float mse = 0.0f;
    float devCen = 0.0f;
    float devCenTan = 0.0f;
    float devVec = 0.0f;

    float xt[D];
    float vt_true[D];
    float ct_true[D];
    float csum[D];
    float u[D];
    float cov_small[D * D];

    float xt2 = 0.0f;
    if (active) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float xv = Xtest[cm_index(D, d, ib)];
            xt[d] = xv;
            xt2 += xv * xv;
            vt_true[d] = Vtrue[cm_index(D, d, ib)];
            ct_true[d] = Ctrue[cm_index(D, d, ib)];
            csum[d] = 0.0f;
            u[d] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < D * D; ++i) cov_small[i] = 0.0f;
    }

    const float rho = (ir < m) ? rhoScale[ir] : 0.0f;
    const float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);

    float min_q = 1.0e30f;
    float sum_quartic = 0.0f;
    int count_pos = 0;

    for (int j0 = 0; j0 < Ntrain; j0 += TILE) {
        int tileN = Ntrain - j0;
        if (tileN > TILE) tileN = TILE;

        for (int t = tid; t < tileN * D; t += BLOCK) {
            int tj = t / D;
            int d = t - tj * D;
            int j = j0 + tj;
            size_t idx = cm_index(D, d, j);
            sX[t] = Xtrain[idx];
            sV[t] = V[idx];
        }
        for (int t = tid; t < tileN; t += BLOCK) {
            int j = j0 + t;
            sVdotX[t] = VdotX[j];
            sXnorm2[t] = Xtrainnorm2[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 1
            for (int tj = 0; tj < tileN; ++tj) {
                int base = tj * D;
                float xtxj = 0.0f;
                float xtvj = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float xv = xt[d];
                    xtxj += xv * sX[base + d];
                    xtvj += xv * sV[base + d];
                }
                float eucl2 = xt2 + sXnorm2[tj] - 2.0f * xtxj;
                eucl2 = fmaxf(eucl2, 0.0f);
                float proj = xtvj - sVdotX[tj];
                float q = (eucl2 + rho * proj * proj) * inv_h2;

                min_q = fminf(min_q, q);
                float rt2 = 1.0f - q * q;
                if (rt2 > 0.0f) {
                    sum_quartic += rt2 * rt2;
                    count_pos += 1;
                }
            }
        }
        __syncthreads();
    }

    const bool sparse = (sum_quartic < sum_sparse_threshold) || (count_pos < count_sparse_threshold);

    float sumW = 0.0f;
    float pred = 0.0f;

    for (int j0 = 0; j0 < Ntrain; j0 += TILE) {
        int tileN = Ntrain - j0;
        if (tileN > TILE) tileN = TILE;

        for (int t = tid; t < tileN * D; t += BLOCK) {
            int tj = t / D;
            int d = t - tj * D;
            int j = j0 + tj;
            size_t idx = cm_index(D, d, j);
            sX[t] = Xtrain[idx];
            sV[t] = V[idx];
            sC[t] = Cen[idx];
        }
        for (int t = tid; t < tileN; t += BLOCK) {
            int j = j0 + t;
            sY[t] = Ytrain[j];
            sVdotX[t] = VdotX[j];
            sXnorm2[t] = Xtrainnorm2[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 1
            for (int tj = 0; tj < tileN; ++tj) {
                int base = tj * D;
                float xtxj = 0.0f;
                float xtvj = 0.0f;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float xv = xt[d];
                    xtxj += xv * sX[base + d];
                    xtvj += xv * sV[base + d];
                }
                float eucl2 = xt2 + sXnorm2[tj] - 2.0f * xtxj;
                eucl2 = fmaxf(eucl2, 0.0f);
                float proj = xtvj - sVdotX[tj];
                float q = (eucl2 + rho * proj * proj) * inv_h2;

                float w;
                if (sparse) {
                    w = expf(fallback_exp_scale * (-q + min_q));
                } else {
                    float rt2 = 1.0f - q * q;
                    w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
                }

                sumW += w;
                pred += w * sY[tj];
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    csum[d] += w * sC[base + d];
                }
#pragma unroll
                for (int a = 0; a < D; ++a) {
                    float va = sV[base + a];
#pragma unroll
                    for (int b = 0; b <= a; ++b) {
                        float vb = sV[base + b];
                        cov_small[a * D + b] += w * va * vb;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        sumW = fmaxf(sumW, eps_weight);
        pred /= sumW;
#pragma unroll
        for (int d = 0; d < D; ++d) csum[d] /= sumW;

        mse = pred - Ytrue[ib];
        mse *= mse;

        float cen2 = 0.0f;
        float cen_tan = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dc = csum[d] - ct_true[d];
            cen2 += dc * dc;
            cen_tan += dc * vt_true[d];
        }
        devCen = sqrtf(cen2);
        devCenTan = fabsf(cen_tan);

#pragma unroll
        for (int a = 0; a < D; ++a) {
#pragma unroll
            for (int b = 0; b <= a; ++b) {
                float v = cov_small[a * D + b] / sumW;
                cov_small[a * D + b] = v;
                cov_small[b * D + a] = v;
            }
        }
        jacobi_top_eigvec_exact_fixed<D>(cov_small, u);

        float devm = 0.0f, devp = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dm = u[d] - vt_true[d];
            float dp = u[d] + vt_true[d];
            devm += dm * dm;
            devp += dp * dp;
        }
        devVec = sqrtf(fminf(devm, devp));
    }

    float sum_mse = warp_allreduce_sum_float(mse);
    float sum_devCen = warp_allreduce_sum_float(devCen);
    float sum_devCenTan = warp_allreduce_sum_float(devCenTan);
    float sum_devVec = warp_allreduce_sum_float(devVec);

    if (lane == 0 && ir < m) {
        size_t idx = (size_t)ir * (size_t)gridX + (size_t)blockIdx.x;
        partialMSE[idx]       = sum_mse;
        partialDevCen[idx]    = sum_devCen;
        partialDevCenTan[idx] = sum_devCenTan;
        partialDevVec[idx]    = sum_devVec;
    }
}


template<int DEXACT, int BLOCK, int TILE>
__global__ void eval_partials_kernel_cm_exact_tiled(
    const float* __restrict__ Xtest,
    const float* __restrict__ Ytrue,
    const float* __restrict__ Vtrue,
    const float* __restrict__ Ctrue,

    const float* __restrict__ Ytrain,
    const float* __restrict__ Xtrain,
    const float* __restrict__ Cen,
    const float* __restrict__ V,
    const float* __restrict__ VdotX,
    const float* __restrict__ Xtrainnorm2,
    const float* __restrict__ rhoScale,

    float h,
    int B, int Ntrain, int m,
    float sum_sparse_threshold,
    int count_sparse_threshold,
    float fallback_exp_scale,
    float eps_weight,
    int gridX,

    float* __restrict__ partialMSE,
    float* __restrict__ partialDevCen,
    float* __restrict__ partialDevCenTan,
    float* __restrict__ partialDevVec)
{
    constexpr int D = DEXACT;
    const int ir  = blockIdx.y;
    const int tid = threadIdx.x;
    const int ib  = blockIdx.x * BLOCK + tid;
    const bool active = (ir < m && ib < B);

    __shared__ float sX[TILE * D];
    __shared__ float sV[TILE * D];
    __shared__ float sC[TILE * D];
    __shared__ float sY[TILE];
    __shared__ float sVdotX[TILE];
    __shared__ float sXnorm2[TILE];

    float mse = 0.0f;
    float devCen = 0.0f;
    float devCenTan = 0.0f;
    float devVec = 0.0f;

    float xt[D];
    float vt_true[D];
    float ct_true[D];
    float csum[D];
    float u[D];
    float cov_small[D * D];

    float xt2 = 0.0f;
    if (active) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float xv = Xtest[cm_index(D, d, ib)];
            xt[d] = xv;
            xt2 += xv * xv;
            vt_true[d] = Vtrue[cm_index(D, d, ib)];
            ct_true[d] = Ctrue[cm_index(D, d, ib)];
            csum[d] = 0.0f;
            u[d] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < D * D; ++i) cov_small[i] = 0.0f;
    }

    const float rho = (ir < m) ? rhoScale[ir] : 0.0f;
    const float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);

    float min_q = 1.0e30f;
    float sum_quartic = 0.0f;
    int count_pos = 0;

    for (int j0 = 0; j0 < Ntrain; j0 += TILE) {
        int tileN = Ntrain - j0;
        if (tileN > TILE) tileN = TILE;

        for (int t = tid; t < tileN * D; t += BLOCK) {
            int tj = t / D;
            int d = t - tj * D;
            int j = j0 + tj;
            size_t idx = cm_index(D, d, j);
            sX[t] = Xtrain[idx];
            sV[t] = V[idx];
        }
        for (int t = tid; t < tileN; t += BLOCK) {
            int j = j0 + t;
            sVdotX[t] = VdotX[j];
            sXnorm2[t] = Xtrainnorm2[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 1
            for (int tj = 0; tj < tileN; ++tj) {
                float xtxj = 0.0f;
                float xtvj = 0.0f;
                int base = tj * D;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float xv = xt[d];
                    xtxj += xv * sX[base + d];
                    xtvj += xv * sV[base + d];
                }
                float eucl2 = xt2 + sXnorm2[tj] - 2.0f * xtxj;
                eucl2 = fmaxf(eucl2, 0.0f);
                float proj = xtvj - sVdotX[tj];
                float q = (eucl2 + rho * proj * proj) * inv_h2;

                min_q = fminf(min_q, q);
                float rt2 = 1.0f - q * q;
                if (rt2 > 0.0f) {
                    sum_quartic += rt2 * rt2;
                    count_pos += 1;
                }
            }
        }
        __syncthreads();
    }

    const bool sparse = (sum_quartic < sum_sparse_threshold) || (count_pos < count_sparse_threshold);

    float sumW = 0.0f;
    float pred = 0.0f;

    for (int j0 = 0; j0 < Ntrain; j0 += TILE) {
        int tileN = Ntrain - j0;
        if (tileN > TILE) tileN = TILE;

        for (int t = tid; t < tileN * D; t += BLOCK) {
            int tj = t / D;
            int d = t - tj * D;
            int j = j0 + tj;
            size_t idx = cm_index(D, d, j);
            sX[t] = Xtrain[idx];
            sV[t] = V[idx];
            sC[t] = Cen[idx];
        }
        for (int t = tid; t < tileN; t += BLOCK) {
            int j = j0 + t;
            sY[t] = Ytrain[j];
            sVdotX[t] = VdotX[j];
            sXnorm2[t] = Xtrainnorm2[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 1
            for (int tj = 0; tj < tileN; ++tj) {
                float xtxj = 0.0f;
                float xtvj = 0.0f;
                int base = tj * D;
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    float xv = xt[d];
                    xtxj += xv * sX[base + d];
                    xtvj += xv * sV[base + d];
                }
                float eucl2 = xt2 + sXnorm2[tj] - 2.0f * xtxj;
                eucl2 = fmaxf(eucl2, 0.0f);
                float proj = xtvj - sVdotX[tj];
                float q = (eucl2 + rho * proj * proj) * inv_h2;

                float w;
                if (sparse) {
                    w = expf(fallback_exp_scale * (-q + min_q));
                } else {
                    float rt2 = 1.0f - q * q;
                    w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
                }

                sumW += w;
                pred += w * sY[tj];
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    csum[d] += w * sC[base + d];
                }

#pragma unroll
                for (int a = 0; a < D; ++a) {
                    float va = sV[base + a];
#pragma unroll
                    for (int b = 0; b <= a; ++b) {
                        float vb = sV[base + b];
                        cov_small[a * D + b] += w * va * vb;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        sumW = fmaxf(sumW, eps_weight);
        pred /= sumW;
#pragma unroll
        for (int d = 0; d < D; ++d) csum[d] /= sumW;

        mse = pred - Ytrue[ib];
        mse *= mse;

        float cen2 = 0.0f;
        float cen_tan = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dc = csum[d] - ct_true[d];
            cen2 += dc * dc;
            cen_tan += dc * vt_true[d];
        }
        devCen = sqrtf(cen2);
        devCenTan = fabsf(cen_tan);

#pragma unroll
        for (int a = 0; a < D; ++a) {
#pragma unroll
            for (int b = 0; b <= a; ++b) {
                float v = cov_small[a * D + b] / sumW;
                cov_small[a * D + b] = v;
                cov_small[b * D + a] = v;
            }
        }
        jacobi_top_eigvec_exact_fixed<D>(cov_small, u);

        float devm = 0.0f, devp = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dm = u[d] - vt_true[d];
            float dp = u[d] + vt_true[d];
            devm += dm * dm;
            devp += dp * dp;
        }
        devVec = sqrtf(fminf(devm, devp));
    }

    __shared__ float sMSE[BLOCK];
    __shared__ float sDevCen[BLOCK];
    __shared__ float sDevCenTan[BLOCK];
    __shared__ float sDevVec[BLOCK];

    sMSE[tid] = mse;
    sDevCen[tid] = devCen;
    sDevCenTan[tid] = devCenTan;
    sDevVec[tid] = devVec;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMSE[tid]       += sMSE[tid + stride];
            sDevCen[tid]    += sDevCen[tid + stride];
            sDevCenTan[tid] += sDevCenTan[tid + stride];
            sDevVec[tid]    += sDevVec[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t idx = (size_t)ir * (size_t)gridX + (size_t)blockIdx.x;
        partialMSE[idx]       = sMSE[0];
        partialDevCen[idx]    = sDevCen[0];
        partialDevCenTan[idx] = sDevCenTan[0];
        partialDevVec[idx]    = sDevVec[0];
    }
}

template<int DEXACT, int BLOCK>
__global__ void eval_partials_kernel_cm_exact(
    const float* __restrict__ Xtest,
    const float* __restrict__ Ytrue,
    const float* __restrict__ Vtrue,
    const float* __restrict__ Ctrue,

    const float* __restrict__ Ytrain,
    const float* __restrict__ Xtrain,
    const float* __restrict__ Cen,
    const float* __restrict__ V,
    const float* __restrict__ VdotX,
    const float* __restrict__ Xtrainnorm2,
    const float* __restrict__ rhoScale,

    float h,
    int B, int Ntrain, int m,
    float sum_sparse_threshold,
    int count_sparse_threshold,
    float fallback_exp_scale,
    float eps_weight,
    int gridX,

    float* __restrict__ partialMSE,
    float* __restrict__ partialDevCen,
    float* __restrict__ partialDevCenTan,
    float* __restrict__ partialDevVec)
{
    constexpr int D = DEXACT;
    const int ir  = blockIdx.y;
    const int tid = threadIdx.x;
    const int ib  = blockIdx.x * BLOCK + tid;

    float mse = 0.0f;
    float devCen = 0.0f;
    float devCenTan = 0.0f;
    float devVec = 0.0f;

    if (ir < m && ib < B) {
        float xt[D];
        float vt_true[D];
        float ct_true[D];
        float csum[D];
        float u[D];
        float cov_small[D * D];

        float xt2 = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float xv = Xtest[cm_index(D, d, ib)];
            xt[d] = xv;
            xt2 += xv * xv;
            vt_true[d] = Vtrue[cm_index(D, d, ib)];
            ct_true[d] = Ctrue[cm_index(D, d, ib)];
            csum[d] = 0.0f;
            u[d] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < D * D; ++i) cov_small[i] = 0.0f;

        const float rho = rhoScale[ir];
        const float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);

        float min_q = 1.0e30f;
        float sum_quartic = 0.0f;
        int count_pos = 0;

        for (int j = 0; j < Ntrain; ++j) {
            float xtxj = 0.0f;
            float xtvj = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                xtxj += xv * Xtrain[cm_index(D, d, j)];
                xtvj += xv * V[cm_index(D, d, j)];
            }
            float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
            eucl2 = fmaxf(eucl2, 0.0f);
            float proj = xtvj - VdotX[j];
            float q = (eucl2 + rho * proj * proj) * inv_h2;

            min_q = fminf(min_q, q);
            float rt2 = 1.0f - q * q;
            if (rt2 > 0.0f) {
                sum_quartic += rt2 * rt2;
                count_pos += 1;
            }
        }

        const bool sparse = (sum_quartic < sum_sparse_threshold) || (count_pos < count_sparse_threshold);

        float sumW = 0.0f;
        float pred = 0.0f;
        for (int j = 0; j < Ntrain; ++j) {
            float xtxj = 0.0f;
            float xtvj = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                xtxj += xv * Xtrain[cm_index(D, d, j)];
                xtvj += xv * V[cm_index(D, d, j)];
            }
            float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
            eucl2 = fmaxf(eucl2, 0.0f);
            float proj = xtvj - VdotX[j];
            float q = (eucl2 + rho * proj * proj) * inv_h2;

            float w;
            if (sparse) {
                w = expf(fallback_exp_scale * (-q + min_q));
            } else {
                float rt2 = 1.0f - q * q;
                w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
            }

            sumW += w;
            pred += w * Ytrain[j];
#pragma unroll
            for (int d = 0; d < D; ++d) {
                csum[d] += w * Cen[cm_index(D, d, j)];
            }

#pragma unroll
            for (int a = 0; a < D; ++a) {
                float va = V[cm_index(D, a, j)];
#pragma unroll
                for (int b = 0; b <= a; ++b) {
                    float vb = V[cm_index(D, b, j)];
                    cov_small[a * D + b] += w * va * vb;
                }
            }
        }

        sumW = fmaxf(sumW, eps_weight);
        pred /= sumW;
#pragma unroll
        for (int d = 0; d < D; ++d) csum[d] /= sumW;

        mse = pred - Ytrue[ib];
        mse *= mse;

        float cen2 = 0.0f;
        float cen_tan = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dc = csum[d] - ct_true[d];
            cen2 += dc * dc;
            cen_tan += dc * vt_true[d];
        }
        devCen = sqrtf(cen2);
        devCenTan = fabsf(cen_tan);

#pragma unroll
        for (int a = 0; a < D; ++a) {
#pragma unroll
            for (int b = 0; b <= a; ++b) {
                float v = cov_small[a * D + b] / sumW;
                cov_small[a * D + b] = v;
                cov_small[b * D + a] = v;
            }
        }
        jacobi_top_eigvec_exact_fixed<D>(cov_small, u);

        float devm = 0.0f, devp = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) {
            float dm = u[d] - vt_true[d];
            float dp = u[d] + vt_true[d];
            devm += dm * dm;
            devp += dp * dp;
        }
        devVec = sqrtf(fminf(devm, devp));
    }

    __shared__ float sMSE[BLOCK];
    __shared__ float sDevCen[BLOCK];
    __shared__ float sDevCenTan[BLOCK];
    __shared__ float sDevVec[BLOCK];

    sMSE[tid] = mse;
    sDevCen[tid] = devCen;
    sDevCenTan[tid] = devCenTan;
    sDevVec[tid] = devVec;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMSE[tid]       += sMSE[tid + stride];
            sDevCen[tid]    += sDevCen[tid + stride];
            sDevCenTan[tid] += sDevCenTan[tid + stride];
            sDevVec[tid]    += sDevVec[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t idx = (size_t)ir * (size_t)gridX + (size_t)blockIdx.x;
        partialMSE[idx]       = sMSE[0];
        partialDevCen[idx]    = sDevCen[0];
        partialDevCenTan[idx] = sDevCenTan[0];
        partialDevVec[idx]    = sDevVec[0];
    }
}

template<int BLOCK>
__global__ void eval_partials_kernel_cm(
    const float* __restrict__ Xtest,   // [D,B] column-major
    const float* __restrict__ Ytrue,   // [B]
    const float* __restrict__ Vtrue,   // [D,B] column-major
    const float* __restrict__ Ctrue,   // [D,B] column-major

    const float* __restrict__ Ytrain,      // [Ntrain]
    const float* __restrict__ Xtrain,      // [D,Ntrain] column-major
    const float* __restrict__ Cen,         // [D,Ntrain] column-major
    const float* __restrict__ V,           // [D,Ntrain] column-major
    const float* __restrict__ VdotX,       // [Ntrain]
    const float* __restrict__ Xtrainnorm2, // [Ntrain]
    const float* __restrict__ rhoScale,    // [m]

    float h,
    int D, int B, int Ntrain, int m,
    int power_iter_steps,
    float sum_sparse_threshold,
    int count_sparse_threshold,
    float fallback_exp_scale,
    float eps_weight,
    int exact_tangent_small_d,
    int gridX,

    float* __restrict__ partialMSE,
    float* __restrict__ partialDevCen,
    float* __restrict__ partialDevCenTan,
    float* __restrict__ partialDevVec)
{
    const int ir  = blockIdx.y;
    const int tid = threadIdx.x;
    const int ib  = blockIdx.x * BLOCK + tid;

    float mse = 0.0f;
    float devCen = 0.0f;
    float devCenTan = 0.0f;
    float devVec = 0.0f;

    if (ir < m && ib < B) {
        float xt[MAX_D_MANOR_EVAL];
        float vt_true[MAX_D_MANOR_EVAL];
        float ct_true[MAX_D_MANOR_EVAL];
        float csum[MAX_D_MANOR_EVAL];
        float u[MAX_D_MANOR_EVAL];
        float unew[MAX_D_MANOR_EVAL];
        float cov_small[EXACT_EIG_D_THRESHOLD * EXACT_EIG_D_THRESHOLD];

        float xt2 = 0.0f;
        for (int d = 0; d < D; ++d) {
            float xv = Xtest[cm_index(D, d, ib)];
            xt[d] = xv;
            xt2 += xv * xv;
            vt_true[d] = Vtrue[cm_index(D, d, ib)];
            ct_true[d] = Ctrue[cm_index(D, d, ib)];
            csum[d] = 0.0f;
            u[d] = 0.0f;
            unew[d] = 0.0f;
        }
        if (exact_tangent_small_d && D <= EXACT_EIG_D_THRESHOLD) {
            for (int i = 0; i < EXACT_EIG_D_THRESHOLD * EXACT_EIG_D_THRESHOLD; ++i) cov_small[i] = 0.0f;
        }

        float rho = rhoScale[ir];
        float inv_h2 = 1.0f / fmaxf(h * h, 1.0e-30f);

        // pass 1: determine sparse regime using X_Train / VdotX
        float min_q = 1.0e30f;
        float sum_quartic = 0.0f;
        int count_pos = 0;

        for (int j = 0; j < Ntrain; ++j) {
            float xtxj = 0.0f;
            float xtvj = 0.0f;
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                xtxj += xv * Xtrain[cm_index(D, d, j)];
                xtvj += xv * V[cm_index(D, d, j)];
            }
            float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
            eucl2 = fmaxf(eucl2, 0.0f);
            float proj = xtvj - VdotX[j];
            float q = (eucl2 + rho * proj * proj) * inv_h2;

            min_q = fminf(min_q, q);
            float rt2 = 1.0f - q * q;
            if (rt2 > 0.0f) {
                sum_quartic += rt2 * rt2;
                count_pos += 1;
            }
        }

        bool sparse = (sum_quartic < sum_sparse_threshold) || (count_pos < count_sparse_threshold);

        // pass 2: weighted prediction / center / tangent covariance
        const bool use_exact_cov = (exact_tangent_small_d && D <= EXACT_EIG_D_THRESHOLD);
        float sumW = 0.0f;
        float pred = 0.0f;
        for (int j = 0; j < Ntrain; ++j) {
            float xtxj = 0.0f;
            float xtvj = 0.0f;
            for (int d = 0; d < D; ++d) {
                float xv = xt[d];
                xtxj += xv * Xtrain[cm_index(D, d, j)];
                xtvj += xv * V[cm_index(D, d, j)];
            }
            float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
            eucl2 = fmaxf(eucl2, 0.0f);
            float proj = xtvj - VdotX[j];
            float q = (eucl2 + rho * proj * proj) * inv_h2;

            float w;
            if (sparse) {
                w = expf(fallback_exp_scale * (-q + min_q));
            } else {
                float rt2 = 1.0f - q * q;
                w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
            }

            sumW += w;
            pred += w * Ytrain[j];
            for (int d = 0; d < D; ++d) {
                csum[d] += w * Cen[cm_index(D, d, j)];
                if (!use_exact_cov) {
                    u[d] += w * V[cm_index(D, d, j)]; // truth-free initializer for large-D fallback only
                }
            }

            if (use_exact_cov) {
                // Covariance is symmetric. Accumulate only the lower triangle here,
                // then mirror once after normalization. This keeps the exact math
                // unchanged while roughly halving the heavy exact-path work.
                for (int a = 0; a < D; ++a) {
                    float va = V[cm_index(D, a, j)];
                    for (int b = 0; b <= a; ++b) {
                        float vb = V[cm_index(D, b, j)];
                        cov_small[a * EXACT_EIG_D_THRESHOLD + b] += w * va * vb;
                    }
                }
            }
        }

        sumW = fmaxf(sumW, eps_weight);
        pred /= sumW;
        for (int d = 0; d < D; ++d) {
            csum[d] /= sumW;
            if (!use_exact_cov) u[d] /= sumW;
        }

        mse = pred - Ytrue[ib];
        mse *= mse;

        float cen2 = 0.0f;
        float cen_tan = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dc = csum[d] - ct_true[d];
            cen2 += dc * dc;
            cen_tan += dc * vt_true[d];
        }
        devCen = sqrtf(cen2);
        devCenTan = fabsf(cen_tan);

        if (use_exact_cov) {
            for (int a = 0; a < D; ++a) {
                for (int b = 0; b <= a; ++b) {
                    float v = cov_small[a * EXACT_EIG_D_THRESHOLD + b] / sumW;
                    cov_small[a * EXACT_EIG_D_THRESHOLD + b] = v;
                    cov_small[b * EXACT_EIG_D_THRESHOLD + a] = v;
                }
            }
            jacobi_top_eigvec_exact_small(cov_small, D, u);
        } else {
            float u2 = 0.0f;
            for (int d = 0; d < D; ++d) u2 += u[d] * u[d];
            if (u2 <= 1.0e-30f) {
                for (int d = 0; d < D; ++d) u[d] = 0.0f;
                u[0] = 1.0f;
            } else {
                float invn = safe_rsqrtf_eval(u2);
                for (int d = 0; d < D; ++d) u[d] *= invn;
            }

            for (int it = 0; it < power_iter_steps; ++it) {
                for (int d = 0; d < D; ++d) unew[d] = 0.0f;

                for (int j = 0; j < Ntrain; ++j) {
                    float xtxj = 0.0f;
                    float xtvj = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        float xv = xt[d];
                        xtxj += xv * Xtrain[cm_index(D, d, j)];
                        xtvj += xv * V[cm_index(D, d, j)];
                    }
                    float eucl2 = xt2 + Xtrainnorm2[j] - 2.0f * xtxj;
                    eucl2 = fmaxf(eucl2, 0.0f);
                    float proj = xtvj - VdotX[j];
                    float q = (eucl2 + rho * proj * proj) * inv_h2;

                    float w;
                    if (sparse) {
                        w = expf(fallback_exp_scale * (-q + min_q));
                    } else {
                        float rt2 = 1.0f - q * q;
                        w = (rt2 > 0.0f) ? (rt2 * rt2) : 0.0f;
                    }

                    float vu = 0.0f;
                    for (int d = 0; d < D; ++d) vu += V[cm_index(D, d, j)] * u[d];
                    float alpha = w * vu;
                    for (int d = 0; d < D; ++d) {
                        unew[d] += alpha * V[cm_index(D, d, j)];
                    }
                }

                float n2 = 0.0f;
                for (int d = 0; d < D; ++d) {
                    unew[d] /= sumW;
                    n2 += unew[d] * unew[d];
                }
                if (n2 <= 1.0e-30f) break;
                float invn = safe_rsqrtf_eval(n2);
                for (int d = 0; d < D; ++d) u[d] = unew[d] * invn;
            }
        }

        float devm = 0.0f, devp = 0.0f;
        for (int d = 0; d < D; ++d) {
            float dm = u[d] - vt_true[d];
            float dp = u[d] + vt_true[d];
            devm += dm * dm;
            devp += dp * dp;
        }
        devVec = sqrtf(fminf(devm, devp));
    }

    __shared__ float sMSE[BLOCK];
    __shared__ float sDevCen[BLOCK];
    __shared__ float sDevCenTan[BLOCK];
    __shared__ float sDevVec[BLOCK];

    sMSE[tid] = mse;
    sDevCen[tid] = devCen;
    sDevCenTan[tid] = devCenTan;
    sDevVec[tid] = devVec;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMSE[tid]       += sMSE[tid + stride];
            sDevCen[tid]    += sDevCen[tid + stride];
            sDevCenTan[tid] += sDevCenTan[tid + stride];
            sDevVec[tid]    += sDevVec[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        size_t idx = (size_t)ir * (size_t)gridX + (size_t)blockIdx.x;
        partialMSE[idx]       = sMSE[0];
        partialDevCen[idx]    = sDevCen[0];
        partialDevCenTan[idx] = sDevCenTan[0];
        partialDevVec[idx]    = sDevVec[0];
    }
}


template<int BLOCK>
__global__ void reduce_partials_kernel4(
    const float* __restrict__ partialMSE,
    const float* __restrict__ partialDevCen,
    const float* __restrict__ partialDevCenTan,
    const float* __restrict__ partialDevVec,
    int m, int gridX,
    float* __restrict__ sumMSE,
    float* __restrict__ sumDevCen,
    float* __restrict__ sumDevCenTan,
    float* __restrict__ sumDevVec)
{
    int ir = blockIdx.x;
    int tid = threadIdx.x;
    if (ir >= m) return;

    float accMSE = 0.0f;
    float accDevCen = 0.0f;
    float accDevCenTan = 0.0f;
    float accDevVec = 0.0f;

    size_t base = (size_t)ir * (size_t)gridX;
    for (int p = tid; p < gridX; p += BLOCK) {
        accMSE       += partialMSE[base + (size_t)p];
        accDevCen    += partialDevCen[base + (size_t)p];
        accDevCenTan += partialDevCenTan[base + (size_t)p];
        accDevVec    += partialDevVec[base + (size_t)p];
    }

    __shared__ float sMSE[BLOCK];
    __shared__ float sDevCen[BLOCK];
    __shared__ float sDevCenTan[BLOCK];
    __shared__ float sDevVec[BLOCK];

    sMSE[tid] = accMSE;
    sDevCen[tid] = accDevCen;
    sDevCenTan[tid] = accDevCenTan;
    sDevVec[tid] = accDevVec;
    __syncthreads();

    for (int stride = BLOCK >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMSE[tid]       += sMSE[tid + stride];
            sDevCen[tid]    += sDevCen[tid + stride];
            sDevCenTan[tid] += sDevCenTan[tid + stride];
            sDevVec[tid]    += sDevVec[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sumMSE[ir]       = sMSE[0];
        sumDevCen[ir]    = sDevCen[0];
        sumDevCenTan[ir] = sDevCenTan[0];
        sumDevVec[ir]    = sDevVec[0];
    }
}

struct ChunkProfileEvents {
    bool valid = false;
    bool has_h2d = false;
    cudaEvent_t e_start = nullptr;
    cudaEvent_t e_after_h2d = nullptr;
    cudaEvent_t e_after_kernel = nullptr;
    cudaEvent_t e_done = nullptr;
};

static void init_chunk_profile_events(ChunkProfileEvents& ev, bool has_h2d)
{
    ev.valid = true;
    ev.has_h2d = has_h2d;
    CUDA_CHECK(cudaEventCreate(&ev.e_start));
    if (has_h2d) CUDA_CHECK(cudaEventCreate(&ev.e_after_h2d));
    CUDA_CHECK(cudaEventCreate(&ev.e_after_kernel));
    CUDA_CHECK(cudaEventCreate(&ev.e_done));
}

static void destroy_chunk_profile_events(ChunkProfileEvents& ev)
{
    if (ev.e_done) CUDA_CHECK(cudaEventDestroy(ev.e_done));
    if (ev.e_after_kernel) CUDA_CHECK(cudaEventDestroy(ev.e_after_kernel));
    if (ev.e_after_h2d) CUDA_CHECK(cudaEventDestroy(ev.e_after_h2d));
    if (ev.e_start) CUDA_CHECK(cudaEventDestroy(ev.e_start));
    ev = ChunkProfileEvents();
}

static void accumulate_chunk_profile_events(ChunkProfileEvents& ev, double& acc_h2d, double& acc_kernel, double& acc_d2h)
{
    if (!ev.valid) return;
    float ms = 0.0f;
    if (ev.has_h2d) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.e_start, ev.e_after_h2d));
        acc_h2d += 1.0e-3 * (double)ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.e_after_h2d, ev.e_after_kernel));
        acc_kernel += 1.0e-3 * (double)ms;
    } else {
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.e_start, ev.e_after_kernel));
        acc_kernel += 1.0e-3 * (double)ms;
    }
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev.e_after_kernel, ev.e_done));
    acc_d2h += 1.0e-3 * (double)ms;
    destroy_chunk_profile_events(ev);
}

static bool use_exact_specialized_multi_hyper_tile_path()
{
    return eval_exact_specialization_enabled() &&
           (G.opts.enable_exact_d20_train_tiling != 0) &&
           (G.opts.enable_exact_d20_multi_hyper_tile != 0) &&
           (G.m > 1);
}

static int eval_gridx_for_chunk(int B)
{
    int samplesPerBlock = EVAL_BLOCK;
    if (use_exact_specialized_multi_hyper_tile_path()) {
        samplesPerBlock = D20_HYPER_SAMPLES_PER_BLOCK;
    }
    return (B + samplesPerBlock - 1) / samplesPerBlock;
}

static mxArray* create_last_profile_struct()
{
    const char* fields[] = {
        "init_sec","cache_test_sec","total_sec","host_pack_sec","stream_wait_sec",
        "h2d_sec","kernel_sec","d2h_sec","chunk_size","num_chunks",
        "used_cached_test","double_buffer_enabled"
    };
    mxArray* out = mxCreateStructMatrix(1, 1, 12, fields);
    mxSetField(out, 0, "init_sec", mxCreateDoubleScalar(G.last_init_sec));
    mxSetField(out, 0, "cache_test_sec", mxCreateDoubleScalar(G.last_cache_test_sec));
    mxSetField(out, 0, "total_sec", mxCreateDoubleScalar(G.last_total_sec));
    mxSetField(out, 0, "host_pack_sec", mxCreateDoubleScalar(G.last_host_pack_sec));
    mxSetField(out, 0, "stream_wait_sec", mxCreateDoubleScalar(G.last_stream_wait_sec));
    mxSetField(out, 0, "h2d_sec", mxCreateDoubleScalar(G.last_h2d_sec));
    mxSetField(out, 0, "kernel_sec", mxCreateDoubleScalar(G.last_kernel_sec));
    mxSetField(out, 0, "d2h_sec", mxCreateDoubleScalar(G.last_d2h_sec));
    mxSetField(out, 0, "chunk_size", mxCreateDoubleScalar((double)G.last_chunkB));
    mxSetField(out, 0, "num_chunks", mxCreateDoubleScalar((double)G.last_num_chunks));
    mxSetField(out, 0, "used_cached_test", mxCreateDoubleScalar((double)G.last_used_cached_test));
    mxSetField(out, 0, "double_buffer_enabled", mxCreateDoubleScalar((double)G.last_double_buffer));
    return out;
}

static void ensure_work_buffers(int chunkB)
{
    ensure_streams();

    int gridX = eval_gridx_for_chunk(chunkB);
    bool need_realloc = (chunkB > G.chunk_cap) || (gridX > G.partial_cap_gridx);

    if (!need_realloc) return;

    free_streamed_buffers();

    G.chunk_cap = chunkB;
    G.partial_cap_gridx = gridX;

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&G.d_Xq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap));
        CUDA_CHECK(cudaMalloc(&G.d_Yq[s], sizeof(float) * (size_t)G.chunk_cap));
        CUDA_CHECK(cudaMalloc(&G.d_Vq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap));
        CUDA_CHECK(cudaMalloc(&G.d_Cq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap));

        CUDA_CHECK(cudaHostAlloc(&G.h_Xq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_Yq[s], sizeof(float) * (size_t)G.chunk_cap, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_Vq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_Cq[s], sizeof(float) * (size_t)G.D * (size_t)G.chunk_cap, cudaHostAllocDefault));

        size_t partialCount = (size_t)G.m * (size_t)G.partial_cap_gridx;
        CUDA_CHECK(cudaMalloc(&G.d_partialMSE[s],       sizeof(float) * partialCount));
        CUDA_CHECK(cudaMalloc(&G.d_partialDevCen[s],    sizeof(float) * partialCount));
        CUDA_CHECK(cudaMalloc(&G.d_partialDevCenTan[s], sizeof(float) * partialCount));
        CUDA_CHECK(cudaMalloc(&G.d_partialDevVec[s],    sizeof(float) * partialCount));

        CUDA_CHECK(cudaMalloc(&G.d_sumMSE[s],       sizeof(float) * (size_t)G.m));
        CUDA_CHECK(cudaMalloc(&G.d_sumDevCen[s],    sizeof(float) * (size_t)G.m));
        CUDA_CHECK(cudaMalloc(&G.d_sumDevCenTan[s], sizeof(float) * (size_t)G.m));
        CUDA_CHECK(cudaMalloc(&G.d_sumDevVec[s],    sizeof(float) * (size_t)G.m));

        CUDA_CHECK(cudaHostAlloc(&G.h_sumMSE[s],       sizeof(float) * (size_t)G.m, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_sumDevCen[s],    sizeof(float) * (size_t)G.m, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_sumDevCenTan[s], sizeof(float) * (size_t)G.m, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&G.h_sumDevVec[s],    sizeof(float) * (size_t)G.m, cudaHostAllocDefault));
    }
}

static int choose_chunk_size(int Ntest, bool usingCachedTest)
{
    if (G.opts.manual_chunk > 0) {
        return std::max(1, std::min(Ntest, G.opts.manual_chunk));
    }

    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

    double safety = G.opts.auto_chunk_safety_bytes;
    if (safety <= 0.0) {
        safety = std::max(512.0 * 1024.0 * 1024.0, 0.08 * (double)totalBytes);
    }
    double usable = std::max(0.0, (double)freeBytes - safety);
    usable *= (double)G.opts.auto_chunk_mem_fraction;

    // Approximate device bytes per sample.
    double bytesPerSample = 0.0;

    // streamed query buffers on device: X,Y,V,C for 2 streams
    if (!usingCachedTest) {
        bytesPerSample += 2.0 * sizeof(float) * (double)(3 * G.D + 1);
    }

    // Partial-buffer footprint depends on how many test samples each block reduces.
    {
        double samplesPerBlockForPartials = (double)EVAL_BLOCK;
        if (use_exact_specialized_multi_hyper_tile_path()) {
            samplesPerBlockForPartials = (double)D20_HYPER_SAMPLES_PER_BLOCK;
        } else if (G.D <= EXACT_EIG_D_THRESHOLD) {
            // exact small-D tiled path still writes partial buffers one row per hyper and x-block.
            samplesPerBlockForPartials = (double)EVAL_BLOCK;
        }
        bytesPerSample += 2.0 * 4.0 * sizeof(float) * ((double)std::max(G.m,1) / samplesPerBlockForPartials);
    }

    // some extra room for outputs / kernel scratch
    bytesPerSample += 64.0;

    int byMem = (int)std::floor(usable / std::max(1.0, bytesPerSample));
    byMem = std::max(1, byMem);

    // Runtime cap from rough pair-count budget: B * Ntrain * m
    double workBudget = G.opts.auto_chunk_target_pairs;
    // Cached-test eval avoids query H2D traffic and usually benefits from larger chunks.
    // Keep the non-cached path unchanged; only relax the pair-budget cap for cached mode.
    if (usingCachedTest && G.opts.manual_chunk <= 0) {
        workBudget *= 4.0;
    }
    int byWork = (int)std::floor(workBudget /
                                 std::max(1.0, (double)G.Ntrain * (double)std::max(G.m,1)));
    byWork = std::max(1, byWork);

    int B = std::min(byMem, byWork);
    B = std::max(G.opts.auto_chunk_min, B);
    B = std::min(G.opts.auto_chunk_max, B);
    B = std::min(Ntest, B);
    B = std::max(1, B);

    return B;
}

static void finalize_stream_accumulate(
    int s,
    std::vector<double>& accMSE,
    std::vector<double>& accDevCen,
    std::vector<double>& accDevCenTan,
    std::vector<double>& accDevVec,
    double* stream_wait_sec = nullptr)
{
    auto t_wait = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(G.streams[s]));
    if (stream_wait_sec) {
        *stream_wait_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wait).count();
    }
    for (int i = 0; i < G.m; ++i) {
        accMSE[i]       += (double)G.h_sumMSE[s][i];
        accDevCen[i]    += (double)G.h_sumDevCen[s][i];
        accDevCenTan[i] += (double)G.h_sumDevCenTan[s][i];
        accDevVec[i]    += (double)G.h_sumDevVec[s][i];
    }
}

template<int DEXACT>
static void launch_exact_specialized_partials(
    int s,
    const float* dX,
    const float* dY,
    const float* dV,
    const float* dC,
    int B,
    int gridX)
{
    dim3 gridPart(gridX, G.m, 1);
    if (G.opts.enable_exact_d20_train_tiling && G.opts.enable_exact_d20_multi_hyper_tile && G.m > 1) {
        dim3 gridMH(gridX, (G.m + D20_HYPER_TILE - 1) / D20_HYPER_TILE, 1);
        eval_partials_kernel_cm_exact_tiled_multi_hyper<DEXACT, D20_TRAIN_TILE, D20_HYPER_TILE, D20_HYPER_SAMPLES_PER_BLOCK><<<gridMH, D20_HYPER_BLOCK_THREADS, 0, G.streams[s]>>>(
            dX, dY, dV, dC,
            G.d_Ytrain, G.d_Xtrain, G.d_Cen, G.d_V, G.d_VdotX, G.d_Xtrainnorm2, G.d_RhoScale,
            G.h, B, G.Ntrain, G.m,
            G.opts.sum_sparse_threshold,
            G.opts.count_sparse_threshold,
            G.opts.fallback_exp_scale,
            G.opts.eps_weight,
            gridX,
            G.d_partialMSE[s],
            G.d_partialDevCen[s],
            G.d_partialDevCenTan[s],
            G.d_partialDevVec[s]);
    } else if (G.opts.enable_exact_d20_train_tiling) {
        eval_partials_kernel_cm_exact_tiled<DEXACT, EVAL_BLOCK, D20_TRAIN_TILE><<<gridPart, EVAL_BLOCK, 0, G.streams[s]>>>(
            dX, dY, dV, dC,
            G.d_Ytrain, G.d_Xtrain, G.d_Cen, G.d_V, G.d_VdotX, G.d_Xtrainnorm2, G.d_RhoScale,
            G.h, B, G.Ntrain, G.m,
            G.opts.sum_sparse_threshold,
            G.opts.count_sparse_threshold,
            G.opts.fallback_exp_scale,
            G.opts.eps_weight,
            gridX,
            G.d_partialMSE[s],
            G.d_partialDevCen[s],
            G.d_partialDevCenTan[s],
            G.d_partialDevVec[s]);
    } else {
        eval_partials_kernel_cm_exact<DEXACT, EVAL_BLOCK><<<gridPart, EVAL_BLOCK, 0, G.streams[s]>>>(
            dX, dY, dV, dC,
            G.d_Ytrain, G.d_Xtrain, G.d_Cen, G.d_V, G.d_VdotX, G.d_Xtrainnorm2, G.d_RhoScale,
            G.h, B, G.Ntrain, G.m,
            G.opts.sum_sparse_threshold,
            G.opts.count_sparse_threshold,
            G.opts.fallback_exp_scale,
            G.opts.eps_weight,
            gridX,
            G.d_partialMSE[s],
            G.d_partialDevCen[s],
            G.d_partialDevCenTan[s],
            G.d_partialDevVec[s]);
    }
}

static bool try_launch_exact_specialized_partials(
    int s,
    const float* dX,
    const float* dY,
    const float* dV,
    const float* dC,
    int B,
    int gridX)
{
    switch (G.D) {
#define MANOR_LAUNCH_EXACT_DIM(d) \
        case d: launch_exact_specialized_partials<d>(s, dX, dY, dV, dC, B, gridX); return true;
        MANOR_EVAL_EXACT_SPECIALIZED_D_LIST(MANOR_LAUNCH_EXACT_DIM)
#undef MANOR_LAUNCH_EXACT_DIM
        default: return false;
    }
}

static void launch_chunk_from_device(
    int s,
    const float* dX,
    const float* dY,
    const float* dV,
    const float* dC,
    int B,
    ChunkProfileEvents* profileEvents = nullptr)
{
    CUDA_CHECK(cudaMemsetAsync(G.d_sumMSE[s],       0, sizeof(float) * (size_t)G.m, G.streams[s]));
    CUDA_CHECK(cudaMemsetAsync(G.d_sumDevCen[s],    0, sizeof(float) * (size_t)G.m, G.streams[s]));
    CUDA_CHECK(cudaMemsetAsync(G.d_sumDevCenTan[s], 0, sizeof(float) * (size_t)G.m, G.streams[s]));
    CUDA_CHECK(cudaMemsetAsync(G.d_sumDevVec[s],    0, sizeof(float) * (size_t)G.m, G.streams[s]));

    int gridX = eval_gridx_for_chunk(B);

    dim3 gridPart(gridX, G.m, 1);
    bool launched_exact = false;
    if (eval_exact_specialization_enabled()) {
        launched_exact = try_launch_exact_specialized_partials(s, dX, dY, dV, dC, B, gridX);
    }
    if (!launched_exact) {
        eval_partials_kernel_cm<EVAL_BLOCK><<<gridPart, EVAL_BLOCK, 0, G.streams[s]>>>(
            dX, dY, dV, dC,
            G.d_Ytrain, G.d_Xtrain, G.d_Cen, G.d_V, G.d_VdotX, G.d_Xtrainnorm2, G.d_RhoScale,
            G.h, G.D, B, G.Ntrain, G.m,
            G.opts.power_iter_steps,
            G.opts.sum_sparse_threshold,
            G.opts.count_sparse_threshold,
            G.opts.fallback_exp_scale,
            G.opts.eps_weight,
            G.opts.exact_tangent_small_d,
            gridX,
            G.d_partialMSE[s],
            G.d_partialDevCen[s],
            G.d_partialDevCenTan[s],
            G.d_partialDevVec[s]);
    }
    CUDA_CHECK(cudaPeekAtLastError());

    reduce_partials_kernel4<EVAL_BLOCK><<<G.m, EVAL_BLOCK, 0, G.streams[s]>>>(
        G.d_partialMSE[s],
        G.d_partialDevCen[s],
        G.d_partialDevCenTan[s],
        G.d_partialDevVec[s],
        G.m, gridX,
        G.d_sumMSE[s],
        G.d_sumDevCen[s],
        G.d_sumDevCenTan[s],
        G.d_sumDevVec[s]);
    CUDA_CHECK(cudaPeekAtLastError());

    if (profileEvents && profileEvents->valid) {
        CUDA_CHECK(cudaEventRecord(profileEvents->e_after_kernel, G.streams[s]));
    }

    CUDA_CHECK(cudaMemcpyAsync(G.h_sumMSE[s],       G.d_sumMSE[s],       sizeof(float) * (size_t)G.m, cudaMemcpyDeviceToHost, G.streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(G.h_sumDevCen[s],    G.d_sumDevCen[s],    sizeof(float) * (size_t)G.m, cudaMemcpyDeviceToHost, G.streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(G.h_sumDevCenTan[s], G.d_sumDevCenTan[s], sizeof(float) * (size_t)G.m, cudaMemcpyDeviceToHost, G.streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(G.h_sumDevVec[s],    G.d_sumDevVec[s],    sizeof(float) * (size_t)G.m, cudaMemcpyDeviceToHost, G.streams[s]));

    if (profileEvents && profileEvents->valid) {
        CUDA_CHECK(cudaEventRecord(profileEvents->e_done, G.streams[s]));
    }
}

static void create_outputs_from_sums(
    mxArray *plhs[],
    const std::vector<double>& accMSE,
    const std::vector<double>& accDevCen,
    const std::vector<double>& accDevCenTan,
    const std::vector<double>& accDevVec,
    double totalCount)
{
    plhs[0] = mxCreateNumericMatrix(G.m, 1, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(G.m, 1, mxSINGLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericMatrix(G.m, 1, mxSINGLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericMatrix(G.m, 1, mxSINGLE_CLASS, mxREAL);
    plhs[4] = mxCreateDoubleScalar(totalCount);

    float* p0 = (float*)mxGetData(plhs[0]);
    float* p1 = (float*)mxGetData(plhs[1]);
    float* p2 = (float*)mxGetData(plhs[2]);
    float* p3 = (float*)mxGetData(plhs[3]);

    for (int i = 0; i < G.m; ++i) {
        p0[i] = (float)accMSE[i];
        p1[i] = (float)accDevCen[i];
        p2[i] = (float)accDevCenTan[i];
        p3[i] = (float)accDevVec[i];
    }
}

static void do_init(const mxArray* estimator, const mxArray* hyperArr, const mxArray* optsArr)
{
    auto t_init_start = std::chrono::steady_clock::now();
    // keep test cache if present; reset estimator-side state only
    if (G.estimator_initialized) {
        free_estimator_buffers();
    }

    const mxArray* mxYpp         = get_required_field(estimator, "Y_pp");
    const mxArray* mxXTrain      = get_required_field(estimator, "X_Train");
    const mxArray* mxCen         = get_required_field(estimator, "Cen_pp");
    const mxArray* mxVec         = get_required_field(estimator, "Vec_pp");
    const mxArray* mxRhok        = get_required_field(estimator, "rhok");
    const mxArray* mxAxialNormal = get_required_field(estimator, "axial_normal");

    if (!is_real_numeric_single_or_double(mxYpp) ||
        !is_real_numeric_single_or_double(mxXTrain) ||
        !is_real_numeric_single_or_double(mxCen) ||
        !is_real_numeric_single_or_double(mxVec) ||
        !is_real_numeric_single_or_double(mxRhok) ||
        !is_real_numeric_single_or_double(mxAxialNormal)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "Estimator fields must be real numeric.");
    }

    int D = (int)mxGetM(mxXTrain);
    int Ntrain = (int)mxGetN(mxXTrain);

    if (D <= 0 || D > MAX_D_MANOR_EVAL) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init",
                          "Ambient dimension D=%d exceeds MAX_D_MANOR_EVAL=%d.",
                          D, MAX_D_MANOR_EVAL);
    }
    if ((int)mxGetM(mxCen) != D || (int)mxGetN(mxCen) != Ntrain) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "Cen_pp size mismatch.");
    }
    if ((int)mxGetM(mxVec) != D || (int)mxGetN(mxVec) != Ntrain) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "Vec_pp size mismatch.");
    }
    if ((int)mxGetNumberOfElements(mxYpp) != Ntrain) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "Y_pp length mismatch.");
    }

    std::vector<float> hyper;
    if (hyperArr && mxGetNumberOfElements(hyperArr) > 0) {
        copy_linear_numeric_to_float_vector(hyperArr, hyper);
    } else {
        const mxArray* mxHyper = get_optional_field(estimator, "hyper_param2");
        if (!mxHyper || mxGetNumberOfElements(mxHyper) == 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:init",
                              "hyper_param2 must be provided either as input or Estimator.hyper_param2.");
        }
        copy_linear_numeric_to_float_vector(mxHyper, hyper);
    }

    int m = (int)hyper.size();
    if (m <= 0) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "hyper_param2 must be non-empty.");
    }

    float rhok = read_scalar_float(mxRhok);
    float h = read_scalar_float(mxAxialNormal);
    if (!(rhok > 0.0f) || !(h > 0.0f)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:init", "rhok and axial_normal must be positive.");
    }

    std::vector<float> rhoScale(m, 0.0f);
    for (int i = 0; i < m; ++i) {
        float rho = rhok * hyper[i];
        if (!(rho > 0.0f)) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:init", "All rhok*hyper_param2 must be positive.");
        }
        rhoScale[i] = 1.0f / (rho * rho) - 1.0f;
    }

    G.opts = parse_opts(optsArr, D);
    G.D = D;
    G.Ntrain = Ntrain;
    G.m = m;
    G.h = h;

    if (G.test_cached && G.Dtest_cached != D) {
        free_test_cache();
    }

    upload_linear_numeric_to_device_float(mxYpp, G.d_Ytrain);
    upload_matrix_cm_numeric_to_device_float(mxXTrain, D, Ntrain, G.d_Xtrain);
    upload_matrix_cm_numeric_to_device_float(mxCen,    D, Ntrain, G.d_Cen);
    upload_matrix_cm_numeric_to_device_float(mxVec,    D, Ntrain, G.d_V);

    CUDA_CHECK(cudaMalloc(&G.d_VdotX,        sizeof(float) * (size_t)Ntrain));
    CUDA_CHECK(cudaMalloc(&G.d_Xtrainnorm2,  sizeof(float) * (size_t)Ntrain));
    CUDA_CHECK(cudaMalloc(&G.d_RhoScale,     sizeof(float) * (size_t)m));
    CUDA_CHECK(cudaMemcpy(G.d_RhoScale, rhoScale.data(), sizeof(float) * (size_t)m, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (Ntrain + threads - 1) / threads;
    train_stats_kernel_cm<<<blocks, threads>>>(
        G.d_Xtrain, G.d_V, D, Ntrain, G.d_Xtrainnorm2, G.d_VdotX);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    ensure_streams();

    if (!g_at_exit_registered) {
        mexAtExit(atExitHandler);
        g_at_exit_registered = true;
    }

    G.estimator_initialized = true;
    touch_lock();
    G.last_init_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init_start).count();
}

static void do_clear_estimator()
{
    if (G.estimator_initialized) {
        free_estimator_buffers();
    }
    maybe_unlock();
}

static void do_cache_test(const mxArray* mxX, const mxArray* mxY, const mxArray* mxV, const mxArray* mxC)
{
    auto t_cache_start = std::chrono::steady_clock::now();
    if (!is_real_numeric_single_or_double(mxX) ||
        !is_real_numeric_single_or_double(mxY) ||
        !is_real_numeric_single_or_double(mxV) ||
        !is_real_numeric_single_or_double(mxC)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test", "Inputs must be real numeric single/double.");
    }

    int D = (int)mxGetM(mxX);
    int N = (int)mxGetN(mxX);
    if (D <= 0 || D > MAX_D_MANOR_EVAL) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test",
                          "Ambient dimension D=%d exceeds MAX_D_MANOR_EVAL=%d.",
                          D, MAX_D_MANOR_EVAL);
    }
    if ((int)mxGetNumberOfElements(mxY) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test", "Y length mismatch.");
    }
    if ((int)mxGetM(mxV) != D || (int)mxGetN(mxV) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test", "V size mismatch.");
    }
    if ((int)mxGetM(mxC) != D || (int)mxGetN(mxC) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test", "C size mismatch.");
    }

    if (G.estimator_initialized && G.D != D) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test",
                          "Cached test dimension D=%d does not match current estimator D=%d.",
                          D, G.D);
    }

    free_test_cache();

    upload_matrix_cm_numeric_to_device_float(mxX, D, N, G.d_Xtest_all);
    upload_linear_numeric_to_device_float(mxY, G.d_Ytrue_all);
    upload_matrix_cm_numeric_to_device_float(mxV, D, N, G.d_Vtrue_all);
    upload_matrix_cm_numeric_to_device_float(mxC, D, N, G.d_Ctrue_all);

    G.test_cached = true;
    G.Dtest_cached = D;
    G.Ntest_cached = N;

    touch_lock();
    G.last_cache_test_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cache_start).count();
}

static void do_clear_test()
{
    if (G.test_cached) {
        free_test_cache();
    }
    maybe_unlock();
}

static void do_eval_full(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!G.estimator_initialized) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "Estimator is not initialized.");
    }
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full",
                          "Usage: manor_1_eval_mex('eval_full', X, Y, V, C)");
    }
    if (nlhs != 5) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full",
                          "Expected 5 outputs: sMSE, sDevCen, sDevCenTan, sDevVec, totalCount");
    }

    const mxArray* mxX = prhs[1];
    const mxArray* mxY = prhs[2];
    const mxArray* mxV = prhs[3];
    const mxArray* mxC = prhs[4];

    if (!is_real_numeric_single_or_double(mxX) ||
        !is_real_numeric_single_or_double(mxY) ||
        !is_real_numeric_single_or_double(mxV) ||
        !is_real_numeric_single_or_double(mxC)) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "Inputs must be real numeric single/double.");
    }

    int D = (int)mxGetM(mxX);
    int N = (int)mxGetN(mxX);
    if (D != G.D) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "X dimension mismatch with estimator.");
    }
    if ((int)mxGetNumberOfElements(mxY) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "Y length mismatch.");
    }
    if ((int)mxGetM(mxV) != D || (int)mxGetN(mxV) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "V size mismatch.");
    }
    if ((int)mxGetM(mxC) != D || (int)mxGetN(mxC) != N) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_full", "C size mismatch.");
    }

    auto t_eval_start = std::chrono::steady_clock::now();
    int chunkB = choose_chunk_size(N, false);
    ensure_work_buffers(chunkB);

    std::vector<double> accMSE(G.m, 0.0), accDevCen(G.m, 0.0), accDevCenTan(G.m, 0.0), accDevVec(G.m, 0.0);
    double totalCount = 0.0;
    double host_pack_sec = 0.0, stream_wait_sec = 0.0, h2d_sec = 0.0, kernel_sec = 0.0, d2h_sec = 0.0;

    bool pending[2] = {false, false};
    ChunkProfileEvents pendingProf[2];

    int nChunks = (N + chunkB - 1) / chunkB;
    for (int c = 0; c < nChunks; ++c) {
        int s = (G.opts.enable_host_double_buffer != 0) ? (c & 1) : 0;

        if (pending[s]) {
            finalize_stream_accumulate(s, accMSE, accDevCen, accDevCenTan, accDevVec, &stream_wait_sec);
            if (G.opts.profile_timers) accumulate_chunk_profile_events(pendingProf[s], h2d_sec, kernel_sec, d2h_sec);
            pending[s] = false;
        }

        int q0 = c * chunkB;
        int B = std::min(chunkB, N - q0);

        auto t_pack = std::chrono::steady_clock::now();
        copy_matrix_chunk_cm_to_pinned(mxX, D, q0, B, G.h_Xq[s]);
        copy_vector_chunk_to_pinned(mxY, q0, B, G.h_Yq[s]);
        copy_matrix_chunk_cm_to_pinned(mxV, D, q0, B, G.h_Vq[s]);
        copy_matrix_chunk_cm_to_pinned(mxC, D, q0, B, G.h_Cq[s]);
        host_pack_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pack).count();

        if (G.opts.profile_timers) {
            init_chunk_profile_events(pendingProf[s], true);
            CUDA_CHECK(cudaEventRecord(pendingProf[s].e_start, G.streams[s]));
        }

        CUDA_CHECK(cudaMemcpyAsync(G.d_Xq[s], G.h_Xq[s],
                                   sizeof(float) * (size_t)D * (size_t)B,
                                   cudaMemcpyHostToDevice, G.streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(G.d_Yq[s], G.h_Yq[s],
                                   sizeof(float) * (size_t)B,
                                   cudaMemcpyHostToDevice, G.streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(G.d_Vq[s], G.h_Vq[s],
                                   sizeof(float) * (size_t)D * (size_t)B,
                                   cudaMemcpyHostToDevice, G.streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(G.d_Cq[s], G.h_Cq[s],
                                   sizeof(float) * (size_t)D * (size_t)B,
                                   cudaMemcpyHostToDevice, G.streams[s]));
        if (G.opts.profile_timers) {
            CUDA_CHECK(cudaEventRecord(pendingProf[s].e_after_h2d, G.streams[s]));
        }

        launch_chunk_from_device(s, G.d_Xq[s], G.d_Yq[s], G.d_Vq[s], G.d_Cq[s], B, G.opts.profile_timers ? &pendingProf[s] : nullptr);
        pending[s] = true;
        totalCount += (double)B;

        if (G.opts.enable_host_double_buffer == 0) {
            finalize_stream_accumulate(s, accMSE, accDevCen, accDevCenTan, accDevVec, &stream_wait_sec);
            if (G.opts.profile_timers) accumulate_chunk_profile_events(pendingProf[s], h2d_sec, kernel_sec, d2h_sec);
            pending[s] = false;
        }
    }

    for (int s = 0; s < 2; ++s) {
        if (pending[s]) {
            finalize_stream_accumulate(s, accMSE, accDevCen, accDevCenTan, accDevVec, &stream_wait_sec);
            if (G.opts.profile_timers) accumulate_chunk_profile_events(pendingProf[s], h2d_sec, kernel_sec, d2h_sec);
            pending[s] = false;
        }
    }

    G.last_total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_eval_start).count();
    G.last_host_pack_sec = host_pack_sec;
    G.last_stream_wait_sec = stream_wait_sec;
    G.last_h2d_sec = h2d_sec;
    G.last_kernel_sec = kernel_sec;
    G.last_d2h_sec = d2h_sec;
    G.last_chunkB = chunkB;
    G.last_num_chunks = nChunks;
    G.last_used_cached_test = 0;
    G.last_double_buffer = (G.opts.enable_host_double_buffer != 0);

    create_outputs_from_sums(plhs, accMSE, accDevCen, accDevCenTan, accDevVec, totalCount);
}

static void do_eval_cached(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (!G.estimator_initialized) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_cached", "Estimator is not initialized.");
    }
    if (!G.test_cached) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_cached", "No cached test data. Call cache_test first.");
    }
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_cached",
                          "Usage: manor_1_eval_mex('eval_cached')");
    }
    if (nlhs != 5) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_cached",
                          "Expected 5 outputs: sMSE, sDevCen, sDevCenTan, sDevVec, totalCount");
    }
    if (G.Dtest_cached != G.D) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:eval_cached",
                          "Cached test dimension does not match estimator dimension.");
    }

    auto t_eval_start = std::chrono::steady_clock::now();
    int N = G.Ntest_cached;
    int chunkB = choose_chunk_size(N, true);
    ensure_work_buffers(chunkB);

    std::vector<double> accMSE(G.m, 0.0), accDevCen(G.m, 0.0), accDevCenTan(G.m, 0.0), accDevVec(G.m, 0.0);
    double totalCount = 0.0;
    double stream_wait_sec = 0.0, kernel_sec = 0.0, d2h_sec = 0.0;

    int s = 0;
    for (int q0 = 0; q0 < N; q0 += chunkB) {
        int B = std::min(chunkB, N - q0);

        const float* dX = G.d_Xtest_all + (size_t)q0 * (size_t)G.D;
        const float* dY = G.d_Ytrue_all + (size_t)q0;
        const float* dV = G.d_Vtrue_all + (size_t)q0 * (size_t)G.D;
        const float* dC = G.d_Ctrue_all + (size_t)q0 * (size_t)G.D;

        ChunkProfileEvents prof;
        if (G.opts.profile_timers) {
            init_chunk_profile_events(prof, false);
            CUDA_CHECK(cudaEventRecord(prof.e_start, G.streams[s]));
        }
        launch_chunk_from_device(s, dX, dY, dV, dC, B, G.opts.profile_timers ? &prof : nullptr);
        finalize_stream_accumulate(s, accMSE, accDevCen, accDevCenTan, accDevVec, &stream_wait_sec);
        if (G.opts.profile_timers) { double h2d_dummy = 0.0; accumulate_chunk_profile_events(prof, h2d_dummy, kernel_sec, d2h_sec); }
        totalCount += (double)B;
    }

    G.last_total_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_eval_start).count();
    G.last_host_pack_sec = 0.0;
    G.last_stream_wait_sec = stream_wait_sec;
    G.last_h2d_sec = 0.0;
    G.last_kernel_sec = kernel_sec;
    G.last_d2h_sec = d2h_sec;
    G.last_chunkB = chunkB;
    G.last_num_chunks = (N + chunkB - 1) / chunkB;
    G.last_used_cached_test = 1;
    G.last_double_buffer = 0;

    create_outputs_from_sums(plhs, accMSE, accDevCen, accDevCenTan, accDevVec, totalCount);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1) {
        mexErrMsgIdAndTxt("manor_1_eval_mex:usage",
                          "Usage: manor_1_eval_mex('init',...), manor_1_eval_mex('eval_full',...), manor_1_eval_mex('cache_test',...), manor_1_eval_mex('eval_cached'), manor_1_eval_mex('get_last_profile'), manor_1_eval_mex('clear_estimator'), manor_1_eval_mex('clear_test'), manor_1_eval_mex('free').");
    }

    std::string cmd = get_command_string(prhs[0]);

    if (cmd == "init") {
        if (nrhs != 4 || nlhs != 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:init",
                              "Usage: manor_1_eval_mex('init', Estimator, hyper_param2, opts)");
        }
        do_init(prhs[1], prhs[2], prhs[3]);
        return;
    }

    if (cmd == "clear_estimator") {
        if (nrhs != 1 || nlhs != 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:clear_estimator",
                              "Usage: manor_1_eval_mex('clear_estimator')");
        }
        do_clear_estimator();
        return;
    }

    if (cmd == "cache_test") {
        if (nrhs != 5 || nlhs != 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:cache_test",
                              "Usage: manor_1_eval_mex('cache_test', X, Y, V, C)");
        }
        do_cache_test(prhs[1], prhs[2], prhs[3], prhs[4]);
        return;
    }

    if (cmd == "clear_test") {
        if (nrhs != 1 || nlhs != 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:clear_test",
                              "Usage: manor_1_eval_mex('clear_test')");
        }
        do_clear_test();
        return;
    }

    if (cmd == "get_last_profile") {
        if (nrhs != 1 || nlhs != 1) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:get_last_profile",
                              "Usage: manor_1_eval_mex('get_last_profile')");
        }
        plhs[0] = create_last_profile_struct();
        return;
    }

    if (cmd == "eval_full") {
        do_eval_full(nlhs, plhs, nrhs, prhs);
        return;
    }

    if (cmd == "eval_cached") {
        do_eval_cached(nlhs, plhs, nrhs, prhs);
        return;
    }

    if (cmd == "free") {
        if (nrhs != 1 || nlhs != 0) {
            mexErrMsgIdAndTxt("manor_1_eval_mex:free",
                              "Usage: manor_1_eval_mex('free')");
        }
        free_all_buffers();
        return;
    }

    mexErrMsgIdAndTxt("manor_1_eval_mex:cmd", "Unknown command '%s'.", cmd.c_str());
}
