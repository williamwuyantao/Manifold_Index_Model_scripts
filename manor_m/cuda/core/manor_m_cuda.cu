#include "manor_m_cuda.h"
#include "manor_m_cuda_internal.h"
#include "manor_m_cuda_resident.h"
#include "manor_m_index_validation.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct ManorMCudaContext {
    int32_t device_ordinal = 0;
    int32_t stream_count = 0;
    bool memory_pools_supported = false;
    bool memory_pool_threshold_configured = false;
    uint64_t original_memory_pool_release_threshold_bytes = 0;
    cudaStream_t streams[8]{};
    ManorMResidentCache resident;
};

thread_local ManorMResidentCache* manor_m_cuda_active_resident = nullptr;

ManorMResidentCache* manor_m_cuda_internal_resident(ManorMCudaContext* context)
{
    return context ? &context->resident : nullptr;
}

namespace {

thread_local std::string g_last_error;

int fail(ManorMCudaStatus status, const char* message)
{
    g_last_error = message == nullptr ? "unknown error" : message;
    return static_cast<int>(status);
}

int cuda_fail(cudaError_t error, const char* operation)
{
    char message[768];
    std::snprintf(
        message,
        sizeof(message),
        "%s failed: %s (%d)",
        operation,
        cudaGetErrorString(error),
        static_cast<int>(error));
    return fail(MANOR_M_CUDA_CUDA_ERROR, message);
}

int ok()
{
    g_last_error.clear();
    return static_cast<int>(MANOR_M_CUDA_SUCCESS);
}

bool checked_add_u64(uint64_t left, uint64_t right, uint64_t* output)
{
    if (left > std::numeric_limits<uint64_t>::max() - right) return false;
    *output = left + right;
    return true;
}

bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t* output)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
        return false;
    *output = left * right;
    return true;
}

int64_t ceil_div_i64(int64_t numerator, int64_t denominator)
{
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

int64_t cap_and_align_rows(
    int64_t rows,
    int64_t row_count,
    int64_t minimum_rows,
    int64_t preferred_rows)
{
    if (row_count == 0) return 0;
    rows = std::max(rows, minimum_rows);
    rows = std::min(rows, row_count);
    if (preferred_rows > 0) rows = std::min(rows, preferred_rows);

    constexpr int64_t alignment = 256;
    if (rows > minimum_rows && rows > alignment) {
        const int64_t aligned = (rows / alignment) * alignment;
        if (aligned >= minimum_rows) rows = aligned;
    }
    return std::max<int64_t>(1, rows);
}

__global__ void smoke_axpy_kernel(
    float alpha, const float* x, const float* y, float* output, int64_t count)
{
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index < count) {
        output[index] = alpha * x[index] + y[index];
    }
}

__global__ void response_point_major_to_soa_kernel(
    const float* point_major,
    float* structure_of_arrays,
    int point_count,
    int response_dimension)
{
    const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const int64_t element_count =
        static_cast<int64_t>(point_count) * response_dimension;
    if (element >= element_count) return;
    const int point = static_cast<int>(element / response_dimension);
    const int coordinate = static_cast<int>(
        element - static_cast<int64_t>(point) * response_dimension);
    structure_of_arrays[
        static_cast<int64_t>(coordinate) * point_count + point]
        = point_major[element];
}

__global__ void response_knn_gather_rows_kernel(
    const int32_t* source, const int32_t* local_rows,
    int32_t row_count, int32_t neighbor_count, int32_t* output)
{
    const size_t total = static_cast<size_t>(row_count) * neighbor_count;
    for (size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         element < total;
         element += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int32_t row = static_cast<int32_t>(element / neighbor_count);
        const int32_t column = static_cast<int32_t>(element % neighbor_count);
        output[element] = source[
            static_cast<size_t>(local_rows[row]) * neighbor_count + column];
    }
}

}  // namespace



namespace manor_knn {

/* ------------------------------------------------------------------ limits */

/* Y is the response; Phase 2 already caps its width at MAX_MY_MANOR = 8. */
static const int kMaxDy = 8;

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

#include "manor_m_knn_ties.cuh"

__global__ void knn_emit_kernel(
    const unsigned long long* __restrict__ sorted_key,
    const int* __restrict__ sorted_value,
    const int* __restrict__ compact_length,
    int slot_count, int compact_cap, int query_first, int K,
    int* __restrict__ index_out,
    unsigned long long* __restrict__ diagnostics, int output_anchor_base = 0)
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
    index_out[static_cast<size_t>(query_first + slot - output_anchor_base) * K + target] = value;
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
    T* release() { T* result = pointer_; pointer_ = nullptr; return result; }
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
 * query_first   first global anchor in the requested contiguous range.
 * query_count   number of anchors in that range.
 */
inline void build_index(const float* Y, int N, int DY, int query_first,
                        int query_count, int K,
                        int* index_out, const Options& options, Stats* stats,
                        bool host_output = false,
                        ManorMCudaKnnTileConsumer consumer = nullptr,
                        void* consumer_user = nullptr,
                        const float* prepared_subsample = nullptr,
                        const float* prepared_aos = nullptr)
{
    if (N < 1) throw Error("response kNN: N must be positive.");
    if (DY < 1 || DY > kMaxDy)
        throw Error("response kNN: the response width must lie in [1," +
                    std::to_string(kMaxDy) + "].");
    if (query_first < 0 || query_first >= N || query_count < 1 ||
        query_count > N - query_first)
        throw Error("response kNN: the contiguous anchor range must lie in [0,N).");
    if (K < 1 || K > N)
        throw Error("response kNN: kSlice must lie in [1,N].");
    detail::ThresholdLauncher threshold = detail::threshold_launcher(DY);
    detail::ScanLauncher scan = detail::scan_launcher(DY);
    if (!threshold || !scan)
        throw Error("response kNN: unsupported response width.");
    if ((prepared_subsample == nullptr) != (prepared_aos == nullptr))
        throw Error("response kNN: prepared layouts must be supplied together.");

    Stats local;
    const bool timing = options.collect_stats;
    detail::Timer timer(timing);

    const int stride = std::min(kSubStride, std::max(1, N / 1024));
    const int Ns = (N - 1) / stride + 1;
    const int cap = static_cast<int>(std::min<int64_t>(N,
        std::max<int64_t>(static_cast<int64_t>(K) + kRefineBand,
                          static_cast<int64_t>(K) * kCapFactor)));
    const int want = static_cast<int>(std::min<int64_t>(N,
        static_cast<int64_t>(K) + kRefineBand));
    const int compact_cap = static_cast<int>(std::min<int64_t>(cap,
        static_cast<int64_t>(want) + kCompactSlack));
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
    const bool tiled_output = host_output || consumer != nullptr;
    const size_t bytes_per_slot =
        static_cast<size_t>(cap) * (sizeof(unsigned int) + sizeof(int)) +
        static_cast<size_t>(compact_cap) *
            (2 * sizeof(unsigned long long) + 2 * sizeof(int)) +
        sort_temp_per_slot +
        8 * sizeof(int) + (tiled_output ? static_cast<size_t>(K) * sizeof(int) : 0);
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
    // All CUB segment/item counts remain int32, including very large N*k.
    chunk = std::min(chunk, std::numeric_limits<int>::max() / compact_cap);

    detail::Buffer<int> output_tile_buffer;
    int* output_tile = tiled_output
        ? output_tile_buffer.allocate(static_cast<size_t>(chunk) * K) : index_out;

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

    float* Ysub = prepared_subsample
        ? const_cast<float*>(prepared_subsample)
        : subsample_buffer.allocate(static_cast<size_t>(Ns) * DY);
    float* Yaos = prepared_aos
        ? const_cast<float*>(prepared_aos)
        : aos_buffer.allocate(static_cast<size_t>(N) * kMaxDy);
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

    if (!prepared_subsample) {
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
    threshold(Ysub, Ns, Y, N, query_first, sample_stride,
              std::min(sample_count, chunk),
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

    for (int local_q0 = 0; local_q0 < query_count; local_q0 += chunk) {
        const int q0 = query_first + local_q0;
        const int slots = std::min(chunk, query_count - local_q0);

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
                sorted_key, sorted_value, compact_length, slots, compact_cap,
                q0, K, output_tile, diagnostics,
                tiled_output ? q0 : query_first);
            MANOR_KNN_CHECK(cudaPeekAtLastError());
            if (host_output) {
                MANOR_KNN_CHECK(cudaMemcpy(
                    index_out + static_cast<size_t>(local_q0) * K, output_tile,
                    static_cast<size_t>(slots) * K * sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
        timer.end(&local.emit_ms);
        if (consumer) {
            const int status = consumer(
                consumer_user, output_tile, q0, slots, K);
            if (status != MANOR_M_CUDA_SUCCESS) {
                throw Error(std::string("response kNN tile consumer failed: ")
                    + manor_m_cuda_last_error());
            }
        }
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

struct ManorMCudaPreparedResponseKnn {
    ManorMCudaContext* context = nullptr;
    const float* response_point_major = nullptr;
    const float* device_point_major = nullptr;
    int point_count = 0;
    int response_dimension = 0;
    int subsample_count = 0;
    manor_knn::detail::Buffer<float> response_soa;
    manor_knn::detail::Buffer<float> response_subsample;
    manor_knn::detail::Buffer<float> response_aos;
};

int manor_m_cuda_internal_prepare_response_knn(
    ManorMCudaContext* context, const float* response_point_major,
    const float* device_response_point_major, int64_t point_count,
    int32_t response_dimension,
    ManorMCudaPreparedResponseKnn** prepared)
{
    if (!prepared) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "prepared response-kNN output must not be null");
    }
    *prepared = nullptr;
    if (!context || !response_point_major || !device_response_point_major ||
            point_count < 1 ||
            point_count > std::numeric_limits<int>::max() ||
            response_dimension < 1 || response_dimension > manor_knn::kMaxDy) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "prepared response-kNN dimensions or pointers are invalid");
    }

    uint64_t response_elements = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(point_count),
            static_cast<uint64_t>(response_dimension), &response_elements) ||
            response_elements > std::numeric_limits<size_t>::max() /
                sizeof(float)) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "prepared response-kNN input size overflowed size_t");
    }
    const size_t response_bytes =
        static_cast<size_t>(response_elements) * sizeof(float);
    if (!context->resident.validated(response_point_major, response_bytes,
            MANOR_M_RESIDENT_FINITE_F32)) {
        for (uint64_t element = 0; element < response_elements; ++element) {
            if (!std::isfinite(response_point_major[element])) {
                return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                    "response kNN input must contain only finite float32 values");
            }
        }
    }

    try {
        cudaError_t error = cudaSetDevice(context->device_ordinal);
        if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");
        for (int stream_index = 0; stream_index < context->stream_count;
                ++stream_index) {
            error = cudaStreamSynchronize(context->streams[stream_index]);
            if (error != cudaSuccess) {
                return cuda_fail(error,
                    "synchronize context before preparing response kNN");
            }
        }

        std::unique_ptr<ManorMCudaPreparedResponseKnn> value(
            new ManorMCudaPreparedResponseKnn());
        value->context = context;
        value->response_point_major = response_point_major;
        value->device_point_major = device_response_point_major;
        value->point_count = static_cast<int>(point_count);
        value->response_dimension = response_dimension;
        const int stride = std::min(manor_knn::kSubStride,
            std::max(1, value->point_count / 1024));
        value->subsample_count =
            (value->point_count - 1) / stride + 1;

        float* soa = value->response_soa.allocate(
            static_cast<size_t>(response_elements));
        float* subsample = value->response_subsample.allocate(
            static_cast<size_t>(value->subsample_count) * response_dimension);
        float* aos = value->response_aos.allocate(
            static_cast<size_t>(value->point_count) * manor_knn::kMaxDy);

        constexpr int threads = 256;
        const uint64_t transpose_blocks =
            (response_elements + threads - 1) / threads;
        if (transpose_blocks > std::numeric_limits<unsigned int>::max()) {
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                "prepared response-kNN transpose grid exceeds the CUDA grid limit");
        }
        response_point_major_to_soa_kernel<<<
            static_cast<unsigned int>(transpose_blocks), threads>>>(
            device_response_point_major, soa, value->point_count,
            response_dimension);
        int blocks = (value->subsample_count * response_dimension +
            threads - 1) / threads;
        manor_knn::knn_subsample_kernel<<<blocks, threads>>>(
            soa, value->point_count, response_dimension, stride,
            value->subsample_count, subsample);
        blocks = (value->point_count + threads - 1) / threads;
        manor_knn::knn_pack_aos_kernel<<<blocks, threads>>>(
            soa, value->point_count, response_dimension, aos);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess)
            return cuda_fail(error, "prepare immutable response-kNN layouts");

        *prepared = value.release();
        return ok();
    } catch (const manor_knn::Error& error) {
        return fail(MANOR_M_CUDA_CUDA_ERROR, error.what());
    } catch (const std::exception& error) {
        return fail(MANOR_M_CUDA_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(MANOR_M_CUDA_INTERNAL_ERROR,
            "unknown exception while preparing response kNN");
    }
}

void manor_m_cuda_internal_destroy_prepared_response_knn(
    ManorMCudaPreparedResponseKnn* prepared)
{
    if (!prepared) return;
    if (prepared->context)
        cudaSetDevice(prepared->context->device_ordinal);
    delete prepared;
}

void manor_m_cuda_internal_knn_memory(int N, int DY, int K, bool tiled_output,
    uint64_t* fixed_bytes, uint64_t* bytes_per_query)
{
    using namespace manor_knn;
    const int stride = std::min(kSubStride, std::max(1, N / 1024));
    const int Ns = (N - 1) / stride + 1;
    const int cap = static_cast<int>(std::min<int64_t>(N,
        std::max<int64_t>(static_cast<int64_t>(K) + kRefineBand,
                          static_cast<int64_t>(K) * kCapFactor)));
    const int want = static_cast<int>(std::min<int64_t>(N,
        static_cast<int64_t>(K) + kRefineBand));
    const int compact = static_cast<int>(std::min<int64_t>(cap,
        static_cast<int64_t>(want) + kCompactSlack));
    *fixed_bytes = static_cast<uint64_t>(Ns) * DY * sizeof(float)
        + static_cast<uint64_t>(N) * kMaxDy * sizeof(float)
        + kDiagSlots * sizeof(unsigned long long) + kSortTempSlack;
    *bytes_per_query = static_cast<uint64_t>(cap) * 8
        + static_cast<uint64_t>(compact) * 36 + 8 * sizeof(int)
        + (tiled_output ? static_cast<uint64_t>(K) * sizeof(int) : 0);
}

extern "C" int manor_m_cuda_response_knn_memory(
    int64_t point_count, int32_t response_dimension,
    int32_t neighbor_count, int32_t tiled_output,
    uint64_t* fixed_bytes, uint64_t* bytes_per_query)
{
    if (!fixed_bytes || !bytes_per_query || point_count < 1 ||
            point_count > std::numeric_limits<int>::max() ||
            response_dimension < 1 || response_dimension > 8 ||
            neighbor_count < 1 || neighbor_count > point_count ||
            (tiled_output != 0 && tiled_output != 1)) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Invalid response-kNN memory-planning request.");
    }
    manor_m_cuda_internal_knn_memory(
        static_cast<int>(point_count), response_dimension, neighbor_count,
        tiled_output != 0, fixed_bytes, bytes_per_query);
    return ok();
}


extern "C" uint32_t manor_m_cuda_abi_version(void)
{
    return MANOR_M_CUDA_ABI_VERSION;
}

extern "C" const char* manor_m_cuda_version(void)
{
    return "0.4.0-dev";
}

int manor_m_cuda_internal_fail(int status, const char* message)
{
    return fail(static_cast<ManorMCudaStatus>(status), message);
}

int manor_m_cuda_internal_context_device(
    const ManorMCudaContext* context, int32_t* device_ordinal)
{
    if (context == nullptr || device_ordinal == nullptr) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "context and device-ordinal output must not be null");
    }
    *device_ordinal = context->device_ordinal;
    return ok();
}

extern "C" const char* manor_m_cuda_status_string(int status)
{
    switch (status) {
        case MANOR_M_CUDA_SUCCESS:
            return "success";
        case MANOR_M_CUDA_INVALID_ARGUMENT:
            return "invalid argument";
        case MANOR_M_CUDA_CUDA_ERROR:
            return "CUDA error";
        case MANOR_M_CUDA_INTERNAL_ERROR:
            return "internal error";
        default:
            return "unknown status";
    }
}

extern "C" const char* manor_m_cuda_last_error(void)
{
    return g_last_error.c_str();
}

extern "C" int manor_m_cuda_device_count(int32_t* count)
{
    if (count == nullptr) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "count must not be null");
    }
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaGetDeviceCount");
    }
    *count = static_cast<int32_t>(device_count);
    return ok();
}

extern "C" int manor_m_cuda_get_device_info(
    int32_t device_ordinal, ManorMCudaDeviceInfo* info)
{
    if (info == nullptr) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "info must not be null");
    }

    int32_t count = 0;
    int status = manor_m_cuda_device_count(&count);
    if (status != MANOR_M_CUDA_SUCCESS) {
        return status;
    }
    if (device_ordinal < 0 || device_ordinal >= count) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "device_ordinal is outside the visible CUDA device range");
    }

    cudaDeviceProp properties{};
    cudaError_t error = cudaGetDeviceProperties(&properties, device_ordinal);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaGetDeviceProperties");
    }
    error = cudaSetDevice(device_ordinal);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaSetDevice");
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaMemGetInfo");
    }

    int driver_version = 0;
    int runtime_version = 0;
    error = cudaDriverGetVersion(&driver_version);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaDriverGetVersion");
    }
    error = cudaRuntimeGetVersion(&runtime_version);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaRuntimeGetVersion");
    }

    std::memset(info, 0, sizeof(*info));
    info->struct_size = static_cast<uint32_t>(sizeof(*info));
    info->abi_version = MANOR_M_CUDA_ABI_VERSION;
    info->device_ordinal = device_ordinal;
    info->compute_capability_major = properties.major;
    info->compute_capability_minor = properties.minor;
    info->multiprocessor_count = properties.multiProcessorCount;
    info->warp_size = properties.warpSize;
    info->max_threads_per_block = properties.maxThreadsPerBlock;
    info->cuda_driver_version = driver_version;
    info->cuda_runtime_version = runtime_version;
    info->total_global_memory_bytes = static_cast<uint64_t>(total_bytes);
    info->free_global_memory_bytes = static_cast<uint64_t>(free_bytes);
    std::strncpy(info->name, properties.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    return ok();
}

extern "C" int manor_m_cuda_set_device(int32_t device_ordinal)
{
    const cudaError_t error = cudaSetDevice(device_ordinal);
    return error == cudaSuccess ? ok() : cuda_fail(error, "cudaSetDevice");
}

extern "C" int manor_m_cuda_synchronize(void)
{
    const cudaError_t error = cudaDeviceSynchronize();
    return error == cudaSuccess
        ? ok()
        : cuda_fail(error, "cudaDeviceSynchronize");
}

extern "C" int manor_m_cuda_plan_tiles(
    const ManorMCudaTilePlanRequest* request,
    ManorMCudaTilePlan* plan)
{
    if (request == nullptr || plan == nullptr) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "tile-plan request and output must not be null");
    }
    if (request->struct_size != sizeof(*request)
        || request->abi_version != MANOR_M_CUDA_ABI_VERSION) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "tile-plan request size or ABI version does not match the core");
    }
    if (request->reference_row_count < 0 || request->query_row_count < 0
        || request->minimum_reference_tile_rows < 0
        || request->minimum_query_tile_rows < 0
        || request->preferred_reference_tile_rows < 0
        || request->preferred_query_tile_rows < 0) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "tile-plan row counts must be nonnegative");
    }
    if (!(request->usable_memory_fraction > 0.0)
        || request->usable_memory_fraction > 1.0
        || !std::isfinite(request->usable_memory_fraction)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "usable_memory_fraction must be finite and in (0, 1]");
    }
    if (request->staging_slot_count < 1 || request->staging_slot_count > 8) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "staging_slot_count must be between 1 and 8");
    }
    if ((request->reference_row_count > 0
            && request->reference_bytes_per_row == 0)
        || (request->query_row_count > 0
            && request->query_bytes_per_row == 0
            && request->scratch_bytes_per_query_row == 0)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "nonempty row sets require nonzero per-row storage");
    }
    if ((request->preferred_reference_tile_rows > 0
            && request->preferred_reference_tile_rows
                < request->minimum_reference_tile_rows)
        || (request->preferred_query_tile_rows > 0
            && request->preferred_query_tile_rows
                < request->minimum_query_tile_rows)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "preferred tile rows cannot be smaller than minimum tile rows");
    }

    cudaError_t error = cudaSetDevice(request->device_ordinal);
    if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");

    size_t free_size = 0;
    size_t total_size = 0;
    error = cudaMemGetInfo(&free_size, &total_size);
    if (error != cudaSuccess) return cuda_fail(error, "cudaMemGetInfo");
    (void)total_size;

    const uint64_t free_bytes = static_cast<uint64_t>(free_size);
    const uint64_t fraction_bytes = static_cast<uint64_t>(
        std::floor(static_cast<long double>(free_bytes)
            * request->usable_memory_fraction));
    if (fraction_bytes <= request->reserve_memory_bytes) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "reserve_memory_bytes leaves no usable device memory");
    }
    const uint64_t usable_bytes =
        fraction_bytes - request->reserve_memory_bytes;

    uint64_t full_reference_bytes = 0;
    uint64_t full_query_bytes = 0;
    uint64_t query_row_bytes = 0;
    uint64_t full_resident_bytes = request->persistent_memory_bytes;
    if (!checked_add_u64(
            request->query_bytes_per_row,
            request->scratch_bytes_per_query_row,
            &query_row_bytes)
        || !checked_mul_u64(
            static_cast<uint64_t>(request->reference_row_count),
            request->reference_bytes_per_row,
            &full_reference_bytes)
        || !checked_mul_u64(
            static_cast<uint64_t>(request->query_row_count),
            query_row_bytes,
            &full_query_bytes)
        || !checked_add_u64(
            full_resident_bytes, full_reference_bytes, &full_resident_bytes)
        || !checked_add_u64(
            full_resident_bytes, full_query_bytes, &full_resident_bytes)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "tile-plan byte calculation overflowed uint64");
    }

    std::memset(plan, 0, sizeof(*plan));
    plan->struct_size = static_cast<uint32_t>(sizeof(*plan));
    plan->abi_version = MANOR_M_CUDA_ABI_VERSION;
    plan->staging_slot_count = request->staging_slot_count;
    plan->device_free_memory_bytes = free_bytes;
    plan->usable_memory_bytes = usable_bytes;
    plan->full_resident_memory_bytes = full_resident_bytes;

    if (full_resident_bytes <= usable_bytes) {
        plan->execution_mode = MANOR_M_CUDA_EXECUTION_RESIDENT;
        plan->estimated_peak_memory_bytes = full_resident_bytes;
        plan->reference_tile_rows = request->reference_row_count;
        plan->query_tile_rows = request->query_row_count;
        plan->reference_tile_count = request->reference_row_count == 0 ? 0 : 1;
        plan->query_tile_count = request->query_row_count == 0 ? 0 : 1;
        return ok();
    }

    if (request->persistent_memory_bytes >= usable_bytes) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "persistent memory alone exceeds the usable device budget");
    }

    const uint64_t slots = static_cast<uint64_t>(request->staging_slot_count);
    uint64_t reference_staged_row_bytes = 0;
    uint64_t query_staged_row_bytes = 0;
    if (!checked_mul_u64(
            request->reference_bytes_per_row,
            slots,
            &reference_staged_row_bytes)
        || !checked_mul_u64(query_row_bytes, slots, &query_staged_row_bytes)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "staged per-row byte calculation overflowed uint64");
    }

    const int64_t minimum_reference_rows = request->reference_row_count == 0
        ? 0
        : std::min(
            request->reference_row_count,
            std::max<int64_t>(1, request->minimum_reference_tile_rows));
    const int64_t minimum_query_rows = request->query_row_count == 0
        ? 0
        : std::min(
            request->query_row_count,
            std::max<int64_t>(1, request->minimum_query_tile_rows));
    uint64_t minimum_reference_bytes = 0;
    uint64_t minimum_query_bytes = 0;
    uint64_t minimum_peak_bytes = request->persistent_memory_bytes;
    if (!checked_mul_u64(
            static_cast<uint64_t>(minimum_reference_rows),
            reference_staged_row_bytes,
            &minimum_reference_bytes)
        || !checked_mul_u64(
            static_cast<uint64_t>(minimum_query_rows),
            query_staged_row_bytes,
            &minimum_query_bytes)
        || !checked_add_u64(
            minimum_peak_bytes, minimum_reference_bytes, &minimum_peak_bytes)
        || !checked_add_u64(
            minimum_peak_bytes, minimum_query_bytes, &minimum_peak_bytes)
        || minimum_peak_bytes > usable_bytes) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "usable device memory cannot fit the minimum requested tiles");
    }

    const uint64_t remaining_bytes = usable_bytes - minimum_peak_bytes;
    uint64_t reference_share = 0;
    uint64_t query_share = 0;
    if (request->reference_row_count > 0 && request->query_row_count > 0) {
        reference_share = remaining_bytes / 2;
        query_share = remaining_bytes - reference_share;
    } else if (request->reference_row_count > 0) {
        reference_share = remaining_bytes;
    } else {
        query_share = remaining_bytes;
    }

    int64_t reference_rows = minimum_reference_rows;
    int64_t query_rows = minimum_query_rows;
    if (reference_staged_row_bytes > 0) {
        reference_rows += static_cast<int64_t>(
            reference_share / reference_staged_row_bytes);
    }
    if (query_staged_row_bytes > 0) {
        query_rows +=
            static_cast<int64_t>(query_share / query_staged_row_bytes);
    }
    reference_rows = cap_and_align_rows(
        reference_rows,
        request->reference_row_count,
        minimum_reference_rows,
        request->preferred_reference_tile_rows);
    query_rows = cap_and_align_rows(
        query_rows,
        request->query_row_count,
        minimum_query_rows,
        request->preferred_query_tile_rows);

    uint64_t reference_tile_bytes = 0;
    uint64_t query_tile_bytes = 0;
    uint64_t estimated_peak_bytes = request->persistent_memory_bytes;
    if (!checked_mul_u64(
            static_cast<uint64_t>(reference_rows),
            reference_staged_row_bytes,
            &reference_tile_bytes)
        || !checked_mul_u64(
            static_cast<uint64_t>(query_rows),
            query_staged_row_bytes,
            &query_tile_bytes)
        || !checked_add_u64(
            estimated_peak_bytes, reference_tile_bytes, &estimated_peak_bytes)
        || !checked_add_u64(
            estimated_peak_bytes, query_tile_bytes, &estimated_peak_bytes)
        || estimated_peak_bytes > usable_bytes) {
        return fail(
            MANOR_M_CUDA_INTERNAL_ERROR,
            "internal tile planner produced an invalid peak estimate");
    }

    plan->execution_mode = MANOR_M_CUDA_EXECUTION_TILED;
    plan->estimated_peak_memory_bytes = estimated_peak_bytes;
    plan->reference_tile_rows = reference_rows;
    plan->query_tile_rows = query_rows;
    plan->reference_tile_count = request->reference_row_count == 0
        ? 0
        : ceil_div_i64(request->reference_row_count, reference_rows);
    plan->query_tile_count = request->query_row_count == 0
        ? 0
        : ceil_div_i64(request->query_row_count, query_rows);
    return ok();
}

extern "C" int manor_m_cuda_context_create(
    const ManorMCudaContextOptions* options,
    ManorMCudaContext** context)
{
    if (options == nullptr || context == nullptr) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "context options and output must not be null");
    }
    *context = nullptr;
    if (options->struct_size != sizeof(*options)
        || options->abi_version != MANOR_M_CUDA_ABI_VERSION) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "context-options size or ABI version does not match the core");
    }
    if (options->stream_count < 1 || options->stream_count > 8) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "stream_count must be between 1 and 8");
    }

    cudaError_t error = cudaSetDevice(options->device_ordinal);
    if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");

    ManorMCudaContext* created = new (std::nothrow) ManorMCudaContext();
    if (created == nullptr) {
        return fail(
            MANOR_M_CUDA_INTERNAL_ERROR,
            "host allocation failed while creating CUDA context");
    }
    created->device_ordinal = options->device_ordinal;
    created->stream_count = options->stream_count;

    for (int index = 0; index < options->stream_count; ++index) {
        cudaStream_t stream = nullptr;
        error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (error != cudaSuccess) {
            for (int prior = 0; prior < index; ++prior)
                cudaStreamDestroy(created->streams[prior]);
            delete created;
            return cuda_fail(error, "cudaStreamCreateWithFlags");
        }
        created->streams[index] = stream;
    }

    int pools_supported = 0;
    error = cudaDeviceGetAttribute(
        &pools_supported,
        cudaDevAttrMemoryPoolsSupported,
        options->device_ordinal);
    if (error != cudaSuccess) {
        for (int index = 0; index < created->stream_count; ++index)
            cudaStreamDestroy(created->streams[index]);
        delete created;
        return cuda_fail(error, "cudaDeviceGetAttribute(memory pools)");
    }
    created->memory_pools_supported = pools_supported != 0;
    if (created->memory_pools_supported
        && options->memory_pool_release_threshold_bytes > 0) {
        cudaMemPool_t pool = nullptr;
        uint64_t threshold = options->memory_pool_release_threshold_bytes;
        error = cudaDeviceGetDefaultMemPool(&pool, options->device_ordinal);
        if (error == cudaSuccess) {
            error = cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReleaseThreshold,
                &created->original_memory_pool_release_threshold_bytes);
        }
        if (error == cudaSuccess) {
            error = cudaMemPoolSetAttribute(
                pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
        if (error != cudaSuccess) {
            for (int index = 0; index < created->stream_count; ++index)
                cudaStreamDestroy(created->streams[index]);
            delete created;
            return cuda_fail(error, "configure CUDA memory pool");
        }
        created->memory_pool_threshold_configured = true;
    }

    *context = created;
    return ok();
}

extern "C" int manor_m_cuda_context_destroy(ManorMCudaContext* context)
{
    if (context == nullptr) return ok();
    cudaError_t first_error = cudaSetDevice(context->device_ordinal);
    if (first_error == cudaSuccess) {
        const cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) first_error = sync_error;
    }
    for (int index = 0; index < context->stream_count; ++index) {
        const cudaStream_t stream = context->streams[index];
        const cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (first_error == cudaSuccess && sync_error != cudaSuccess)
            first_error = sync_error;
        const cudaError_t destroy_error = cudaStreamDestroy(stream);
        if (first_error == cudaSuccess && destroy_error != cudaSuccess)
            first_error = destroy_error;
    }
    if (context->memory_pool_threshold_configured) {
        cudaMemPool_t pool = nullptr;
        cudaError_t pool_error = cudaDeviceGetDefaultMemPool(
            &pool, context->device_ordinal);
        if (pool_error == cudaSuccess)
            pool_error = cudaMemPoolTrimTo(pool, 0);
        if (pool_error == cudaSuccess) {
            pool_error = cudaMemPoolSetAttribute(
                pool, cudaMemPoolAttrReleaseThreshold,
                &context->original_memory_pool_release_threshold_bytes);
        }
        if (first_error == cudaSuccess && pool_error != cudaSuccess)
            first_error = pool_error;
    }
    context->resident.clear("");
    delete context;
    return first_error == cudaSuccess
        ? ok()
        : cuda_fail(first_error, "destroy CUDA context");
}

extern "C" int manor_m_cuda_resident_begin(
    ManorMCudaContext* context, uint64_t maximum_retained_bytes)
{
    if (!context || !maximum_retained_bytes || context->resident.enabled)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "A nonzero cache budget and an inactive, live context are required.");
    const cudaError_t error = cudaSetDevice(context->device_ordinal);
    if (error != cudaSuccess) return cuda_fail(error, "select resident device");
    context->resident = ManorMResidentCache{};
    context->resident.budget = maximum_retained_bytes;
    context->resident.enabled = true;
    return ok();
}

extern "C" int manor_m_cuda_resident_put(
    ManorMCudaContext* context, const char* name, const void* host, uint64_t bytes)
{
    if (!context || !context->resident.enabled || !name || !*name || !host || !bytes
        || bytes > std::numeric_limits<size_t>::max())
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Invalid resident input or inactive context.");
    void* device = nullptr;
    try {
        auto& cache = context->resident;
        if (bytes > cache.budget || cache.bytes() > cache.budget - bytes)
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Resident byte budget exceeded.");
        for (const auto& entry : cache.entries)
            if (entry->name == name || entry->host == host)
                return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                    "Resident input exists; clear its group before replacing it.");
        const std::string resident_name(name);
        const bool finite_f32 = resident_name.compare(0, 5, "data.") == 0 ||
            resident_name.compare(0, 6, "query.") == 0;
        if (finite_f32) {
            if ((bytes % sizeof(float)) != 0)
                return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                    "Validated resident float32 input has a non-float byte count.");
            const float* values = static_cast<const float*>(host);
            const size_t count = static_cast<size_t>(bytes / sizeof(float));
            for (size_t index = 0; index < count; ++index)
                if (!std::isfinite(values[index]))
                    return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                        "Resident dataset/query input must contain only finite float32 values.");
        }
        cudaError_t error = cudaSetDevice(context->device_ordinal);
        if (error != cudaSuccess) return cuda_fail(error, "select resident device");
        error = cudaMalloc(&device, static_cast<size_t>(bytes));
        if (error != cudaSuccess) return cuda_fail(error, "allocate resident input");
        error = cudaMemcpy(device, host, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) { cudaFree(device); return cuda_fail(error, "upload resident input"); }
        cache.adopt(name, host, device, static_cast<size_t>(bytes));
        device = nullptr;
        if (finite_f32)
            cache.record_validation(name, host, static_cast<size_t>(bytes),
                MANOR_M_RESIDENT_FINITE_F32);
        cache.h2d_bytes += bytes;
        return ok();
    } catch (const std::exception& e) {
        if (device) cudaFree(device);
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, e.what());
    }
}

extern "C" int manor_m_cuda_resident_validate_index(
    ManorMCudaContext* context, const int32_t* index,
    int64_t n, int32_t k)
{
    if (!context || !context->resident.enabled || !index || n < 1 ||
            n > INT32_MAX || k < 1 || k > n)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Invalid immutable resident index validation request.");
    uint64_t elements = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(n), static_cast<uint64_t>(k),
            &elements) || elements > std::numeric_limits<size_t>::max() / sizeof(int32_t))
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Resident index validation byte count overflowed size_t.");
    const size_t bytes = static_cast<size_t>(elements) * sizeof(int32_t);
    try {
        auto& cache = context->resident;
        // An explicit re-registration is a new validation request even when
        // the pointer is unchanged. This catches deliberate mutation before
        // re-registration without charging every subsequent Phase2 cell.
        if (!manor_m_index_range_valid(index, static_cast<size_t>(elements),
                static_cast<int32_t>(n)))
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                "Immutable resident index contains an out-of-range value.");
        cache.record_validation("phase1.response_index.validation", index, bytes,
            MANOR_M_RESIDENT_INDEX_RANGE, n);
        return ok();
    } catch (const std::exception& e) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, e.what());
    }
}

extern "C" int manor_m_cuda_resident_validate_compact_index(
    ManorMCudaContext* context, const int32_t* index,
    int64_t rows, int32_t k, int64_t n)
{
    if (!context || !context->resident.enabled || !index || rows < 1 ||
            rows > n || n < 1 || n > INT32_MAX || k < 1 || k > n)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Invalid immutable compact index validation request.");
    uint64_t elements = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(rows),
            static_cast<uint64_t>(k), &elements) ||
            elements > std::numeric_limits<size_t>::max() / sizeof(int32_t))
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Compact index validation byte count overflowed size_t.");
    const size_t bytes = static_cast<size_t>(elements) * sizeof(int32_t);
    try {
        if (!manor_m_index_range_valid(index, static_cast<size_t>(elements),
                static_cast<int32_t>(n)))
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                "Immutable compact index contains an out-of-range value.");
        context->resident.record_validation(
            "phase1.compact_index.validation", index, bytes,
            MANOR_M_RESIDENT_INDEX_RANGE, n);
        return ok();
    } catch (const std::exception& e) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, e.what());
    }
}

extern "C" int manor_m_cuda_resident_prepare_owner_index(
    ManorMCudaContext* context, const int32_t* index, const uint8_t* owner,
    int64_t n, int32_t k)
{
    if (!context || !index || !owner || n < 1 || n > INT32_MAX || k < 1 || k > n)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Invalid resident owner-index shape.");
    void* device = nullptr;
    try {
        cudaError_t error = cudaSetDevice(context->device_ordinal);
        if (error != cudaSuccess) return cuda_fail(error, "select resident device");
        auto& c = context->resident;
        if (!c.find(index, static_cast<size_t>(n) * k * sizeof(int32_t)) ||
            !c.find(owner, static_cast<size_t>(n)))
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Register immutable full index and owner mask first.");
        size_t m = 0;
        for (int64_t i = 0; i < n; ++i) if (owner[i]) ++m;
        if (!m) return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Phase1 has no owners.");
        if (m == static_cast<size_t>(n)) return ok();
        const size_t bytes = m * static_cast<size_t>(k) * sizeof(int32_t);
        if (bytes > c.budget || c.bytes() > c.budget - bytes)
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Packed owner index exceeds resident byte budget.");
        std::vector<int32_t> packed(m * static_cast<size_t>(k));
        size_t q = 0;
        for (int64_t i = 0; i < n; ++i) if (owner[i]) {
            for (int32_t j = 0; j < k; ++j) {
                const int32_t value = index[static_cast<size_t>(i) * k + j];
                if (value < 0 || value >= n)
                    return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Invalid resident neighbor index.");
                packed[q++] = value;
            }
        }
        error = cudaMalloc(&device, bytes);
        if (error != cudaSuccess) return cuda_fail(error, "allocate packed owner index");
        error = cudaMemcpy(device, packed.data(), bytes, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) { cudaFree(device); return cuda_fail(error, "upload packed owner index"); }
        c.adopt("phase1.compact_index", packed.data(), device, bytes);
        device = nullptr;
        auto& entry = *c.entries.back();
        entry.host_index = std::move(packed);
        entry.source_index = index; entry.source_owner = owner;
        c.record_validation("phase1.compact_index.validation", entry.host,
            bytes, MANOR_M_RESIDENT_INDEX_RANGE, n);
        c.h2d_bytes += bytes;
        return ok();
    } catch (const std::exception& e) {
        if (device) cudaFree(device);
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, e.what());
    }
}

extern "C" int manor_m_cuda_resident_clear(ManorMCudaContext* context, const char* group)
{
    if (!context || !group)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Context and resident group are required.");
    cudaError_t error = cudaSetDevice(context->device_ordinal);
    if (error != cudaSuccess) return cuda_fail(error, "select resident device");
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) return cuda_fail(error, "synchronize resident release");
    context->resident.clear(group);
    if (!*group) context->resident.enabled = false;
    return ok();
}

extern "C" int manor_m_cuda_resident_stats(
    ManorMCudaContext* context, ManorMCudaResidentStats* stats)
{
    if (!context || !stats)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "Context and resident stats are required.");
    const auto& c = context->resident;
    *stats = {sizeof(*stats), 1, c.enabled ? 1ull : 0ull,
        static_cast<uint64_t>(c.entries.size()), c.bytes(), c.h2d_bytes,
        c.generated_bytes, c.hits, c.skipped_h2d_bytes, c.reset_d2d_bytes};
    return ok();
}

static int response_knn_impl(
    ManorMCudaContext* context,
    const float* response_point_major,
    int64_t point_count,
    int32_t response_dimension,
    int64_t query_first,
    int64_t query_count,
    int32_t neighbor_count,
    int32_t* index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats, bool host_output,
    ManorMCudaKnnTileConsumer consumer = nullptr,
    void* consumer_user = nullptr,
    const float* device_point_major_override = nullptr,
    ManorMCudaPreparedResponseKnn* prepared_override = nullptr)
{
    static_assert(sizeof(int) == sizeof(int32_t),
        "The exact-kNN implementation requires a 32-bit int.");
    if (context == nullptr || response_point_major == nullptr
        || (index_zero_based == nullptr && consumer == nullptr)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "context, response input, and kNN output must not be null");
    }
    if (point_count < 1
        || point_count > std::numeric_limits<int>::max()
        || query_first < 0 || query_first >= point_count
        || query_count < 1 || query_count > point_count - query_first
        || query_count > std::numeric_limits<int>::max()
        || response_dimension < 1 || response_dimension > 8
        || neighbor_count < 1 || neighbor_count > point_count) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "response kNN dimensions are outside the supported bounds");
    }

    manor_knn::Options knn_options;
    if (options != nullptr) {
        if (options->struct_size != sizeof(*options)
            || options->abi_version != MANOR_M_CUDA_ABI_VERSION) {
            return fail(
                MANOR_M_CUDA_INVALID_ARGUMENT,
                "response-kNN options size or ABI version does not match");
        }
        if (options->max_retry_rounds < 1) {
            return fail(
                MANOR_M_CUDA_INVALID_ARGUMENT,
                "response-kNN max_retry_rounds must be positive");
        }
        if (options->scratch_memory_bytes > 0) {
            if (options->scratch_memory_bytes
                > std::numeric_limits<size_t>::max()) {
                return fail(
                    MANOR_M_CUDA_INVALID_ARGUMENT,
                    "response-kNN scratch byte count overflows size_t");
            }
            knn_options.scratch_bytes =
                static_cast<size_t>(options->scratch_memory_bytes);
        }
        knn_options.max_retry_rounds = options->max_retry_rounds;
        knn_options.collect_stats = options->collect_stats != 0;
    }
    if (stats != nullptr) knn_options.collect_stats = true;

    uint64_t response_elements = 0;
    uint64_t index_elements = 0;
    if (!checked_mul_u64(
            static_cast<uint64_t>(point_count),
            static_cast<uint64_t>(response_dimension),
            &response_elements)
        || !checked_mul_u64(
            static_cast<uint64_t>(query_count),
            static_cast<uint64_t>(neighbor_count),
            &index_elements)
        || response_elements > std::numeric_limits<size_t>::max()
                / sizeof(float)
        || index_elements > std::numeric_limits<size_t>::max()
                / sizeof(int32_t)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "response-kNN input or output byte count overflowed size_t");
    }

    const size_t response_bytes =
        static_cast<size_t>(response_elements) * sizeof(float);
    if (prepared_override &&
            (prepared_override->context != context ||
             prepared_override->response_point_major != response_point_major ||
             prepared_override->point_count != point_count ||
             prepared_override->response_dimension != response_dimension ||
             (device_point_major_override &&
              prepared_override->device_point_major !=
                  device_point_major_override))) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "prepared response-kNN identity does not match this request");
    }
    if (!prepared_override && !context->resident.validated(
            response_point_major, response_bytes,
            MANOR_M_RESIDENT_FINITE_F32)) {
        for (uint64_t element = 0; element < response_elements; ++element) {
            if (!std::isfinite(response_point_major[element])) {
                return fail(
                    MANOR_M_CUDA_INVALID_ARGUMENT,
                    "response kNN input must contain only finite float32 values");
            }
        }
    }

    if (stats != nullptr) {
        std::memset(stats, 0, sizeof(*stats));
        stats->struct_size = static_cast<uint32_t>(sizeof(*stats));
        stats->abi_version = MANOR_M_CUDA_ABI_VERSION;
        stats->device_ordinal = context->device_ordinal;
        stats->response_dimension = response_dimension;
        stats->point_count = point_count;
        stats->query_count = query_count;
        stats->neighbor_count = neighbor_count;
    }

    try {
        cudaError_t error = cudaSetDevice(context->device_ordinal);
        if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");
        for (int stream_index = 0;
             stream_index < context->stream_count;
             ++stream_index) {
            error = cudaStreamSynchronize(context->streams[stream_index]);
            if (error != cudaSuccess)
                return cuda_fail(error, "synchronize context before response kNN");
        }

        manor_knn::detail::Buffer<float> device_point_major;
        manor_knn::detail::Buffer<float> device_soa;
        manor_knn::detail::Buffer<int> device_index;
        ManorMResidentCallScope resident_scope(context);
        float* point_pointer = prepared_override
            ? const_cast<float*>(prepared_override->device_point_major)
            : const_cast<float*>(device_point_major_override);
        if (!point_pointer) point_pointer = static_cast<float*>(
            manor_m_resident_find(response_point_major, response_bytes));
        if (!point_pointer) point_pointer = device_point_major.allocate(
            static_cast<size_t>(response_elements));
        float* soa_pointer = prepared_override
            ? prepared_override->response_soa.get()
            : device_soa.allocate(static_cast<size_t>(response_elements));
        int* index_pointer = consumer ? nullptr : host_output ? index_zero_based
            : device_index.allocate(static_cast<size_t>(index_elements));
        if (!prepared_override && !device_point_major_override) {
            error = manor_m_resident_upload(
                point_pointer,
                response_point_major,
                response_bytes);
            if (error != cudaSuccess)
                return cuda_fail(error, "upload response kNN input");
        }

        if (!prepared_override) {
            constexpr int threads = 256;
            const uint64_t block_count =
                (response_elements + threads - 1) / threads;
            if (block_count > std::numeric_limits<unsigned int>::max()) {
                return fail(
                    MANOR_M_CUDA_INVALID_ARGUMENT,
                    "response-kNN transpose grid exceeds the CUDA grid limit");
            }
            response_point_major_to_soa_kernel<<<
                static_cast<unsigned int>(block_count), threads>>>(
                point_pointer,
                soa_pointer,
                static_cast<int>(point_count),
                response_dimension);
            error = cudaPeekAtLastError();
            if (error != cudaSuccess)
                return cuda_fail(error, "stage response kNN input as SoA");
        }

        manor_knn::Stats raw_stats;
        manor_knn::build_index(
            soa_pointer,
            static_cast<int>(point_count),
            response_dimension,
            static_cast<int>(query_first),
            static_cast<int>(query_count),
            neighbor_count,
            index_pointer,
            knn_options,
            stats == nullptr ? nullptr : &raw_stats, host_output,
            consumer, consumer_user,
            prepared_override
                ? prepared_override->response_subsample.get() : nullptr,
            prepared_override
                ? prepared_override->response_aos.get() : nullptr);
        error = (host_output || consumer) ? cudaSuccess : cudaMemcpy(
            index_zero_based,
            index_pointer,
            static_cast<size_t>(index_elements) * sizeof(int32_t),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            return cuda_fail(error, "download response kNN index");

        if (context->resident.enabled && !host_output && !consumer) {
            const size_t index_bytes = static_cast<size_t>(index_elements) * sizeof(int32_t);
            context->resident.adopt("phase1.response_index", index_zero_based,
                index_pointer, index_bytes);
            device_index.release();
            context->resident.generated_bytes += index_bytes;
        }

        if (stats != nullptr) {
            stats->bootstrap_milliseconds = raw_stats.bootstrap_ms;
            stats->threshold_milliseconds = raw_stats.threshold_ms;
            stats->scan_milliseconds = raw_stats.scan_ms;
            stats->retry_milliseconds = raw_stats.retry_ms;
            stats->select_milliseconds = raw_stats.select_ms;
            stats->refine_milliseconds = raw_stats.refine_ms;
            stats->sort_milliseconds = raw_stats.sort_ms;
            stats->emit_milliseconds = raw_stats.emit_ms;
            stats->total_milliseconds = raw_stats.total_ms;
            stats->retry_rounds = raw_stats.retry_rounds;
            stats->retry_queries = raw_stats.retry_queries;
            stats->owner_misplaced = raw_stats.owner_misplaced;
            stats->exact_ties = raw_stats.exact_ties;
            stats->compact_overflow = raw_stats.compact_overflow;
            stats->mean_candidates = raw_stats.mean_candidates;
            stats->max_candidates = raw_stats.max_candidates;
        }
        return ok();
    } catch (const manor_knn::Error& error) {
        return fail(MANOR_M_CUDA_CUDA_ERROR, error.what());
    } catch (const std::exception& error) {
        return fail(MANOR_M_CUDA_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(
            MANOR_M_CUDA_INTERNAL_ERROR,
            "unknown exception in response kNN");
    }
}

int manor_m_cuda_internal_response_knn_tiles(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int64_t query_first,
    int64_t query_count, int32_t neighbor_count,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user)
{
    if (!consumer) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "response kNN tile consumer must not be null");
    }
    return response_knn_impl(
        context, response_point_major, point_count, response_dimension,
        query_first, query_count, neighbor_count, nullptr, options, stats,
        false, consumer, user);
}

int manor_m_cuda_internal_response_knn_tiles_device(
    ManorMCudaContext* context, const float* response_point_major,
    const float* device_response_point_major, int64_t point_count,
    int32_t response_dimension, int64_t query_first, int64_t query_count,
    int32_t neighbor_count, const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user)
{
    if (!device_response_point_major)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "device response input must not be null");
    return response_knn_impl(
        context, response_point_major, point_count, response_dimension,
        query_first, query_count, neighbor_count, nullptr, options, stats,
        false, consumer, user, device_response_point_major);
}

int manor_m_cuda_internal_response_knn_tiles_prepared(
    ManorMCudaPreparedResponseKnn* prepared,
    const float* device_response_point_major,
    int64_t point_count, int32_t response_dimension,
    int64_t query_first, int64_t query_count, int32_t neighbor_count,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user)
{
    if (!prepared || !consumer ||
            device_response_point_major != prepared->device_point_major ||
            point_count != prepared->point_count ||
            response_dimension != prepared->response_dimension) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "prepared response kNN, device input, and tile consumer must match");
    }
    return response_knn_impl(
        prepared->context, prepared->response_point_major,
        prepared->point_count, prepared->response_dimension,
        query_first, query_count, neighbor_count, nullptr, options, stats,
        false, consumer, user, prepared->device_point_major, prepared);
}

namespace {

struct CompactOwnerKnnConsumerState {
    const uint8_t* owner_mask = nullptr;
    int32_t* host_output = nullptr;
    int64_t expected_owners = 0;
    int64_t emitted_owners = 0;
    int32_t neighbor_count = 0;
    int32_t capacity_rows = 0;
    int32_t* device_rows = nullptr;
    int32_t* device_output = nullptr;
    std::vector<int32_t> local_rows;

    ~CompactOwnerKnnConsumerState()
    {
        if (device_rows) cudaFree(device_rows);
        if (device_output) cudaFree(device_output);
    }
};

int compact_owner_knn_consumer(
    void* opaque, const int32_t* device_index, int32_t query_first,
    int32_t query_count, int32_t neighbor_count)
{
    auto& state = *static_cast<CompactOwnerKnnConsumerState*>(opaque);
    if (!device_index || query_first < 0 || query_count < 1 ||
            neighbor_count != state.neighbor_count)
        return fail(MANOR_M_CUDA_INTERNAL_ERROR,
            "Compact-owner kNN consumer received an invalid tile.");
    state.local_rows.clear();
    state.local_rows.reserve(static_cast<size_t>(query_count));
    for (int32_t local = 0; local < query_count; ++local)
        if (state.owner_mask[static_cast<size_t>(query_first) + local])
            state.local_rows.push_back(local);
    if (state.local_rows.empty()) return MANOR_M_CUDA_SUCCESS;
    const int32_t selected = static_cast<int32_t>(state.local_rows.size());
    if (state.emitted_owners > state.expected_owners - selected)
        return fail(MANOR_M_CUDA_INTERNAL_ERROR,
            "Compact-owner kNN emitted more rows than declared.");

    if (selected > state.capacity_rows) {
        if (state.device_rows) cudaFree(state.device_rows);
        if (state.device_output) cudaFree(state.device_output);
        state.device_rows = nullptr;
        state.device_output = nullptr;
        cudaError_t error = cudaMalloc(
            reinterpret_cast<void**>(&state.device_rows),
            static_cast<size_t>(selected) * sizeof(int32_t));
        if (error != cudaSuccess)
            return cuda_fail(error, "allocate compact-owner row IDs");
        error = cudaMalloc(
            reinterpret_cast<void**>(&state.device_output),
            static_cast<size_t>(selected) * neighbor_count * sizeof(int32_t));
        if (error != cudaSuccess)
            return cuda_fail(error, "allocate compact-owner kNN tile");
        state.capacity_rows = selected;
    }
    cudaError_t error = cudaMemcpy(
        state.device_rows, state.local_rows.data(),
        static_cast<size_t>(selected) * sizeof(int32_t),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        return cuda_fail(error, "upload compact-owner row IDs");
    const size_t elements = static_cast<size_t>(selected) * neighbor_count;
    constexpr unsigned threads = 256;
    const size_t required_blocks = (elements + threads - 1) / threads;
    const unsigned blocks = static_cast<unsigned>(
        std::max<size_t>(1, std::min<size_t>(required_blocks, 65535)));
    response_knn_gather_rows_kernel<<<blocks, threads>>>(
        device_index, state.device_rows, selected, neighbor_count,
        state.device_output);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        return cuda_fail(error, "gather compact-owner kNN rows");
    error = cudaMemcpy(
        state.host_output + static_cast<size_t>(state.emitted_owners) * neighbor_count,
        state.device_output, elements * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
        return cuda_fail(error, "download compact-owner kNN rows");
    state.emitted_owners += selected;
    return MANOR_M_CUDA_SUCCESS;
}

}  // namespace

extern "C" int manor_m_cuda_response_knn(
    ManorMCudaContext* c, const float* y, int64_t n, int32_t my,
    int64_t q, int32_t k, int32_t* index,
    const ManorMCudaResponseKnnOptions* options, ManorMCudaResponseKnnStats* stats)
{
    return response_knn_impl(c, y, n, my, 0, q, k, index, options, stats, false);
}

extern "C" int manor_m_cuda_response_knn_host(
    ManorMCudaContext* c, const float* y, int64_t n, int32_t my,
    int64_t q, int32_t k, int32_t* index,
    const ManorMCudaResponseKnnOptions* options, ManorMCudaResponseKnnStats* stats)
{
    return response_knn_impl(c, y, n, my, 0, q, k, index, options, stats, true);
}

extern "C" int manor_m_cuda_response_knn_host_range(
    ManorMCudaContext* c, const float* y, int64_t n, int32_t my,
    int64_t query_first, int64_t query_count, int32_t k, int32_t* index,
    const ManorMCudaResponseKnnOptions* options, ManorMCudaResponseKnnStats* stats)
{
    return response_knn_impl(
        c, y, n, my, query_first, query_count, k, index, options, stats, true);
}

extern "C" int manor_m_cuda_response_knn_compact_owners_host(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int32_t neighbor_count,
    const uint8_t* owner_mask, int64_t owner_count,
    int32_t* compact_index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats)
{
    if (!context || !response_point_major || !owner_mask ||
            !compact_index_zero_based || point_count < 1 ||
            point_count > INT32_MAX || owner_count < 1 ||
            owner_count > point_count || neighbor_count < 1 ||
            neighbor_count > point_count)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Invalid compact-owner response-kNN request.");
    int64_t actual_owners = 0;
    for (int64_t index = 0; index < point_count; ++index)
        actual_owners += owner_mask[index] != 0;
    if (actual_owners != owner_count)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Compact-owner count does not match the owner mask.");
    uint64_t output_elements = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(owner_count),
            static_cast<uint64_t>(neighbor_count), &output_elements) ||
            output_elements > std::numeric_limits<size_t>::max() / sizeof(int32_t))
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Compact-owner kNN output byte count overflowed size_t.");

    ManorMCudaResponseKnnOptions adjusted{};
    adjusted.struct_size = sizeof(adjusted);
    adjusted.abi_version = MANOR_M_CUDA_ABI_VERSION;
    adjusted.scratch_memory_bytes = static_cast<uint64_t>(7) << 30;
    adjusted.max_retry_rounds = 12;
    adjusted.collect_stats = stats != nullptr;
    if (options) {
        if (options->struct_size != sizeof(*options) ||
                options->abi_version != MANOR_M_CUDA_ABI_VERSION ||
                options->max_retry_rounds < 1)
            return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
                "Compact-owner response-kNN options do not match this ABI.");
        adjusted = *options;
        if (!adjusted.scratch_memory_bytes)
            adjusted.scratch_memory_bytes = static_cast<uint64_t>(7) << 30;
    }
    uint64_t fixed_bytes = 0, knn_bytes_per_query = 0;
    manor_m_cuda_internal_knn_memory(
        static_cast<int>(point_count), response_dimension, neighbor_count,
        true, &fixed_bytes, &knn_bytes_per_query);
    const uint64_t gather_bytes_per_query =
        static_cast<uint64_t>(neighbor_count) * sizeof(int32_t) + sizeof(int32_t);
    uint64_t bytes_per_query = 0;
    if (!checked_add_u64(knn_bytes_per_query, gather_bytes_per_query,
            &bytes_per_query) || adjusted.scratch_memory_bytes < fixed_bytes ||
            adjusted.scratch_memory_bytes - fixed_bytes < bytes_per_query)
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Compact-owner response-kNN scratch cannot hold one query.");
    uint64_t query_capacity =
        (adjusted.scratch_memory_bytes - fixed_bytes) / bytes_per_query;
    query_capacity = std::min<uint64_t>(query_capacity,
        static_cast<uint64_t>(point_count));
    uint64_t knn_budget = 0, variable_budget = 0;
    if (!checked_mul_u64(query_capacity, knn_bytes_per_query,
            &variable_budget) ||
            !checked_add_u64(fixed_bytes, variable_budget, &knn_budget))
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Compact-owner response-kNN scratch arithmetic overflowed.");
    adjusted.scratch_memory_bytes = knn_budget;

    CompactOwnerKnnConsumerState consumer;
    consumer.owner_mask = owner_mask;
    consumer.host_output = compact_index_zero_based;
    consumer.expected_owners = owner_count;
    consumer.neighbor_count = neighbor_count;
    const int status = manor_m_cuda_internal_response_knn_tiles(
        context, response_point_major, point_count, response_dimension,
        0, point_count, neighbor_count, &adjusted, stats,
        compact_owner_knn_consumer, &consumer);
    if (status != MANOR_M_CUDA_SUCCESS) return status;
    if (consumer.emitted_owners != owner_count)
        return fail(MANOR_M_CUDA_INTERNAL_ERROR,
            "Compact-owner response-kNN did not emit every owner row.");
    return ok();
}

extern "C" int manor_m_cuda_smoke_tiled_axpy(
    ManorMCudaContext* context,
    float alpha,
    const float* x,
    const float* y,
    float* output,
    int64_t element_count,
    int64_t tile_elements,
    ManorMCudaTiledStats* stats)
{
    if (context == nullptr || stats == nullptr) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "context and tiled stats must not be null");
    }
    if (element_count < 0 || tile_elements < 0) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "element_count and tile_elements must be nonnegative");
    }
    if (element_count > 0 && (x == nullptr || y == nullptr || output == nullptr)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "x, y, and output must not be null when element_count is positive");
    }

    std::memset(stats, 0, sizeof(*stats));
    stats->struct_size = static_cast<uint32_t>(sizeof(*stats));
    stats->abi_version = MANOR_M_CUDA_ABI_VERSION;
    stats->device_ordinal = context->device_ordinal;
    stats->element_count = element_count;
    if (element_count == 0) return ok();

    cudaError_t error = cudaSetDevice(context->device_ordinal);
    if (error != cudaSuccess) return cuda_fail(error, "cudaSetDevice");

    if (tile_elements == 0) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess) return cuda_fail(error, "cudaMemGetInfo");
        (void)total_bytes;
        constexpr int64_t default_tile_elements = 8LL * 1024 * 1024;
        const uint64_t denominator = 2ULL * 3ULL * sizeof(float) * 4ULL;
        const int64_t memory_tile_elements = static_cast<int64_t>(
            static_cast<uint64_t>(free_bytes) / denominator);
        tile_elements = std::min(
            element_count,
            std::max<int64_t>(1, std::min(
                default_tile_elements, memory_tile_elements)));
    } else {
        tile_elements = std::min(tile_elements, element_count);
    }

    if (static_cast<uint64_t>(tile_elements)
        > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "tile byte count overflows size_t");
    }
    const size_t tile_bytes = static_cast<size_t>(tile_elements) * sizeof(float);
    const int64_t tile_count = ceil_div_i64(element_count, tile_elements);
    const int slot_count = std::min<int64_t>(
        std::min<int32_t>(2, context->stream_count), tile_count);

    struct Slot {
        cudaStream_t stream = nullptr;
        cudaEvent_t complete = nullptr;
        float* host_x = nullptr;
        float* host_y = nullptr;
        float* host_output = nullptr;
        float* device_x = nullptr;
        float* device_y = nullptr;
        float* device_output = nullptr;
        int64_t output_offset = 0;
        int64_t output_count = 0;
        bool pending = false;
    };
    Slot slots[2];

    auto cleanup = [&]() {
        for (int index = 0; index < slot_count; ++index) {
            Slot& slot = slots[index];
            if (slot.stream != nullptr) cudaStreamSynchronize(slot.stream);
            if (slot.complete != nullptr) cudaEventDestroy(slot.complete);
            if (slot.device_output != nullptr) {
                if (context->memory_pools_supported)
                    cudaFreeAsync(slot.device_output, slot.stream);
                else
                    cudaFree(slot.device_output);
            }
            if (slot.device_y != nullptr) {
                if (context->memory_pools_supported)
                    cudaFreeAsync(slot.device_y, slot.stream);
                else
                    cudaFree(slot.device_y);
            }
            if (slot.device_x != nullptr) {
                if (context->memory_pools_supported)
                    cudaFreeAsync(slot.device_x, slot.stream);
                else
                    cudaFree(slot.device_x);
            }
            if (context->memory_pools_supported && slot.stream != nullptr)
                cudaStreamSynchronize(slot.stream);
            if (slot.host_output != nullptr) cudaFreeHost(slot.host_output);
            if (slot.host_y != nullptr) cudaFreeHost(slot.host_y);
            if (slot.host_x != nullptr) cudaFreeHost(slot.host_x);
        }
    };
    auto allocation_error = [&](cudaError_t status, const char* operation) {
        cleanup();
        return cuda_fail(status, operation);
    };

    for (int index = 0; index < slot_count; ++index) {
        Slot& slot = slots[index];
        slot.stream = context->streams[index];
        error = cudaEventCreateWithFlags(&slot.complete, cudaEventDisableTiming);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaEventCreateWithFlags");
        error = cudaMallocHost(reinterpret_cast<void**>(&slot.host_x), tile_bytes);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMallocHost(x)");
        error = cudaMallocHost(reinterpret_cast<void**>(&slot.host_y), tile_bytes);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMallocHost(y)");
        error = cudaMallocHost(
            reinterpret_cast<void**>(&slot.host_output), tile_bytes);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMallocHost(output)");

        auto allocate_device = [&](float** pointer, const char* operation) {
            if (context->memory_pools_supported)
                error = cudaMallocAsync(
                    reinterpret_cast<void**>(pointer), tile_bytes, slot.stream);
            else
                error = cudaMalloc(reinterpret_cast<void**>(pointer), tile_bytes);
            if (error == cudaSuccess && context->memory_pools_supported)
                error = cudaStreamSynchronize(slot.stream);
            return error == cudaSuccess
                ? static_cast<int>(MANOR_M_CUDA_SUCCESS)
                : allocation_error(error, operation);
        };
        int status = allocate_device(&slot.device_x, "allocate tiled x");
        if (status != MANOR_M_CUDA_SUCCESS) return status;
        status = allocate_device(&slot.device_y, "allocate tiled y");
        if (status != MANOR_M_CUDA_SUCCESS) return status;
        status = allocate_device(
            &slot.device_output, "allocate tiled output");
        if (status != MANOR_M_CUDA_SUCCESS) return status;
    }

    const auto wall_start = std::chrono::steady_clock::now();
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        Slot& slot = slots[tile_index % slot_count];
        if (slot.pending) {
            error = cudaEventSynchronize(slot.complete);
            if (error != cudaSuccess)
                return allocation_error(error, "wait for tiled output");
            std::memcpy(
                output + slot.output_offset,
                slot.host_output,
                static_cast<size_t>(slot.output_count) * sizeof(float));
            slot.pending = false;
        }

        const int64_t offset = tile_index * tile_elements;
        const int64_t current_count =
            std::min(tile_elements, element_count - offset);
        const size_t current_bytes =
            static_cast<size_t>(current_count) * sizeof(float);
        std::memcpy(slot.host_x, x + offset, current_bytes);
        std::memcpy(slot.host_y, y + offset, current_bytes);

        error = cudaMemcpyAsync(
            slot.device_x,
            slot.host_x,
            current_bytes,
            cudaMemcpyHostToDevice,
            slot.stream);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMemcpyAsync(x)");
        error = cudaMemcpyAsync(
            slot.device_y,
            slot.host_y,
            current_bytes,
            cudaMemcpyHostToDevice,
            slot.stream);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMemcpyAsync(y)");

        constexpr int threads = 256;
        const int64_t block_count = (current_count + threads - 1) / threads;
        if (block_count > std::numeric_limits<unsigned int>::max()) {
            cleanup();
            return fail(
                MANOR_M_CUDA_INVALID_ARGUMENT,
                "tiled launch grid is too large");
        }
        smoke_axpy_kernel<<<
            static_cast<unsigned int>(block_count), threads, 0, slot.stream>>>(
            alpha,
            slot.device_x,
            slot.device_y,
            slot.device_output,
            current_count);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return allocation_error(error, "tiled AXPY kernel launch");
        error = cudaMemcpyAsync(
            slot.host_output,
            slot.device_output,
            current_bytes,
            cudaMemcpyDeviceToHost,
            slot.stream);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaMemcpyAsync(output)");
        error = cudaEventRecord(slot.complete, slot.stream);
        if (error != cudaSuccess)
            return allocation_error(error, "cudaEventRecord(tiled output)");
        slot.output_offset = offset;
        slot.output_count = current_count;
        slot.pending = true;
    }

    for (int index = 0; index < slot_count; ++index) {
        Slot& slot = slots[index];
        if (!slot.pending) continue;
        error = cudaEventSynchronize(slot.complete);
        if (error != cudaSuccess)
            return allocation_error(error, "finalize tiled output");
        std::memcpy(
            output + slot.output_offset,
            slot.host_output,
            static_cast<size_t>(slot.output_count) * sizeof(float));
        slot.pending = false;
    }
    const auto wall_stop = std::chrono::steady_clock::now();

    uint64_t staging_bytes = 0;
    if (!checked_mul_u64(
            static_cast<uint64_t>(slot_count) * 3ULL,
            static_cast<uint64_t>(tile_bytes),
            &staging_bytes)) {
        cleanup();
        return fail(
            MANOR_M_CUDA_INTERNAL_ERROR,
            "tiled staging-byte calculation overflowed uint64");
    }
    stats->staging_slot_count = slot_count;
    stats->tile_elements = tile_elements;
    stats->tile_count = tile_count;
    stats->peak_device_staging_bytes = staging_bytes;
    stats->peak_pinned_host_staging_bytes = staging_bytes;
    stats->wall_milliseconds =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    cleanup();
    return ok();
}

extern "C" int manor_m_cuda_smoke_axpy(
    int32_t device_ordinal,
    float alpha,
    const float* x,
    const float* y,
    float* output,
    int64_t element_count)
{
    if (element_count < 0) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "element_count must be nonnegative");
    }
    if (element_count == 0) {
        return ok();
    }
    if (x == nullptr || y == nullptr || output == nullptr) {
        return fail(
            MANOR_M_CUDA_INVALID_ARGUMENT,
            "x, y, and output must not be null when element_count is positive");
    }
    if (static_cast<uint64_t>(element_count)
        > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "array size overflows size_t");
    }

    cudaError_t error = cudaSetDevice(device_ordinal);
    if (error != cudaSuccess) {
        return cuda_fail(error, "cudaSetDevice");
    }

    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_output = nullptr;

    auto cleanup = [&]() {
        if (device_output != nullptr) cudaFree(device_output);
        if (device_y != nullptr) cudaFree(device_y);
        if (device_x != nullptr) cudaFree(device_x);
    };

    error = cudaMalloc(reinterpret_cast<void**>(&device_x), bytes);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMalloc(x)");
    }
    error = cudaMalloc(reinterpret_cast<void**>(&device_y), bytes);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMalloc(y)");
    }
    error = cudaMalloc(reinterpret_cast<void**>(&device_output), bytes);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMalloc(output)");
    }
    error = cudaMemcpy(device_x, x, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMemcpy(x)");
    }
    error = cudaMemcpy(device_y, y, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMemcpy(y)");
    }

    constexpr int threads = 256;
    const int64_t block_count = (element_count + threads - 1) / threads;
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        cleanup();
        return fail(MANOR_M_CUDA_INVALID_ARGUMENT, "launch grid is too large");
    }
    smoke_axpy_kernel<<<static_cast<unsigned int>(block_count), threads>>>(
        alpha, device_x, device_y, device_output, element_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "smoke_axpy_kernel launch");
    }
    error = cudaMemcpy(output, device_output, bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        cleanup();
        return cuda_fail(error, "cudaMemcpy(output)");
    }

    cleanup();
    return ok();
}
