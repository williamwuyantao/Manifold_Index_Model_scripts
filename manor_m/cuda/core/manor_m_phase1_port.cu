/* Phase-1 migration translation unit.
 *
 * The production kernels are still owned by the verified MATLAB source while
 * parity work is in progress.  Compiling that source in core-library mode
 * excludes all mxArray parsing/packing and its duplicate response-kNN engine,
 * leaving the unchanged Stable-H selection, extraction, and owner-aggregation
 * kernels available to the language-neutral C ABI implemented below.
 *
 * Once the MEX adapter itself calls the C ABI and same-cell parity passes, the
 * kernel body will move physically here and the MATLAB file will become a thin
 * adapter.  Until then this include is deliberate: it prevents a hand-copied
 * fork from drifting before the parity gate exists.
 */

#include "manor_m_cuda.h"
#include "manor_m_cuda_internal.h"
#include "manor_m_cuda_resident.h"
#include <memory>
#include <cstring>
#include <climits>
#include <type_traits>

#define MANOR_M_PHASE1_CORE_LIBRARY 1
#include "../../matlab/cuda_mex/manor_m_phase1_mex.cu"

namespace {

struct Phase1CoreRequest {
    int D = 0;
    int MM = 0;
    int N = 0;
    int anchors = 0;
    int kSlice = 0;
    int S = 0;
    int R = 0;
    StablePhase1Opts options;
};

static void phase1_require(bool condition, const char* message)
{
    if (!condition)
        throw StableMexError("manor_m_cuda:phase1Argument", message);
}

#include "manor_m_phase1_capacity.cuh"
#include "manor_m_phase1_index_stream.cuh"
#include "manor_m_phase1_query_local_mask.cuh"

static StablePhase1Opts phase1_core_options(
    const ManorMCudaPhase1Options* source)
{
    StablePhase1Opts result;
    /* adaptive_parse_opts makes this unbounded when MATLAB does not supply an
       explicit maximum.  Match that production default, not Phase1Opts' old
       single-scale value. */
    result.memory.max_query_block = std::numeric_limits<int>::max();
    if (!source) return result;
    phase1_require(
        source->struct_size == sizeof(*source) &&
            source->abi_version == MANOR_M_CUDA_ABI_VERSION,
        "Phase-1 options size or ABI version does not match the core.");
    result.memory.gpu_mem_fraction = source->gpu_memory_fraction;
    result.memory.reserve_fraction = source->reserve_memory_fraction;
    result.memory.reserve_bytes = static_cast<double>(source->reserve_memory_bytes);
    result.memory.min_query_block = source->minimum_query_block;
    result.memory.max_query_block = source->maximum_query_block;
    result.memory.profile_timers = source->collect_profile != 0;
    result.heavy_factor = source->heavy_factor;
    result.jitter = source->jitter;
    result.window = source->stable_window;
    result.neighbor_scaled_max = source->neighbor_scaled_max;
    result.split_scaled_max = source->split_scaled_max;
    result.split_seed = source->split_seed;
    result.cut_max_fraction = source->cut_max_fraction;
    result.selection_mode = source->selection_mode;
    result.idx_zero_based = true;
    result.idx_shared_slice = source->index_is_shared_slice != 0;
    result.h_null_correction = source->h_null_correction != 0;
    return result;
}

static Phase1CoreRequest phase1_validate(
    const float* X,
    int64_t point_count,
    int32_t D,
    int32_t MM,
    const int32_t* index,
    int64_t active_anchor_count,
    int32_t kSlice,
    const int32_t* response_counts,
    int32_t scale_count,
    const int32_t* cuts,
    int32_t cuts_per_scale,
    const ManorMCudaPhase1Options* options,
    ManorMCudaPhase1Output* output,
    bool generated_index = false)
{
    phase1_require(X && (index || generated_index) && response_counts && cuts && output,
        "Phase-1 inputs and output descriptor must not be null.");
    phase1_require(
        output->struct_size == sizeof(*output) &&
            output->abi_version == MANOR_M_CUDA_ABI_VERSION,
        "Phase-1 output size or ABI version does not match the core.");
    phase1_require(
        output->tangent_point_major && output->center_point_major &&
            output->radius && output->axial_tangent &&
            output->neighbor_count && output->owner_mask,
        "Phase-1 tangent, center, radius, axial, neighbor-count, and owner outputs are required.");
    phase1_require(point_count >= 1 &&
        point_count <= std::numeric_limits<int>::max(),
        "Phase-1 point_count lies outside the supported int32 range.");
    phase1_require(active_anchor_count >= 1 &&
        active_anchor_count <= point_count &&
        active_anchor_count <= std::numeric_limits<int>::max(),
        "Phase-1 active_anchor_count must lie in [1,point_count].");
    phase1_require(D >= 2 && D <= MAX_D_MANOR,
        "Phase-1 ambient dimension is unsupported.");
    phase1_require(MM >= 1 && MM < D && MM <= MAX_MM_MANOR,
        "Phase-1 manifold dimension must satisfy 1 <= MM < D.");
    phase1_require(kSlice >= 1 && kSlice <= point_count,
        "Phase-1 neighbor_count must lie in [1,point_count].");
    phase1_require(scale_count >= 1 && scale_count <= ADAPTIVE_MAX_SCALES,
        "Phase-1 scale_count lies outside the supported range.");
    phase1_require(cuts_per_scale >= 1 && cuts_per_scale <= MAX_R_MANOR,
        "Phase-1 cuts_per_scale lies outside the supported range.");

    Phase1CoreRequest request;
    request.D = D;
    request.MM = MM;
    request.N = static_cast<int>(point_count);
    request.anchors = static_cast<int>(active_anchor_count);
    request.kSlice = kSlice;
    request.S = scale_count;
    request.R = cuts_per_scale;
    request.options = phase1_core_options(options);
    phase1_require(!generated_index || !request.options.idx_shared_slice,
        "Generated exact-kNN Phase 1 does not accept a shared response slice.");
    request.options.memory.max_query_block = std::min(
        request.options.memory.max_query_block, request.anchors);

    phase1_require(std::isfinite(request.options.heavy_factor) &&
        request.options.heavy_factor >= 0.0,
        "Phase-1 heavy_factor must be finite and nonnegative.");
    phase1_require(std::isfinite(request.options.jitter) &&
        request.options.jitter >= 0.0f,
        "Phase-1 jitter must be finite and nonnegative.");
    phase1_require(request.options.window >= 1 &&
        (request.options.window & 1) == 1 &&
        request.options.window <= request.R,
        "Phase-1 stable_window must be odd and no larger than cuts_per_scale.");
    phase1_require(std::isfinite(request.options.neighbor_scaled_max) &&
        request.options.neighbor_scaled_max >= 0.0f &&
        std::isfinite(request.options.split_scaled_max) &&
        request.options.split_scaled_max >= 0.0f,
        "Phase-1 Stable-H thresholds must be finite and nonnegative.");
    phase1_require(std::isfinite(request.options.memory.gpu_mem_fraction) &&
        request.options.memory.gpu_mem_fraction > 0.0f &&
        std::isfinite(request.options.memory.reserve_fraction) &&
        request.options.memory.reserve_fraction >= 0.0f &&
        std::isfinite(request.options.memory.reserve_bytes) &&
        request.options.memory.reserve_bytes >= 0.0 &&
        request.options.memory.min_query_block >= 1 &&
        request.options.memory.max_query_block >=
            request.options.memory.min_query_block,
        "Phase-1 GPU memory or query-block options are invalid.");
    phase1_require(std::isfinite(request.options.cut_max_fraction) &&
        request.options.cut_max_fraction > 0.0 &&
        request.options.cut_max_fraction <= 1.0,
        "Phase-1 cut_max_fraction must lie in (0,1].");
    phase1_require(request.options.selection_mode == STABLE_POSITIVE_H ||
        request.options.selection_mode == TOP5_ABS_H_MEDIAN_CUT,
        "Phase-1 selection_mode is unsupported.");
    phase1_require(request.S == 1 ||
        request.options.selection_mode == STABLE_POSITIVE_H,
        "Multi-scale Phase 1 supports only positive Stable-H selection.");

    const size_t x_count = static_cast<size_t>(request.N) * request.D;
    for (size_t element = 0; element < x_count; ++element)
        phase1_require(std::isfinite(X[element]),
            "Phase-1 X must contain only finite float32 values.");

    if (!generated_index) {
        const size_t index_count = request.options.idx_shared_slice
            ? static_cast<size_t>(request.kSlice)
            : static_cast<size_t>(request.anchors) * request.kSlice;
        std::vector<uint8_t> shared_seen;
        if (request.options.idx_shared_slice)
            shared_seen.assign(static_cast<size_t>(request.N), 0u);
        for (size_t element = 0; element < index_count; ++element) {
            int32_t value = index[element];
            phase1_require(value >= 0 && value < request.N,
                "Phase-1 response index contains an invalid zero-based value.");
            if (request.options.idx_shared_slice) {
                phase1_require(!shared_seen[static_cast<size_t>(value)],
                    "A shared Phase-1 response slice must contain unique indices.");
                shared_seen[static_cast<size_t>(value)] = 1u;
            }
        }
    }

    int32_t previous_count = 0;
    for (int scale = 0; scale < request.S; ++scale) {
        const int32_t count = response_counts[scale];
        phase1_require(count >= 9 * request.D && count <= request.kSlice &&
            (scale == 0 || count > previous_count),
            "Phase-1 response_counts must increase strictly within [9D,k].");
        previous_count = count;
        const int maximum_cut = static_cast<int>(std::floor(
            static_cast<double>(count) * request.options.cut_max_fraction));
        int32_t previous_cut = 0;
        for (int cut = 0; cut < request.R; ++cut) {
            const int32_t value = cuts[static_cast<size_t>(scale) * request.R + cut];
            phase1_require(value >= 3 * request.D && value <= maximum_cut &&
                (cut == 0 || value > previous_cut),
                "Each Phase-1 cut ladder must increase within [3D,floor(count*cut_max_fraction)].");
            previous_cut = value;
        }
    }
    phase1_require(previous_count == request.kSlice,
        "The final Phase-1 response count must equal neighbor_count.");
    return request;
}

template <typename Function>
static int phase1_consume_generated_knn_tile(
    void* user, const int32_t* device_index, int32_t query_first,
    int32_t query_count, int32_t neighbor_count)
{
    (*static_cast<Function*>(user))(
        device_index, query_first, query_count, neighbor_count);
    return MANOR_M_CUDA_SUCCESS;
}

static double phase1_profile_value(
    const std::vector<StableProfileSpans::SpanTotal>& totals,
    const char* label)
{
    for (const auto& entry : totals)
        if (std::strcmp(entry.label, label) == 0) return entry.ms;
    return 0.0;
}

template <typename T>
static void phase1_download(T* destination, const T* source, size_t count)
{
    if (!destination || count == 0) return;
    STABLE_CUDA_CHECK(cudaMemcpy(
        destination, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}

static size_t phase1_hybrid_extra_global_bytes(
    const Phase1CoreRequest& request, bool bounded_temporaries = true)
{
    const size_t n = static_cast<size_t>(request.N);
    const size_t anchors = static_cast<size_t>(request.anchors);
    const size_t mask_words = stable_mask_words(request.kSlice);
    /* A second selected record, raw tangent/center extraction, scalar fields,
       and owner-rule state remain global.  Bounded calls recycle both the
       temporary selector record and the fallback neighborhood mask per query
       block; unbounded legacy calls retain their original full-N allocation. */
    size_t bytes = n * (51 + 4 * request.D * (request.MM + 1)) + 32;
    if (!bounded_temporaries) {
        bytes = checked_add_size(bytes,
            n * p1_temporary_selector_bytes_per_row
                + 4 * anchors * mask_words,
            "unbounded hybrid Phase-1 extra bytes");
    }
    return bytes;
}

static AdaptiveMemoryBudget phase1_add_hybrid_budget(
    AdaptiveMemoryBudget budget, const Phase1CoreRequest& request, int tmax)
{
    auto& stable = budget.stable;
    const size_t extra = phase1_hybrid_extra_global_bytes(request, false);
    stable.fixed_bytes = checked_add_size(
        stable.fixed_bytes, extra, "hybrid Phase-1 fixed bytes");
    int low = 1;
    int high = stable.upper_bound;
    int best = 0;
    while (low <= high) {
        const int candidate = low + (high - low) / 2;
        const StableQueryMemory query = stable_query_memory(
            candidate, request.D, request.kSlice, tmax);
        const size_t required = checked_add_size(
            stable.fixed_bytes, query.total_bytes,
            "hybrid Phase-1 candidate bytes");
        if (required <= stable.usable_bytes) {
            best = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    const int minimum = std::min(
        stable.upper_bound,
        std::max(1, request.options.memory.min_query_block));
    phase1_require(best >= minimum,
        "GPU budget cannot fit fused hybrid selector state and min_query_block.");
    stable.BQ = best;
    stable.selected_query = stable_query_memory(
        best, request.D, request.kSlice, tmax);
    stable.required_bytes = checked_add_size(
        stable.fixed_bytes, stable.selected_query.total_bytes,
        "hybrid Phase-1 selected bytes");
    stable.next_required_bytes = 0;
    if (best < stable.upper_bound) {
        const StableQueryMemory next = stable_query_memory(
            best + 1, request.D, request.kSlice, tmax);
        stable.next_required_bytes = checked_add_size(
            stable.fixed_bytes, next.total_bytes,
            "hybrid Phase-1 next-query bytes");
    }
    budget.extra_fixed_bytes = checked_add_size(
        budget.extra_fixed_bytes, extra, "hybrid Phase-1 extra bytes");
    return budget;
}

__global__ void phase1_hybrid_merge_vector_kernel(
    float* destination, const float* fallback,
    const uint8_t* primary_owner, const uint8_t* fallback_owner,
    int point_count, int elements_per_point)
{
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    const size_t count = static_cast<size_t>(point_count) * elements_per_point;
    if (element >= count) return;
    const int point = static_cast<int>(element / elements_per_point);
    if (!primary_owner[point] && fallback_owner[point])
        destination[element] = fallback[element];
}

__global__ void phase1_hybrid_merge_scalar_kernel(
    int point_count,
    uint8_t* owner,
    float* radius, float* axial, int32_t* neighbor_count,
    uint32_t* selected_cut, float* geometry, float* stable_h,
    float* neighbor_scaled, float* split_scaled,
    float* split_h_a, float* split_h_b,
    uint8_t* rejection, int32_t* selected_index,
    const uint8_t* fallback_owner,
    const float* fallback_radius, const float* fallback_axial,
    const int32_t* fallback_neighbor_count,
    const uint32_t* fallback_selected_cut,
    const float* fallback_geometry, const float* fallback_stable_h,
    const float* fallback_neighbor_scaled, const float* fallback_split_scaled,
    const float* fallback_split_h_a, const float* fallback_split_h_b,
    const uint8_t* fallback_rejection,
    const int32_t* fallback_selected_index)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count || owner[point] || !fallback_owner[point]) return;
    radius[point] = fallback_radius[point];
    axial[point] = fallback_axial[point];
    neighbor_count[point] = fallback_neighbor_count[point];
    selected_cut[point] = fallback_selected_cut[point];
    geometry[point] = fallback_geometry[point];
    stable_h[point] = fallback_stable_h[point];
    neighbor_scaled[point] = fallback_neighbor_scaled[point];
    split_scaled[point] = fallback_split_scaled[point];
    split_h_a[point] = fallback_split_h_a[point];
    split_h_b[point] = fallback_split_h_b[point];
    rejection[point] = fallback_rejection[point];
    selected_index[point] = fallback_selected_index[point];
    owner[point] = 1u;
}

}  // namespace

static int phase1_plan_impl(
    ManorMCudaContext* context, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    int32_t scales, int32_t cuts, int32_t maximum_cut,
    const ManorMCudaP1CapacityOptions* o, ManorMCudaP1CapacityPlan* plan,
    bool fused_hybrid)
{
    try {
        p1_check_capacity_options(o);
        phase1_require(context && x && response && plan && n > 0 && n <= INT_MAX &&
            d >= 2 && d <= MAX_D_MANOR && mm >= 1 && mm < d && mm <= MAX_MM_MANOR &&
            my >= 1 && my <= 8 && k >= 9 * d && k <= n &&
            scales > 0 && scales <= ADAPTIVE_MAX_SCALES && cuts > 0 && cuts <= MAX_R_MANOR &&
            maximum_cut > 0 && maximum_cut <= k, "Invalid P1 capacity shape.");
        phase1_require(!fused_hybrid || scales == 1,
            "Fused hybrid Phase-1 capacity planning requires one response scale.");
        int32_t device = 0;
        int status = manor_m_cuda_internal_context_device(context, &device);
        if (status) return status;
        STABLE_CUDA_CHECK(cudaSetDevice(device));
        ManorMResidentCallScope scope(context);
        size_t free = 0, total = 0;
        STABLE_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        Phase1CoreRequest r;
        r.N = r.anchors = static_cast<int>(n); r.D = d; r.MM = mm;
        r.kSlice = k; r.S = scales; r.R = cuts;
        r.options.memory.max_query_block = INT_MAX;
        // Include optional null-calibration storage conservatively; only O(SR).
        r.options.h_null_correction = true;
        const size_t index = size_t(n) * k * sizeof(int32_t);
        const size_t xb = size_t(n) * d * sizeof(float);
        const size_t yb = size_t(n) * my * sizeof(float);
        const size_t x_new = manor_m_resident_find(x, xb) ? 0 : xb;
        const size_t y_new = manor_m_resident_find(response, yb) ? 0 : yb;
        const size_t usable = p1_usable(free, total, *o);
        auto* cache = manor_m_cuda_active_resident;
        const uint64_t retained = cache && cache->enabled ? cache->bytes() : 0;
        const uint64_t model = size_t(n) * (4 * d * (mm + 1) + 1);
        for (int mode : {1, 2, 3, 4}) {
            if (o->mode && mode != o->mode) continue;
            const bool streamed = mode == 2 || mode == 3;
            const bool generated = mode == 4;
            const bool query_local_main_mask = mode == 3 || generated;
            uint64_t knn_fixed = 0, knn_per_query = 0;
            manor_m_cuda_internal_knn_memory(
                r.N, my, k, generated || streamed,
                &knn_fixed, &knn_per_query);
            Phase1CoreRequest planned = r;
            if (generated) {
                if (o->knn_scratch_bytes < knn_fixed + knn_per_query)
                    continue;
                const uint64_t scratch_rows =
                    (o->knn_scratch_bytes - knn_fixed) / knn_per_query;
                planned.options.memory.max_query_block = static_cast<int>(
                    std::min<uint64_t>(scratch_rows,
                        static_cast<uint64_t>(planned.anchors)));
            }
            const uint64_t projected = retained + model
                + ((streamed || generated) ? 0 : index);
            if (projected > o->retained_limit_bytes) continue;
            const size_t hybrid_extra = fused_hybrid
                ? phase1_hybrid_extra_global_bytes(planned) : 0;
            const size_t global_state = checked_add_size(
                p1_global_bytes(planned, query_local_main_mask), hybrid_extra,
                "planned Phase-1 global state bytes");
            const size_t floor = checked_add_size(
                global_state, x_new + ((streamed || generated) ? 0 : index),
                "planned Phase-1 fixed bytes");
            const size_t generated_fixed = generated
                ? checked_add_size(y_new + yb, static_cast<size_t>(knn_fixed),
                    "generated exact-kNN fixed bytes")
                : 0;
            const auto budget = p1_bounded_budget(
                planned, maximum_cut,
                checked_add_size(floor, generated_fixed,
                    "planned generated Phase-1 fixed bytes"),
                streamed, free, total, *o,
                fused_hybrid, query_local_main_mask,
                generated ? static_cast<size_t>(knn_per_query) : 0);
            if (!budget.stable.BQ) continue;
            size_t scratch = 0;
            size_t knn_required = 0;
            if (generated) {
                scratch = checked_add_size(static_cast<size_t>(knn_fixed),
                    checked_mul_size(static_cast<size_t>(budget.stable.BQ),
                        static_cast<size_t>(knn_per_query),
                        "generated exact-kNN scratch"),
                    "generated exact-kNN scratch");
                knn_required = y_new + yb + scratch;
            } else {
                const size_t knn_other = y_new + yb
                    + (streamed ? 0 : index) + p1_allocation_margin;
                if (knn_other >= usable ||
                        knn_fixed + knn_per_query > usable - knn_other)
                    continue;
                // Bound by actual useful rows as well as the preferred scratch
                // ceiling; do not reserve 7 GiB for a tiny N request.
                scratch = std::min<uint64_t>(o->knn_scratch_bytes,
                    std::min<uint64_t>(usable - knn_other,
                        knn_fixed + n * knn_per_query));
                if (scratch < knn_fixed + knn_per_query) continue;
                knn_required = knn_other + scratch;
            }
            *plan = {};
            plan->struct_size = sizeof(*plan); plan->extension_version = 1;
            plan->mode = mode; plan->query_block = budget.stable.BQ;
            plan->free_bytes = free; plan->total_bytes = total; plan->usable_bytes = usable;
            plan->full_index_bytes = index;
            plan->index_device_bytes = generated ? 0
                : streamed ? p1_staging_bytes(r, plan->query_block) : index;
            plan->pinned_bytes = streamed ? plan->index_device_bytes : 0;
            plan->knn_scratch_bytes = scratch;
            plan->knn_required_bytes = knn_required;
            plan->phase1_required_bytes = budget.stable.required_bytes + p1_allocation_margin;
            plan->global_state_bytes = global_state;
            plan->estimated_peak_bytes = total - free
                + (generated ? plan->phase1_required_bytes
                    : std::max(plan->knn_required_bytes,
                        plan->phase1_required_bytes));
            plan->projected_retained_bytes = projected;
            return manor_m_cuda_internal_fail(MANOR_M_CUDA_SUCCESS, "");
        }
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Neither resident nor bounded-index P1 fits the measured GPU/retained/pinned budgets; "
            "global model state cannot fit even with query-local neighborhood reconstruction.");
    } catch (const std::exception& error) {
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_INVALID_ARGUMENT, error.what());
    }
}

extern "C" int manor_m_cuda_plan_phase1(
    ManorMCudaContext* context, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    int32_t scales, int32_t cuts, int32_t maximum_cut,
    const ManorMCudaP1CapacityOptions* o, ManorMCudaP1CapacityPlan* plan)
{
    return phase1_plan_impl(
        context, x, response, n, d, my, mm, k, scales, cuts, maximum_cut,
        o, plan, false);
}

extern "C" int manor_m_cuda_plan_phase1_hybrid(
    ManorMCudaContext* context, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    int32_t scales, int32_t cuts, int32_t maximum_cut,
    const ManorMCudaP1CapacityOptions* o, ManorMCudaP1CapacityPlan* plan)
{
    return phase1_plan_impl(
        context, x, response, n, d, my, mm, k, scales, cuts, maximum_cut,
        o, plan, true);
}

template <typename Phase1Rank>
static int phase1_impl(
    ManorMCudaContext* context,
    const float* x_point_major,
    int64_t point_count,
    int32_t ambient_dimension,
    int32_t manifold_dimension,
    const int32_t* index_zero_based,
    int64_t active_anchor_count,
    int32_t neighbor_count,
    const int32_t* response_counts,
    int32_t scale_count,
    const int32_t* cuts_scale_major,
    int32_t cuts_per_scale,
    const ManorMCudaPhase1Options* options,
    ManorMCudaPhase1Output* output,
    ManorMCudaPhase1Stats* stats,
    bool fused_hybrid,
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* actual_plan,
    const float* generated_response = nullptr,
    int32_t generated_response_dimension = 0)
{
    try {
        phase1_require(context != nullptr,
            "Phase-1 context must not be null.");
        const bool generated_index = capacity && capacity->mode == 4;
        Phase1CoreRequest request = phase1_validate(
            x_point_major, point_count, ambient_dimension,
            manifold_dimension, index_zero_based, active_anchor_count,
            neighbor_count, response_counts, scale_count, cuts_scale_major,
            cuts_per_scale, options, output, generated_index);
        phase1_require(!generated_index ||
                (generated_response && generated_response_dimension >= 1 &&
                 generated_response_dimension <= 8),
            "Generated exact-kNN Phase 1 requires response[N,MY].");
        if (fused_hybrid) {
            phase1_require(request.S == 1,
                "Fused hybrid Phase 1 currently requires one response scale.");
            phase1_require(request.options.selection_mode == STABLE_POSITIVE_H,
                "Fused hybrid Phase 1 requires primary selection_mode=0.");
            request.options.selection_mode = HYBRID_STABLE_AND_TOP5;
        }

        int32_t device_ordinal = 0;
        int status = manor_m_cuda_internal_context_device(
            context, &device_ordinal);
        if (status != MANOR_M_CUDA_SUCCESS) return status;
        STABLE_CUDA_CHECK(cudaSetDevice(device_ordinal));
        ManorMResidentCallScope resident_scope(context);
        /* The extracted production kernels use the legacy default stream.
           Drain the context's prior work before entering that exact ordering. */
        STABLE_CUDA_CHECK(cudaDeviceSynchronize());

        if (stats) {
            std::memset(stats, 0, sizeof(*stats));
            stats->struct_size = sizeof(*stats);
            stats->abi_version = MANOR_M_CUDA_ABI_VERSION;
            stats->device_ordinal = device_ordinal;
            stats->ambient_dimension = request.D;
            stats->manifold_dimension = request.MM;
            stats->point_count = request.N;
            stats->active_anchor_count = request.anchors;
            stats->scale_count = request.S;
        }

        const auto total_start = std::chrono::steady_clock::now();
        StableProfileSpans profile;
        profile.enabled = request.options.memory.profile_timers || stats;

        std::vector<int32_t> tmax_host(request.S);
        int tmax = 0;
        for (int scale = 0; scale < request.S; ++scale) {
            tmax_host[scale] = cuts_scale_major[
                static_cast<size_t>(scale) * request.R + request.R - 1];
            tmax = std::max(tmax, static_cast<int>(tmax_host[scale]));
        }

        const size_t selection_shared_bytes =
            compute_selection_shared_bytes(
                request.D, request.MM, request.options.window);
        const size_t final_extract_shared_bytes =
            compute_final_extract_shared_bytes(request.D, request.MM);
        const size_t owner_aggregate_shared_bytes =
            compute_owner_aggregate_shared_bytes(request.D);
        const size_t owner_finalize_shared_bytes =
            compute_owner_finalize_shared_bytes(request.D, request.MM);
        cudaDeviceProp properties{};
        STABLE_CUDA_CHECK(cudaGetDeviceProperties(
            &properties, device_ordinal));
        phase1_require(selection_shared_bytes <=
            static_cast<size_t>(properties.sharedMemPerBlockOptin),
            "Phase-1 Stable-H selector exceeds opt-in shared memory.");
        STABLE_CUDA_CHECK(cudaFuncSetAttribute(
            stable_h_select_kernel<Phase1Rank>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(selection_shared_bytes)));
        STABLE_CUDA_CHECK(cudaFuncSetCacheConfig(
            stable_h_select_kernel<Phase1Rank>, cudaFuncCachePreferShared));

        const size_t x_bytes = size_t(request.N) * request.D * sizeof(float);
        const size_t full_index_bytes = size_t(request.options.idx_shared_slice ? 1 : request.anchors)
            * request.kSlice * sizeof(int32_t);
        const bool cached_x = manor_m_resident_find(x_point_major, x_bytes) != nullptr;
        const bool cached_index = index_zero_based &&
            manor_m_resident_find(index_zero_based, full_index_bytes) != nullptr;
        const bool streamed = capacity &&
            (capacity->mode == 2 || capacity->mode == 3) &&
            !request.options.idx_shared_slice;
        const bool query_local_main_mask = capacity &&
            (capacity->mode == 3 || capacity->mode == 4);
        AdaptiveMemoryBudget memory_budget{};
        if (capacity) {
            p1_check_capacity_options(capacity);
            phase1_require(capacity->mode != 0, "P1 execution requires the preflight's explicit mode.");
            phase1_require(!(streamed || generated_index) || !cached_index,
                "Release the full device index before bounded/generated P1.");
            size_t free = 0, total = 0;
            STABLE_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            size_t fixed = p1_global_bytes(request, query_local_main_mask)
                + (fused_hybrid ? phase1_hybrid_extra_global_bytes(request) : 0)
                + (cached_x ? 0 : x_bytes)
                + (streamed || generated_index || cached_index
                    ? 0 : full_index_bytes);
            uint64_t generated_knn_fixed = 0;
            uint64_t generated_knn_per_query = 0;
            size_t generated_response_bytes = 0;
            bool cached_response = false;
            if (generated_index) {
                generated_response_bytes = static_cast<size_t>(request.N)
                    * generated_response_dimension * sizeof(float);
                cached_response = manor_m_resident_find(
                    generated_response, generated_response_bytes) != nullptr;
                manor_m_cuda_internal_knn_memory(
                    request.N, generated_response_dimension, request.kSlice,
                    true, &generated_knn_fixed,
                    &generated_knn_per_query);
                fixed = checked_add_size(fixed,
                    static_cast<size_t>(generated_knn_fixed)
                        + generated_response_bytes
                        + (cached_response ? 0 : generated_response_bytes),
                    "generated exact-kNN fixed bytes");
            }
            memory_budget = p1_bounded_budget(
                request, tmax, fixed, streamed, free, total, *capacity,
                fused_hybrid, query_local_main_mask,
                generated_index
                    ? static_cast<size_t>(generated_knn_per_query) : 0);
            phase1_require(memory_budget.stable.BQ >= request.options.memory.min_query_block,
                "P1 global state and minimum useful query block exceed the available budget.");
            if (actual_plan) {
                *actual_plan = {};
                actual_plan->struct_size = sizeof(*actual_plan); actual_plan->extension_version = 1;
                actual_plan->mode = capacity->mode;
                actual_plan->query_block = memory_budget.stable.BQ;
                actual_plan->free_bytes = free; actual_plan->total_bytes = total;
                actual_plan->usable_bytes = p1_usable(free, total, *capacity);
                actual_plan->full_index_bytes = full_index_bytes;
                actual_plan->index_device_bytes = generated_index ? 0
                    : streamed
                        ? p1_staging_bytes(request, actual_plan->query_block)
                        : full_index_bytes;
                actual_plan->pinned_bytes = streamed ? actual_plan->index_device_bytes : 0;
                actual_plan->global_state_bytes = p1_global_bytes(
                        request, query_local_main_mask)
                    + (fused_hybrid
                        ? phase1_hybrid_extra_global_bytes(request) : 0);
                actual_plan->phase1_required_bytes = memory_budget.stable.required_bytes + p1_allocation_margin;
                if (generated_index) {
                    actual_plan->knn_scratch_bytes = checked_add_size(
                        static_cast<size_t>(generated_knn_fixed),
                        checked_mul_size(
                            static_cast<size_t>(actual_plan->query_block),
                            static_cast<size_t>(generated_knn_per_query),
                            "generated exact-kNN scratch"),
                        "generated exact-kNN scratch");
                    actual_plan->knn_required_bytes =
                        generated_response_bytes
                        + (cached_response ? 0 : generated_response_bytes)
                        + actual_plan->knn_scratch_bytes;
                }
                actual_plan->estimated_peak_bytes = total - free + actual_plan->phase1_required_bytes;
            }
        } else {
            memory_budget = adaptive_choose_query_block(
                request.D, request.MM, request.N, request.anchors,
                request.kSlice, tmax, request.R,
                request.options.idx_shared_slice, request.options.memory);
            if (fused_hybrid)
                memory_budget = phase1_add_hybrid_budget(
                    memory_budget, request, tmax);
        }
        const int BQ = memory_budget.stable.BQ;
        const int max_items_int = checked_product_int(
            BQ, request.kSlice, "BQ*kSlice");
        const int max_top_int = checked_product_int(BQ, tmax, "BQ*tmax");
        const size_t max_items = static_cast<size_t>(max_items_int);
        const size_t max_top = static_cast<size_t>(max_top_int);
        const size_t basis_count = static_cast<size_t>(request.D) *
            request.MM * request.N;
        const size_t x_count = static_cast<size_t>(request.D) * request.N;
        const int mask_words = static_cast<int>(stable_mask_words(
            request.kSlice));
        const bool bounded_temporaries = capacity != nullptr;
        const size_t temporary_count = bounded_temporaries
            ? static_cast<size_t>(BQ) : static_cast<size_t>(request.N);
        const size_t main_mask_count = query_local_main_mask
            ? static_cast<size_t>(BQ) * mask_words
            : static_cast<size_t>(request.anchors) * mask_words;
        const size_t fallback_mask_count = bounded_temporaries
            ? static_cast<size_t>(BQ) * mask_words
            : static_cast<size_t>(request.anchors) * mask_words;
        const size_t index_count = request.options.idx_shared_slice
            ? static_cast<size_t>(request.kSlice)
            : static_cast<size_t>(request.anchors) * request.kSlice;

        StableCudaBuffer<float> X_resource, keys_in_resource, keys_out_resource;
        StableCudaBuffer<float> top_keys_resource, tangent_resource;
        StableCudaBuffer<float> center_resource, radius_resource, axial_resource;
        StableCudaBuffer<float> neighborhood_radius_limit_resource;
        StableCudaBuffer<float> aggregated_tangent_resource;
        StableCudaBuffer<float> aggregated_center_resource;
        StableCudaBuffer<float> aggregate_covariance_resource;
        StableCudaBuffer<float> aggregate_center_sum_resource;
        StableCudaBuffer<int> idx_resource, cuts_resource, offsets_resource;
        StableCudaBuffer<float> h_null_resource;
        StableCudaBuffer<int32_t> response_counts_resource, tmax_resource;
        StableCudaBuffer<Phase1Rank> values_in_resource, values_out_resource;
        StableCudaBuffer<Phase1Rank> top_values_resource;
        StableCudaBuffer<uint8_t> temporary_covered_resource;
        StableCudaBuffer<uint8_t> selected_covered_resource, owner_resource;
        StableCudaBuffer<uint8_t> rejection_resource;
        StableCudaBuffer<uint32_t> temporary_cut_resource;
        StableCudaBuffer<uint32_t> selected_cut_resource;
        StableCudaBuffer<int32_t> temporary_index_resource;
        StableCudaBuffer<int32_t> selected_scale_resource;
        StableCudaBuffer<int32_t> selected_index_resource;
        StableCudaBuffer<int32_t> num_neigh_resource;
        StableCudaBuffer<int32_t> aggregate_count_resource;
        StableCudaBuffer<uint32_t> neighborhood_mask_resource;
        StableCudaBuffer<unsigned long long> owner_stats_resource;
        StableCudaBuffer<double> support_threshold_resource;
        StableCudaBuffer<unsigned char> sort_temp_resource;
        StableCudaBuffer<float> temporary_geom_resource;
        StableCudaBuffer<float> temporary_stable_h_resource;
        StableCudaBuffer<float> temporary_neighbor_resource;
        StableCudaBuffer<float> temporary_split_resource;
        StableCudaBuffer<float> temporary_h_a_resource;
        StableCudaBuffer<float> temporary_h_b_resource;
        StableCudaBuffer<float> selected_geom_resource;
        StableCudaBuffer<float> selected_stable_h_resource;
        StableCudaBuffer<float> selected_neighbor_resource;
        StableCudaBuffer<float> selected_split_resource;
        StableCudaBuffer<float> selected_h_a_resource;
        StableCudaBuffer<float> selected_h_b_resource;
        StableCudaBuffer<float> fallback_temporary_geom_resource;
        StableCudaBuffer<float> fallback_temporary_stable_h_resource;
        StableCudaBuffer<float> fallback_temporary_neighbor_resource;
        StableCudaBuffer<float> fallback_temporary_split_resource;
        StableCudaBuffer<float> fallback_temporary_h_a_resource;
        StableCudaBuffer<float> fallback_temporary_h_b_resource;
        StableCudaBuffer<float> fallback_selected_geom_resource;
        StableCudaBuffer<float> fallback_selected_stable_h_resource;
        StableCudaBuffer<float> fallback_selected_neighbor_resource;
        StableCudaBuffer<float> fallback_selected_split_resource;
        StableCudaBuffer<float> fallback_selected_h_a_resource;
        StableCudaBuffer<float> fallback_selected_h_b_resource;
        StableCudaBuffer<int32_t> fallback_temporary_index_resource;
        StableCudaBuffer<uint32_t> fallback_temporary_cut_resource;
        StableCudaBuffer<uint8_t> fallback_temporary_covered_resource;
        StableCudaBuffer<int32_t> fallback_selected_scale_resource;
        StableCudaBuffer<int32_t> fallback_selected_index_resource;
        StableCudaBuffer<uint32_t> fallback_selected_cut_resource;
        StableCudaBuffer<uint8_t> fallback_selected_covered_resource;
        StableCudaBuffer<float> fallback_tangent_resource;
        StableCudaBuffer<float> fallback_center_resource;
        StableCudaBuffer<float> fallback_radius_resource;
        StableCudaBuffer<float> fallback_axial_resource;
        StableCudaBuffer<int32_t> fallback_num_neigh_resource;
        StableCudaBuffer<uint32_t> fallback_neighborhood_mask_resource;
        StableCudaBuffer<uint8_t> fallback_owner_resource;
        StableCudaBuffer<uint8_t> fallback_rejection_resource;
        StableCudaBuffer<unsigned long long> fallback_owner_stats_resource;
        StableCudaBuffer<double> fallback_support_threshold_resource;

        float* d_X = static_cast<float*>(manor_m_resident_find(
            x_point_major, x_count * sizeof(float)));
        int* d_idx = static_cast<int*>(manor_m_resident_find(
            index_zero_based, index_count * sizeof(int32_t)));
        if (!d_X) d_X = X_resource.allocate(x_count);
        if (!d_idx && !streamed && !generated_index)
            d_idx = idx_resource.allocate(index_count);
        std::unique_ptr<Phase1IndexStream> index_stream;
        if (streamed) {
            index_stream.reset(new Phase1IndexStream(request.anchors, request.kSlice, BQ));
            index_stream->allocate();
        }
        int* d_cuts = cuts_resource.allocate(
            static_cast<size_t>(request.R) * request.S);
        int32_t* d_response_counts = response_counts_resource.allocate(request.S);
        int32_t* d_tmax_by_scale = tmax_resource.allocate(request.S);
        float* d_keys_in = keys_in_resource.allocate(max_items);
        float* d_keys_out = keys_out_resource.allocate(max_items);
        Phase1Rank* d_values_in = values_in_resource.allocate(max_items);
        Phase1Rank* d_values_out = values_out_resource.allocate(max_items);
        float* d_top_keys = top_keys_resource.allocate(max_top);
        Phase1Rank* d_top_values = top_values_resource.allocate(max_top);
        int* d_offsets = offsets_resource.allocate(static_cast<size_t>(BQ) + 1);

        float* d_temporary_geom = temporary_geom_resource.allocate(temporary_count);
        float* d_temporary_stable_h =
            temporary_stable_h_resource.allocate(temporary_count);
        float* d_temporary_neighbor =
            temporary_neighbor_resource.allocate(temporary_count);
        float* d_temporary_split = temporary_split_resource.allocate(temporary_count);
        float* d_temporary_h_a = temporary_h_a_resource.allocate(temporary_count);
        float* d_temporary_h_b = temporary_h_b_resource.allocate(temporary_count);
        float* d_selected_geom = selected_geom_resource.allocate(request.N);
        float* d_selected_stable_h =
            selected_stable_h_resource.allocate(request.N);
        float* d_selected_neighbor =
            selected_neighbor_resource.allocate(request.N);
        float* d_selected_split = selected_split_resource.allocate(request.N);
        float* d_selected_h_a = selected_h_a_resource.allocate(request.N);
        float* d_selected_h_b = selected_h_b_resource.allocate(request.N);
        int32_t* d_temporary_index =
            temporary_index_resource.allocate(temporary_count);
        uint32_t* d_temporary_cut = temporary_cut_resource.allocate(temporary_count);
        uint8_t* d_temporary_covered =
            temporary_covered_resource.allocate(temporary_count);
        int32_t* d_selected_scale = selected_scale_resource.allocate(request.N);
        int32_t* d_selected_index = selected_index_resource.allocate(request.N);
        uint32_t* d_selected_cut = selected_cut_resource.allocate(request.N);
        uint8_t* d_selected_covered =
            selected_covered_resource.allocate(request.N);
        float* d_tangent = tangent_resource.allocate(basis_count);
        float* d_center = center_resource.allocate(x_count);
        float* d_radius = radius_resource.allocate(request.N);
        float* d_axial = axial_resource.allocate(request.N);
        int32_t* d_num_neigh = num_neigh_resource.allocate(request.N);
        uint32_t* d_neighborhood_mask = neighborhood_mask_resource.allocate(
            main_mask_count);
        float* d_neighborhood_radius_limit = query_local_main_mask
            ? neighborhood_radius_limit_resource.allocate(request.anchors)
            : nullptr;
        uint8_t* d_owner = owner_resource.allocate(request.N);
        uint8_t* d_rejection = rejection_resource.allocate(request.N);
        unsigned long long* d_owner_stats = owner_stats_resource.allocate(3);
        double* d_support_threshold = support_threshold_resource.allocate(1);
        float* d_aggregated_tangent =
            aggregated_tangent_resource.allocate(basis_count);
        float* d_aggregated_center =
            aggregated_center_resource.allocate(x_count);
        float* d_aggregate_covariance = aggregate_covariance_resource.allocate(
            static_cast<size_t>(BQ) * request.D * request.D);
        float* d_aggregate_center_sum = aggregate_center_sum_resource.allocate(
            static_cast<size_t>(BQ) * request.D);
        int32_t* d_aggregate_count = aggregate_count_resource.allocate(BQ);
        float* d_fallback_temporary_geom = fused_hybrid ?
            fallback_temporary_geom_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_temporary_stable_h = fused_hybrid ?
            fallback_temporary_stable_h_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_temporary_neighbor = fused_hybrid ?
            fallback_temporary_neighbor_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_temporary_split = fused_hybrid ?
            fallback_temporary_split_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_temporary_h_a = fused_hybrid ?
            fallback_temporary_h_a_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_temporary_h_b = fused_hybrid ?
            fallback_temporary_h_b_resource.allocate(temporary_count) : nullptr;
        float* d_fallback_selected_geom = fused_hybrid ?
            fallback_selected_geom_resource.allocate(request.N) : nullptr;
        float* d_fallback_selected_stable_h = fused_hybrid ?
            fallback_selected_stable_h_resource.allocate(request.N) : nullptr;
        float* d_fallback_selected_neighbor = fused_hybrid ?
            fallback_selected_neighbor_resource.allocate(request.N) : nullptr;
        float* d_fallback_selected_split = fused_hybrid ?
            fallback_selected_split_resource.allocate(request.N) : nullptr;
        float* d_fallback_selected_h_a = fused_hybrid ?
            fallback_selected_h_a_resource.allocate(request.N) : nullptr;
        float* d_fallback_selected_h_b = fused_hybrid ?
            fallback_selected_h_b_resource.allocate(request.N) : nullptr;
        int32_t* d_fallback_temporary_index = fused_hybrid ?
            fallback_temporary_index_resource.allocate(temporary_count) : nullptr;
        uint32_t* d_fallback_temporary_cut = fused_hybrid ?
            fallback_temporary_cut_resource.allocate(temporary_count) : nullptr;
        uint8_t* d_fallback_temporary_covered = fused_hybrid ?
            fallback_temporary_covered_resource.allocate(temporary_count) : nullptr;
        int32_t* d_fallback_selected_scale = fused_hybrid ?
            fallback_selected_scale_resource.allocate(request.N) : nullptr;
        int32_t* d_fallback_selected_index = fused_hybrid ?
            fallback_selected_index_resource.allocate(request.N) : nullptr;
        uint32_t* d_fallback_selected_cut = fused_hybrid ?
            fallback_selected_cut_resource.allocate(request.N) : nullptr;
        uint8_t* d_fallback_selected_covered = fused_hybrid ?
            fallback_selected_covered_resource.allocate(request.N) : nullptr;
        float* d_fallback_tangent = fused_hybrid ?
            fallback_tangent_resource.allocate(basis_count) : nullptr;
        float* d_fallback_center = fused_hybrid ?
            fallback_center_resource.allocate(x_count) : nullptr;
        float* d_fallback_radius = fused_hybrid ?
            fallback_radius_resource.allocate(request.N) : nullptr;
        float* d_fallback_axial = fused_hybrid ?
            fallback_axial_resource.allocate(request.N) : nullptr;
        int32_t* d_fallback_num_neigh = fused_hybrid ?
            fallback_num_neigh_resource.allocate(request.N) : nullptr;
        uint32_t* d_fallback_neighborhood_mask = fused_hybrid ?
            fallback_neighborhood_mask_resource.allocate(fallback_mask_count) : nullptr;
        uint8_t* d_fallback_owner = fused_hybrid ?
            fallback_owner_resource.allocate(request.N) : nullptr;
        uint8_t* d_fallback_rejection = fused_hybrid ?
            fallback_rejection_resource.allocate(request.N) : nullptr;
        unsigned long long* d_fallback_owner_stats = fused_hybrid ?
            fallback_owner_stats_resource.allocate(3) : nullptr;
        double* d_fallback_support_threshold = fused_hybrid ?
            fallback_support_threshold_resource.allocate(1) : nullptr;

        profile.begin("upload");
        STABLE_CUDA_CHECK(manor_m_resident_upload(
            d_X, x_point_major, x_count * sizeof(float)));
        if (!streamed && !generated_index) STABLE_CUDA_CHECK(manor_m_resident_upload(
            d_idx, index_zero_based, index_count * sizeof(int32_t)));
        STABLE_CUDA_CHECK(cudaMemcpy(
            d_cuts, cuts_scale_major,
            static_cast<size_t>(request.R) * request.S * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        float* d_h_null = nullptr;
        if (request.options.h_null_correction) {
            std::vector<float> h_null_host(
                static_cast<size_t>(request.R) * request.S);
            h_null_fill_offsets(
                request.D, request.MM, cuts_scale_major,
                request.R * request.S, h_null_host.data(),
                "manor_m_cuda:phase1HNullCalibration");
            d_h_null = h_null_resource.allocate(h_null_host.size());
            STABLE_CUDA_CHECK(cudaMemcpy(
                d_h_null, h_null_host.data(),
                h_null_host.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        STABLE_CUDA_CHECK(cudaMemcpy(
            d_response_counts, response_counts,
            static_cast<size_t>(request.S) * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        STABLE_CUDA_CHECK(cudaMemcpy(
            d_tmax_by_scale, tmax_host.data(),
            static_cast<size_t>(request.S) * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        STABLE_CUDA_CHECK(cudaMemset(
            d_owner_stats, 0, 3 * sizeof(unsigned long long)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_scale, 0xff, request.N * sizeof(int32_t)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_index, 0xff, request.N * sizeof(int32_t)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_cut, 0, request.N * sizeof(uint32_t)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_covered, 0, request.N * sizeof(uint8_t)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_geom, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_stable_h, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_neighbor, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_split, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_h_a, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_selected_h_b, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_tangent, 0, basis_count * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(d_center, 0, x_count * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_radius, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_axial, 0, request.N * sizeof(float)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_num_neigh, 0, request.N * sizeof(int32_t)));
        STABLE_CUDA_CHECK(cudaMemset(
            d_neighborhood_mask, 0,
            main_mask_count * sizeof(uint32_t)));
        if (query_local_main_mask) {
            STABLE_CUDA_CHECK(cudaMemset(
                d_neighborhood_radius_limit, 0,
                static_cast<size_t>(request.anchors) * sizeof(float)));
        }
        if (fused_hybrid) {
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_owner_stats, 0,
                3 * sizeof(unsigned long long)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_scale, 0xff,
                request.N * sizeof(int32_t)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_index, 0xff,
                request.N * sizeof(int32_t)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_cut, 0,
                request.N * sizeof(uint32_t)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_covered, 0,
                request.N * sizeof(uint8_t)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_geom, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_stable_h, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_neighbor, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_split, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_h_a, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_selected_h_b, 0,
                request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_tangent, 0, basis_count * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_center, 0, x_count * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_radius, 0, request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_axial, 0, request.N * sizeof(float)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_num_neigh, 0,
                request.N * sizeof(int32_t)));
            STABLE_CUDA_CHECK(cudaMemset(
                d_fallback_neighborhood_mask, 0,
                fallback_mask_count * sizeof(uint32_t)));
        }
        std::vector<int> offsets_host(static_cast<size_t>(BQ) + 1);
        for (int block = 0; block <= BQ; ++block)
            offsets_host[block] = block * request.kSlice;
        STABLE_CUDA_CHECK(cudaMemcpy(
            d_offsets, offsets_host.data(),
            (static_cast<size_t>(BQ) + 1) * sizeof(int),
            cudaMemcpyHostToDevice));
        profile.end();

        size_t sort_temp_bytes = 0;
        STABLE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, sort_temp_bytes, d_keys_in, d_keys_out,
            d_values_in, d_values_out, max_items_int, BQ,
            d_offsets, d_offsets + 1));
        void* d_sort_temp = sort_temp_resource.allocate(sort_temp_bytes);
        phase1_require(sort_temp_bytes ==
            memory_budget.stable.selected_query.sort_temp_bytes,
            "CUB sort workspace differs from the Phase-1 memory budget.");

        constexpr int threads = 256;
        auto run_selection_tile = [&](const int32_t* tile_index,
                                      int q0, int B, int supplied_k) {
            phase1_require(supplied_k == request.kSlice,
                "Generated exact-kNN tile has the wrong neighbor count.");
            phase1_require(B >= 1 && B <= BQ && q0 >= 0 &&
                    q0 + B <= request.anchors,
                "Generated exact-kNN tile lies outside the planned P1 block.");
            const int* d_idx = tile_index;
            const int index_base = streamed || generated_index ? q0 : 0;
            const int item_count = checked_product_int(
                B, request.kSlice, "B*kSlice");
            const int selector_blocks = (B + threads - 1) / threads;
            const dim3 distance_grid(
                (request.kSlice + threads - 1) / threads, B);
            profile.begin("dist_idx");
            phase1_dist_idx_kernel<<<distance_grid, threads,
                sizeof(float) * request.D>>>(
                d_X, d_idx, request.D, request.N, q0, B,
                request.kSlice, request.options.idx_shared_slice,
                d_keys_in, d_values_in, index_base);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            profile.end();

            profile.begin("sort");
            STABLE_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
                d_sort_temp, sort_temp_bytes, d_keys_in, d_keys_out,
                d_values_in, d_values_out, item_count, B,
                d_offsets, d_offsets + 1));
            profile.end();

            for (int scale = 0; scale < request.S; ++scale) {
                const int scale_tmax = tmax_host[scale];
                profile.begin("scale_topk");
                adaptive_compact_nested_topk_kernel<<<B, 32>>>(
                    d_keys_out, d_values_out, B, request.kSlice,
                    response_counts[scale], scale_tmax, scale_tmax,
                    d_top_keys, d_top_values);
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
                profile.end();

                profile.begin("scale_select");
                stable_h_select_kernel<<<B, 32, selection_shared_bytes>>>(
                    d_X, d_idx, d_top_values, d_cuts + scale * request.R,
                    request.D, request.MM, request.kSlice, scale_tmax,
                    request.R, request.options.idx_shared_slice, q0, B,
                    request.options.jitter, request.options.window,
                    request.options.neighbor_scaled_max,
                    request.options.split_scaled_max,
                    request.options.split_seed,
                    request.options.selection_mode,
                    d_h_null ? d_h_null + scale * request.R : nullptr,
                    d_temporary_index, d_temporary_cut,
                    d_temporary_covered, d_temporary_geom,
                    d_temporary_stable_h, d_temporary_neighbor,
                    d_temporary_split, d_temporary_h_a, d_temporary_h_b,
                    d_fallback_temporary_index, d_fallback_temporary_cut,
                    d_fallback_temporary_covered, d_fallback_temporary_geom,
                    d_fallback_temporary_stable_h,
                    d_fallback_temporary_neighbor,
                    d_fallback_temporary_split, d_fallback_temporary_h_a,
                    d_fallback_temporary_h_b, index_base,
                    bounded_temporaries ? q0 : 0);
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
                profile.end();

                profile.begin("scale_stream");
                adaptive_stream_best_kernel<<<selector_blocks, threads>>>(
                    scale, q0, B, d_temporary_index, d_temporary_cut,
                    d_temporary_covered, d_temporary_geom,
                    d_temporary_stable_h, d_temporary_neighbor,
                    d_temporary_split, d_temporary_h_a, d_temporary_h_b,
                    d_selected_scale, d_selected_index, d_selected_cut,
                    d_selected_covered, d_selected_geom,
                    d_selected_stable_h, d_selected_neighbor,
                    d_selected_split, d_selected_h_a, d_selected_h_b,
                    bounded_temporaries ? q0 : 0);
                /* Bounded calls recycle the temporary record at every query
                   block; unbounded compatibility calls keep global indexing. */
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
                if (fused_hybrid) {
                    adaptive_stream_best_kernel<<<selector_blocks, threads>>>(
                        scale, q0, B,
                        d_fallback_temporary_index,
                        d_fallback_temporary_cut,
                        d_fallback_temporary_covered,
                        d_fallback_temporary_geom,
                        d_fallback_temporary_stable_h,
                        d_fallback_temporary_neighbor,
                        d_fallback_temporary_split,
                        d_fallback_temporary_h_a,
                        d_fallback_temporary_h_b,
                        d_fallback_selected_scale,
                        d_fallback_selected_index,
                        d_fallback_selected_cut,
                        d_fallback_selected_covered,
                        d_fallback_selected_geom,
                        d_fallback_selected_stable_h,
                        d_fallback_selected_neighbor,
                        d_fallback_selected_split,
                        d_fallback_selected_h_a,
                        d_fallback_selected_h_b,
                        bounded_temporaries ? q0 : 0);
                    STABLE_CUDA_CHECK(cudaPeekAtLastError());
                }
                profile.end();
            }

            profile.begin("final_extract");
            adaptive_compact_selected_topk_kernel<<<B, 32>>>(
                d_keys_out, d_values_out, d_selected_scale,
                d_response_counts, d_tmax_by_scale, q0, B,
                request.kSlice, tmax, d_top_keys, d_top_values);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            adaptive_final_extract_kernel<<<B, 32,
                final_extract_shared_bytes>>>(
                d_X, d_idx, d_top_keys, d_top_values, d_cuts,
                d_response_counts, d_selected_scale, d_selected_index,
                d_selected_geom, request.R, request.D, request.MM,
                request.kSlice, tmax, q0, B,
                request.options.idx_shared_slice, request.options.jitter,
                d_tangent, d_center, d_radius, d_axial, d_num_neigh,
                d_neighborhood_mask, mask_words, index_base,
                query_local_main_mask ? q0 : 0);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            if (query_local_main_mask) {
                const int radius_threads = 256;
                phase1_capture_radius_limit_kernel<<<
                    (B + radius_threads - 1) / radius_threads,
                    radius_threads>>>(
                    d_top_keys, d_selected_cut, q0, B, tmax,
                    d_neighborhood_radius_limit);
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
            }
            if (fused_hybrid) {
                adaptive_compact_selected_topk_kernel<<<B, 32>>>(
                    d_keys_out, d_values_out, d_fallback_selected_scale,
                    d_response_counts, d_tmax_by_scale, q0, B,
                    request.kSlice, tmax, d_top_keys, d_top_values);
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
                adaptive_final_extract_kernel<<<B, 32,
                    final_extract_shared_bytes>>>(
                    d_X, d_idx, d_top_keys, d_top_values, d_cuts,
                    d_response_counts, d_fallback_selected_scale,
                    d_fallback_selected_index, d_fallback_selected_geom,
                    request.R, request.D, request.MM,
                    request.kSlice, tmax, q0, B,
                    request.options.idx_shared_slice, request.options.jitter,
                    d_fallback_tangent, d_fallback_center,
                    d_fallback_radius, d_fallback_axial,
                    d_fallback_num_neigh, d_fallback_neighborhood_mask,
                    mask_words, index_base,
                    bounded_temporaries ? q0 : 0);
                STABLE_CUDA_CHECK(cudaPeekAtLastError());
            }
            profile.end();
        };

        auto run_generated_tiles = [&](auto& consumer) {
            uint64_t knn_fixed = 0, knn_per_query = 0;
            manor_m_cuda_internal_knn_memory(
                request.N, generated_response_dimension, request.kSlice,
                true, &knn_fixed, &knn_per_query);
            const uint64_t scratch = knn_fixed
                + static_cast<uint64_t>(BQ) * knn_per_query;
            phase1_require(capacity && scratch <= capacity->knn_scratch_bytes,
                "Generated exact-kNN scratch exceeds the planned ceiling.");
            ManorMCudaResponseKnnOptions knn_options{};
            knn_options.struct_size = sizeof(knn_options);
            knn_options.abi_version = MANOR_M_CUDA_ABI_VERSION;
            knn_options.scratch_memory_bytes = scratch;
            knn_options.max_retry_rounds = 12;
            using Consumer = std::remove_reference_t<decltype(consumer)>;
            const int generated_status = manor_m_cuda_internal_response_knn_tiles(
                context, generated_response, request.N,
                generated_response_dimension, 0, request.anchors,
                request.kSlice, &knn_options, nullptr,
                &phase1_consume_generated_knn_tile<Consumer>, &consumer);
            phase1_require(generated_status == MANOR_M_CUDA_SUCCESS,
                manor_m_cuda_last_error());
        };

        if (generated_index) {
            run_generated_tiles(run_selection_tile);
        } else {
            for (int q0 = 0; q0 < request.anchors; q0 += BQ) {
                const int B = std::min(BQ, request.anchors - q0);
                if (streamed)
                    d_idx = index_stream->prepare(index_zero_based, q0);
                run_selection_tile(d_idx, q0, B, request.kSlice);
                if (streamed) index_stream->consumed();
            }
        }

        const int blocks = (request.N + threads - 1) / threads;
        profile.begin("owner_rule");
        adaptive_owner_stats_kernel<<<blocks, threads>>>(
            d_selected_covered, d_num_neigh, request.N, d_owner_stats);
        adaptive_owner_threshold_kernel<<<1, 1>>>(
            d_owner_stats, request.D, request.options.heavy_factor,
            d_support_threshold);
        adaptive_owner_mask_kernel<<<blocks, threads>>>(
            d_selected_covered, d_num_neigh, d_support_threshold,
            request.N, d_owner, d_rejection, d_owner_stats);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        if (fused_hybrid) {
            adaptive_owner_stats_kernel<<<blocks, threads>>>(
                d_fallback_selected_covered, d_fallback_num_neigh,
                request.N, d_fallback_owner_stats);
            adaptive_owner_threshold_kernel<<<1, 1>>>(
                d_fallback_owner_stats, request.D,
                request.options.heavy_factor,
                d_fallback_support_threshold);
            adaptive_owner_mask_kernel<<<blocks, threads>>>(
                d_fallback_selected_covered, d_fallback_num_neigh,
                d_fallback_support_threshold, request.N,
                d_fallback_owner, d_fallback_rejection,
                d_fallback_owner_stats);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
        }
        profile.end();

        auto run_aggregation_tile = [&](const int32_t* tile_index,
                                        int q0, int B, int supplied_k) {
            phase1_require(supplied_k == request.kSlice,
                "Generated exact-kNN aggregation tile has the wrong neighbor count.");
            phase1_require(B >= 1 && B <= BQ && q0 >= 0 &&
                    q0 + B <= request.anchors,
                "Generated exact-kNN aggregation tile lies outside the planned P1 block.");
            const int* d_idx = tile_index;
            profile.begin("owner_aggregate");
            if (query_local_main_mask) {
                phase1_owner_aggregate_from_radius_kernel_mm<<<B, 32,
                    owner_aggregate_shared_bytes>>>(
                    d_neighborhood_radius_limit, d_selected_scale,
                    d_response_counts, d_owner, d_tangent, d_center, d_X,
                    d_idx, request.D, request.MM, request.N,
                    request.kSlice, request.options.idx_shared_slice, q0, B,
                    d_aggregate_covariance, d_aggregate_center_sum,
                    d_aggregate_count,
                    streamed || generated_index ? q0 : 0);
            } else {
                owner_aggregate_kernel_mm<<<B, 32,
                    owner_aggregate_shared_bytes>>>(
                    d_neighborhood_mask, d_owner, d_tangent, d_center, d_idx,
                    request.D, request.MM, request.N, request.kSlice,
                    mask_words, request.options.idx_shared_slice, q0, B,
                    d_aggregate_covariance, d_aggregate_center_sum,
                    d_aggregate_count,
                    streamed || generated_index ? q0 : 0);
            }
            adaptive_owner_finalize_kernel_mm<<<B, 32,
                owner_finalize_shared_bytes>>>(
                d_aggregate_covariance, d_aggregate_center_sum,
                d_aggregate_count, d_X, d_tangent, d_center, d_owner,
                request.D, request.MM, q0, B, d_support_threshold,
                d_aggregated_tangent, d_aggregated_center);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
            profile.end();
        };
        if (generated_index) {
            run_generated_tiles(run_aggregation_tile);
        } else {
            for (int q0 = 0; q0 < request.anchors; q0 += BQ) {
                const int B = std::min(BQ, request.anchors - q0);
                if (streamed)
                    d_idx = index_stream->prepare(index_zero_based, q0);
                run_aggregation_tile(d_idx, q0, B, request.kSlice);
                if (streamed) index_stream->consumed();
            }
        }

        profile.begin("mask_nonowners");
        mask_nonowners_kernel<<<blocks, threads>>>(
            d_owner, request.D, request.MM, request.N,
            d_aggregated_tangent, d_aggregated_center);
        STABLE_CUDA_CHECK(cudaPeekAtLastError());
        if (fused_hybrid) {
            const size_t tangent_elements = basis_count;
            const size_t center_elements = x_count;
            phase1_hybrid_merge_vector_kernel<<<
                static_cast<unsigned int>((tangent_elements + threads - 1) / threads),
                threads>>>(
                    d_aggregated_tangent, d_fallback_tangent,
                    d_owner, d_fallback_owner, request.N,
                    request.D * request.MM);
            phase1_hybrid_merge_vector_kernel<<<
                static_cast<unsigned int>((center_elements + threads - 1) / threads),
                threads>>>(
                    d_aggregated_center, d_fallback_center,
                    d_owner, d_fallback_owner, request.N, request.D);
            phase1_hybrid_merge_scalar_kernel<<<blocks, threads>>>(
                request.N, d_owner,
                d_radius, d_axial, d_num_neigh,
                d_selected_cut, d_selected_geom, d_selected_stable_h,
                d_selected_neighbor, d_selected_split,
                d_selected_h_a, d_selected_h_b,
                d_rejection, d_selected_index,
                d_fallback_owner,
                d_fallback_radius, d_fallback_axial,
                d_fallback_num_neigh, d_fallback_selected_cut,
                d_fallback_selected_geom, d_fallback_selected_stable_h,
                d_fallback_selected_neighbor, d_fallback_selected_split,
                d_fallback_selected_h_a, d_fallback_selected_h_b,
                d_fallback_rejection, d_fallback_selected_index);
            STABLE_CUDA_CHECK(cudaPeekAtLastError());
        }
        profile.end();
        STABLE_CUDA_CHECK(cudaDeviceSynchronize());

        const auto download_start = std::chrono::steady_clock::now();
        phase1_download(
            output->tangent_point_major, d_aggregated_tangent, basis_count);
        phase1_download(output->center_point_major, d_aggregated_center, x_count);
        phase1_download(output->radius, d_radius, request.N);
        phase1_download(output->axial_tangent, d_axial, request.N);
        phase1_download(output->neighbor_count, d_num_neigh, request.N);
        phase1_download(output->owner_mask, d_owner, request.N);
        phase1_download(
            output->stable_covered, d_selected_covered, request.N);
        phase1_download(output->selected_cut, d_selected_cut, request.N);
        phase1_download(output->geometry_score, d_selected_geom, request.N);
        phase1_download(output->stable_h, d_selected_stable_h, request.N);
        phase1_download(
            output->neighbor_scaled, d_selected_neighbor, request.N);
        phase1_download(output->split_scaled, d_selected_split, request.N);
        phase1_download(output->split_h_a, d_selected_h_a, request.N);
        phase1_download(output->split_h_b, d_selected_h_b, request.N);
        phase1_download(output->rejection_code, d_rejection, request.N);
        phase1_download(
            output->raw_tangent_point_major, d_tangent, basis_count);
        phase1_download(output->raw_center_point_major, d_center, x_count);

        std::vector<int32_t> selected_scale_host(request.N);
        std::vector<int32_t> selected_index_host;
        phase1_download(
            selected_scale_host.data(), d_selected_scale, request.N);
        if (output->selected_index_one_based) {
            selected_index_host.resize(request.N);
            phase1_download(
                selected_index_host.data(), d_selected_index, request.N);
        }
        for (int anchor = 0; anchor < request.N; ++anchor) {
            const int scale = selected_scale_host[anchor];
            if (output->selected_scale_one_based)
                output->selected_scale_one_based[anchor] =
                    scale >= 0 ? static_cast<uint32_t>(scale + 1) : 0u;
            if (output->selected_response_count)
                output->selected_response_count[anchor] =
                    scale >= 0 ? static_cast<uint32_t>(response_counts[scale]) : 0u;
            if (output->selected_index_one_based)
                output->selected_index_one_based[anchor] =
                    selected_index_host[anchor] >= 0
                        ? static_cast<uint32_t>(selected_index_host[anchor] + 1)
                        : 0u;
        }

        std::array<unsigned long long, 3> owner_stats_host{};
        double support_threshold_host = 0.0;
        STABLE_CUDA_CHECK(cudaMemcpy(
            owner_stats_host.data(), d_owner_stats,
            3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        STABLE_CUDA_CHECK(cudaMemcpy(
            &support_threshold_host, d_support_threshold, sizeof(double),
            cudaMemcpyDeviceToHost));
        uint64_t reported_owner_count = owner_stats_host[2];
        if (fused_hybrid) {
            reported_owner_count = 0;
            for (int anchor = 0; anchor < request.N; ++anchor)
                reported_owner_count += output->owner_mask[anchor] != 0;
        }
        const double download_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - download_start).count();

        const auto profile_totals = profile.totals();
        if (stats) {
            stats->query_block = BQ;
            stats->query_block_upper_bound =
                memory_budget.stable.upper_bound;
            stats->stable_covered_count = owner_stats_host[0];
            stats->owner_count = reported_owner_count;
            stats->support_threshold = support_threshold_host;
            stats->total_milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - total_start).count();
            stats->upload_milliseconds =
                phase1_profile_value(profile_totals, "upload");
            stats->distance_milliseconds =
                phase1_profile_value(profile_totals, "dist_idx");
            stats->sort_milliseconds =
                phase1_profile_value(profile_totals, "sort");
            stats->scale_topk_milliseconds =
                phase1_profile_value(profile_totals, "scale_topk");
            stats->scale_select_milliseconds =
                phase1_profile_value(profile_totals, "scale_select");
            stats->scale_stream_milliseconds =
                phase1_profile_value(profile_totals, "scale_stream");
            stats->final_extract_milliseconds =
                phase1_profile_value(profile_totals, "final_extract");
            stats->owner_rule_milliseconds =
                phase1_profile_value(profile_totals, "owner_rule");
            stats->owner_aggregate_milliseconds =
                phase1_profile_value(profile_totals, "owner_aggregate");
            stats->mask_nonowners_milliseconds =
                phase1_profile_value(profile_totals, "mask_nonowners");
            stats->download_milliseconds = download_ms;
            stats->memory_free_bytes = memory_budget.stable.free_bytes;
            stats->memory_total_bytes = memory_budget.stable.total_bytes;
            stats->memory_required_bytes = memory_budget.stable.required_bytes;
            stats->memory_next_required_bytes =
                memory_budget.stable.next_required_bytes;
        }
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_SUCCESS, "");
    } catch (const StableMexError& error) {
        const std::string identifier(error.identifier());
        /* Only legacy launch/runtime failures use the :cuda identifier.
           The public core namespace itself contains "cuda", so a broad
           substring test would misclassify ordinary argument errors. */
        const int status = identifier.find(":cuda") != std::string::npos
            ? MANOR_M_CUDA_CUDA_ERROR
            : MANOR_M_CUDA_INVALID_ARGUMENT;
        return manor_m_cuda_internal_fail(status, error.what());
    } catch (const std::exception& error) {
        return manor_m_cuda_internal_fail(
            MANOR_M_CUDA_INTERNAL_ERROR, error.what());
    } catch (...) {
        return manor_m_cuda_internal_fail(
            MANOR_M_CUDA_INTERNAL_ERROR,
            "Unknown exception in shared Phase 1.");
    }
}

extern "C" int manor_m_cuda_phase1_precomputed(
    ManorMCudaContext* c, const float* x, int64_t n, int32_t d, int32_t mm,
    const int32_t* index, int64_t anchors, int32_t k, const int32_t* counts,
    int32_t scales, const int32_t* cuts, int32_t cut_count,
    const ManorMCudaPhase1Options* o, ManorMCudaPhase1Output* out,
    ManorMCudaPhase1Stats* stats)
{
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, index, anchors, k,
            counts, scales, cuts, cut_count, o, out, stats, false, nullptr,
            nullptr);
    return phase1_impl<uint32_t>(c, x, n, d, mm, index, anchors, k, counts,
        scales, cuts, cut_count, o, out, stats, false, nullptr, nullptr);
}

extern "C" int manor_m_cuda_phase1_hybrid_precomputed(
    ManorMCudaContext* c, const float* x, int64_t n, int32_t d, int32_t mm,
    const int32_t* index, int64_t anchors, int32_t k, const int32_t* counts,
    int32_t scales, const int32_t* cuts, int32_t cut_count,
    const ManorMCudaPhase1Options* o, ManorMCudaPhase1Output* out,
    ManorMCudaPhase1Stats* stats)
{
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, index, anchors, k,
            counts, scales, cuts, cut_count, o, out, stats, true, nullptr,
            nullptr);
    return phase1_impl<uint32_t>(c, x, n, d, mm, index, anchors, k, counts,
        scales, cuts, cut_count, o, out, stats, true, nullptr, nullptr);
}

extern "C" int manor_m_cuda_phase1_bounded(
    ManorMCudaContext* c, const float* x, int64_t n, int32_t d, int32_t mm,
    const int32_t* index, int64_t anchors, int32_t k, const int32_t* counts,
    int32_t scales, const int32_t* cuts, int32_t cut_count,
    const ManorMCudaPhase1Options* o, ManorMCudaPhase1Output* out,
    ManorMCudaPhase1Stats* stats, const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan)
{
    if (!capacity || !plan) return manor_m_cuda_internal_fail(
        MANOR_M_CUDA_INVALID_ARGUMENT, "Bounded P1 requires capacity and plan descriptors.");
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, index, anchors, k,
            counts, scales, cuts, cut_count, o, out, stats, false, capacity,
            plan);
    return phase1_impl<uint32_t>(c, x, n, d, mm, index, anchors, k, counts,
        scales, cuts, cut_count, o, out, stats, false, capacity, plan);
}

extern "C" int manor_m_cuda_phase1_hybrid_bounded(
    ManorMCudaContext* c, const float* x, int64_t n, int32_t d, int32_t mm,
    const int32_t* index, int64_t anchors, int32_t k, const int32_t* counts,
    int32_t scales, const int32_t* cuts, int32_t cut_count,
    const ManorMCudaPhase1Options* o, ManorMCudaPhase1Output* out,
    ManorMCudaPhase1Stats* stats, const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan)
{
    if (!capacity || !plan) return manor_m_cuda_internal_fail(
        MANOR_M_CUDA_INVALID_ARGUMENT,
        "Bounded fused hybrid P1 requires capacity and plan descriptors.");
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, index, anchors, k,
            counts, scales, cuts, cut_count, o, out, stats, true, capacity,
            plan);
    return phase1_impl<uint32_t>(c, x, n, d, mm, index, anchors, k, counts,
        scales, cuts, cut_count, o, out, stats, true, capacity, plan);
}

extern "C" int manor_m_cuda_phase1_response_bounded(
    ManorMCudaContext* c, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    const int32_t* counts, int32_t scales, const int32_t* cuts,
    int32_t cut_count, const ManorMCudaPhase1Options* o,
    ManorMCudaPhase1Output* out, ManorMCudaPhase1Stats* stats,
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan)
{
    if (!capacity || !plan || capacity->mode != 4)
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Generated exact-kNN P1 requires an explicit mode-4 capacity plan.");
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, nullptr, n, k, counts,
            scales, cuts, cut_count, o, out, stats, false, capacity, plan,
            response, my);
    return phase1_impl<uint32_t>(c, x, n, d, mm, nullptr, n, k, counts,
        scales, cuts, cut_count, o, out, stats, false, capacity, plan,
        response, my);
}

extern "C" int manor_m_cuda_phase1_response_hybrid_bounded(
    ManorMCudaContext* c, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    const int32_t* counts, int32_t scales, const int32_t* cuts,
    int32_t cut_count, const ManorMCudaPhase1Options* o,
    ManorMCudaPhase1Output* out, ManorMCudaPhase1Stats* stats,
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan)
{
    if (!capacity || !plan || capacity->mode != 4)
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_INVALID_ARGUMENT,
            "Generated exact-kNN fused P1 requires an explicit mode-4 capacity plan.");
    if (k <= static_cast<int32_t>(std::numeric_limits<uint16_t>::max()))
        return phase1_impl<uint16_t>(c, x, n, d, mm, nullptr, n, k, counts,
            scales, cuts, cut_count, o, out, stats, true, capacity, plan,
            response, my);
    return phase1_impl<uint32_t>(c, x, n, d, mm, nullptr, n, k, counts,
        scales, cuts, cut_count, o, out, stats, true, capacity, plan,
        response, my);
}
