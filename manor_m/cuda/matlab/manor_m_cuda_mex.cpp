#include "mex.h"
#include "manor_m_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void check_status(int status, const char* operation)
{
    if (status == MANOR_M_CUDA_SUCCESS) return;
    const char* detail = manor_m_cuda_last_error();
    throw std::runtime_error(
        std::string(operation) + ": " + manor_m_cuda_status_string(status)
        + (detail != nullptr && detail[0] != '\0' ? std::string(": ") + detail : ""));
}

std::string command_string(const mxArray* value)
{
    if (!mxIsChar(value)) {
        throw std::runtime_error("The first input must be a command string.");
    }
    char* raw = mxArrayToUTF8String(value);
    if (raw == nullptr) throw std::runtime_error("Could not decode command string.");
    std::string command(raw);
    mxFree(raw);
    return command;
}

int32_t integer_scalar(const mxArray* value, const char* name)
{
    if (!mxIsNumeric(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
        throw std::runtime_error(std::string(name) + " must be a real numeric scalar.");
    }
    const double raw = mxGetScalar(value);
    if (!std::isfinite(raw)
        || raw < static_cast<double>(std::numeric_limits<int32_t>::min())
        || raw > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string(name) + " must be an exact int32 value.");
    }
    const int32_t converted = static_cast<int32_t>(raw);
    if (static_cast<double>(converted) != raw) {
        throw std::runtime_error(std::string(name) + " must be an exact int32 value.");
    }
    return converted;
}

double real_scalar(const mxArray* value, const char* name)
{
    if (!value || !mxIsNumeric(value) || mxIsComplex(value)
        || mxGetNumberOfElements(value) != 1) {
        throw std::runtime_error(std::string(name) + " must be a real numeric scalar.");
    }
    const double result = mxGetScalar(value);
    if (!std::isfinite(result)) {
        throw std::runtime_error(std::string(name) + " must be finite.");
    }
    return result;
}

bool logical_scalar(const mxArray* value, const char* name)
{
    if (mxIsLogicalScalar(value)) return mxIsLogicalScalarTrue(value);
    return real_scalar(value, name) != 0.0;
}

uint64_t uint64_scalar(const mxArray* value, const char* name)
{
    if (mxIsUint64(value) && !mxIsComplex(value)
        && mxGetNumberOfElements(value) == 1) {
        return static_cast<const uint64_T*>(mxGetData(value))[0];
    }
    const double raw = real_scalar(value, name);
    if (raw < 0.0 || std::floor(raw) != raw || raw > 9007199254740992.0) {
        throw std::runtime_error(
            std::string(name) + " must be an exact nonnegative integer no larger than 2^53.");
    }
    return static_cast<uint64_t>(raw);
}

const mxArray* option_field(const mxArray* options, const char* name)
{
    return options == nullptr ? nullptr : mxGetField(options, 0, name);
}

std::unordered_map<int32_t, ManorMCudaContext*> contexts;
bool context_cleanup_registered = false;

void destroy_contexts()
{
    for (auto& entry : contexts) {
        if (entry.second != nullptr) manor_m_cuda_context_destroy(entry.second);
    }
    contexts.clear();
    context_cleanup_registered = false;
}

ManorMCudaContext* context_for_device(int32_t device_ordinal)
{
    const auto existing = contexts.find(device_ordinal);
    if (existing != contexts.end()) return existing->second;
    ManorMCudaContextOptions options{};
    options.struct_size = sizeof(options);
    options.abi_version = MANOR_M_CUDA_ABI_VERSION;
    options.device_ordinal = device_ordinal;
    options.stream_count = 2;
    options.memory_pool_release_threshold_bytes = 0;
    ManorMCudaContext* context = nullptr;
    check_status(manor_m_cuda_context_create(&options, &context), "context_create");
    contexts.emplace(device_ordinal, context);
    if (!context_cleanup_registered) {
        mexAtExit(destroy_contexts);
        context_cleanup_registered = true;
    }
    return context;
}

struct ParsedPhase1Options {
    ManorMCudaPhase1Options core{};
    int64_t active_anchor_count = 0;
    bool return_debug = false;
};

mxArray* response_knn_profile(const ManorMCudaResponseKnnStats& stats)
{
    const char* fields[] = {
        "device_ordinal", "response_dimension", "point_count", "query_count",
        "neighbor_count", "bootstrap_ms", "threshold_ms", "scan_ms",
        "retry_ms", "select_ms", "refine_ms", "sort_ms", "emit_ms",
        "total_ms", "retry_rounds", "retry_queries", "owner_misplaced",
        "exact_ties", "compact_overflow", "mean_candidates", "max_candidates"};
    mxArray* result = mxCreateStructMatrix(1, 1, 21, fields);
    auto set = [&](const char* name, double value) {
        mxSetField(result, 0, name, mxCreateDoubleScalar(value));
    };
    set("device_ordinal", stats.device_ordinal);
    set("response_dimension", stats.response_dimension);
    set("point_count", static_cast<double>(stats.point_count));
    set("query_count", static_cast<double>(stats.query_count));
    set("neighbor_count", stats.neighbor_count);
    set("bootstrap_ms", stats.bootstrap_milliseconds);
    set("threshold_ms", stats.threshold_milliseconds);
    set("scan_ms", stats.scan_milliseconds);
    set("retry_ms", stats.retry_milliseconds);
    set("select_ms", stats.select_milliseconds);
    set("refine_ms", stats.refine_milliseconds);
    set("sort_ms", stats.sort_milliseconds);
    set("emit_ms", stats.emit_milliseconds);
    set("total_ms", stats.total_milliseconds);
    set("retry_rounds", static_cast<double>(stats.retry_rounds));
    set("retry_queries", static_cast<double>(stats.retry_queries));
    set("owner_misplaced", static_cast<double>(stats.owner_misplaced));
    set("exact_ties", static_cast<double>(stats.exact_ties));
    set("compact_overflow", static_cast<double>(stats.compact_overflow));
    set("mean_candidates", stats.mean_candidates);
    set("max_candidates", stats.max_candidates);
    return result;
}

mxArray* response_knn_output(int nrhs, const mxArray* prhs[])
{
    if (nrhs < 3 || nrhs > 5) {
        throw std::runtime_error(
            "response_knn requires response, neighbor_count, optional options, "
            "and optional device_ordinal.");
    }
    const mxArray* response = prhs[1];
    if (!mxIsSingle(response) || mxIsComplex(response) || mxIsSparse(response)
        || mxGetNumberOfDimensions(response) != 2) {
        throw std::runtime_error(
            "response must be a dense real single [response_dimension,N] matrix.");
    }
    const int64_t point_count = static_cast<int64_t>(mxGetN(response));
    const int32_t response_dimension = static_cast<int32_t>(mxGetM(response));
    const int32_t neighbor_count = integer_scalar(prhs[2], "neighbor_count");
    const mxArray* option_source = nrhs >= 4 && !mxIsEmpty(prhs[3]) ? prhs[3] : nullptr;
    if (option_source && (!mxIsStruct(option_source)
        || mxGetNumberOfElements(option_source) != 1)) {
        throw std::runtime_error("response_knn options must be empty or a scalar struct.");
    }
    const int32_t device = nrhs == 5 ? integer_scalar(prhs[4], "device_ordinal") : 0;
    int64_t query_count = point_count;
    ManorMCudaResponseKnnOptions options{};
    options.struct_size = sizeof(options);
    options.abi_version = MANOR_M_CUDA_ABI_VERSION;
    options.scratch_memory_bytes = 7ull * 1024ull * 1024ull * 1024ull;
    options.max_retry_rounds = 12;
    options.collect_stats = 1;
    if (option_source) {
        if (const mxArray* value = option_field(option_source, "query_count"))
            query_count = integer_scalar(value, "query_count");
        if (const mxArray* value = option_field(option_source, "scratch_memory_bytes"))
            options.scratch_memory_bytes = uint64_scalar(value, "scratch_memory_bytes");
        if (const mxArray* value = option_field(option_source, "max_retry_rounds"))
            options.max_retry_rounds = integer_scalar(value, "max_retry_rounds");
        if (const mxArray* value = option_field(option_source, "collect_stats"))
            options.collect_stats = logical_scalar(value, "collect_stats");
    }
    if (point_count < 1 || response_dimension < 1
        || query_count < 1 || query_count > point_count
        || neighbor_count < 1 || neighbor_count > point_count) {
        throw std::runtime_error(
            "response dimensions, query_count, and neighbor_count are inconsistent.");
    }

    ManorMCudaContext* context = context_for_device(device);
    mxArray* index = mxCreateNumericMatrix(
        static_cast<mwSize>(neighbor_count), static_cast<mwSize>(query_count),
        mxINT32_CLASS, mxREAL);
    ManorMCudaResponseKnnStats stats{};
    const int status = manor_m_cuda_response_knn(
        context,
        static_cast<const float*>(mxGetData(response)),
        point_count,
        response_dimension,
        query_count,
        neighbor_count,
        static_cast<int32_t*>(mxGetData(index)),
        &options,
        &stats);
    if (status != MANOR_M_CUDA_SUCCESS) {
        const char* detail = manor_m_cuda_last_error();
        const std::string message = std::string("response_knn: ")
            + manor_m_cuda_status_string(status)
            + (detail && detail[0] ? std::string(": ") + detail : "");
        mxDestroyArray(index);
        throw std::runtime_error(message);
    }
    const char* fields[] = {"index_zero_based", "profile"};
    mxArray* result = mxCreateStructMatrix(1, 1, 2, fields);
    mxSetField(result, 0, "index_zero_based", index);
    mxSetField(result, 0, "profile", response_knn_profile(stats));
    return result;
}

ParsedPhase1Options phase1_options(const mxArray* source, int64_t point_count)
{
    if (source && (!mxIsStruct(source) || mxGetNumberOfElements(source) != 1)) {
        throw std::runtime_error("Phase-1 options must be an empty value or scalar struct.");
    }
    ParsedPhase1Options result;
    result.core.struct_size = sizeof(result.core);
    result.core.abi_version = MANOR_M_CUDA_ABI_VERSION;
    result.core.gpu_memory_fraction = 0.92f;
    result.core.reserve_memory_fraction = 0.08f;
    result.core.reserve_memory_bytes = 6ull * 1024ull * 1024ull * 1024ull;
    result.core.heavy_factor = 0.10f;
    result.core.jitter = 1.0e-6f;
    result.core.neighbor_scaled_max = 1.0f;
    result.core.split_scaled_max = 6.0f;
    result.core.minimum_query_block = 128;
    result.core.maximum_query_block = std::numeric_limits<int32_t>::max();
    result.core.stable_window = 7;
    result.core.selection_mode = 0;
    result.core.split_seed = 730021;
    result.core.cut_max_fraction = 0.8;
    result.core.index_is_shared_slice = 0;
    result.core.h_null_correction = 0;
    result.core.collect_profile = 0;
    result.active_anchor_count = point_count;
    if (!source) return result;

    auto set_float = [&](const char* name, float& target) {
        if (const mxArray* value = option_field(source, name))
            target = static_cast<float>(real_scalar(value, name));
    };
    set_float("gpu_mem_fraction", result.core.gpu_memory_fraction);
    set_float("reserve_fraction", result.core.reserve_memory_fraction);
    set_float("heavy_factor", result.core.heavy_factor);
    set_float("jitter", result.core.jitter);
    set_float("neighbor_scaled_max", result.core.neighbor_scaled_max);
    set_float("split_scaled_max", result.core.split_scaled_max);
    if (const mxArray* value = option_field(source, "reserve_bytes"))
        result.core.reserve_memory_bytes = uint64_scalar(value, "reserve_bytes");
    if (const mxArray* value = option_field(source, "min_query_block"))
        result.core.minimum_query_block = integer_scalar(value, "min_query_block");
    if (const mxArray* value = option_field(source, "max_query_block"))
        result.core.maximum_query_block = integer_scalar(value, "max_query_block");
    if (const mxArray* value = option_field(source, "window"))
        result.core.stable_window = integer_scalar(value, "window");
    if (const mxArray* value = option_field(source, "selection_mode"))
        result.core.selection_mode = integer_scalar(value, "selection_mode");
    if (const mxArray* value = option_field(source, "split_seed"))
        result.core.split_seed = uint64_scalar(value, "split_seed");
    if (const mxArray* value = option_field(source, "cut_max_fraction"))
        result.core.cut_max_fraction = real_scalar(value, "cut_max_fraction");
    if (const mxArray* value = option_field(source, "idx_zero_based")) {
        if (!logical_scalar(value, "idx_zero_based")) {
            throw std::runtime_error(
                "The shared Phase-1 adapter requires a zero-based int32 response index.");
        }
    }
    if (const mxArray* value = option_field(source, "idx_shared_slice"))
        result.core.index_is_shared_slice = logical_scalar(value, "idx_shared_slice");
    if (const mxArray* value = option_field(source, "h_null_correction"))
        result.core.h_null_correction = logical_scalar(value, "h_null_correction");
    if (const mxArray* value = option_field(source, "profile_timers"))
        result.core.collect_profile = logical_scalar(value, "profile_timers");
    if (const mxArray* value = option_field(source, "return_debug"))
        result.return_debug = logical_scalar(value, "return_debug");
    if (const mxArray* value = option_field(source, "active_anchor_count"))
        result.active_anchor_count = integer_scalar(value, "active_anchor_count");
    return result;
}

mxArray* phase1_retention(
    const ManorMCudaPhase1Stats& stats,
    const ManorMCudaPhase1Options& options)
{
    const char* fields[] = {
        "stable_covered_count", "stable_covered_fraction", "owner_count",
        "owner_fraction", "support_threshold", "heavy_factor", "jitter",
        "neighbor_scaled_max", "split_scaled_max", "window", "split_seed",
        "cut_max_fraction", "scale_count", "active_anchor_count"};
    mxArray* result = mxCreateStructMatrix(1, 1, 14, fields);
    mxSetField(result, 0, "stable_covered_count",
        mxCreateDoubleScalar(static_cast<double>(stats.stable_covered_count)));
    mxSetField(result, 0, "stable_covered_fraction",
        mxCreateDoubleScalar(static_cast<double>(stats.stable_covered_count) / stats.point_count));
    mxSetField(result, 0, "owner_count",
        mxCreateDoubleScalar(static_cast<double>(stats.owner_count)));
    mxSetField(result, 0, "owner_fraction",
        mxCreateDoubleScalar(static_cast<double>(stats.owner_count) / stats.point_count));
    mxSetField(result, 0, "support_threshold", mxCreateDoubleScalar(stats.support_threshold));
    mxSetField(result, 0, "heavy_factor", mxCreateDoubleScalar(options.heavy_factor));
    mxSetField(result, 0, "jitter", mxCreateDoubleScalar(options.jitter));
    mxSetField(result, 0, "neighbor_scaled_max",
        mxCreateDoubleScalar(options.neighbor_scaled_max));
    mxSetField(result, 0, "split_scaled_max",
        mxCreateDoubleScalar(options.split_scaled_max));
    mxSetField(result, 0, "window", mxCreateDoubleScalar(options.stable_window));
    mxArray* seed = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    static_cast<uint64_T*>(mxGetData(seed))[0] = options.split_seed;
    mxSetField(result, 0, "split_seed", seed);
    mxSetField(result, 0, "cut_max_fraction",
        mxCreateDoubleScalar(options.cut_max_fraction));
    mxSetField(result, 0, "scale_count", mxCreateDoubleScalar(stats.scale_count));
    mxSetField(result, 0, "active_anchor_count",
        mxCreateDoubleScalar(static_cast<double>(stats.active_anchor_count)));
    return result;
}

mxArray* phase1_profile(
    const ManorMCudaPhase1Stats& stats,
    const ManorMCudaPhase1Options& options,
    int64_t index_count)
{
    const char* fields[] = {
        "total_sec", "BQ", "BQ_upper_bound", "scale_count",
        "selection_shared_bytes", "idx_shared_slice", "memory_index_bytes",
        "memory_free_bytes", "memory_total_bytes", "memory_reserve_bytes",
        "memory_usable_bytes", "memory_fixed_bytes", "memory_adaptive_extra_bytes",
        "memory_sort_payload_bytes", "memory_sort_temp_bytes",
        "memory_aggregate_scratch_bytes", "memory_query_bytes",
        "memory_required_bytes", "memory_next_required_bytes",
        "active_anchor_count", "timer_labels", "timer_ms"};
    mxArray* result = mxCreateStructMatrix(1, 1, 22, fields);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double reserve = std::max(
        static_cast<double>(options.reserve_memory_bytes),
        static_cast<double>(stats.memory_total_bytes) * options.reserve_memory_fraction);
    const double usable = std::max(
        0.0, static_cast<double>(stats.memory_free_bytes) - reserve)
        * options.gpu_memory_fraction;
    mxSetField(result, 0, "total_sec", mxCreateDoubleScalar(stats.total_milliseconds / 1000.0));
    mxSetField(result, 0, "BQ", mxCreateDoubleScalar(stats.query_block));
    mxSetField(result, 0, "BQ_upper_bound",
        mxCreateDoubleScalar(stats.query_block_upper_bound));
    mxSetField(result, 0, "scale_count", mxCreateDoubleScalar(stats.scale_count));
    mxSetField(result, 0, "selection_shared_bytes", mxCreateDoubleScalar(nan));
    mxSetField(result, 0, "idx_shared_slice",
        mxCreateLogicalScalar(options.index_is_shared_slice != 0));
    mxSetField(result, 0, "memory_index_bytes",
        mxCreateDoubleScalar(static_cast<double>(index_count) * sizeof(int32_t)));
    mxSetField(result, 0, "memory_free_bytes",
        mxCreateDoubleScalar(static_cast<double>(stats.memory_free_bytes)));
    mxSetField(result, 0, "memory_total_bytes",
        mxCreateDoubleScalar(static_cast<double>(stats.memory_total_bytes)));
    mxSetField(result, 0, "memory_reserve_bytes", mxCreateDoubleScalar(reserve));
    mxSetField(result, 0, "memory_usable_bytes", mxCreateDoubleScalar(usable));
    const char* unavailable[] = {
        "memory_fixed_bytes", "memory_adaptive_extra_bytes",
        "memory_sort_payload_bytes", "memory_sort_temp_bytes",
        "memory_aggregate_scratch_bytes", "memory_query_bytes"};
    for (const char* field : unavailable)
        mxSetField(result, 0, field, mxCreateDoubleScalar(nan));
    mxSetField(result, 0, "memory_required_bytes",
        mxCreateDoubleScalar(static_cast<double>(stats.memory_required_bytes)));
    mxSetField(result, 0, "memory_next_required_bytes",
        mxCreateDoubleScalar(static_cast<double>(stats.memory_next_required_bytes)));
    mxSetField(result, 0, "active_anchor_count",
        mxCreateDoubleScalar(static_cast<double>(stats.active_anchor_count)));

    const char* labels[] = {
        "upload", "dist_idx", "sort", "scale_topk", "scale_select",
        "scale_stream", "final_extract", "owner_rule", "owner_aggregate",
        "mask_nonowners", "download"};
    const double values[] = {
        stats.upload_milliseconds, stats.distance_milliseconds,
        stats.sort_milliseconds, stats.scale_topk_milliseconds,
        stats.scale_select_milliseconds, stats.scale_stream_milliseconds,
        stats.final_extract_milliseconds, stats.owner_rule_milliseconds,
        stats.owner_aggregate_milliseconds, stats.mask_nonowners_milliseconds,
        stats.download_milliseconds};
    constexpr mwSize count = sizeof(values) / sizeof(values[0]);
    mxArray* timer_labels = mxCreateCellMatrix(1, count);
    mxArray* timer_values = mxCreateDoubleMatrix(1, count, mxREAL);
    for (mwSize index = 0; index < count; ++index) {
        mxSetCell(timer_labels, index, mxCreateString(labels[index]));
        mxGetPr(timer_values)[index] = values[index];
    }
    mxSetField(result, 0, "timer_labels", timer_labels);
    mxSetField(result, 0, "timer_ms", timer_values);
    return result;
}

mxArray* phase1_precomputed_output(int nrhs, const mxArray* prhs[])
{
    if (nrhs < 7 || nrhs > 8) {
        throw std::runtime_error(
            "phase1_precomputed requires X, index, response_counts, cuts, MM, "
            "options, and optional device_ordinal.");
    }
    const mxArray* x = prhs[1];
    const mxArray* index = prhs[2];
    const mxArray* counts = prhs[3];
    const mxArray* cuts = prhs[4];
    if (!mxIsSingle(x) || mxIsComplex(x) || mxIsSparse(x)
        || mxGetNumberOfDimensions(x) != 2) {
        throw std::runtime_error("X must be a dense real single [D,N] matrix.");
    }
    if (!mxIsInt32(index) || mxIsComplex(index) || mxIsSparse(index)
        || mxGetNumberOfDimensions(index) != 2) {
        throw std::runtime_error(
            "index must be a dense zero-based int32 [k,N] matrix.");
    }
    if (!mxIsInt32(counts) || mxIsComplex(counts) || mxIsSparse(counts)
        || (mxGetM(counts) != 1 && mxGetN(counts) != 1)) {
        throw std::runtime_error("response_counts must be a real int32 vector.");
    }
    if (!mxIsInt32(cuts) || mxIsComplex(cuts) || mxIsSparse(cuts)
        || mxGetNumberOfDimensions(cuts) != 2) {
        throw std::runtime_error("cuts must be a real int32 [R,S] matrix.");
    }
    const int64_t point_count = static_cast<int64_t>(mxGetN(x));
    const int32_t ambient_dimension = static_cast<int32_t>(mxGetM(x));
    const int32_t manifold_dimension = integer_scalar(prhs[5], "MM");
    const mxArray* option_source = mxIsEmpty(prhs[6]) ? nullptr : prhs[6];
    ParsedPhase1Options parsed = phase1_options(option_source, point_count);
    const int32_t device = nrhs == 8 ? integer_scalar(prhs[7], "device_ordinal") : 0;
    const int32_t neighbor_count = static_cast<int32_t>(mxGetM(index));
    const int64_t index_columns = static_cast<int64_t>(mxGetN(index));
    if (parsed.core.index_is_shared_slice) {
        if (index_columns != 1)
            throw std::runtime_error("A shared Phase-1 index must have shape [k,1].");
    } else if (index_columns != parsed.active_anchor_count) {
        throw std::runtime_error(
            "A non-shared Phase-1 index must have one column per active anchor.");
    }
    const int32_t scale_count = static_cast<int32_t>(mxGetNumberOfElements(counts));
    const int32_t cuts_per_scale = static_cast<int32_t>(mxGetM(cuts));
    if (static_cast<int32_t>(mxGetN(cuts)) != scale_count)
        throw std::runtime_error("cuts must have one column per response scale.");

    // Create the device-affine context before allocating the large MATLAB
    // outputs.  If context creation fails, no partially built mxArray tree
    // needs to be unwound.
    ManorMCudaContext* context = context_for_device(device);

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
    mxArray* result = mxCreateStructMatrix(
        1, 1,
        parsed.return_debug
            ? static_cast<int>(sizeof(debug_fields) / sizeof(debug_fields[0]))
            : static_cast<int>(sizeof(basic_fields) / sizeof(basic_fields[0])),
        parsed.return_debug ? debug_fields : basic_fields);

    mwSize tangent_dimensions[3] = {
        static_cast<mwSize>(ambient_dimension),
        static_cast<mwSize>(manifold_dimension),
        static_cast<mwSize>(point_count)};
    auto numeric = [&](const char* name, mwSize rows, mxClassID type) {
        mxArray* value = mxCreateNumericMatrix(rows, point_count, type, mxREAL);
        mxSetField(result, 0, name, value);
        return value;
    };
    mxArray* tangent = mxCreateNumericArray(3, tangent_dimensions, mxSINGLE_CLASS, mxREAL);
    mxSetField(result, 0, "Tan0", tangent);
    mxArray* center = numeric("Cen0", ambient_dimension, mxSINGLE_CLASS);
    mxArray* radius = numeric("Rad0", 1, mxSINGLE_CLASS);
    mxArray* axial = numeric("axial_tan0", 1, mxSINGLE_CLASS);
    mxArray* neighbors = numeric("num_neigh0", 1, mxINT32_CLASS);
    mxArray* owner = mxCreateLogicalMatrix(1, point_count);
    mxArray* covered = mxCreateLogicalMatrix(1, point_count);
    mxSetField(result, 0, "owner_mask0", owner);
    mxSetField(result, 0, "stable_covered", covered);
    mxArray* selected_scale = numeric("selected_scale0", 1, mxUINT32_CLASS);
    mxArray* selected_response = numeric(
        "selected_response_count0", 1, mxUINT32_CLASS);
    mxArray* selected_cut = numeric("selected_cut0", 1, mxUINT32_CLASS);
    mxArray* geometry = numeric("geom0", 1, mxSINGLE_CLASS);
    mxArray* stable_h = numeric("stable_h0", 1, mxSINGLE_CLASS);
    mxArray* neighbor_scaled = numeric("neighbor_scaled0", 1, mxSINGLE_CLASS);
    mxArray* split_scaled = numeric("split_scaled0", 1, mxSINGLE_CLASS);
    mxArray* split_h_a = numeric("split_h_a0", 1, mxSINGLE_CLASS);
    mxArray* split_h_b = numeric("split_h_b0", 1, mxSINGLE_CLASS);
    mxArray* rejection = numeric("rejection_code0", 1, mxUINT8_CLASS);
    mxArray* raw_tangent = nullptr;
    mxArray* raw_center = nullptr;
    mxArray* selected_index = nullptr;
    if (parsed.return_debug) {
        raw_tangent = mxCreateNumericArray(
            3, tangent_dimensions, mxSINGLE_CLASS, mxREAL);
        mxSetField(result, 0, "TanRaw0", raw_tangent);
        raw_center = numeric("CenRaw0", ambient_dimension, mxSINGLE_CLASS);
        selected_index = numeric("selected_index0", 1, mxUINT32_CLASS);
    }

    ManorMCudaPhase1Output output{};
    output.struct_size = sizeof(output);
    output.abi_version = MANOR_M_CUDA_ABI_VERSION;
    output.tangent_point_major = static_cast<float*>(mxGetData(tangent));
    output.center_point_major = static_cast<float*>(mxGetData(center));
    output.radius = static_cast<float*>(mxGetData(radius));
    output.axial_tangent = static_cast<float*>(mxGetData(axial));
    output.neighbor_count = static_cast<int32_t*>(mxGetData(neighbors));
    output.owner_mask = reinterpret_cast<uint8_t*>(mxGetLogicals(owner));
    output.stable_covered = reinterpret_cast<uint8_t*>(mxGetLogicals(covered));
    output.selected_scale_one_based = static_cast<uint32_t*>(mxGetData(selected_scale));
    output.selected_response_count = static_cast<uint32_t*>(mxGetData(selected_response));
    output.selected_cut = static_cast<uint32_t*>(mxGetData(selected_cut));
    output.geometry_score = static_cast<float*>(mxGetData(geometry));
    output.stable_h = static_cast<float*>(mxGetData(stable_h));
    output.neighbor_scaled = static_cast<float*>(mxGetData(neighbor_scaled));
    output.split_scaled = static_cast<float*>(mxGetData(split_scaled));
    output.split_h_a = static_cast<float*>(mxGetData(split_h_a));
    output.split_h_b = static_cast<float*>(mxGetData(split_h_b));
    output.rejection_code = static_cast<uint8_t*>(mxGetData(rejection));
    output.raw_tangent_point_major = raw_tangent
        ? static_cast<float*>(mxGetData(raw_tangent)) : nullptr;
    output.raw_center_point_major = raw_center
        ? static_cast<float*>(mxGetData(raw_center)) : nullptr;
    output.selected_index_one_based = selected_index
        ? static_cast<uint32_t*>(mxGetData(selected_index)) : nullptr;
    ManorMCudaPhase1Stats stats{};
    const int64_t index_count = parsed.core.index_is_shared_slice
        ? neighbor_count : parsed.active_anchor_count * neighbor_count;
    const int status = manor_m_cuda_phase1_precomputed(
        context,
        static_cast<const float*>(mxGetData(x)),
        point_count,
        ambient_dimension,
        manifold_dimension,
        static_cast<const int32_t*>(mxGetData(index)),
        parsed.active_anchor_count,
        neighbor_count,
        static_cast<const int32_t*>(mxGetData(counts)),
        scale_count,
        static_cast<const int32_t*>(mxGetData(cuts)),
        cuts_per_scale,
        &parsed.core,
        &output,
        &stats);
    if (status != MANOR_M_CUDA_SUCCESS) {
        const char* detail = manor_m_cuda_last_error();
        const std::string message = std::string("phase1_precomputed: ")
            + manor_m_cuda_status_string(status)
            + (detail && detail[0] ? std::string(": ") + detail : "");
        mxDestroyArray(result);
        throw std::runtime_error(message);
    }
    mxSetField(result, 0, "retention", phase1_retention(stats, parsed.core));
    mxSetField(result, 0, "profile", phase1_profile(stats, parsed.core, index_count));
    return result;
}

std::vector<float> float_vector(
    const mxArray* value, const char* name, bool positive)
{
    if (!value || mxIsComplex(value) || mxIsSparse(value)
        || (!mxIsSingle(value) && !mxIsDouble(value))
        || mxGetNumberOfElements(value) < 1
        || (mxGetM(value) != 1 && mxGetN(value) != 1)) {
        throw std::runtime_error(
            std::string(name) + " must be a nonempty real single/double vector.");
    }
    const mwSize count = mxGetNumberOfElements(value);
    std::vector<float> result(static_cast<size_t>(count));
    for (mwSize index = 0; index < count; ++index) {
        const double raw = mxIsSingle(value)
            ? static_cast<const float*>(mxGetData(value))[index]
            : mxGetPr(value)[index];
        const float converted = static_cast<float>(raw);
        if (!std::isfinite(raw) || !std::isfinite(converted)
            || (positive && converted <= 0.0f)) {
            throw std::runtime_error(
                std::string(name) + (positive
                    ? " entries must be finite and positive."
                    : " entries must be finite."));
        }
        result[static_cast<size_t>(index)] = converted;
    }
    return result;
}

std::vector<int32_t> int32_vector(const mxArray* value, const char* name)
{
    if (!value || mxIsComplex(value) || mxIsSparse(value)
        || !mxIsNumeric(value) || mxGetNumberOfElements(value) < 1
        || (mxGetM(value) != 1 && mxGetN(value) != 1)) {
        throw std::runtime_error(
            std::string(name) + " must be a nonempty real numeric vector.");
    }
    const mwSize count = mxGetNumberOfElements(value);
    std::vector<int32_t> result(static_cast<size_t>(count));
    for (mwSize index = 0; index < count; ++index) {
        double raw = 0.0;
        switch (mxGetClassID(value)) {
        case mxINT32_CLASS:
            raw = static_cast<const int32_t*>(mxGetData(value))[index]; break;
        case mxUINT32_CLASS:
            raw = static_cast<const uint32_t*>(mxGetData(value))[index]; break;
        case mxSINGLE_CLASS:
            raw = static_cast<const float*>(mxGetData(value))[index]; break;
        case mxDOUBLE_CLASS:
            raw = mxGetPr(value)[index]; break;
        default:
            throw std::runtime_error(
                std::string(name) + " must use int32, uint32, single, or double.");
        }
        if (!std::isfinite(raw) || std::floor(raw) != raw
            || raw < static_cast<double>(std::numeric_limits<int32_t>::min())
            || raw > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                std::string(name) + " entries must be exact int32 values.");
        }
        result[static_cast<size_t>(index)] = static_cast<int32_t>(raw);
    }
    return result;
}

struct ParsedPhase2Options {
    ManorMCudaPhase2Options core{};
    std::vector<float> h_schedule;
    bool return_debug = false;
    int32_t device_ordinal = 0;
};

ParsedPhase2Options phase2_options(const mxArray* source)
{
    if (!source || !mxIsStruct(source) || mxGetNumberOfElements(source) != 1) {
        throw std::runtime_error("Phase-2 options must be a scalar struct.");
    }
    ParsedPhase2Options result;
    result.core.struct_size = sizeof(result.core);
    result.core.abi_version = MANOR_M_CUDA_ABI_VERSION;
    result.core.lambda = 1.0e-5f;
    result.core.jitter = 1.0e-6f;
    result.core.gpu_memory_fraction = 0.92f;
    result.core.reserve_memory_fraction = 0.08f;
    result.core.reserve_memory_bytes = 0;
    result.core.minimum_query_block = 1;
    result.core.maximum_query_block = 131072;
    result.core.regression_mode = 0;
    result.core.iteration_count = 1;
    result.core.heavy_factor = 0.10f;
    result.core.temporary_heavy_factor = 0.20f;
    result.core.index_is_shared_slice = 0;
    result.core.evaluation_uses_owner_mask = 1;
    result.core.collect_profile = 0;
    result.core.use_async_allocator = 0;

    auto set_float = [&](const char* name, float& target) {
        if (const mxArray* value = option_field(source, name))
            target = static_cast<float>(real_scalar(value, name));
    };
    set_float("lambda", result.core.lambda);
    set_float("jitter", result.core.jitter);
    set_float("gpu_mem_fraction", result.core.gpu_memory_fraction);
    set_float("reserve_fraction", result.core.reserve_memory_fraction);
    set_float("heavy_factor", result.core.heavy_factor);
    set_float("heavy_factor_tmp", result.core.temporary_heavy_factor);
    if (const mxArray* value = option_field(source, "reserve_bytes"))
        result.core.reserve_memory_bytes = uint64_scalar(value, "reserve_bytes");
    if (const mxArray* value = option_field(source, "min_query_block"))
        result.core.minimum_query_block = integer_scalar(value, "min_query_block");
    if (const mxArray* value = option_field(source, "max_query_block"))
        result.core.maximum_query_block = integer_scalar(value, "max_query_block");
    if (const mxArray* value = option_field(source, "regression_mode"))
        result.core.regression_mode = integer_scalar(value, "regression_mode");
    if (const mxArray* value = option_field(source, "K"))
        result.core.iteration_count = integer_scalar(value, "K");
    if (const mxArray* value = option_field(source, "iterations"))
        result.core.iteration_count = integer_scalar(value, "iterations");
    if (const mxArray* value = option_field(source, "hk_iter"))
        result.h_schedule = float_vector(value, "hk_iter", true);
    if (const mxArray* value = option_field(source, "idx_zero_based")) {
        if (!logical_scalar(value, "idx_zero_based")) {
            throw std::runtime_error(
                "The shared Phase-2 adapter requires a zero-based int32 index.");
        }
    }
    if (const mxArray* value = option_field(source, "idx_shared_slice"))
        result.core.index_is_shared_slice = logical_scalar(value, "idx_shared_slice");
    if (const mxArray* value = option_field(source, "fused_eval_use_owner_mask"))
        result.core.evaluation_uses_owner_mask = logical_scalar(
            value, "fused_eval_use_owner_mask");
    if (const mxArray* value = option_field(source, "profile_timers"))
        result.core.collect_profile = logical_scalar(value, "profile_timers");
    if (const mxArray* value = option_field(source, "async_allocator"))
        result.core.use_async_allocator = logical_scalar(
            value, "async_allocator");
    if (const mxArray* value = option_field(source, "return_debug"))
        result.return_debug = logical_scalar(value, "return_debug");
    if (const mxArray* value = option_field(source, "device_ordinal"))
        result.device_ordinal = integer_scalar(value, "device_ordinal");

    if (result.core.lambda < 0.0f || result.core.jitter < 0.0f
        || result.core.gpu_memory_fraction <= 0.0f
        || result.core.gpu_memory_fraction > 1.0f
        || result.core.reserve_memory_fraction < 0.0f
        || result.core.reserve_memory_fraction >= 1.0f
        || result.core.minimum_query_block < 1
        || result.core.maximum_query_block < result.core.minimum_query_block
        || (result.core.regression_mode != 0 && result.core.regression_mode != 1)
        || result.core.iteration_count < 1
        || result.core.heavy_factor < 0.0f
        || result.core.temporary_heavy_factor < 0.0f
        || (result.core.use_async_allocator != 0
            && result.core.use_async_allocator != 1)) {
        throw std::runtime_error("Invalid Phase-2 numerical or memory options.");
    }
    if (!result.h_schedule.empty()
        && result.h_schedule.size() != 1
        && result.h_schedule.size()
            != static_cast<size_t>(result.core.iteration_count)) {
        throw std::runtime_error(
            "hk_iter must be scalar or have exactly K/iterations entries.");
    }
    result.core.h_schedule = result.h_schedule.empty()
        ? nullptr : result.h_schedule.data();
    result.core.h_schedule_count = static_cast<int32_t>(result.h_schedule.size());
    return result;
}

struct ParsedPhase2Evaluation {
    ManorMCudaPhase2Evaluation core{};
    std::vector<float> rho_scales{1.0f};
    std::vector<float> h_scales{1.0f};
    std::vector<int32_t> evaluated_iterations;
    bool enabled = false;
    bool test_metrics = false;
    bool validation = false;
    bool eval_k0 = false;
};

ParsedPhase2Evaluation phase2_evaluation_options(const mxArray* source)
{
    if (source && (!mxIsStruct(source) || mxGetNumberOfElements(source) != 1)) {
        throw std::runtime_error(
            "Phase-2 evaluation options must be empty or a scalar struct.");
    }
    ParsedPhase2Evaluation result;
    result.core.struct_size = sizeof(result.core);
    result.core.abi_version = MANOR_M_CUDA_ABI_VERSION;
    result.core.sum_sparse_threshold = 10.0f;
    result.core.count_sparse_threshold = 0;
    result.core.fallback_exponent_scale = 2.0f;
    result.core.epsilon_weight = 1.0e-20f;
    result.core.warps_per_block = 4;
    result.core.center_rho_scale = 4.0f;
    result.core.center_h_scale = 1.25f;
    if (source) {
        auto set_float = [&](const char* name, float& target) {
            if (const mxArray* value = option_field(source, name))
                target = static_cast<float>(real_scalar(value, name));
        };
        set_float("sum_sparse_threshold", result.core.sum_sparse_threshold);
        if (const mxArray* value = option_field(source, "count_sparse_threshold"))
            result.core.count_sparse_threshold = integer_scalar(
                value, "count_sparse_threshold");
        set_float("fallback_exp_scale", result.core.fallback_exponent_scale);
        set_float("eps_weight", result.core.epsilon_weight);
        if (const mxArray* value = option_field(source, "warps_per_block"))
            result.core.warps_per_block = integer_scalar(value, "warps_per_block");
        if (const mxArray* value = option_field(source, "eval_rho_scale"))
            result.rho_scales = float_vector(value, "eval_rho_scale", true);
        if (const mxArray* value = option_field(source, "hyper_param_rho"))
            result.rho_scales = float_vector(value, "hyper_param_rho", true);
        if (const mxArray* value = option_field(source, "eval_h_scale"))
            result.h_scales = float_vector(value, "eval_h_scale", true);
        if (const mxArray* value = option_field(source, "hyper_param_R"))
            result.h_scales = float_vector(value, "hyper_param_R", true);
        if (const mxArray* value = option_field(source, "eval_k_arr"))
            result.evaluated_iterations = int32_vector(value, "eval_k_arr");
        if (const mxArray* value = option_field(source, "eval_K_arr"))
            result.evaluated_iterations = int32_vector(value, "eval_K_arr");
        set_float("center_eval_rho_scale", result.core.center_rho_scale);
        set_float("center_eval_h_scale", result.core.center_h_scale);
    }
    result.core.sum_sparse_threshold = std::max(
        0.0f, result.core.sum_sparse_threshold);
    result.core.count_sparse_threshold = std::max(
        0, result.core.count_sparse_threshold);
    result.core.fallback_exponent_scale = std::max(
        0.0f, result.core.fallback_exponent_scale);
    result.core.epsilon_weight = std::max(1.0e-30f, result.core.epsilon_weight);
    if (result.core.warps_per_block != 2 && result.core.warps_per_block != 4)
        result.core.warps_per_block = 4;
    if (!std::isfinite(result.core.center_rho_scale)
        || result.core.center_rho_scale <= 0.0f
        || !std::isfinite(result.core.center_h_scale)
        || result.core.center_h_scale <= 0.0f) {
        throw std::runtime_error(
            "center evaluation scales must be finite and positive.");
    }
    for (int32_t iteration : result.evaluated_iterations)
        result.eval_k0 = result.eval_k0 || iteration == 0;
    result.core.rho_scales = result.rho_scales.data();
    result.core.rho_scale_count = static_cast<int32_t>(result.rho_scales.size());
    result.core.h_scales = result.h_scales.data();
    result.core.h_scale_count = static_cast<int32_t>(result.h_scales.size());
    result.core.evaluated_iterations = result.evaluated_iterations.empty()
        ? nullptr : result.evaluated_iterations.data();
    result.core.evaluated_iteration_count = static_cast<int32_t>(
        result.evaluated_iterations.size());
    return result;
}

bool nonempty(const mxArray* value)
{
    return value != nullptr && mxGetNumberOfElements(value) != 0;
}

void require_single(const mxArray* value, const char* name)
{
    if (!value || !mxIsSingle(value) || mxIsComplex(value) || mxIsSparse(value))
        throw std::runtime_error(std::string(name) + " must be dense real single.");
}

mxArray* phase2_profile(
    const ManorMCudaPhase2Stats& stats,
    const ManorMCudaPhase2Options& options,
    const ManorMCudaPhase2Evaluation* evaluation,
    int rho_schedule_length)
{
    const char* fields[] = {
        "BQ", "owner_count", "memory_required_bytes", "memory_usable_bytes",
        "cub_temp_bytes", "K", "history_bytes", "eval_block_threads",
        "rhok_schedule_length", "idx_shared_slice", "memory_index_bytes",
        "fused_eval_use_owner_mask", "mse_only_eval", "tangent_only_eval",
        "response_eval", "tangent_metric_eval", "center_eval",
        "center_tangent_metric_eval", "center_eval_h_scale",
        "center_eval_rho_scale", "async_allocator", "timer_labels",
        "timer_ms", "host_prep_ms",
        "device_loop_ms", "output_pack_ms", "sparse_pairs_total",
        "host_cuda_allocate_ms", "host_cuda_free_ms",
        "host_cuda_event_destroy_ms"};
    mxArray* result = mxCreateStructMatrix(1, 1, 30, fields);
    auto set = [&](const char* name, double value) {
        mxSetField(result, 0, name, mxCreateDoubleScalar(value));
    };
    set("BQ", stats.query_block);
    set("owner_count", static_cast<double>(stats.owner_count));
    set("memory_required_bytes", static_cast<double>(stats.memory_required_bytes));
    set("memory_usable_bytes", static_cast<double>(stats.memory_usable_bytes));
    set("cub_temp_bytes", 0.0);
    set("K", stats.iteration_count);
    set("history_bytes", static_cast<double>(stats.history_bytes));
    set("eval_block_threads", evaluation ? evaluation->warps_per_block * 32.0 : 0.0);
    set("rhok_schedule_length", rho_schedule_length);
    mxSetField(result, 0, "idx_shared_slice",
        mxCreateLogicalScalar(options.index_is_shared_slice != 0));
    set("memory_index_bytes", static_cast<double>(stats.memory_index_bytes));
    mxSetField(result, 0, "fused_eval_use_owner_mask",
        mxCreateLogicalScalar(options.evaluation_uses_owner_mask != 0));
    const bool response = stats.response_evaluated != 0;
    const bool tangent = stats.tangent_evaluated != 0;
    const bool center = stats.center_evaluated != 0;
    mxSetField(result, 0, "mse_only_eval",
        mxCreateLogicalScalar(response && !tangent && !center));
    mxSetField(result, 0, "tangent_only_eval",
        mxCreateLogicalScalar(tangent && !response && !center));
    mxSetField(result, 0, "response_eval", mxCreateLogicalScalar(response));
    mxSetField(result, 0, "tangent_metric_eval", mxCreateLogicalScalar(tangent));
    mxSetField(result, 0, "center_eval", mxCreateLogicalScalar(center));
    mxSetField(result, 0, "center_tangent_metric_eval",
        mxCreateLogicalScalar(stats.center_tangent_evaluated != 0));
    set("center_eval_h_scale", evaluation ? evaluation->center_h_scale : 0.0);
    set("center_eval_rho_scale", evaluation ? evaluation->center_rho_scale : 0.0);
    mxSetField(result, 0, "async_allocator",
        mxCreateLogicalScalar(options.use_async_allocator != 0));
    const char* timer_names[] = {
        "upload_init", "eval_stats", "eval_main", "eval_pair",
        "valid_main", "valid_pair", "iter_setup", "regression",
        "mask_chain", "agg_accumulate", "agg_finalize", "snapshots",
        "eval_store", "valid_store"};
    const double timer_values[] = {
        stats.timer_upload_init_milliseconds,
        stats.timer_eval_stats_milliseconds,
        stats.timer_eval_main_milliseconds,
        stats.timer_eval_pair_milliseconds,
        stats.timer_valid_main_milliseconds,
        stats.timer_valid_pair_milliseconds,
        stats.timer_iter_setup_milliseconds,
        stats.timer_regression_milliseconds,
        stats.timer_mask_chain_milliseconds,
        stats.timer_aggregation_accumulate_milliseconds,
        stats.timer_aggregation_finalize_milliseconds,
        stats.timer_snapshots_milliseconds,
        stats.timer_eval_store_milliseconds,
        stats.timer_valid_store_milliseconds};
    constexpr mwSize timer_count = sizeof(timer_names) / sizeof(timer_names[0]);
    mxArray* timer_labels = mxCreateCellMatrix(1, timer_count);
    mxArray* timer_ms = mxCreateDoubleMatrix(1, timer_count, mxREAL);
    for (mwSize index = 0; index < timer_count; ++index) {
        mxSetCell(timer_labels, index, mxCreateString(timer_names[index]));
        mxGetPr(timer_ms)[index] = timer_values[index];
    }
    mxSetField(result, 0, "timer_labels", timer_labels);
    mxSetField(result, 0, "timer_ms", timer_ms);
    set("host_prep_ms", stats.host_prepare_milliseconds);
    set("device_loop_ms", stats.device_loop_milliseconds);
    set("output_pack_ms", stats.output_pack_milliseconds);
    set("sparse_pairs_total", static_cast<double>(stats.sparse_pair_count));
    set("host_cuda_allocate_ms", stats.host_cuda_allocate_milliseconds);
    set("host_cuda_free_ms", stats.host_cuda_free_milliseconds);
    set("host_cuda_event_destroy_ms",
        stats.host_cuda_event_destroy_milliseconds);
    return result;
}

mxArray* phase2_output(int nrhs, const mxArray* prhs[])
{
    if (nrhs < 11 || nrhs > 18) {
        throw std::runtime_error(
            "phase2 requires X, Y, zero-based idx, Tan, Cen, owner_mask0, hk, "
            "rhok, MM, opts, then optional X_Test, Y_Test, Tan_Test, "
            "C_Test, eval_opts, X_Valid, and Y_Valid.");
    }
    const mxArray* x = prhs[1];
    const mxArray* response = prhs[2];
    const mxArray* index = prhs[3];
    const mxArray* tangent = prhs[4];
    const mxArray* center = prhs[5];
    const mxArray* owner = prhs[6];
    require_single(x, "X");
    require_single(response, "Y");
    require_single(tangent, "Tan");
    require_single(center, "Cen");
    if (mxGetNumberOfDimensions(x) != 2 || mxGetNumberOfDimensions(response) != 2
        || mxGetNumberOfDimensions(center) != 2) {
        throw std::runtime_error("X, Y, and Cen must be two-dimensional.");
    }
    if (!mxIsInt32(index) || mxIsComplex(index) || mxIsSparse(index)
        || mxGetNumberOfDimensions(index) != 2) {
        throw std::runtime_error("idx must be dense zero-based int32 [k,N] or [k,1].");
    }
    if (!mxIsLogical(owner) || mxIsComplex(owner) || mxIsSparse(owner))
        throw std::runtime_error("owner_mask0 must be a dense logical row vector.");

    const int64_t point_count = static_cast<int64_t>(mxGetN(x));
    const int32_t ambient_dimension = static_cast<int32_t>(mxGetM(x));
    const int32_t response_dimension = static_cast<int32_t>(mxGetM(response));
    const int32_t manifold_dimension = integer_scalar(prhs[9], "MM");
    if (point_count < 1 || mxGetN(response) != static_cast<mwSize>(point_count)
        || mxGetM(center) != static_cast<mwSize>(ambient_dimension)
        || mxGetN(center) != static_cast<mwSize>(point_count)
        || mxGetM(owner) != 1 || mxGetN(owner) != static_cast<mwSize>(point_count)
        || mxGetNumberOfDimensions(tangent) != 3) {
        throw std::runtime_error("Phase-2 training dimensions are inconsistent.");
    }
    const mwSize* tangent_dimensions = mxGetDimensions(tangent);
    if (tangent_dimensions[0] != static_cast<mwSize>(ambient_dimension)
        || tangent_dimensions[1] != static_cast<mwSize>(manifold_dimension)
        || tangent_dimensions[2] != static_cast<mwSize>(point_count)) {
        throw std::runtime_error("Tan must have shape [D,MM,N].");
    }

    ParsedPhase2Options parsed = phase2_options(prhs[10]);
    parsed.core.h_schedule = parsed.h_schedule.empty()
        ? nullptr : parsed.h_schedule.data();
    const int32_t neighbor_count = static_cast<int32_t>(mxGetM(index));
    const int64_t index_columns = static_cast<int64_t>(mxGetN(index));
    if ((parsed.core.index_is_shared_slice && index_columns != 1)
        || (!parsed.core.index_is_shared_slice && index_columns != point_count)) {
        throw std::runtime_error(
            "idx must be [k,1] for a shared slice or [k,N] otherwise.");
    }
    const float base_h = static_cast<float>(real_scalar(prhs[7], "hk"));
    std::vector<float> rho_schedule = float_vector(prhs[8], "rhok", true);
    if (rho_schedule.size() != 1
        && rho_schedule.size() != static_cast<size_t>(parsed.core.iteration_count)) {
        throw std::runtime_error(
            "rhok must be scalar or have exactly K/iterations entries.");
    }

    const mxArray* test_x = nrhs >= 12 ? prhs[11] : nullptr;
    const mxArray* test_y = nrhs >= 13 ? prhs[12] : nullptr;
    const mxArray* test_tangent = nrhs >= 14 ? prhs[13] : nullptr;
    const mxArray* test_center = nrhs >= 15 ? prhs[14] : nullptr;
    const mxArray* eval_options = nrhs >= 16 && nonempty(prhs[15]) ? prhs[15] : nullptr;
    const mxArray* validation_x = nrhs >= 17 ? prhs[16] : nullptr;
    const mxArray* validation_y = nrhs >= 18 ? prhs[17] : nullptr;
    ParsedPhase2Evaluation evaluation = phase2_evaluation_options(eval_options);
    evaluation.core.rho_scales = evaluation.rho_scales.data();
    evaluation.core.h_scales = evaluation.h_scales.data();
    evaluation.core.evaluated_iterations = evaluation.evaluated_iterations.empty()
        ? nullptr : evaluation.evaluated_iterations.data();
    const bool has_test_y = nonempty(test_y);
    const bool has_test_tangent = nonempty(test_tangent);
    const bool has_test_center = nonempty(test_center);
    evaluation.test_metrics = has_test_y || has_test_tangent || has_test_center;
    if (evaluation.test_metrics) {
        if (!nonempty(test_x))
            throw std::runtime_error("Testing truth requires nonempty X_Test.");
        require_single(test_x, "X_Test");
        if (mxGetNumberOfDimensions(test_x) != 2
            || mxGetM(test_x) != static_cast<mwSize>(ambient_dimension))
            throw std::runtime_error("X_Test must have shape [D,Ntest].");
        evaluation.core.test_point_count = static_cast<int64_t>(mxGetN(test_x));
        evaluation.core.test_x_point_major = static_cast<const float*>(mxGetData(test_x));
    }
    if (has_test_y) {
        require_single(test_y, "Y_Test");
        if (mxGetNumberOfDimensions(test_y) != 2
            || mxGetM(test_y) != static_cast<mwSize>(response_dimension)
            || mxGetN(test_y) != static_cast<mwSize>(evaluation.core.test_point_count))
            throw std::runtime_error("Y_Test must have shape [MY,Ntest].");
        evaluation.core.test_y_point_major = static_cast<const float*>(mxGetData(test_y));
    }
    if (has_test_tangent) {
        require_single(test_tangent, "Tan_Test");
        if (mxGetNumberOfDimensions(test_tangent) != 3)
            throw std::runtime_error("Tan_Test must have shape [D,MM,Ntest].");
        const mwSize* dimensions = mxGetDimensions(test_tangent);
        if (dimensions[0] != static_cast<mwSize>(ambient_dimension)
            || dimensions[1] != static_cast<mwSize>(manifold_dimension)
            || dimensions[2] != static_cast<mwSize>(evaluation.core.test_point_count))
            throw std::runtime_error("Tan_Test must have shape [D,MM,Ntest].");
        evaluation.core.test_tangent_point_major =
            static_cast<const float*>(mxGetData(test_tangent));
    }
    if (has_test_center) {
        require_single(test_center, "C_Test");
        if (mxGetNumberOfDimensions(test_center) != 2
            || mxGetM(test_center) != static_cast<mwSize>(ambient_dimension)
            || mxGetN(test_center) != static_cast<mwSize>(evaluation.core.test_point_count))
            throw std::runtime_error("C_Test must have shape [D,Ntest].");
        evaluation.core.test_center_point_major =
            static_cast<const float*>(mxGetData(test_center));
    }
    if (nonempty(validation_x) != nonempty(validation_y))
        throw std::runtime_error("X_Valid and Y_Valid must be supplied together.");
    evaluation.validation = nonempty(validation_x);
    if (evaluation.validation) {
        require_single(validation_x, "X_Valid");
        require_single(validation_y, "Y_Valid");
        if (mxGetNumberOfDimensions(validation_x) != 2
            || mxGetNumberOfDimensions(validation_y) != 2
            || mxGetM(validation_x) != static_cast<mwSize>(ambient_dimension)
            || mxGetM(validation_y) != static_cast<mwSize>(response_dimension)
            || mxGetN(validation_x) != mxGetN(validation_y)) {
            throw std::runtime_error(
                "X_Valid and Y_Valid must have shapes [D,Nvalid] and [MY,Nvalid].");
        }
        evaluation.core.validation_point_count = static_cast<int64_t>(
            mxGetN(validation_x));
        evaluation.core.validation_x_point_major =
            static_cast<const float*>(mxGetData(validation_x));
        evaluation.core.validation_y_point_major =
            static_cast<const float*>(mxGetData(validation_y));
    }
    evaluation.enabled = evaluation.test_metrics || evaluation.validation;
    for (int32_t iteration : evaluation.evaluated_iterations) {
        if (iteration < 0 || iteration > parsed.core.iteration_count)
            throw std::runtime_error("eval_k_arr entries must lie in [0,K].");
    }

    ManorMCudaContext* context = context_for_device(parsed.device_ordinal);
    const char* fields[] = {
        "Tan_next", "Cen_next", "num_neigh", "heavy_mask", "keep_mask",
        "heavy_mask_iter", "keep_mask_iter", "num_neigh_iter",
        "valid_iterations", "stopped_no_heavy", "fused_eval_executed",
        "owner_count", "solver_failed", "profile", "grad_debug", "EvalMSE",
        "EvalMSECoordinate", "EvalDevCen", "EvalDevCenTan", "EvalDevTan",
        "EvalMSE0", "EvalMSECoordinate0", "EvalDevCen0", "EvalDevCenTan0",
        "EvalDevTan0", "EvalValidMSE", "EvalValidMSECoordinate",
        "EvalValidMSE0", "EvalValidMSECoordinate0", "fused_validation_count"};
    mxArray* result = mxCreateStructMatrix(1, 1, 30, fields);
    auto empty_single = [&]() {
        return mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
    };
    auto set_empty = [&](const char* name) {
        mxSetField(result, 0, name, empty_single());
    };
    mwSize output_tangent_dimensions[3] = {
        static_cast<mwSize>(ambient_dimension),
        static_cast<mwSize>(manifold_dimension),
        static_cast<mwSize>(point_count)};
    mxArray* tangent_next = mxCreateNumericArray(
        3, output_tangent_dimensions, mxSINGLE_CLASS, mxREAL);
    mxArray* center_next = mxCreateNumericMatrix(
        ambient_dimension, static_cast<mwSize>(point_count), mxSINGLE_CLASS, mxREAL);
    mxArray* neighbors = mxCreateNumericMatrix(
        1, static_cast<mwSize>(point_count), mxINT32_CLASS, mxREAL);
    mxArray* heavy = mxCreateLogicalMatrix(1, static_cast<mwSize>(point_count));
    mxArray* keep = mxCreateLogicalMatrix(1, static_cast<mwSize>(point_count));
    mxArray* solver = mxCreateLogicalMatrix(1, static_cast<mwSize>(point_count));
    mwSize history_dimensions[2] = {
        static_cast<mwSize>(point_count),
        static_cast<mwSize>(parsed.core.iteration_count)};
    mxArray* heavy_history = mxCreateLogicalArray(2, history_dimensions);
    mxArray* keep_history = mxCreateLogicalArray(2, history_dimensions);
    mxArray* neighbor_history = mxCreateNumericArray(
        2, history_dimensions, mxINT32_CLASS, mxREAL);
    mxSetField(result, 0, "Tan_next", tangent_next);
    mxSetField(result, 0, "Cen_next", center_next);
    mxSetField(result, 0, "num_neigh", neighbors);
    mxSetField(result, 0, "heavy_mask", heavy);
    mxSetField(result, 0, "keep_mask", keep);
    mxSetField(result, 0, "solver_failed", solver);
    mxSetField(result, 0, "heavy_mask_iter", heavy_history);
    mxSetField(result, 0, "keep_mask_iter", keep_history);
    mxSetField(result, 0, "num_neigh_iter", neighbor_history);

    mxArray* gradient = nullptr;
    if (parsed.return_debug) {
        mwSize dimensions[3] = {
            static_cast<mwSize>(ambient_dimension),
            static_cast<mwSize>(response_dimension),
            static_cast<mwSize>(point_count)};
        gradient = mxCreateNumericArray(3, dimensions, mxSINGLE_CLASS, mxREAL);
        mxSetField(result, 0, "grad_debug", gradient);
    } else {
        set_empty("grad_debug");
    }

    const mwSize n_rho = static_cast<mwSize>(evaluation.rho_scales.size());
    const mwSize n_h = static_cast<mwSize>(evaluation.h_scales.size());
    mxArray* eval_mse = nullptr;
    mxArray* eval_coordinate = nullptr;
    mxArray* eval_center = nullptr;
    mxArray* eval_center_tangent = nullptr;
    mxArray* eval_tangent = nullptr;
    mxArray* eval_mse0 = nullptr;
    mxArray* eval_coordinate0 = nullptr;
    mxArray* eval_center0 = nullptr;
    mxArray* eval_center_tangent0 = nullptr;
    mxArray* eval_tangent0 = nullptr;
    mxArray* valid_mse = nullptr;
    mxArray* valid_coordinate = nullptr;
    mxArray* valid_mse0 = nullptr;
    mxArray* valid_coordinate0 = nullptr;
    if (evaluation.enabled) {
        mwSize dimensions[3] = {
            static_cast<mwSize>(parsed.core.iteration_count), n_rho, n_h};
        mwSize coordinate_dimensions[4] = {
            static_cast<mwSize>(parsed.core.iteration_count), n_rho, n_h,
            static_cast<mwSize>(response_dimension)};
        auto make_eval = [&](const char* name) {
            mxArray* value = mxCreateNumericArray(3, dimensions, mxSINGLE_CLASS, mxREAL);
            mxSetField(result, 0, name, value);
            return value;
        };
        eval_mse = make_eval("EvalMSE");
        eval_coordinate = mxCreateNumericArray(
            4, coordinate_dimensions, mxSINGLE_CLASS, mxREAL);
        mxSetField(result, 0, "EvalMSECoordinate", eval_coordinate);
        eval_center = make_eval("EvalDevCen");
        eval_center_tangent = make_eval("EvalDevCenTan");
        eval_tangent = make_eval("EvalDevTan");
        if (evaluation.eval_k0 && evaluation.test_metrics) {
            mwSize k0_dimensions[3] = {1, n_rho, n_h};
            mwSize k0_coordinate_dimensions[4] = {
                1, n_rho, n_h, static_cast<mwSize>(response_dimension)};
            auto make_k0 = [&](const char* name) {
                mxArray* value = mxCreateNumericArray(
                    3, k0_dimensions, mxSINGLE_CLASS, mxREAL);
                mxSetField(result, 0, name, value);
                return value;
            };
            eval_mse0 = make_k0("EvalMSE0");
            eval_coordinate0 = mxCreateNumericArray(
                4, k0_coordinate_dimensions, mxSINGLE_CLASS, mxREAL);
            mxSetField(result, 0, "EvalMSECoordinate0", eval_coordinate0);
            eval_center0 = make_k0("EvalDevCen0");
            eval_center_tangent0 = make_k0("EvalDevCenTan0");
            eval_tangent0 = make_k0("EvalDevTan0");
        } else {
            set_empty("EvalMSE0"); set_empty("EvalMSECoordinate0");
            set_empty("EvalDevCen0"); set_empty("EvalDevCenTan0");
            set_empty("EvalDevTan0");
        }
        if (evaluation.validation) {
            valid_mse = make_eval("EvalValidMSE");
            valid_coordinate = mxCreateNumericArray(
                4, coordinate_dimensions, mxSINGLE_CLASS, mxREAL);
            mxSetField(result, 0, "EvalValidMSECoordinate", valid_coordinate);
            if (evaluation.eval_k0) {
                mwSize k0_dimensions[3] = {1, n_rho, n_h};
                mwSize k0_coordinate_dimensions[4] = {
                    1, n_rho, n_h, static_cast<mwSize>(response_dimension)};
                valid_mse0 = mxCreateNumericArray(
                    3, k0_dimensions, mxSINGLE_CLASS, mxREAL);
                valid_coordinate0 = mxCreateNumericArray(
                    4, k0_coordinate_dimensions, mxSINGLE_CLASS, mxREAL);
                mxSetField(result, 0, "EvalValidMSE0", valid_mse0);
                mxSetField(result, 0, "EvalValidMSECoordinate0", valid_coordinate0);
            } else {
                set_empty("EvalValidMSE0");
                set_empty("EvalValidMSECoordinate0");
            }
        } else {
            set_empty("EvalValidMSE"); set_empty("EvalValidMSECoordinate");
            set_empty("EvalValidMSE0"); set_empty("EvalValidMSECoordinate0");
        }
    } else {
        const char* empty_fields[] = {
            "EvalMSE", "EvalMSECoordinate", "EvalDevCen", "EvalDevCenTan",
            "EvalDevTan", "EvalMSE0", "EvalMSECoordinate0", "EvalDevCen0",
            "EvalDevCenTan0", "EvalDevTan0", "EvalValidMSE",
            "EvalValidMSECoordinate", "EvalValidMSE0", "EvalValidMSECoordinate0"};
        for (const char* name : empty_fields) set_empty(name);
    }

    ManorMCudaPhase2Input input{};
    input.struct_size = sizeof(input);
    input.abi_version = MANOR_M_CUDA_ABI_VERSION;
    input.x_point_major = static_cast<const float*>(mxGetData(x));
    input.response_point_major = static_cast<const float*>(mxGetData(response));
    input.index_zero_based = static_cast<const int32_t*>(mxGetData(index));
    input.tangent_point_major = static_cast<const float*>(mxGetData(tangent));
    input.center_point_major = static_cast<const float*>(mxGetData(center));
    input.owner_mask = reinterpret_cast<const uint8_t*>(mxGetLogicals(owner));
    input.point_count = point_count;
    input.ambient_dimension = ambient_dimension;
    input.manifold_dimension = manifold_dimension;
    input.response_dimension = response_dimension;
    input.neighbor_count = neighbor_count;
    input.base_h = base_h;
    input.rho_schedule = rho_schedule.data();
    input.rho_schedule_count = static_cast<int32_t>(rho_schedule.size());

    ManorMCudaPhase2Output output{};
    output.struct_size = sizeof(output);
    output.abi_version = MANOR_M_CUDA_ABI_VERSION;
    output.tangent_point_major = static_cast<float*>(mxGetData(tangent_next));
    output.center_point_major = static_cast<float*>(mxGetData(center_next));
    output.neighbor_count = static_cast<int32_t*>(mxGetData(neighbors));
    output.heavy_mask = reinterpret_cast<uint8_t*>(mxGetLogicals(heavy));
    output.keep_mask = reinterpret_cast<uint8_t*>(mxGetLogicals(keep));
    output.solver_failed = reinterpret_cast<uint8_t*>(mxGetLogicals(solver));
    output.neighbor_count_history = static_cast<int32_t*>(mxGetData(neighbor_history));
    output.heavy_mask_history = reinterpret_cast<uint8_t*>(mxGetLogicals(heavy_history));
    output.keep_mask_history = reinterpret_cast<uint8_t*>(mxGetLogicals(keep_history));
    output.gradient_debug_point_major = gradient
        ? static_cast<float*>(mxGetData(gradient)) : nullptr;
    output.eval_mse = eval_mse ? static_cast<float*>(mxGetData(eval_mse)) : nullptr;
    output.eval_mse_coordinate = eval_coordinate
        ? static_cast<float*>(mxGetData(eval_coordinate)) : nullptr;
    output.eval_dev_center = eval_center
        ? static_cast<float*>(mxGetData(eval_center)) : nullptr;
    output.eval_dev_center_tangent = eval_center_tangent
        ? static_cast<float*>(mxGetData(eval_center_tangent)) : nullptr;
    output.eval_dev_tangent = eval_tangent
        ? static_cast<float*>(mxGetData(eval_tangent)) : nullptr;
    output.eval_mse_k0 = eval_mse0 ? static_cast<float*>(mxGetData(eval_mse0)) : nullptr;
    output.eval_mse_coordinate_k0 = eval_coordinate0
        ? static_cast<float*>(mxGetData(eval_coordinate0)) : nullptr;
    output.eval_dev_center_k0 = eval_center0
        ? static_cast<float*>(mxGetData(eval_center0)) : nullptr;
    output.eval_dev_center_tangent_k0 = eval_center_tangent0
        ? static_cast<float*>(mxGetData(eval_center_tangent0)) : nullptr;
    output.eval_dev_tangent_k0 = eval_tangent0
        ? static_cast<float*>(mxGetData(eval_tangent0)) : nullptr;
    output.validation_mse = valid_mse
        ? static_cast<float*>(mxGetData(valid_mse)) : nullptr;
    output.validation_mse_coordinate = valid_coordinate
        ? static_cast<float*>(mxGetData(valid_coordinate)) : nullptr;
    output.validation_mse_k0 = valid_mse0
        ? static_cast<float*>(mxGetData(valid_mse0)) : nullptr;
    output.validation_mse_coordinate_k0 = valid_coordinate0
        ? static_cast<float*>(mxGetData(valid_coordinate0)) : nullptr;

    ManorMCudaPhase2Stats stats{};
    const int status = manor_m_cuda_phase2(
        context, &input, &parsed.core,
        evaluation.enabled ? &evaluation.core : nullptr,
        &output, &stats);
    if (status != MANOR_M_CUDA_SUCCESS) {
        const char* detail = manor_m_cuda_last_error();
        const std::string message = std::string("phase2: ")
            + manor_m_cuda_status_string(status)
            + (detail && detail[0] ? std::string(": ") + detail : "");
        mxDestroyArray(result);
        throw std::runtime_error(message);
    }
    mxSetField(result, 0, "valid_iterations",
        mxCreateDoubleScalar(stats.valid_iterations));
    mxSetField(result, 0, "stopped_no_heavy",
        mxCreateLogicalScalar(stats.stopped_no_heavy != 0));
    mxSetField(result, 0, "fused_eval_executed",
        mxCreateDoubleScalar(stats.evaluation_iterations_executed));
    mxSetField(result, 0, "owner_count",
        mxCreateDoubleScalar(static_cast<double>(stats.owner_count)));
    mxSetField(result, 0, "profile",
        phase2_profile(stats, parsed.core,
            evaluation.enabled ? &evaluation.core : nullptr,
            static_cast<int>(rho_schedule.size())));
    mxSetField(result, 0, "fused_validation_count",
        mxCreateDoubleScalar(static_cast<double>(stats.validation_point_count)));
    return result;
}

mxArray* device_info_output(int32_t ordinal)
{
    ManorMCudaDeviceInfo info{};
    check_status(manor_m_cuda_get_device_info(ordinal, &info), "device_info");
    const char* fields[] = {
        "abi_version", "device_ordinal", "name", "compute_capability",
        "multiprocessor_count", "warp_size", "max_threads_per_block",
        "cuda_driver_version", "cuda_runtime_version",
        "total_global_memory_bytes", "free_global_memory_bytes"};
    mxArray* output = mxCreateStructMatrix(1, 1, 11, fields);
    mxSetField(output, 0, "abi_version", mxCreateDoubleScalar(info.abi_version));
    mxSetField(output, 0, "device_ordinal", mxCreateDoubleScalar(info.device_ordinal));
    mxSetField(output, 0, "name", mxCreateString(info.name));
    mxSetField(
        output,
        0,
        "compute_capability",
        mxCreateDoubleScalar(
            static_cast<double>(info.compute_capability_major)
            + static_cast<double>(info.compute_capability_minor) / 10.0));
    mxSetField(output, 0, "multiprocessor_count", mxCreateDoubleScalar(info.multiprocessor_count));
    mxSetField(output, 0, "warp_size", mxCreateDoubleScalar(info.warp_size));
    mxSetField(output, 0, "max_threads_per_block", mxCreateDoubleScalar(info.max_threads_per_block));
    mxSetField(output, 0, "cuda_driver_version", mxCreateDoubleScalar(info.cuda_driver_version));
    mxSetField(output, 0, "cuda_runtime_version", mxCreateDoubleScalar(info.cuda_runtime_version));
    mxSetField(
        output,
        0,
        "total_global_memory_bytes",
        mxCreateDoubleScalar(static_cast<double>(info.total_global_memory_bytes)));
    mxSetField(
        output,
        0,
        "free_global_memory_bytes",
        mxCreateDoubleScalar(static_cast<double>(info.free_global_memory_bytes)));
    return output;
}

mxArray* smoke_axpy_output(
    const mxArray* x,
    const mxArray* y,
    float alpha,
    int32_t device_ordinal)
{
    if (!mxIsSingle(x) || mxIsComplex(x) || !mxIsSingle(y) || mxIsComplex(y)) {
        throw std::runtime_error("x and y must be real single arrays.");
    }
    const mwSize count = mxGetNumberOfElements(x);
    if (mxGetNumberOfElements(y) != count) {
        throw std::runtime_error("x and y must have the same number of elements.");
    }
    const mwSize dimensions = mxGetNumberOfDimensions(x);
    mxArray* output = mxCreateNumericArray(
        dimensions, mxGetDimensions(x), mxSINGLE_CLASS, mxREAL);
    check_status(
        manor_m_cuda_smoke_axpy(
            device_ordinal,
            alpha,
            static_cast<const float*>(mxGetData(x)),
            static_cast<const float*>(mxGetData(y)),
            static_cast<float*>(mxGetData(output)),
            static_cast<int64_t>(count)),
        "smoke_axpy");
    return output;
}

}  // namespace

extern "C" void mexFunction(
    int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    try {
        if (nrhs < 1) {
            throw std::runtime_error(
                "Usage: manor_m_cuda_mex(command, ...). Commands: version, "
                "device_count, info, smoke_axpy, response_knn, phase1_precomputed, phase2, "
                "reset_contexts.");
        }
        if (nlhs > 1) throw std::runtime_error("At most one output is supported.");
        const std::string command = command_string(prhs[0]);

        mxArray* output = nullptr;
        if (command == "version") {
            if (nrhs != 1) throw std::runtime_error("version takes no arguments.");
            output = mxCreateString(manor_m_cuda_version());
        } else if (command == "device_count") {
            if (nrhs != 1) throw std::runtime_error("device_count takes no arguments.");
            int32_t count = 0;
            check_status(manor_m_cuda_device_count(&count), "device_count");
            output = mxCreateDoubleScalar(count);
        } else if (command == "info") {
            if (nrhs != 2) throw std::runtime_error("info requires device_ordinal.");
            output = device_info_output(integer_scalar(prhs[1], "device_ordinal"));
        } else if (command == "smoke_axpy") {
            if (nrhs < 4 || nrhs > 5) {
                throw std::runtime_error(
                    "smoke_axpy requires x, y, alpha, and optional device_ordinal.");
            }
            if (!mxIsNumeric(prhs[3]) || mxIsComplex(prhs[3])
                || mxGetNumberOfElements(prhs[3]) != 1) {
                throw std::runtime_error("alpha must be a real numeric scalar.");
            }
            const int32_t device = nrhs == 5
                ? integer_scalar(prhs[4], "device_ordinal")
                : 0;
            output = smoke_axpy_output(
                prhs[1], prhs[2], static_cast<float>(mxGetScalar(prhs[3])), device);
        } else if (command == "response_knn") {
            output = response_knn_output(nrhs, prhs);
        } else if (command == "phase1_precomputed") {
            output = phase1_precomputed_output(nrhs, prhs);
        } else if (command == "phase2") {
            output = phase2_output(nrhs, prhs);
        } else if (command == "reset_contexts") {
            if (nrhs != 1) {
                throw std::runtime_error("reset_contexts takes no arguments.");
            }
            destroy_contexts();
        } else {
            throw std::runtime_error("Unknown command: " + command);
        }

        if (nlhs == 1) {
            plhs[0] = output;
        } else if (output != nullptr) {
            mxDestroyArray(output);
        }
    } catch (const std::exception& error) {
        mexErrMsgIdAndTxt("manor_m:cuda:adapter", "%s", error.what());
    } catch (...) {
        mexErrMsgIdAndTxt("manor_m:cuda:adapter", "Unknown adapter failure.");
    }
}
