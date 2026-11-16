/* Phase-2 migration translation unit.
 *
 * The verified production execution body is compiled into the language-
 * neutral library through a private, header-only mxArray compatibility layer.
 * No MATLAB header or library is linked.  This transition mechanism keeps
 * every CUDA launch, reduction, optional-metric rule, and numerical ordering
 * in one source while MATLAB/Python parity is established.  The public ABI
 * below accepts only plain C descriptors and caller-owned host buffers.
 */

#include "manor_m_cuda.h"
#include "manor_m_cuda_internal.h"
#include "manor_m_cuda_resident.h"

#define MANOR_M_PHASE2_CORE_LIBRARY 1
#define mexFunction manor_m_phase2_minimex_entry
#include "../../matlab/cuda_mex/manor_m_phase2_mex.cu"
#undef mexFunction

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

thread_local ManorMGeneratedPhase2Source*
    manor_m_cuda_active_generated_phase2 = nullptr;

namespace {

class MiniArrayOwner {
public:
    ~MiniArrayOwner()
    {
        for (mxArray* value : values_) mxDestroyArray(value);
    }

    mxArray* add(mxArray* value)
    {
        values_.push_back(value);
        return value;
    }

private:
    std::vector<mxArray*> values_;
};

static void phase2_require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

static bool finite_positive(float value)
{
    return std::isfinite(value) && value > 0.0f;
}

static mxArray* phase2_view(
    mxClassID class_id, const void* data, std::vector<mwSize> dimensions)
{
    return manor_m_minimex_view(
        class_id, const_cast<void*>(data), std::move(dimensions));
}

static void phase2_set_double(mxArray* options, const char* name, double value)
{
    mxSetField(options, 0, name, mxCreateDoubleScalar(value));
}

static void phase2_set_logical(mxArray* options, const char* name, bool value)
{
    mxSetField(options, 0, name, mxCreateLogicalScalar(value));
}

static void phase2_set_view(
    mxArray* options, const char* name, mxClassID class_id,
    const void* data, std::vector<mwSize> dimensions)
{
    mxSetField(options, 0, name,
        phase2_view(class_id, data, std::move(dimensions)));
}

static const mxArray* phase2_field(const mxArray* structure, const char* name)
{
    return mxGetField(structure, 0, name);
}

static double phase2_scalar_field(
    const mxArray* structure, const char* name, double fallback = 0.0)
{
    const mxArray* value = phase2_field(structure, name);
    return value && mxGetNumberOfElements(value) == 1
        ? mxGetScalar(value) : fallback;
}

static double phase2_timer_total(const mxArray* profile, const char* label)
{
    const mxArray* labels = phase2_field(profile, "timer_labels");
    const mxArray* values = phase2_field(profile, "timer_ms");
    if (!labels || labels->class_id != mxCELL_CLASS ||
        !values || !mxIsDouble(values) ||
        labels->cells.size() != mxGetNumberOfElements(values)) {
        return 0.0;
    }
    const double* milliseconds = mxGetPr(values);
    for (size_t index = 0; index < labels->cells.size(); ++index) {
        const mxArray* value = labels->cells[index];
        if (value && value->class_id == mxCHAR_CLASS && value->text == label)
            return milliseconds[index];
    }
    return 0.0;
}

template <typename T>
static void phase2_copy_field(
    const mxArray* result, const char* name, mxClassID class_id,
    T* destination, size_t count)
{
    if (!destination) return;
    const mxArray* source = phase2_field(result, name);
    phase2_require(source != nullptr, "Requested Phase-2 output field is missing.");
    phase2_require(mxGetClassID(source) == class_id,
        "Requested Phase-2 output field has an unexpected type.");
    phase2_require(mxGetNumberOfElements(source) == count,
        "Requested Phase-2 output field has an unexpected size.");
    if (count > 0 && destination != mxGetData(source))
        std::memcpy(destination, mxGetData(source), count * sizeof(T));
}

static size_t checked_product(size_t left, size_t right, const char* message)
{
    phase2_require(right == 0 || left <= std::numeric_limits<size_t>::max() / right,
        message);
    return left * right;
}

struct Phase2Shape {
    int N = 0;
    int D = 0;
    int MM = 0;
    int MY = 0;
    int k = 0;
    int K = 0;
    int n_rho = 1;
    int n_h = 1;
    bool evaluate = false;
    bool validate = false;
    bool validation_tangent = false;
    bool test_metrics = false;
    bool eval_k0 = false;
};

static Phase2Shape phase2_validate(
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    bool generated_response_index)
{
    phase2_require(input && options && output,
        "Phase-2 input, options, and output descriptors are required.");
    phase2_require(input->struct_size == sizeof(*input) &&
        input->abi_version == MANOR_M_CUDA_ABI_VERSION,
        "Phase-2 input size or ABI version does not match the core.");
    phase2_require(options->struct_size == sizeof(*options) &&
        options->abi_version == MANOR_M_CUDA_ABI_VERSION,
        "Phase-2 options size or ABI version does not match the core.");
    phase2_require((output->struct_size == sizeof(*output) ||
        output->struct_size == offsetof(ManorMCudaPhase2Output, validation_dev_tangent)) &&
        output->abi_version == MANOR_M_CUDA_ABI_VERSION,
        "Phase-2 output size or ABI version does not match the core.");
    phase2_require(input->x_point_major && input->response_point_major &&
        (generated_response_index || input->index_zero_based) &&
        input->tangent_point_major &&
        input->center_point_major && input->owner_mask,
        "Phase-2 training inputs must not be null.");
    const bool evaluation_only = (options->reserved & 2) != 0;
    if (evaluation_only) {
        phase2_require(!output->tangent_point_major &&
                !output->center_point_major && !output->neighbor_count &&
                !output->heavy_mask && !output->keep_mask &&
                !output->solver_failed,
            "Evaluation-only Phase-2 state/status outputs must be null.");
    } else {
        phase2_require(output->tangent_point_major &&
                output->center_point_major,
            "Phase-2 final tangent and center outputs are required.");
    }
    phase2_require(input->point_count >= 1 &&
        input->point_count <= std::numeric_limits<int>::max(),
        "Phase-2 point_count lies outside the supported int32 range.");
    phase2_require(input->ambient_dimension >= 2 &&
        input->ambient_dimension <= MAX_D_MANOR,
        "Phase-2 ambient dimension is unsupported.");
    phase2_require(input->manifold_dimension >= 1 &&
        input->manifold_dimension < input->ambient_dimension &&
        input->manifold_dimension <= MAX_MM_MANOR,
        "Phase-2 manifold dimension is unsupported.");
    phase2_require(input->response_dimension >= 1 &&
        input->response_dimension <= MAX_MY_MANOR,
        "Phase-2 response dimension is unsupported.");
    phase2_require(input->neighbor_count >= 1 &&
        input->neighbor_count <= input->point_count,
        "Phase-2 neighbor_count lies outside [1,point_count].");
    phase2_require(finite_positive(input->base_h),
        "Phase-2 base_h must be finite and positive.");
    phase2_require(input->rho_schedule &&
        (input->rho_schedule_count == 1 ||
         input->rho_schedule_count == options->iteration_count),
        "Phase-2 rho schedule must be scalar or have iteration_count entries.");
    for (int index = 0; index < input->rho_schedule_count; ++index)
        phase2_require(finite_positive(input->rho_schedule[index]),
            "Phase-2 rho schedule entries must be finite and positive.");
    phase2_require(options->iteration_count >= 1,
        "Phase-2 iteration_count must be positive.");
    phase2_require(std::isfinite(options->lambda) && options->lambda >= 0.0f &&
        std::isfinite(options->jitter) && options->jitter >= 0.0f &&
        std::isfinite(options->gpu_memory_fraction) &&
        options->gpu_memory_fraction > 0.0f &&
        options->gpu_memory_fraction <= 1.0f &&
        std::isfinite(options->reserve_memory_fraction) &&
        options->reserve_memory_fraction >= 0.0f &&
        options->reserve_memory_fraction < 1.0f &&
        options->minimum_query_block >= 1 &&
        options->maximum_query_block >= options->minimum_query_block &&
        (options->regression_mode == 0 || options->regression_mode == 1) &&
        std::isfinite(options->heavy_factor) && options->heavy_factor >= 0.0f &&
        std::isfinite(options->temporary_heavy_factor) &&
        options->temporary_heavy_factor >= 0.0f,
        "Phase-2 numerical or memory options are invalid.");
    phase2_require(options->h_schedule_count == 0 ||
        ((options->h_schedule_count == 1 ||
          options->h_schedule_count == options->iteration_count) &&
         options->h_schedule),
        "Phase-2 h schedule must be absent, scalar, or iteration-sized.");
    for (int index = 0; index < options->h_schedule_count; ++index)
        phase2_require(finite_positive(options->h_schedule[index]),
            "Phase-2 h schedule entries must be finite and positive.");
    phase2_require(options->index_is_shared_slice == 0 ||
        options->index_is_shared_slice == 1,
        "Phase-2 index_is_shared_slice must be zero or one.");
    phase2_require(options->use_async_allocator == 0 ||
        options->use_async_allocator == 1,
        "Phase-2 use_async_allocator must be zero or one.");

    Phase2Shape shape;
    shape.N = static_cast<int>(input->point_count);
    shape.D = input->ambient_dimension;
    shape.MM = input->manifold_dimension;
    shape.MY = input->response_dimension;
    shape.k = input->neighbor_count;
    shape.K = options->iteration_count;

    if (evaluation) {
        phase2_require((evaluation->struct_size == sizeof(*evaluation) ||
            evaluation->struct_size == offsetof(ManorMCudaPhase2Evaluation,
                validation_tangent_point_major)) &&
            evaluation->abi_version == MANOR_M_CUDA_ABI_VERSION,
            "Phase-2 evaluation size or ABI version does not match the core.");
        const bool has_test_truth = evaluation->test_y_point_major ||
            evaluation->test_tangent_point_major ||
            evaluation->test_center_point_major;
        const bool has_validation = evaluation->validation_x_point_major ||
            evaluation->validation_y_point_major ||
            evaluation->validation_point_count != 0;
        if (has_test_truth) {
            phase2_require(evaluation->test_x_point_major &&
                evaluation->test_point_count >= 1 &&
                evaluation->test_point_count <= std::numeric_limits<int>::max(),
                "Testing truth requires a nonempty test X array.");
        }
        if (has_validation) {
            phase2_require(evaluation->validation_x_point_major &&
                evaluation->validation_y_point_major &&
                evaluation->validation_point_count >= 1 &&
                evaluation->validation_point_count <= std::numeric_limits<int>::max(),
                "Validation requires nonempty X and Y arrays.");
        }
        shape.evaluate = has_test_truth || has_validation;
        shape.validate = has_validation;
        shape.validation_tangent = evaluation->struct_size == sizeof(*evaluation) &&
            evaluation->validation_tangent_point_major;
        phase2_require(!shape.validation_tangent || (has_validation &&
            output->struct_size == sizeof(*output)),
            "Validation tangent requires validation X/Y and an extended output descriptor.");
        shape.test_metrics = has_test_truth;
        shape.n_rho = evaluation->rho_scale_count > 0
            ? evaluation->rho_scale_count : 1;
        shape.n_h = evaluation->h_scale_count > 0
            ? evaluation->h_scale_count : 1;
        phase2_require(shape.n_rho >= 1 && shape.n_h >= 1,
            "Evaluation scale counts must be positive.");
        phase2_require(evaluation->rho_scale_count == 0 || evaluation->rho_scales,
            "Evaluation rho scales pointer is null.");
        phase2_require(evaluation->h_scale_count == 0 || evaluation->h_scales,
            "Evaluation h scales pointer is null.");
        for (int index = 0; index < evaluation->rho_scale_count; ++index)
            phase2_require(finite_positive(evaluation->rho_scales[index]),
                "Evaluation rho scales must be finite and positive.");
        for (int index = 0; index < evaluation->h_scale_count; ++index)
            phase2_require(finite_positive(evaluation->h_scales[index]),
                "Evaluation h scales must be finite and positive.");
        phase2_require(evaluation->evaluated_iteration_count == 0 ||
            evaluation->evaluated_iterations,
            "Evaluated-iteration pointer is null.");
        for (int index = 0; index < evaluation->evaluated_iteration_count; ++index) {
            const int value = evaluation->evaluated_iterations[index];
            phase2_require(value >= 0 && value <= shape.K,
                "Evaluated iterations must lie in [0,K].");
            shape.eval_k0 = shape.eval_k0 || value == 0;
        }
    }
    return shape;
}

static mxArray* phase2_build_options(
    const ManorMCudaPhase2Options& source,
    const ManorMCudaPhase2Output& output,
    bool fused_eval,
    bool fused_validation,
    int32_t aggregation_tile_owners,
    int32_t index_tile_owners,
    bool compact_owner_index,
    bool generated_response_index)
{
    const char* fields[] = {
        "lambda", "jitter", "gpu_mem_fraction", "reserve_fraction",
        "reserve_bytes", "min_query_block", "max_query_block",
        "regression_mode", "iterations", "heavy_factor",
        "heavy_factor_tmp", "hk_iter", "fused_eval", "fused_validation",
        "idx_zero_based", "idx_shared_slice", "return_debug", "return_histories", "silent",
        "fused_eval_use_owner_mask", "profile_timers", "async_allocator",
        "aggregation_tile_owners", "index_tile_owners",
        "idx_compact_owner_order", "idx_generated_response",
        "evaluation_only"};
    mxArray* result = mxCreateStructMatrix(1, 1, 27, fields);
    phase2_set_double(result, "lambda", source.lambda);
    phase2_set_double(result, "jitter", source.jitter);
    phase2_set_double(result, "gpu_mem_fraction", source.gpu_memory_fraction);
    phase2_set_double(result, "reserve_fraction", source.reserve_memory_fraction);
    phase2_set_double(result, "reserve_bytes",
        static_cast<double>(source.reserve_memory_bytes));
    phase2_set_double(result, "min_query_block", source.minimum_query_block);
    phase2_set_double(result, "max_query_block", source.maximum_query_block);
    phase2_set_double(result, "regression_mode", source.regression_mode);
    phase2_set_double(result, "iterations", source.iteration_count);
    phase2_set_double(result, "heavy_factor", source.heavy_factor);
    phase2_set_double(result, "heavy_factor_tmp", source.temporary_heavy_factor);
    if (source.h_schedule_count > 0)
        phase2_set_view(result, "hk_iter", mxSINGLE_CLASS, source.h_schedule,
            {1, static_cast<mwSize>(source.h_schedule_count)});
    phase2_set_logical(result, "fused_eval", fused_eval);
    phase2_set_logical(result, "fused_validation", fused_validation);
    phase2_set_logical(result, "idx_zero_based", true);
    phase2_set_logical(result, "idx_shared_slice",
        source.index_is_shared_slice != 0);
    phase2_set_logical(result, "return_debug",
        output.gradient_debug_point_major != nullptr);
    phase2_set_logical(result, "return_histories",
        output.neighbor_count_history != nullptr ||
        output.heavy_mask_history != nullptr ||
        output.keep_mask_history != nullptr);
    phase2_set_logical(result, "silent", true);
    phase2_set_logical(result, "fused_eval_use_owner_mask",
        source.evaluation_uses_owner_mask != 0);
    phase2_set_logical(result, "profile_timers", source.collect_profile != 0);
    phase2_set_logical(result, "async_allocator",
        source.use_async_allocator != 0);
    phase2_set_double(result, "aggregation_tile_owners", aggregation_tile_owners);
    phase2_set_double(result, "index_tile_owners", index_tile_owners);
    phase2_set_logical(result, "idx_compact_owner_order", compact_owner_index);
    phase2_set_logical(result, "idx_generated_response",
        generated_response_index);
    phase2_set_logical(result, "evaluation_only",
        (source.reserved & 2) != 0);
    return result;
}

static mxArray* phase2_build_eval_options(
    const ManorMCudaPhase2Evaluation& source)
{
    const char* fields[] = {
        "sum_sparse_threshold", "count_sparse_threshold",
        "fallback_exp_scale", "eps_weight", "warps_per_block",
        "eval_rho_scale", "eval_h_scale", "eval_k_arr",
        "center_eval_rho_scale", "center_eval_h_scale"};
    mxArray* result = mxCreateStructMatrix(1, 1, 10, fields);
    phase2_set_double(result, "sum_sparse_threshold",
        source.sum_sparse_threshold);
    phase2_set_double(result, "count_sparse_threshold",
        source.count_sparse_threshold);
    if (finite_positive(source.fallback_exponent_scale))
        phase2_set_double(result, "fallback_exp_scale",
            source.fallback_exponent_scale);
    if (finite_positive(source.epsilon_weight))
        phase2_set_double(result, "eps_weight", source.epsilon_weight);
    if (source.warps_per_block != 0)
        phase2_set_double(result, "warps_per_block", source.warps_per_block);
    if (source.rho_scale_count > 0)
        phase2_set_view(result, "eval_rho_scale", mxSINGLE_CLASS,
            source.rho_scales, {1, static_cast<mwSize>(source.rho_scale_count)});
    if (source.h_scale_count > 0)
        phase2_set_view(result, "eval_h_scale", mxSINGLE_CLASS,
            source.h_scales, {1, static_cast<mwSize>(source.h_scale_count)});
    if (source.evaluated_iteration_count > 0)
        phase2_set_view(result, "eval_k_arr", mxINT32_CLASS,
            source.evaluated_iterations,
            {1, static_cast<mwSize>(source.evaluated_iteration_count)});
    if (finite_positive(source.center_rho_scale))
        phase2_set_double(result, "center_eval_rho_scale",
            source.center_rho_scale);
    if (finite_positive(source.center_h_scale))
        phase2_set_double(result, "center_eval_h_scale",
            source.center_h_scale);
    return result;
}

static void phase2_copy_outputs(
    const mxArray* result,
    const Phase2Shape& shape,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t device_ordinal,
    double total_milliseconds)
{
    const size_t N = static_cast<size_t>(shape.N);
    const size_t K = static_cast<size_t>(shape.K);
    const size_t tangent_count = checked_product(
        checked_product(N, static_cast<size_t>(shape.MM), "Phase-2 tangent size overflow."),
        static_cast<size_t>(shape.D), "Phase-2 tangent size overflow.");
    const size_t center_count = checked_product(
        N, static_cast<size_t>(shape.D), "Phase-2 center size overflow.");
    const size_t history_count = checked_product(
        N, K, "Phase-2 history size overflow.");
    phase2_copy_field(result, "Tan_next", mxSINGLE_CLASS,
        output->tangent_point_major, tangent_count);
    phase2_copy_field(result, "Cen_next", mxSINGLE_CLASS,
        output->center_point_major, center_count);
    phase2_copy_field(result, "num_neigh", mxINT32_CLASS,
        output->neighbor_count, N);
    phase2_copy_field(result, "heavy_mask", mxLOGICAL_CLASS,
        output->heavy_mask, N);
    phase2_copy_field(result, "keep_mask", mxLOGICAL_CLASS,
        output->keep_mask, N);
    phase2_copy_field(result, "solver_failed", mxLOGICAL_CLASS,
        output->solver_failed, N);
    phase2_copy_field(result, "num_neigh_iter", mxINT32_CLASS,
        output->neighbor_count_history, history_count);
    phase2_copy_field(result, "heavy_mask_iter", mxLOGICAL_CLASS,
        output->heavy_mask_history, history_count);
    phase2_copy_field(result, "keep_mask_iter", mxLOGICAL_CLASS,
        output->keep_mask_history, history_count);
    phase2_copy_field(result, "grad_debug", mxSINGLE_CLASS,
        output->gradient_debug_point_major,
        checked_product(center_count, static_cast<size_t>(shape.MY),
            "Phase-2 gradient size overflow."));

    const size_t slots = checked_product(
        static_cast<size_t>(shape.n_rho), static_cast<size_t>(shape.n_h),
        "Phase-2 evaluation-grid size overflow.");
    const size_t history_slots = checked_product(K, slots,
        "Phase-2 evaluation-history size overflow.");
    const size_t coordinate_slots = checked_product(
        history_slots, static_cast<size_t>(shape.MY),
        "Phase-2 coordinate-history size overflow.");
    if (shape.evaluate) {
        phase2_copy_field(result, "EvalMSE", mxSINGLE_CLASS,
            output->eval_mse, history_slots);
        phase2_copy_field(result, "EvalMSECoordinate", mxSINGLE_CLASS,
            output->eval_mse_coordinate, coordinate_slots);
        phase2_copy_field(result, "EvalDevCen", mxSINGLE_CLASS,
            output->eval_dev_center, history_slots);
        phase2_copy_field(result, "EvalDevCenTan", mxSINGLE_CLASS,
            output->eval_dev_center_tangent, history_slots);
        phase2_copy_field(result, "EvalDevTan", mxSINGLE_CLASS,
            output->eval_dev_tangent, history_slots);
        if (shape.eval_k0 && shape.test_metrics) {
            phase2_copy_field(result, "EvalMSE0", mxSINGLE_CLASS,
                output->eval_mse_k0, slots);
            phase2_copy_field(result, "EvalMSECoordinate0", mxSINGLE_CLASS,
                output->eval_mse_coordinate_k0,
                checked_product(slots, static_cast<size_t>(shape.MY),
                    "Phase-2 K0 coordinate size overflow."));
            phase2_copy_field(result, "EvalDevCen0", mxSINGLE_CLASS,
                output->eval_dev_center_k0, slots);
            phase2_copy_field(result, "EvalDevCenTan0", mxSINGLE_CLASS,
                output->eval_dev_center_tangent_k0, slots);
            phase2_copy_field(result, "EvalDevTan0", mxSINGLE_CLASS,
                output->eval_dev_tangent_k0, slots);
        }
        if (shape.validate) {
            if (shape.validation_tangent) {
                phase2_copy_field(result, "EvalValidDevTan", mxSINGLE_CLASS,
                    output->validation_dev_tangent, history_slots);
                if (shape.eval_k0)
                    phase2_copy_field(result, "EvalValidDevTan0", mxSINGLE_CLASS,
                        output->validation_dev_tangent_k0, slots);
            }
            phase2_copy_field(result, "EvalValidMSE", mxSINGLE_CLASS,
                output->validation_mse, history_slots);
            phase2_copy_field(result, "EvalValidMSECoordinate", mxSINGLE_CLASS,
                output->validation_mse_coordinate, coordinate_slots);
            if (shape.eval_k0) {
                phase2_copy_field(result, "EvalValidMSE0", mxSINGLE_CLASS,
                    output->validation_mse_k0, slots);
                phase2_copy_field(result, "EvalValidMSECoordinate0", mxSINGLE_CLASS,
                    output->validation_mse_coordinate_k0,
                    checked_product(slots, static_cast<size_t>(shape.MY),
                        "Phase-2 validation K0 coordinate size overflow."));
            }
        }
    }

    if (!stats) return;
    const mxArray* profile = phase2_field(result, "profile");
    stats->struct_size = sizeof(*stats);
    stats->abi_version = MANOR_M_CUDA_ABI_VERSION;
    stats->device_ordinal = device_ordinal;
    stats->ambient_dimension = shape.D;
    stats->manifold_dimension = shape.MM;
    stats->response_dimension = shape.MY;
    stats->query_block = static_cast<int32_t>(
        phase2_scalar_field(profile, "BQ"));
    stats->iteration_count = shape.K;
    stats->valid_iterations = static_cast<int32_t>(
        phase2_scalar_field(result, "valid_iterations"));
    stats->stopped_no_heavy = static_cast<int32_t>(
        phase2_scalar_field(result, "stopped_no_heavy"));
    stats->evaluation_iterations_executed = static_cast<int32_t>(
        phase2_scalar_field(result, "fused_eval_executed"));
    stats->point_count = shape.N;
    stats->owner_count = static_cast<int64_t>(
        phase2_scalar_field(result, "owner_count"));
    stats->validation_point_count = shape.validate && evaluation
        ? evaluation->validation_point_count : 0;
    stats->memory_required_bytes = static_cast<uint64_t>(
        phase2_scalar_field(profile, "memory_required_bytes"));
    stats->memory_usable_bytes = static_cast<uint64_t>(
        phase2_scalar_field(profile, "memory_usable_bytes"));
    stats->memory_index_bytes = static_cast<uint64_t>(
        phase2_scalar_field(profile, "memory_index_bytes"));
    stats->history_bytes = static_cast<uint64_t>(
        phase2_scalar_field(profile, "history_bytes"));
    stats->total_milliseconds = total_milliseconds;
    stats->host_prepare_milliseconds = phase2_scalar_field(
        profile, "host_prep_ms");
    stats->device_loop_milliseconds = phase2_scalar_field(
        profile, "device_loop_ms");
    stats->output_pack_milliseconds = phase2_scalar_field(
        profile, "output_pack_ms");
    stats->sparse_pair_count = static_cast<int64_t>(
        phase2_scalar_field(profile, "sparse_pairs_total"));
    stats->response_evaluated = evaluation && evaluation->test_y_point_major;
    stats->tangent_evaluated = evaluation && evaluation->test_tangent_point_major;
    stats->center_evaluated = evaluation && evaluation->test_center_point_major;
    stats->center_tangent_evaluated = stats->tangent_evaluated &&
        stats->center_evaluated;
    stats->timer_upload_init_milliseconds = phase2_timer_total(
        profile, "upload_init");
    stats->timer_eval_stats_milliseconds = phase2_timer_total(
        profile, "eval_stats");
    stats->timer_eval_main_milliseconds = phase2_timer_total(
        profile, "eval_main");
    stats->timer_eval_pair_milliseconds = phase2_timer_total(
        profile, "eval_pair");
    stats->timer_valid_main_milliseconds = phase2_timer_total(
        profile, "valid_main");
    stats->timer_valid_pair_milliseconds = phase2_timer_total(
        profile, "valid_pair");
    stats->timer_iter_setup_milliseconds = phase2_timer_total(
        profile, "iter_setup");
    stats->timer_regression_milliseconds = phase2_timer_total(
        profile, "regression");
    stats->timer_mask_chain_milliseconds = phase2_timer_total(
        profile, "mask_chain");
    stats->timer_aggregation_accumulate_milliseconds = phase2_timer_total(
        profile, "agg_accumulate");
    stats->timer_aggregation_finalize_milliseconds = phase2_timer_total(
        profile, "agg_finalize");
    stats->timer_snapshots_milliseconds = phase2_timer_total(
        profile, "snapshots");
    stats->timer_eval_store_milliseconds = phase2_timer_total(
        profile, "eval_store");
    stats->timer_valid_store_milliseconds = phase2_timer_total(
        profile, "valid_store");
    stats->host_cuda_allocate_milliseconds =
        owner_phase2_core_cuda_allocate_milliseconds();
    stats->host_cuda_free_milliseconds =
        owner_phase2_core_cuda_free_milliseconds();
    stats->host_cuda_event_destroy_milliseconds =
        owner_phase2_core_cuda_event_destroy_milliseconds();
}

}  // namespace

static int phase2_execute(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners,
    int32_t index_tile_owners,
    ManorMCudaIndexStreamStats* index_stats,
    bool compact_owner_index,
    bool generated_response_index,
    uint64_t knn_scratch_memory_bytes,
    ManorMCudaGeneratedIndexStats* generated_stats)
{
    try {
        const bool evaluation_only = options && (options->reserved & 2) != 0;
        phase2_require(context != nullptr, "Phase-2 context must not be null.");
        phase2_require(input != nullptr,
            "Phase-2 input descriptor must not be null.");
        phase2_require(aggregation_tile_owners >= 0,
            "Aggregation tile owner count must be nonnegative.");
        phase2_require(index_tile_owners >= 0,
            "Index tile owner count must be nonnegative.");
        phase2_require(!generated_response_index ||
                input->index_zero_based == nullptr,
            "Generated Phase-2 requires a null index pointer.");
        phase2_require(generated_response_index ||
                input->index_zero_based != nullptr,
            "Ordinary Phase-2 requires an explicit index pointer.");
        phase2_require(!generated_response_index || evaluation_only ||
                knn_scratch_memory_bytes > 0,
            "Generated Phase-2 requires a nonzero exact-kNN scratch budget.");
        phase2_require(generated_response_index ||
                knn_scratch_memory_bytes == 0,
            "An exact-kNN scratch budget is valid only for generated Phase-2.");
        phase2_require(!generated_response_index ||
                (!compact_owner_index && index_tile_owners == 0 &&
                 index_stats == nullptr),
            "Generated Phase-2 cannot also use host index streaming.");
        phase2_require(generated_response_index || generated_stats == nullptr,
            "Generated-index stats are valid only for generated Phase-2.");
        Phase2Shape shape = phase2_validate(
            input, options, evaluation, output, generated_response_index);
        phase2_require(!compact_owner_index || !options->index_is_shared_slice,
            "A compact owner index cannot also be a shared index slice.");
        phase2_require(!generated_response_index ||
                !options->index_is_shared_slice,
            "Generated Phase-2 cannot use a shared index slice.");
        int32_t compact_owner_count = shape.N;
        if (compact_owner_index || generated_response_index) {
            compact_owner_count = 0;
            for (int index = 0; index < shape.N; ++index)
                compact_owner_count += input->owner_mask[index] != 0;
            phase2_require(compact_owner_count > 0,
                "Phase-2 requires at least one owner.");
        }
        int32_t device_ordinal = 0;
        int status = manor_m_cuda_internal_context_device(context, &device_ordinal);
        if (status != MANOR_M_CUDA_SUCCESS) return status;
        cudaError_t cuda_status = cudaSetDevice(device_ordinal);
        if (cuda_status != cudaSuccess)
            return manor_m_cuda_internal_fail(MANOR_M_CUDA_CUDA_ERROR,
                cudaGetErrorString(cuda_status));
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess)
            return manor_m_cuda_internal_fail(MANOR_M_CUDA_CUDA_ERROR,
                cudaGetErrorString(cuda_status));

        if (stats) std::memset(stats, 0, sizeof(*stats));
        if (index_stats) std::memset(index_stats, 0, sizeof(*index_stats));
        if (generated_stats) {
            std::memset(generated_stats, 0, sizeof(*generated_stats));
            generated_stats->struct_size = sizeof(*generated_stats);
            generated_stats->extension_version = 1;
            generated_stats->enabled = 1;
            generated_stats->logical_full_index_bytes = checked_product(
                checked_product(static_cast<size_t>(shape.N),
                    static_cast<size_t>(shape.k),
                    "Generated full-index logical size overflow."),
                sizeof(int32_t),
                "Generated full-index logical byte count overflow.");
            generated_stats->logical_compact_index_bytes = checked_product(
                checked_product(static_cast<size_t>(compact_owner_count),
                    static_cast<size_t>(shape.k),
                    "Generated compact-index logical size overflow."),
                sizeof(int32_t),
                "Generated compact-index logical byte count overflow.");
            generated_stats->scratch_memory_bytes = knn_scratch_memory_bytes;
            generated_stats->response_soa_bytes = evaluation_only ? 0 :
                checked_product(
                    checked_product(static_cast<size_t>(shape.N),
                        static_cast<size_t>(shape.MY),
                        "Generated response SoA size overflow."),
                    sizeof(float),
                    "Generated response SoA byte count overflow.");
        }
        MiniArrayOwner owner;
        std::vector<const mxArray*> arguments;
        arguments.reserve(17);
        auto add = [&](mxArray* value) {
            arguments.push_back(owner.add(value));
        };
        add(phase2_view(mxSINGLE_CLASS, input->x_point_major,
            {static_cast<mwSize>(shape.D), static_cast<mwSize>(shape.N)}));
        add(phase2_view(mxSINGLE_CLASS, input->response_point_major,
            {static_cast<mwSize>(shape.MY), static_cast<mwSize>(shape.N)}));
        int32_t generated_index_shape_token = 0;
        add(phase2_view(mxINT32_CLASS,
            generated_response_index ? &generated_index_shape_token :
                input->index_zero_based,
            {static_cast<mwSize>(shape.k),
             static_cast<mwSize>(generated_response_index ? 1 :
                 options->index_is_shared_slice ? 1 :
                 (compact_owner_index ? compact_owner_count : shape.N))}));
        add(phase2_view(mxSINGLE_CLASS, input->tangent_point_major,
            {static_cast<mwSize>(shape.D), static_cast<mwSize>(shape.MM),
             static_cast<mwSize>(shape.N)}));
        add(phase2_view(mxSINGLE_CLASS, input->center_point_major,
            {static_cast<mwSize>(shape.D), static_cast<mwSize>(shape.N)}));
        add(phase2_view(mxLOGICAL_CLASS, input->owner_mask,
            {1, static_cast<mwSize>(shape.N)}));
        add(mxCreateDoubleScalar(input->base_h));
        add(phase2_view(mxSINGLE_CLASS, input->rho_schedule,
            {1, static_cast<mwSize>(input->rho_schedule_count)}));
        add(mxCreateDoubleScalar(shape.MM));
        add(phase2_build_options(*options, *output,
            shape.evaluate, shape.validate, aggregation_tile_owners,
            index_tile_owners, compact_owner_index,
            generated_response_index));

        if (shape.evaluate) {
            const int64_t test_count = evaluation->test_point_count;
            add(phase2_view(mxSINGLE_CLASS, evaluation->test_x_point_major,
                {static_cast<mwSize>(shape.D), static_cast<mwSize>(test_count)}));
            add(phase2_view(mxSINGLE_CLASS, evaluation->test_y_point_major,
                {static_cast<mwSize>(shape.MY),
                 static_cast<mwSize>(evaluation->test_y_point_major ? test_count : 0)}));
            add(phase2_view(mxSINGLE_CLASS, evaluation->test_tangent_point_major,
                {static_cast<mwSize>(shape.D), static_cast<mwSize>(shape.MM),
                 static_cast<mwSize>(evaluation->test_tangent_point_major ? test_count : 0)}));
            add(phase2_view(mxSINGLE_CLASS, evaluation->test_center_point_major,
                {static_cast<mwSize>(shape.D),
                 static_cast<mwSize>(evaluation->test_center_point_major ? test_count : 0)}));
            add(phase2_build_eval_options(*evaluation));
            if (shape.validate) {
                add(phase2_view(mxSINGLE_CLASS,
                    evaluation->validation_x_point_major,
                    {static_cast<mwSize>(shape.D),
                     static_cast<mwSize>(evaluation->validation_point_count)}));
                add(phase2_view(mxSINGLE_CLASS,
                    evaluation->validation_y_point_major,
                    {static_cast<mwSize>(shape.MY),
                     static_cast<mwSize>(evaluation->validation_point_count)}));
                if (shape.validation_tangent)
                    add(phase2_view(mxSINGLE_CLASS,
                        evaluation->validation_tangent_point_major,
                        {static_cast<mwSize>(shape.D), static_cast<mwSize>(shape.MM),
                         static_cast<mwSize>(evaluation->validation_point_count)}));
            }
        }

        ManorMResidentCallScope resident_scope(context);
        struct OutputScope {
            decltype(manor_m_minimex_output_bindings) previous;
            OutputScope() : previous(std::move(manor_m_minimex_output_bindings)) {
                manor_m_minimex_output_bindings.clear();
            }
            ~OutputScope() { manor_m_minimex_output_bindings = std::move(previous); }
        } output_scope;
        phase2_require((options->reserved & ~3) == 0,
            "Unknown Phase2 extension flag.");
        if ((options->reserved & 1) != 0) {
            auto bind = [&](const char* name, void* ptr, size_t count, mxClassID type) {
                if (ptr) manor_m_minimex_output_bindings.emplace(name,
                    ManorMMiniMexOutputBinding{ptr, checked_product(count,
                        manor_m_minimex_class_bytes(type), "Output binding overflow."), type});
            };
            const size_t n = static_cast<size_t>(shape.N);
            bind("Tan_next", output->tangent_point_major, n * shape.D * shape.MM, mxSINGLE_CLASS);
            bind("Cen_next", output->center_point_major, n * shape.D, mxSINGLE_CLASS);
            bind("num_neigh", output->neighbor_count, n, mxINT32_CLASS);
            bind("heavy_mask", output->heavy_mask, n, mxLOGICAL_CLASS);
            bind("keep_mask", output->keep_mask, n, mxLOGICAL_CLASS);
            bind("solver_failed", output->solver_failed, n, mxLOGICAL_CLASS);
            bind("num_neigh_iter", output->neighbor_count_history, n * shape.K, mxINT32_CLASS);
            bind("heavy_mask_iter", output->heavy_mask_history, n * shape.K, mxLOGICAL_CLASS);
            bind("keep_mask_iter", output->keep_mask_history, n * shape.K, mxLOGICAL_CLASS);
        }
        ManorMGeneratedPhase2Source generated_source;
        generated_source.context = generated_response_index ? context : nullptr;
        generated_source.response_point_major = generated_response_index ?
            input->response_point_major : nullptr;
        generated_source.scratch_memory_bytes = generated_response_index ?
            knn_scratch_memory_bytes : 0;
        generated_source.stats = generated_response_index ? generated_stats : nullptr;
        struct GeneratedPhase2Scope {
            ManorMGeneratedPhase2Source* previous;
            ManorMGeneratedPhase2Source* current;
            explicit GeneratedPhase2Scope(ManorMGeneratedPhase2Source* value)
                : previous(manor_m_cuda_active_generated_phase2), current(value)
            {
                manor_m_cuda_active_generated_phase2 = value;
            }
            ~GeneratedPhase2Scope()
            {
                if (current && current->prepared) {
                    manor_m_cuda_internal_destroy_prepared_response_knn(
                        current->prepared);
                    current->prepared = nullptr;
                }
                manor_m_cuda_active_generated_phase2 = previous;
            }
        } generated_scope(generated_response_index ? &generated_source : nullptr);
        mxArray* result = nullptr;
        const auto start = std::chrono::steady_clock::now();
        manor_m_phase2_minimex_entry(
            1, &result, static_cast<int>(arguments.size()), arguments.data());
        std::unique_ptr<mxArray> result_owner(result);
#if MANOR_M_GENERATED_PREPARED_RESPONSE
        phase2_require(!generated_response_index ||
                generated_stats->generation_passes == 0 ||
                generated_source.preparation_count == 1,
            "Generated Phase-2 must prepare immutable response layouts exactly once.");
#endif
        const double total_milliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        phase2_require(result != nullptr,
            "Phase-2 execution returned no result object.");
        phase2_copy_outputs(result, shape, evaluation, output, stats,
            device_ordinal, total_milliseconds);
        if (index_stats) {
            const mxArray* profile = mxGetField(result, 0, "profile");
            index_stats->struct_size = sizeof(*index_stats);
            index_stats->extension_version = 1;
            index_stats->enabled = phase2_scalar_field(profile, "index_streamed");
            index_stats->logical_index_bytes = phase2_scalar_field(profile, "index_logical_bytes");
            index_stats->device_index_bytes = phase2_scalar_field(profile, "memory_index_bytes");
            index_stats->pinned_host_bytes = phase2_scalar_field(profile, "index_pinned_bytes");
            index_stats->tile_owners = phase2_scalar_field(profile, "index_tile_owners");
            index_stats->tiles_per_pass = phase2_scalar_field(profile, "index_tiles_per_pass");
            index_stats->buffer_count = phase2_scalar_field(profile, "index_buffer_count");
            index_stats->upload_calls = phase2_scalar_field(profile, "index_upload_calls");
            index_stats->host_to_device_bytes = phase2_scalar_field(profile, "index_h2d_bytes");
            index_stats->host_fill_milliseconds = phase2_scalar_field(profile, "index_host_fill_ms");
            index_stats->host_reuse_wait_milliseconds = phase2_scalar_field(profile, "index_host_reuse_wait_ms");
            index_stats->copy_milliseconds = phase2_scalar_field(profile, "index_copy_ms");
        }
        return manor_m_cuda_internal_fail(MANOR_M_CUDA_SUCCESS, "");
    } catch (const ManorMMiniMexError& error) {
        const bool cuda_error = error.identifier().find(":cuda") != std::string::npos ||
            error.identifier().find(":memory") != std::string::npos;
        return manor_m_cuda_internal_fail(
            cuda_error ? MANOR_M_CUDA_CUDA_ERROR : MANOR_M_CUDA_INVALID_ARGUMENT,
            error.what());
    } catch (const std::invalid_argument& error) {
        return manor_m_cuda_internal_fail(
            MANOR_M_CUDA_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return manor_m_cuda_internal_fail(
            MANOR_M_CUDA_INTERNAL_ERROR, error.what());
    } catch (...) {
        return manor_m_cuda_internal_fail(
            MANOR_M_CUDA_INTERNAL_ERROR, "Unknown Phase-2 failure.");
    }
}

extern "C" int manor_m_cuda_validation_tangent_version(void) { return 1; }

extern "C" const char* manor_m_cuda_metric_sum_contract(void)
{
    return "fp32-query256-tree-v1";
}

extern "C" const char* manor_m_cuda_devtan_evaluation_version(void)
{
#if MANOR_M_DEVTAN_ORTHONORMAL_FP64
    return "orthonormal-fp64-v2";
#else
    return "legacy-fp32-v1";
#endif
}

extern "C" int manor_m_cuda_phase2(
    ManorMCudaContext* context, const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output, ManorMCudaPhase2Stats* stats)
{
    return phase2_execute(context, input, options, evaluation, output, stats,
        0, 0, nullptr, false, false, 0, nullptr);
}

extern "C" int manor_m_cuda_phase2_tiled(
    ManorMCudaContext* context, const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output, ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners)
{
    return phase2_execute(context, input, options, evaluation, output, stats,
        aggregation_tile_owners, 0, nullptr, false, false, 0, nullptr);
}

extern "C" int manor_m_cuda_phase2_index_tiled(
    ManorMCudaContext* context, const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output, ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners, int32_t index_tile_owners,
    ManorMCudaIndexStreamStats* index_stats)
{
    return phase2_execute(context, input, options, evaluation, output, stats,
        aggregation_tile_owners, index_tile_owners, index_stats, false,
        false, 0, nullptr);
}

extern "C" int manor_m_cuda_phase2_compact_index_tiled(
    ManorMCudaContext* context, const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output, ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners, int32_t index_tile_owners,
    ManorMCudaIndexStreamStats* index_stats)
{
    return phase2_execute(context, input, options, evaluation, output, stats,
        aggregation_tile_owners, index_tile_owners, index_stats, true,
        false, 0, nullptr);
}

extern "C" int manor_m_cuda_phase2_response_tiled(
    ManorMCudaContext* context, const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output, ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners,
    uint64_t knn_scratch_memory_bytes,
    ManorMCudaGeneratedIndexStats* generated_stats)
{
    return phase2_execute(context, input, options, evaluation, output, stats,
        aggregation_tile_owners, 0, nullptr, false, true,
        knn_scratch_memory_bytes, generated_stats);
}
