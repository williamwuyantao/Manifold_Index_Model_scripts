#ifndef MANOR_M_CUDA_H
#define MANOR_M_CUDA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(MANOR_M_CUDA_BUILDING_LIBRARY)
#    define MANOR_M_CUDA_API __declspec(dllexport)
#  else
#    define MANOR_M_CUDA_API __declspec(dllimport)
#  endif
#else
#  define MANOR_M_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Increment only when the binary layout or function contract changes. */
#define MANOR_M_CUDA_ABI_VERSION 3u

typedef enum ManorMCudaStatus {
    MANOR_M_CUDA_SUCCESS = 0,
    MANOR_M_CUDA_INVALID_ARGUMENT = 1,
    MANOR_M_CUDA_CUDA_ERROR = 2,
    MANOR_M_CUDA_INTERNAL_ERROR = 3
} ManorMCudaStatus;

typedef struct ManorMCudaDeviceInfo {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    int32_t multiprocessor_count;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int32_t cuda_driver_version;
    int32_t cuda_runtime_version;
    uint64_t total_global_memory_bytes;
    uint64_t free_global_memory_bytes;
    char name[256];
} ManorMCudaDeviceInfo;

typedef enum ManorMCudaExecutionMode {
    MANOR_M_CUDA_EXECUTION_RESIDENT = 0,
    MANOR_M_CUDA_EXECUTION_TILED = 1
} ManorMCudaExecutionMode;

/* Generic Phase working-set planner.  The Phase implementation supplies its
 * exact persistent and per-row byte counts, so this ABI does not bake an
 * approximate Manor-M memory formula into the runtime. */
typedef struct ManorMCudaTilePlanRequest {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t staging_slot_count;
    double usable_memory_fraction;
    uint64_t reserve_memory_bytes;
    uint64_t persistent_memory_bytes;
    int64_t reference_row_count;
    uint64_t reference_bytes_per_row;
    int64_t query_row_count;
    uint64_t query_bytes_per_row;
    uint64_t scratch_bytes_per_query_row;
    int64_t minimum_reference_tile_rows;
    int64_t minimum_query_tile_rows;
    int64_t preferred_reference_tile_rows;
    int64_t preferred_query_tile_rows;
} ManorMCudaTilePlanRequest;

typedef struct ManorMCudaTilePlan {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t execution_mode;
    int32_t staging_slot_count;
    uint64_t device_free_memory_bytes;
    uint64_t usable_memory_bytes;
    uint64_t full_resident_memory_bytes;
    uint64_t estimated_peak_memory_bytes;
    int64_t reference_tile_rows;
    int64_t query_tile_rows;
    int64_t reference_tile_count;
    int64_t query_tile_count;
} ManorMCudaTilePlan;

typedef struct ManorMCudaContextOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t stream_count;
    uint64_t memory_pool_release_threshold_bytes;
} ManorMCudaContextOptions;

typedef struct ManorMCudaTiledStats {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t staging_slot_count;
    int64_t element_count;
    int64_t tile_elements;
    int64_t tile_count;
    uint64_t peak_device_staging_bytes;
    uint64_t peak_pinned_host_staging_bytes;
    double wall_milliseconds;
} ManorMCudaTiledStats;

/* Exact response-space kNN used by Phase 1.  response_point_major is the
 * caller-owned host array [point_count, response_dimension] in C order, which
 * is byte-identical to MATLAB's column-major [response_dimension, point_count]
 * storage.  index_zero_based is [query_count, neighbor_count] in C order,
 * likewise byte-identical to MATLAB's [neighbor_count, query_count] int32
 * layout. */
typedef struct ManorMCudaResponseKnnOptions {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t scratch_memory_bytes;
    int32_t max_retry_rounds;
    int32_t collect_stats;
} ManorMCudaResponseKnnOptions;

typedef struct ManorMCudaResponseKnnStats {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t response_dimension;
    int64_t point_count;
    int64_t query_count;
    int32_t neighbor_count;
    int32_t reserved;
    double bootstrap_milliseconds;
    double threshold_milliseconds;
    double scan_milliseconds;
    double retry_milliseconds;
    double select_milliseconds;
    double refine_milliseconds;
    double sort_milliseconds;
    double emit_milliseconds;
    double total_milliseconds;
    int64_t retry_rounds;
    int64_t retry_queries;
    uint64_t owner_misplaced;
    uint64_t exact_ties;
    uint64_t compact_overflow;
    double mean_candidates;
    double max_candidates;
} ManorMCudaResponseKnnStats;

/* Phase 1 with a caller-supplied exact response-neighbour index.  This is the
 * numerical-parity API: no approximation, cross-cell cache, or hidden
 * reparameterization is introduced.  The Python adapter may obtain the index
 * from manor_m_cuda_response_knn in the same process before calling Phase 1.
 *
 * Layouts are chosen so the same bytes serve both adapters:
 *   x_point_major             C [N,D] == MATLAB column-major [D,N]
 *   index_zero_based          C [A,k] == MATLAB column-major [k,A]
 *   cuts_scale_major          C [S,R] == MATLAB column-major [R,S]
 *   tangent_point_major       C [N,MM,D] == MATLAB [D,MM,N]
 */
typedef struct ManorMCudaPhase1Options {
    uint32_t struct_size;
    uint32_t abi_version;
    float gpu_memory_fraction;
    float reserve_memory_fraction;
    uint64_t reserve_memory_bytes;
    float heavy_factor;
    float jitter;
    float neighbor_scaled_max;
    float split_scaled_max;
    int32_t minimum_query_block;
    int32_t maximum_query_block;
    int32_t stable_window;
    int32_t selection_mode;
    uint64_t split_seed;
    double cut_max_fraction;
    int32_t index_is_shared_slice;
    int32_t h_null_correction;
    int32_t collect_profile;
    int32_t reserved;
} ManorMCudaPhase1Options;

/* All pointers are caller-owned host buffers.  Null diagnostic pointers skip
 * only their host download; the algorithm's internal selection is unchanged.
 * The six primary outputs through owner_mask are required. */
typedef struct ManorMCudaPhase1Output {
    uint32_t struct_size;
    uint32_t abi_version;
    float* tangent_point_major;
    float* center_point_major;
    float* radius;
    float* axial_tangent;
    int32_t* neighbor_count;
    uint8_t* owner_mask;
    uint8_t* stable_covered;
    uint32_t* selected_scale_one_based;
    uint32_t* selected_response_count;
    uint32_t* selected_cut;
    float* geometry_score;
    float* stable_h;
    float* neighbor_scaled;
    float* split_scaled;
    float* split_h_a;
    float* split_h_b;
    uint8_t* rejection_code;
    float* raw_tangent_point_major;
    float* raw_center_point_major;
    uint32_t* selected_index_one_based;
} ManorMCudaPhase1Output;

typedef struct ManorMCudaPhase1Stats {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t ambient_dimension;
    int32_t manifold_dimension;
    int32_t query_block;
    int32_t query_block_upper_bound;
    int32_t scale_count;
    int64_t point_count;
    int64_t active_anchor_count;
    uint64_t stable_covered_count;
    uint64_t owner_count;
    double support_threshold;
    double total_milliseconds;
    double upload_milliseconds;
    double distance_milliseconds;
    double sort_milliseconds;
    double scale_topk_milliseconds;
    double scale_select_milliseconds;
    double scale_stream_milliseconds;
    double final_extract_milliseconds;
    double owner_rule_milliseconds;
    double owner_aggregate_milliseconds;
    double mask_nonowners_milliseconds;
    double download_milliseconds;
    uint64_t memory_free_bytes;
    uint64_t memory_total_bytes;
    uint64_t memory_required_bytes;
    uint64_t memory_next_required_bytes;
} ManorMCudaPhase1Stats;

/* Phase 2 consumes the Phase-1 owner state.  All layouts are point-major in
 * C and therefore byte-identical to the corresponding MATLAB column-major
 * arrays:
 *   X/Cen                C [N,D]       == MATLAB [D,N]
 *   Y                    C [N,MY]      == MATLAB [MY,N]
 *   Tan                  C [N,MM,D]    == MATLAB [D,MM,N]
 *   index_zero_based     C [N,k]       == MATLAB [k,N]
 *   iteration histories C [K,N]       == MATLAB [N,K]
 *
 * Index data are always int32 and zero based at the shared boundary. */
typedef struct ManorMCudaPhase2Input {
    uint32_t struct_size;
    uint32_t abi_version;
    const float* x_point_major;
    const float* response_point_major;
    const int32_t* index_zero_based;
    const float* tangent_point_major;
    const float* center_point_major;
    const uint8_t* owner_mask;
    int64_t point_count;
    int32_t ambient_dimension;
    int32_t manifold_dimension;
    int32_t response_dimension;
    int32_t neighbor_count;
    float base_h;
    const float* rho_schedule;
    int32_t rho_schedule_count;
    int32_t reserved;
} ManorMCudaPhase2Input;

typedef struct ManorMCudaPhase2Options {
    uint32_t struct_size;
    uint32_t abi_version;
    float lambda;
    float jitter;
    float gpu_memory_fraction;
    float reserve_memory_fraction;
    uint64_t reserve_memory_bytes;
    int32_t minimum_query_block;
    int32_t maximum_query_block;
    int32_t regression_mode;
    int32_t iteration_count;
    float heavy_factor;
    float temporary_heavy_factor;
    const float* h_schedule;
    int32_t h_schedule_count;
    int32_t index_is_shared_slice;
    int32_t evaluation_uses_owner_mask;
    int32_t collect_profile;
    int32_t use_async_allocator;
    /* Extension bit 0: direct caller-owned large output buffers.
     * Extension bit 1: evaluate the supplied state at K=0 only; no Phase-2
     * iteration or response-index generation is allowed to run.  Both are
     * opt-in and disabled by default. */
    int32_t reserved;
} ManorMCudaPhase2Options;

/* Truth-array presence controls metric work independently.  test_x is
 * required whenever any testing truth is supplied.  Null test_y skips relMSE;
 * null test_tangent skips DevTan; null test_center skips DevCen and
 * DevCenTan.  Validation runs only when both validation_x and validation_y
 * are non-null.  A null descriptor skips all evaluation. */
typedef struct ManorMCudaPhase2Evaluation {
    uint32_t struct_size;
    uint32_t abi_version;
    const float* test_x_point_major;
    const float* test_y_point_major;
    const float* test_tangent_point_major;
    const float* test_center_point_major;
    int64_t test_point_count;
    const float* validation_x_point_major;
    const float* validation_y_point_major;
    int64_t validation_point_count;
    const float* rho_scales;
    int32_t rho_scale_count;
    const float* h_scales;
    int32_t h_scale_count;
    const int32_t* evaluated_iterations;
    int32_t evaluated_iteration_count;
    float sum_sparse_threshold;
    int32_t count_sparse_threshold;
    float fallback_exponent_scale;
    float epsilon_weight;
    int32_t warps_per_block;
    float center_rho_scale;
    float center_h_scale;
    int32_t reserved;
    /* Optional append-only extension. Old descriptor prefixes remain valid. */
    const float* validation_tangent_point_major;
} ManorMCudaPhase2Evaluation;

/* Null diagnostic pointers suppress only the corresponding host copy.  The
 * final tangent and center are required for updating calls and deliberately
 * absent for evaluation-only calls.  Evaluation histories have shape
 * [K,n_rho,n_h], coordinate histories append MY, and K0 histories have first
 * dimension one. */
typedef struct ManorMCudaPhase2Output {
    uint32_t struct_size;
    uint32_t abi_version;
    /* State/status outputs are required for an updating Phase-2 call.  An
     * evaluation-only call (options.reserved bit 1) must leave the first six
     * pointers null: it scores the supplied checkpoint in place and does not
     * materialize a duplicate N-sized next state on either device or host. */
    float* tangent_point_major;
    float* center_point_major;
    int32_t* neighbor_count;
    uint8_t* heavy_mask;
    uint8_t* keep_mask;
    uint8_t* solver_failed;
    int32_t* neighbor_count_history;
    uint8_t* heavy_mask_history;
    uint8_t* keep_mask_history;
    float* gradient_debug_point_major;
    float* eval_mse;
    float* eval_mse_coordinate;
    float* eval_dev_center;
    float* eval_dev_center_tangent;
    float* eval_dev_tangent;
    float* eval_mse_k0;
    float* eval_mse_coordinate_k0;
    float* eval_dev_center_k0;
    float* eval_dev_center_tangent_k0;
    float* eval_dev_tangent_k0;
    float* validation_mse;
    float* validation_mse_coordinate;
    float* validation_mse_k0;
    float* validation_mse_coordinate_k0;
    float* validation_dev_tangent;
    float* validation_dev_tangent_k0;
} ManorMCudaPhase2Output;

/* Nonzero when the append-only validation tangent fields are supported. */
MANOR_M_CUDA_API int manor_m_cuda_validation_tangent_version(void);
/* Read-only feature identifiers for portable experiment manifests. */
MANOR_M_CUDA_API const char* manor_m_cuda_metric_sum_contract(void);
MANOR_M_CUDA_API const char* manor_m_cuda_devtan_evaluation_version(void);

typedef struct ManorMCudaPhase2Stats {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    int32_t ambient_dimension;
    int32_t manifold_dimension;
    int32_t response_dimension;
    int32_t query_block;
    int32_t iteration_count;
    int32_t valid_iterations;
    int32_t stopped_no_heavy;
    int32_t evaluation_iterations_executed;
    int64_t point_count;
    int64_t owner_count;
    int64_t validation_point_count;
    uint64_t memory_required_bytes;
    uint64_t memory_usable_bytes;
    uint64_t memory_index_bytes;
    uint64_t history_bytes;
    double total_milliseconds;
    double host_prepare_milliseconds;
    double device_loop_milliseconds;
    double output_pack_milliseconds;
    int64_t sparse_pair_count;
    int32_t response_evaluated;
    int32_t tangent_evaluated;
    int32_t center_evaluated;
    int32_t center_tangent_evaluated;
    double timer_upload_init_milliseconds;
    double timer_eval_stats_milliseconds;
    double timer_eval_main_milliseconds;
    double timer_eval_pair_milliseconds;
    double timer_valid_main_milliseconds;
    double timer_valid_pair_milliseconds;
    double timer_iter_setup_milliseconds;
    double timer_regression_milliseconds;
    double timer_mask_chain_milliseconds;
    double timer_aggregation_accumulate_milliseconds;
    double timer_aggregation_finalize_milliseconds;
    double timer_snapshots_milliseconds;
    double timer_eval_store_milliseconds;
    double timer_valid_store_milliseconds;
    double host_cuda_allocate_milliseconds;
    double host_cuda_free_milliseconds;
    double host_cuda_event_destroy_milliseconds;
} ManorMCudaPhase2Stats;

typedef struct ManorMCudaContext ManorMCudaContext;

/* Optional resident-input extension, version 1; the existing ABI-3 structs
 * are unchanged. Disabled by default. Buffers are immutable and caller-owned
 * on the host until clear/destroy. Adapters must own snapshots and keep their
 * identity/lifetime valid; a pointer alone is NOT a dataset identity.
 * All calls use the same live, device-affine Context and thread. */
typedef struct ManorMCudaResidentStats {
    uint32_t struct_size;
    uint32_t extension_version;
    uint64_t enabled;
    uint64_t buffer_count;
    uint64_t retained_bytes;
    uint64_t host_to_device_bytes;
    uint64_t generated_device_bytes;
    uint64_t cache_hits;
    uint64_t skipped_host_to_device_bytes;
    uint64_t initial_state_device_copy_bytes;
} ManorMCudaResidentStats;

MANOR_M_CUDA_API int manor_m_cuda_resident_begin(
    ManorMCudaContext* context, uint64_t maximum_retained_bytes);
MANOR_M_CUDA_API int manor_m_cuda_resident_put(
    ManorMCudaContext* context, const char* name, const void* immutable_host,
    uint64_t bytes);
/* Validate an explicitly session-owned immutable exact index once. Phase-2
 * may then omit repeated O(N*k) host scans only for this exact pointer, byte
 * count and upper bound. Clearing the phase1 resident group revokes it. */
MANOR_M_CUDA_API int manor_m_cuda_resident_validate_index(
    ManorMCudaContext* context, const int32_t* immutable_index,
    int64_t point_count, int32_t neighbor_count);
/* Validate a compact owner-order index once.  row_count controls the input
 * byte count, while point_count remains the global neighbor-ID upper bound. */
MANOR_M_CUDA_API int manor_m_cuda_resident_validate_compact_index(
    ManorMCudaContext* context, const int32_t* immutable_index,
    int64_t row_count, int32_t neighbor_count, int64_t point_count);
MANOR_M_CUDA_API int manor_m_cuda_resident_prepare_owner_index(
    ManorMCudaContext* context, const int32_t* index, const uint8_t* owner,
    int64_t point_count, int32_t neighbor_count);
/* Empty group clears everything and disables reuse. A named group such as
 * "phase1" evicts only that group while keeping dataset inputs resident. */
MANOR_M_CUDA_API int manor_m_cuda_resident_clear(
    ManorMCudaContext* context, const char* group);
MANOR_M_CUDA_API int manor_m_cuda_resident_stats(
    ManorMCudaContext* context, ManorMCudaResidentStats* stats);

MANOR_M_CUDA_API uint32_t manor_m_cuda_abi_version(void);
MANOR_M_CUDA_API const char* manor_m_cuda_version(void);
MANOR_M_CUDA_API const char* manor_m_cuda_status_string(int status);

/* The returned pointer is thread-local and remains valid until the next call
 * into this library on the same thread. */
MANOR_M_CUDA_API const char* manor_m_cuda_last_error(void);

MANOR_M_CUDA_API int manor_m_cuda_device_count(int32_t* count);
MANOR_M_CUDA_API int manor_m_cuda_get_device_info(
    int32_t device_ordinal, ManorMCudaDeviceInfo* info);
MANOR_M_CUDA_API int manor_m_cuda_set_device(int32_t device_ordinal);
MANOR_M_CUDA_API int manor_m_cuda_synchronize(void);

MANOR_M_CUDA_API int manor_m_cuda_plan_tiles(
    const ManorMCudaTilePlanRequest* request,
    ManorMCudaTilePlan* plan);

/* Contexts are deliberately device-affine and not thread-safe.  Production
 * uses one long-lived process/context per GPU. */
MANOR_M_CUDA_API int manor_m_cuda_context_create(
    const ManorMCudaContextOptions* options,
    ManorMCudaContext** context);
MANOR_M_CUDA_API int manor_m_cuda_context_destroy(
    ManorMCudaContext* context);

/* End-to-end adapter smoke. Inputs and output are caller-owned host arrays.
 * This is deliberately small; it is not the Manor-M phase API. */
MANOR_M_CUDA_API int manor_m_cuda_smoke_axpy(
    int32_t device_ordinal,
    float alpha,
    const float* x,
    const float* y,
    float* output,
    int64_t element_count);

/* Out-of-core adapter smoke.  Only tile_elements reside on the GPU at once;
 * two pinned staging slots overlap copies and execution when requested by the
 * context.  This validates the reusable execution engine, not Manor-M Phase
 * 1/2 numerical parity. */
MANOR_M_CUDA_API int manor_m_cuda_smoke_tiled_axpy(
    ManorMCudaContext* context,
    float alpha,
    const float* x,
    const float* y,
    float* output,
    int64_t element_count,
    int64_t tile_elements,
    ManorMCudaTiledStats* stats);

/* Phase-1 migration checkpoint: exact threshold-first response kNN.  This
 * function intentionally preserves the legacy direct-float32 distances and
 * float64 survivor rerank; TF32/GEMM substitutions are not permitted here. */
MANOR_M_CUDA_API int manor_m_cuda_response_knn(
    ManorMCudaContext* context,
    const float* response_point_major,
    int64_t point_count,
    int32_t response_dimension,
    int64_t query_count,
    int32_t neighbor_count,
    int32_t* index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats);

MANOR_M_CUDA_API int manor_m_cuda_phase1_precomputed(
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
    ManorMCudaPhase1Stats* stats);

/* Exact one-pass hybrid Phase 1. The public options remain selection_mode=0;
 * the core emits the Stable-H primary and top-5-|H| fallback decisions during
 * one selector pass, applies the unchanged owner rules, and returns the same
 * merged model as the legacy two-call adapter. One response scale is required. */
MANOR_M_CUDA_API int manor_m_cuda_phase1_hybrid_precomputed(
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
    ManorMCudaPhase1Stats* stats);

/* Additive P1 capacity extension. Existing ABI-3 layouts remain unchanged.
 * working_bytes is an optional allocation ceiling in addition to real free
 * GPU memory, excluding buffers already resident in this context. Zero uses
 * actual free memory. mode: 0=automatic, 1=require resident, 2=require streamed
 * with a global neighborhood mask, 3=require streamed with query-local
 * neighborhood reconstruction, 4=require exact response-kNN generation
 * directly into query-local P1 consumers (no complete host/device N*k
 * index). The latter four are controlled comparison modes.
 * Streamed indices are caller-owned host storage, possibly file-backed;
 * the complete N*k index is never allocated on the device in that mode. */
typedef struct ManorMCudaP1CapacityOptions {
    uint32_t struct_size, extension_version;
    uint64_t working_bytes, pinned_bytes, retained_limit_bytes;
    uint64_t knn_scratch_bytes, reserve_bytes;
    float memory_fraction, reserve_fraction;
    int32_t mode, maximum_query_block;
} ManorMCudaP1CapacityOptions;

typedef struct ManorMCudaP1CapacityPlan {
    uint32_t struct_size, extension_version;
    int32_t mode, query_block;
    uint64_t free_bytes, total_bytes, usable_bytes, full_index_bytes;
    uint64_t index_device_bytes, pinned_bytes, knn_scratch_bytes;
    uint64_t knn_required_bytes, phase1_required_bytes, global_state_bytes;
    uint64_t estimated_peak_bytes, projected_retained_bytes;
} ManorMCudaP1CapacityPlan;

MANOR_M_CUDA_API int manor_m_cuda_plan_phase1(
    ManorMCudaContext* context, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    int32_t scale_count, int32_t cuts_per_scale, int32_t maximum_cut,
    const ManorMCudaP1CapacityOptions* options,
    ManorMCudaP1CapacityPlan* plan);

/* Capacity plan for the exact fused Stable-H + top-5 fallback call.  This is
 * separate from the ordinary planner because the fused call retains a second
 * selector/raw-model record while avoiding the second distance/sort pass. */
MANOR_M_CUDA_API int manor_m_cuda_plan_phase1_hybrid(
    ManorMCudaContext* context, const float* x, const float* response,
    int64_t n, int32_t d, int32_t my, int32_t mm, int32_t k,
    int32_t scale_count, int32_t cuts_per_scale, int32_t maximum_cut,
    const ManorMCudaP1CapacityOptions* options,
    ManorMCudaP1CapacityPlan* plan);

MANOR_M_CUDA_API int manor_m_cuda_phase1_hybrid_bounded(
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
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan);

/* Exact memory-bounded P1 directly from the response coordinates. These
 * additive entries are the no-full-index fallback selected by capacity mode
 * 4. Exact response kNN tiles are generated twice: once for selection and
 * once after the global owner rule for aggregation. Faster resident/host
 * streamed paths remain preferred whenever their complete index fits. */
MANOR_M_CUDA_API int manor_m_cuda_phase1_response_bounded(
    ManorMCudaContext* context,
    const float* x_point_major,
    const float* response_point_major,
    int64_t point_count,
    int32_t ambient_dimension,
    int32_t response_dimension,
    int32_t manifold_dimension,
    int32_t neighbor_count,
    const int32_t* response_counts,
    int32_t scale_count,
    const int32_t* cuts_scale_major,
    int32_t cuts_per_scale,
    const ManorMCudaPhase1Options* options,
    ManorMCudaPhase1Output* output,
    ManorMCudaPhase1Stats* stats,
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan);

MANOR_M_CUDA_API int manor_m_cuda_phase1_response_hybrid_bounded(
    ManorMCudaContext* context,
    const float* x_point_major,
    const float* response_point_major,
    int64_t point_count,
    int32_t ambient_dimension,
    int32_t response_dimension,
    int32_t manifold_dimension,
    int32_t neighbor_count,
    const int32_t* response_counts,
    int32_t scale_count,
    const int32_t* cuts_scale_major,
    int32_t cuts_per_scale,
    const ManorMCudaPhase1Options* options,
    ManorMCudaPhase1Output* output,
    ManorMCudaPhase1Stats* stats,
    const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* plan);

MANOR_M_CUDA_API int manor_m_cuda_response_knn_host(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int64_t query_count,
    int32_t neighbor_count, int32_t* host_index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats);

/* Additive exact range form for bounded callers.  The returned C-order array
 * has query_count rows beginning at the global anchor query_first; self-first
 * and global-ID tie ordering are identical to the prefix API.  This entry
 * point is the transport primitive for Phase-1/2 consumers that must not
 * materialize the complete N*k host index. */
MANOR_M_CUDA_API int manor_m_cuda_response_knn_host_range(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int64_t query_first,
    int64_t query_count, int32_t neighbor_count,
    int32_t* host_index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats);

/* Exact allocation arithmetic used by every response-kNN entry point.  The
 * fixed term is paid once per call and bytes_per_query is the additional
 * scratch needed for one generated output row.  A production planner can
 * therefore select the largest query tile that fits the currently observed
 * GPU budget without timing trial kernels or duplicating private constants. */
MANOR_M_CUDA_API int manor_m_cuda_response_knn_memory(
    int64_t point_count, int32_t response_dimension,
    int32_t neighbor_count, int32_t tiled_output,
    uint64_t* fixed_bytes, uint64_t* bytes_per_query);

/* Build only the exact rows selected by owner_mask, in increasing global
 * owner order.  kNN remains exact and query-tiled; neither a complete device
 * nor host N*k index is materialized.  The scratch ceiling includes the
 * bounded gather workspace, so production chunk selection is arithmetic and
 * never performs throughput trial runs. */
MANOR_M_CUDA_API int manor_m_cuda_response_knn_compact_owners_host(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int32_t neighbor_count,
    const uint8_t* owner_mask, int64_t owner_count,
    int32_t* compact_index_zero_based,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats);

MANOR_M_CUDA_API int manor_m_cuda_phase1_bounded(
    ManorMCudaContext* context, const float* x_point_major,
    int64_t point_count, int32_t ambient_dimension, int32_t manifold_dimension,
    const int32_t* index_zero_based, int64_t active_anchor_count,
    int32_t neighbor_count, const int32_t* response_counts, int32_t scale_count,
    const int32_t* cuts_scale_major, int32_t cuts_per_scale,
    const ManorMCudaPhase1Options* options, ManorMCudaPhase1Output* output,
    ManorMCudaPhase1Stats* stats, const ManorMCudaP1CapacityOptions* capacity,
    ManorMCudaP1CapacityPlan* actual_plan);

MANOR_M_CUDA_API int manor_m_cuda_phase2(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats);

/* Additive entry point: existing descriptors, binary layouts and phase2 calls
 * remain unchanged. Zero retains full resident scratch; a positive count caps
 * independent owner aggregation scratch only. Indices and model state remain
 * resident. Negative counts are invalid and never silently fall back. */
MANOR_M_CUDA_API int manor_m_cuda_phase2_tiled(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners);

/* Additive bounded-index extension. Original ABI-3 descriptors stay intact.
 * Explicitly clear phase1.response_index and phase1.compact_index first;
 * their CPU snapshots and the other resident data/P1 state must remain alive.
 * Whole owner rows are copied in order through at most two pinned/device
 * buffers. Shared k-vectors remain resident. P1 still needs its full index.
 * Index H2D is counted both here and in resident host_to_device_bytes.
 * copy_milliseconds is summed DMA work, not nonoverlapped wall time. */
typedef struct ManorMCudaIndexStreamStats {
    uint32_t struct_size;
    uint32_t extension_version;
    uint64_t enabled;
    uint64_t logical_index_bytes;
    uint64_t device_index_bytes;
    uint64_t pinned_host_bytes;
    uint64_t tile_owners;
    uint64_t tiles_per_pass;
    uint64_t buffer_count;
    uint64_t upload_calls;
    uint64_t host_to_device_bytes;
    double host_fill_milliseconds;
    double host_reuse_wait_milliseconds;
    double copy_milliseconds;
} ManorMCudaIndexStreamStats;

/* Exact generated-index Phase-2 evidence.  This path consumes response-kNN
 * tiles directly on the device and never materializes a complete N*k or M*k
 * index on either host or device.  The response SoA and the declared kNN
 * scratch coexist with the ordinary Phase-2 working set and are included in
 * the native capacity check. */
typedef struct ManorMCudaGeneratedIndexStats {
    uint32_t struct_size;
    uint32_t extension_version;
    uint64_t enabled;
    uint64_t logical_full_index_bytes;
    uint64_t logical_compact_index_bytes;
    uint64_t materialized_host_index_bytes;
    uint64_t materialized_device_index_bytes;
    uint64_t scratch_memory_bytes;
    uint64_t response_soa_bytes;
    uint64_t generation_passes;
    uint64_t query_tiles;
    uint64_t owner_rows;
    uint64_t peak_query_rows;
    uint64_t peak_owner_rows;
    uint64_t peak_device_index_tile_bytes;
    uint64_t owner_row_map_h2d_bytes;
    double knn_reported_milliseconds;
    double generation_wall_milliseconds;
} ManorMCudaGeneratedIndexStats;

MANOR_M_CUDA_API int manor_m_cuda_phase2_index_tiled(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners,
    int32_t index_tile_owners,
    ManorMCudaIndexStreamStats* index_stats);

/* Exact large-N extension. input->index_zero_based is an [owner_count,k]
 * C-contiguous int32 array ordered by increasing global owner id, rather than
 * a complete [point_count,k] array. The owner mask remains the source of truth
 * and the core validates both the row count and every global neighbor id.
 * This removes the complete host N*k prerequisite without changing Phase-2
 * arithmetic. A positive index_tile_owners streams the compact rows through
 * the existing double-buffered H2D path; zero keeps the compact M*k array
 * resident for the call. Existing entry points retain their original shape. */
MANOR_M_CUDA_API int manor_m_cuda_phase2_compact_index_tiled(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners,
    int32_t index_tile_owners,
    ManorMCudaIndexStreamStats* index_stats);

/* Exact capacity fallback for generated Phase-1 runs.  The caller supplies no
 * index (input->index_zero_based must be null); input->neighbor_count remains
 * the exact k.  Each global iteration regenerates the deterministic response
 * kNN stream once for regression and once for aggregation.  No complete host
 * or retained device index is created.  The scratch budget selects the largest
 * exact device tile that fits; when the entire index fits, one full-size tile
 * is intentionally used to avoid needless launch overhead.  Existing resident
 * and compact-index entry points remain unchanged. */
MANOR_M_CUDA_API int manor_m_cuda_phase2_response_tiled(
    ManorMCudaContext* context,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMCudaPhase2Evaluation* evaluation,
    ManorMCudaPhase2Output* output,
    ManorMCudaPhase2Stats* stats,
    int32_t aggregation_tile_owners,
    uint64_t knn_scratch_memory_bytes,
    ManorMCudaGeneratedIndexStats* generated_stats);

#ifdef __cplusplus
}
#endif

#endif
