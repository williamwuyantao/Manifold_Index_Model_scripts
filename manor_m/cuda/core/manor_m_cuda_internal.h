#ifndef MANOR_M_CUDA_INTERNAL_H
#define MANOR_M_CUDA_INTERNAL_H

#include "manor_m_cuda.h"

/* Private cross-translation-unit hooks.  Visibility remains hidden with the
 * library target's default settings; adapters must use only manor_m_cuda.h. */
int manor_m_cuda_internal_fail(int status, const char* message);
int manor_m_cuda_internal_context_device(
    const ManorMCudaContext* context, int32_t* device_ordinal);

/* Same allocation arithmetic used by the exact kNN builder. */
void manor_m_cuda_internal_knn_memory(int n, int my, int k, bool tiled_output,
    uint64_t* fixed_bytes, uint64_t* bytes_per_query);

/* Private exact-kNN tile stream. The consumer sees one contiguous device
 * tile in global-anchor order; the pointer remains valid only for the call.
 * Returning a nonzero core status aborts generation without materializing a
 * complete device or host N*k index. */
typedef int (*ManorMCudaKnnTileConsumer)(
    void* user, const int32_t* device_index, int32_t query_first,
    int32_t query_count, int32_t neighbor_count);

/* Prepared immutable response layouts for repeated exact-kNN passes inside
 * one Phase-2 call.  The object owns only device-side layouts derived from the
 * already-resident response; it never owns or materializes an N*k index. */
struct ManorMCudaPreparedResponseKnn;

int manor_m_cuda_internal_prepare_response_knn(
    ManorMCudaContext* context, const float* response_point_major,
    const float* device_response_point_major, int64_t point_count,
    int32_t response_dimension,
    ManorMCudaPreparedResponseKnn** prepared);

void manor_m_cuda_internal_destroy_prepared_response_knn(
    ManorMCudaPreparedResponseKnn* prepared);

int manor_m_cuda_internal_response_knn_tiles(
    ManorMCudaContext* context, const float* response_point_major,
    int64_t point_count, int32_t response_dimension, int64_t query_first,
    int64_t query_count, int32_t neighbor_count,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user);

/* Variant used by the in-call Phase-2 generator.  The point-major response is
 * already resident in device memory, so this avoids a duplicate allocation
 * and upload while retaining host validation and the exact same kNN builder. */
int manor_m_cuda_internal_response_knn_tiles_device(
    ManorMCudaContext* context, const float* response_point_major,
    const float* device_response_point_major, int64_t point_count,
    int32_t response_dimension, int64_t query_first, int64_t query_count,
    int32_t neighbor_count, const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user);

/* Reuses one prepared response SoA/subsample/AoS across repeated Phase-2
 * generation passes.  Query tiles and their transient output remain bounded
 * by the caller's exact scratch budget. */
int manor_m_cuda_internal_response_knn_tiles_prepared(
    ManorMCudaPreparedResponseKnn* prepared,
    const float* device_response_point_major,
    int64_t point_count, int32_t response_dimension,
    int64_t query_first, int64_t query_count, int32_t neighbor_count,
    const ManorMCudaResponseKnnOptions* options,
    ManorMCudaResponseKnnStats* stats,
    ManorMCudaKnnTileConsumer consumer, void* user);

struct ManorMGeneratedPhase2Source {
    ManorMCudaContext* context = nullptr;
    const float* response_point_major = nullptr;
    uint64_t scratch_memory_bytes = 0;
    ManorMCudaGeneratedIndexStats* stats = nullptr;
    ManorMCudaPreparedResponseKnn* prepared = nullptr;
    uint64_t preparation_count = 0;
};

extern thread_local ManorMGeneratedPhase2Source*
    manor_m_cuda_active_generated_phase2;

#endif
