#pragma once
#include "../core/manor_m_cuda.h"

typedef struct ManorMMultiGpu ManorMMultiGpu;
typedef struct ManorMMultiMemory {
    uint32_t struct_size, version;
    uint64_t working_bytes, reserve_bytes, pinned_bytes_per_device;
    int32_t maximum_tile_owners;
    int32_t flags; // bit 0: bounded host relay; bit 1: streamed index control
} ManorMMultiMemory;

typedef struct ManorMMultiStats {
    uint32_t struct_size, version;
    int32_t device_count, owner_count, valid_iterations, stopped_no_heavy;
    uint64_t logical_index_bytes, uploaded_index_bytes;
    uint64_t peer_bytes, relayed_bytes, pinned_bytes;
    double regression_ms, masks_ms, aggregation_ms, communication_ms;
} ManorMMultiStats;

#ifdef __cplusplus
extern "C" {
#endif
MANOR_M_CUDA_API const char* manor_m_multi_error(void);
MANOR_M_CUDA_API int manor_m_multi_minimum_bytes(
    int32_t n, int32_t d, int32_t mm, int32_t my, int32_t owners,
    int32_t neighbors, uint64_t* required);
MANOR_M_CUDA_API int manor_m_multi_create(
    ManorMCudaContext* const* contexts, int32_t device_count,
    const ManorMCudaPhase2Input* input,
    const ManorMCudaPhase2Options* options,
    const ManorMMultiMemory* memory, ManorMMultiGpu** result);
MANOR_M_CUDA_API int manor_m_multi_step(
    ManorMMultiGpu* model, float h, float rho, ManorMMultiStats* stats);
MANOR_M_CUDA_API int manor_m_multi_snapshot(
    ManorMMultiGpu* model, float* tangent, float* center,
    int32_t* neighbors, uint8_t* heavy, uint8_t* keep, uint8_t* failed);
MANOR_M_CUDA_API int manor_m_multi_stats(
    ManorMMultiGpu* model, ManorMMultiStats* stats);
MANOR_M_CUDA_API int manor_m_multi_readout_cache(
    ManorMMultiGpu* model, int32_t device_slot, uint64_t* bytes, int32_t* state_iteration);
MANOR_M_CUDA_API int manor_m_multi_evaluate(
    ManorMMultiGpu* model, int32_t device_slot,
    const ManorMCudaPhase2Evaluation* evaluation, float h, float rho,
    int32_t initial_normalization, float* mse, float* coordinate, float* tangent,
    float* query_contributions);
MANOR_M_CUDA_API int manor_m_multi_reduce_queries(
    ManorMMultiGpu* model, int32_t device_slot, const float* query_contributions,
    int32_t queries, int32_t slots, int32_t initial_normalization,
    float* mse, float* coordinate, float* tangent);
MANOR_M_CUDA_API int manor_m_multi_destroy(ManorMMultiGpu* model);
#ifdef __cplusplus
}
#endif
