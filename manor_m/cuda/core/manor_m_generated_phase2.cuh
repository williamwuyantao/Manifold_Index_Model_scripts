#pragma once

#include "manor_m_cuda_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef MANOR_M_GENERATED_PREPARED_RESPONSE
#define MANOR_M_GENERATED_PREPARED_RESPONSE 0
#endif

/* Device-only bridge between exact response-kNN tiles and Phase 2.  The kNN
 * tile contains every query in a contiguous global range.  Phase 2 owns only
 * a sorted subset of those queries, so a tiny row-map selects owner rows
 * without copying the k-vectors or constructing an M*k compact array. */
template <typename Consumer>
class OwnerGeneratedPhase2Visitor {
public:
    OwnerGeneratedPhase2Visitor(
        const int* owner_ids, int owner_count, Consumer consumer,
        ManorMCudaGeneratedIndexStats* stats)
        : owner_ids_(owner_ids), owner_count_(owner_count),
          consumer_(std::move(consumer)), stats_(stats)
    {
    }

    OwnerGeneratedPhase2Visitor(const OwnerGeneratedPhase2Visitor&) = delete;
    OwnerGeneratedPhase2Visitor& operator=(
        const OwnerGeneratedPhase2Visitor&) = delete;

    ~OwnerGeneratedPhase2Visitor()
    {
        if (!device_rows_) return;
        const auto started = std::chrono::steady_clock::now();
        cudaFree(device_rows_);
        owner_cuda_free_milliseconds += owner_host_elapsed_milliseconds(started);
    }

    static int accept(
        void* opaque, const int32_t* device_index, int32_t query_first,
        int32_t query_count, int32_t neighbor_count)
    {
        auto& self = *static_cast<OwnerGeneratedPhase2Visitor*>(opaque);
        if (!device_index || query_first < 0 || query_count < 1 ||
                neighbor_count < 1)
            return manor_m_cuda_internal_fail(
                MANOR_M_CUDA_INTERNAL_ERROR,
                "Generated Phase-2 received an invalid exact-kNN tile.");

        const int64_t query_end = static_cast<int64_t>(query_first) + query_count;
        const int owner_first = self.emitted_owners_;
        self.local_rows_.clear();
        while (self.emitted_owners_ < self.owner_count_) {
            const int owner = self.owner_ids_[self.emitted_owners_];
            if (owner < query_first)
                return manor_m_cuda_internal_fail(
                    MANOR_M_CUDA_INTERNAL_ERROR,
                    "Generated Phase-2 owner order moved behind its query tile.");
            if (static_cast<int64_t>(owner) >= query_end) break;
            self.local_rows_.push_back(owner - query_first);
            ++self.emitted_owners_;
        }

        if (self.stats_) {
            ++self.stats_->query_tiles;
            self.stats_->peak_query_rows = std::max<uint64_t>(
                self.stats_->peak_query_rows,
                static_cast<uint64_t>(query_count));
            self.stats_->peak_device_index_tile_bytes = std::max<uint64_t>(
                self.stats_->peak_device_index_tile_bytes,
                static_cast<uint64_t>(query_count) *
                    static_cast<uint64_t>(neighbor_count) * sizeof(int32_t));
        }
        if (self.local_rows_.empty()) return MANOR_M_CUDA_SUCCESS;

        const int selected = static_cast<int>(self.local_rows_.size());
        self.ensure_capacity(selected);
        owner_cuda_check(cudaMemcpy(
            self.device_rows_, self.local_rows_.data(),
            static_cast<size_t>(selected) * sizeof(int32_t),
            cudaMemcpyHostToDevice), "upload generated-index owner row map");
        if (self.stats_) {
            self.stats_->owner_rows += static_cast<uint64_t>(selected);
            self.stats_->peak_owner_rows = std::max<uint64_t>(
                self.stats_->peak_owner_rows,
                static_cast<uint64_t>(selected));
            self.stats_->owner_row_map_h2d_bytes +=
                static_cast<uint64_t>(selected) * sizeof(int32_t);
        }
        self.consumer_(device_index, self.device_rows_, owner_first, selected);
        return MANOR_M_CUDA_SUCCESS;
    }

    void require_complete() const
    {
        if (emitted_owners_ != owner_count_)
            throw OwnerMexError(
                "manor_m_phase2_mex:generatedIndex",
                "Generated Phase-2 did not visit every owner row.");
    }

private:
    void ensure_capacity(int selected)
    {
        if (selected <= capacity_) return;
        if (device_rows_) {
            const auto free_started = std::chrono::steady_clock::now();
            owner_cuda_check(cudaFree(device_rows_),
                "free generated-index owner row map");
            owner_cuda_free_milliseconds +=
                owner_host_elapsed_milliseconds(free_started);
            device_rows_ = nullptr;
        }
        const auto allocate_started = std::chrono::steady_clock::now();
        owner_cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&device_rows_),
            static_cast<size_t>(selected) * sizeof(int32_t)),
            "allocate generated-index owner row map");
        owner_cuda_allocate_milliseconds +=
            owner_host_elapsed_milliseconds(allocate_started);
        capacity_ = selected;
    }

    const int* owner_ids_ = nullptr;
    int owner_count_ = 0;
    int emitted_owners_ = 0;
    int capacity_ = 0;
    int32_t* device_rows_ = nullptr;
    std::vector<int32_t> local_rows_;
    Consumer consumer_;
    ManorMCudaGeneratedIndexStats* stats_ = nullptr;
};

template <typename Consumer>
static void owner_generated_phase2_for_each(
    const float* device_response_point_major, const int* owner_ids,
    int owner_count, int point_count, int response_dimension,
    int neighbor_count, Consumer&& consumer)
{
    ManorMGeneratedPhase2Source* source =
        manor_m_cuda_active_generated_phase2;
    if (!source || !source->context || !source->response_point_major ||
            !device_response_point_major || !owner_ids || owner_count < 1 ||
            point_count < 1 || response_dimension < 1 || neighbor_count < 1 ||
            !source->scratch_memory_bytes)
        throw OwnerMexError(
            "manor_m_phase2_mex:generatedIndex",
            "Generated Phase-2 source is incomplete.");

    using ConsumerType = std::decay_t<Consumer>;
    OwnerGeneratedPhase2Visitor<ConsumerType> visitor(
        owner_ids, owner_count, std::forward<Consumer>(consumer), source->stats);
    ManorMCudaResponseKnnOptions options{};
    options.struct_size = sizeof(options);
    options.abi_version = MANOR_M_CUDA_ABI_VERSION;
    options.scratch_memory_bytes = source->scratch_memory_bytes;
    options.max_retry_rounds = 12;
    options.collect_stats = 1;
    ManorMCudaResponseKnnStats pass_stats{};
    const auto started = std::chrono::steady_clock::now();
#if MANOR_M_GENERATED_PREPARED_RESPONSE
    if (!source->prepared) {
        const int prepare_status = manor_m_cuda_internal_prepare_response_knn(
            source->context, source->response_point_major,
            device_response_point_major, point_count, response_dimension,
            &source->prepared);
        if (prepare_status != MANOR_M_CUDA_SUCCESS)
            throw OwnerMexError(
                "manor_m_phase2_mex:generatedIndex",
                std::string("Prepare exact generated Phase-2 response failed: ") +
                    manor_m_cuda_last_error());
        ++source->preparation_count;
    }
    const int status = manor_m_cuda_internal_response_knn_tiles_prepared(
        source->prepared, device_response_point_major,
        point_count, response_dimension, 0, point_count, neighbor_count,
        &options, &pass_stats,
        &OwnerGeneratedPhase2Visitor<ConsumerType>::accept, &visitor);
#else
    const int status = manor_m_cuda_internal_response_knn_tiles_device(
        source->context, source->response_point_major,
        device_response_point_major, point_count, response_dimension,
        0, point_count, neighbor_count, &options, &pass_stats,
        &OwnerGeneratedPhase2Visitor<ConsumerType>::accept, &visitor);
#endif
    const double wall = owner_host_elapsed_milliseconds(started);
    if (status != MANOR_M_CUDA_SUCCESS)
        throw OwnerMexError(
            "manor_m_phase2_mex:generatedIndex",
            std::string("Exact generated Phase-2 index failed: ") +
                manor_m_cuda_last_error());
    visitor.require_complete();
    if (source->stats) {
        ++source->stats->generation_passes;
        source->stats->knn_reported_milliseconds += pass_stats.total_milliseconds;
        source->stats->generation_wall_milliseconds += wall;
    }
}
