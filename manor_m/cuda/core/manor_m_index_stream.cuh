#pragma once

#include "manor_m_index_tiles.h"
#include <cstdint>
#include <cstring>

// Included after owner_cuda_check and its host timing helpers. Copies run on
// a private nonblocking stream. ready orders H2D before compute; consumed
// prevents overwriting either the pinned source or device tile too early.
// There are at most two pinned host buffers and two device buffers per call.
class OwnerIndexStream {
public:
    explicit OwnerIndexStream(const OwnerIndexTiles& layout) : layout_(layout) {}
    OwnerIndexStream(const OwnerIndexStream&) = delete;
    OwnerIndexStream& operator=(const OwnerIndexStream&) = delete;
    ~OwnerIndexStream()
    {
        // Also safe on a partially constructed/failed call. Compute can still
        // reference a tile, so neither a pinned source nor device tile is freed
        // until both streams have drained. This does not run on resident calls.
        if (copy_stream_) {
            cudaStreamSynchronize(copy_stream_);
            cudaStreamSynchronize(nullptr);
        }
        for (auto& s : slots_) {
            if (s.start) cudaEventDestroy(s.start);
            if (s.ready) cudaEventDestroy(s.ready);
            if (s.consumed) cudaEventDestroy(s.consumed);
            if (s.device) cudaFree(s.device);
            if (s.host) cudaFreeHost(s.host);
        }
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
    }

    void allocate()
    {
        if (!layout_.buffers) return;
        const auto start = std::chrono::steady_clock::now();
        owner_cuda_check(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking),
            "create index copy stream");
        for (int i = 0; i < layout_.buffers; ++i) {
            auto& s = slots_[i];
            owner_cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&s.host),
                layout_.buffer_bytes, cudaHostAllocDefault), "allocate pinned index tile");
            owner_cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.device),
                layout_.buffer_bytes), "allocate device index tile");
            owner_cuda_check(cudaEventCreate(&s.start), "create index copy start");
            owner_cuda_check(cudaEventCreate(&s.ready), "create index copy ready");
            owner_cuda_check(cudaEventCreateWithFlags(&s.consumed, cudaEventDisableTiming),
                "create index tile consumed event");
        }
        owner_cuda_allocate_milliseconds += owner_host_elapsed_milliseconds(start);
    }

    const int* prepare(const int* validated_host, int base,
                       const int* owner_rows = nullptr)
    {
        if (!layout_.streamed || !layout_.buffers || active_ >= 0)
            throw OwnerMexError("manor_m_phase2_mex:indexTile", "Invalid index tile lifecycle.");
        active_ = static_cast<int>(upload_calls % layout_.buffers);
        auto& s = slots_[active_];
        if (s.in_flight) {
            const auto start = std::chrono::steady_clock::now();
            owner_cuda_check(cudaEventSynchronize(s.consumed), "wait for consumed index tile");
            host_reuse_wait_milliseconds += owner_host_elapsed_milliseconds(start);
            collect(s);
        }
        const size_t bytes = layout_.bytes_at(base);
        const auto start = std::chrono::steady_clock::now();
        layout_.copy_at(s.host, validated_host, base, owner_rows);
        host_fill_milliseconds += owner_host_elapsed_milliseconds(start);
        owner_cuda_check(cudaEventRecord(s.start, copy_stream_), "index copy start");
        owner_cuda_check(cudaMemcpyAsync(s.device, s.host, bytes,
            cudaMemcpyHostToDevice, copy_stream_), "upload index tile");
        ++upload_calls;
        host_to_device_bytes += bytes;
#if defined(MANOR_M_PHASE2_CORE_LIBRARY)
        if (manor_m_cuda_active_resident && manor_m_cuda_active_resident->enabled)
            manor_m_cuda_active_resident->h2d_bytes += bytes;
#endif
        owner_cuda_check(cudaEventRecord(s.ready, copy_stream_), "index copy ready");
        owner_cuda_check(cudaStreamWaitEvent(nullptr, s.ready, 0), "compute wait for index tile");
        return s.device;
    }

    void consumed()
    {
        if (active_ < 0)
            throw OwnerMexError("manor_m_phase2_mex:indexTile", "No active index tile.");
        auto& s = slots_[active_];
        owner_cuda_check(cudaEventRecord(s.consumed, nullptr), "index tile consumed");
        s.in_flight = true;
        active_ = -1;
    }

    void finish()
    {
        if (active_ >= 0)
            throw OwnerMexError("manor_m_phase2_mex:indexTile", "Unconsumed index tile.");
        for (int i = 0; i < layout_.buffers; ++i) {
            auto& s = slots_[i];
            if (s.in_flight) {
                owner_cuda_check(cudaEventSynchronize(s.consumed), "finish index tile");
                collect(s);
            }
        }
    }

    uint64_t upload_calls = 0, host_to_device_bytes = 0;
    double host_fill_milliseconds = 0.0, host_reuse_wait_milliseconds = 0.0;
    double copy_milliseconds = 0.0; // Sum of DMA durations, not extra wall time.

private:
    struct Slot {
        int* host = nullptr;
        int* device = nullptr;
        cudaEvent_t start = nullptr, ready = nullptr, consumed = nullptr;
        bool in_flight = false;
    } slots_[2];
    const OwnerIndexTiles& layout_;
    cudaStream_t copy_stream_ = nullptr;
    int active_ = -1;

    void collect(Slot& s)
    {
        float elapsed = 0.0f;
        owner_cuda_check(cudaEventElapsedTime(&elapsed, s.start, s.ready), "index DMA elapsed");
        copy_milliseconds += elapsed;
        s.in_flight = false;
    }
};
