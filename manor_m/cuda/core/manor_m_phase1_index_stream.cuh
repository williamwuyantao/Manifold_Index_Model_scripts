#pragma once

#include <cstring>

// P1 consumes all local rows, then computes the global owner rule, then
// consumes the same rows again for owner aggregation. Global point IDs never
// change. Two bounded pinned/device slots overlap the next copy with compute.
class Phase1IndexStream {
public:
    Phase1IndexStream(int rows, int k, int tile)
        : rows_(rows), k_(k), tile_(tile), count_(std::min(2, (rows - 1) / tile + 1)) {}
    Phase1IndexStream(const Phase1IndexStream&) = delete;
    ~Phase1IndexStream()
    {
        if (copy_) { cudaStreamSynchronize(copy_); cudaStreamSynchronize(nullptr); }
        for (auto& s : slots_) {
            if (s.ready) cudaEventDestroy(s.ready);
            if (s.used) cudaEventDestroy(s.used);
            if (s.gpu) cudaFree(s.gpu);
            if (s.host) cudaFreeHost(s.host);
        }
        if (copy_) cudaStreamDestroy(copy_);
    }
    void allocate()
    {
        STABLE_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
        const size_t bytes = static_cast<size_t>(tile_) * k_ * sizeof(int);
        for (int i = 0; i < count_; ++i) {
            auto& s = slots_[i];
            STABLE_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&s.host), bytes, cudaHostAllocDefault));
            STABLE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.gpu), bytes));
            STABLE_CUDA_CHECK(cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming));
            STABLE_CUDA_CHECK(cudaEventCreateWithFlags(&s.used, cudaEventDisableTiming));
        }
    }
    int* prepare(const int* host, int first)
    {
        auto& s = slots_[calls_++ % count_];
        if (s.active) STABLE_CUDA_CHECK(cudaEventSynchronize(s.used));
        const size_t bytes = static_cast<size_t>(std::min(tile_, rows_ - first)) * k_ * sizeof(int);
        std::memcpy(s.host, host + static_cast<size_t>(first) * k_, bytes);
        STABLE_CUDA_CHECK(cudaMemcpyAsync(s.gpu, s.host, bytes, cudaMemcpyHostToDevice, copy_));
        STABLE_CUDA_CHECK(cudaEventRecord(s.ready, copy_));
        STABLE_CUDA_CHECK(cudaStreamWaitEvent(nullptr, s.ready, 0));
        current_ = &s;
        transferred_bytes += bytes;
        if (manor_m_cuda_active_resident && manor_m_cuda_active_resident->enabled)
            manor_m_cuda_active_resident->h2d_bytes += bytes;
        return s.gpu;
    }
    void consumed()
    {
        STABLE_CUDA_CHECK(cudaEventRecord(current_->used, nullptr));
        current_->active = true;
        current_ = nullptr;
    }
    uint64_t transferred_bytes = 0;
private:
    struct Slot {
        int* host = nullptr;
        int* gpu = nullptr;
        cudaEvent_t ready = nullptr, used = nullptr;
        bool active = false;
    } slots_[2];
    Slot* current_ = nullptr;
    cudaStream_t copy_ = nullptr;
    int rows_, k_, tile_, count_;
    uint64_t calls_ = 0;
};
