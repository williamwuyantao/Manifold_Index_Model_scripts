#ifndef MANOR_M_CUDA_RESIDENT_H
#define MANOR_M_CUDA_RESIDENT_H

// Private, explicitly enabled, context-owned immutable input storage. No global
// dataset cache: different Contexts, processes and GPUs never share entries.
#include "manor_m_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct ManorMResidentEntry {
    std::string name;
    const void* host = nullptr;
    void* device = nullptr;
    size_t bytes = 0;
    const void* source_index = nullptr;
    const void* source_owner = nullptr;
    std::vector<int32_t> host_index;
    ~ManorMResidentEntry() { if (device) cudaFree(device); }
};

enum ManorMResidentValidationKind : uint32_t {
    MANOR_M_RESIDENT_FINITE_F32 = 1u,
    MANOR_M_RESIDENT_INDEX_RANGE = 2u,
};

struct ManorMResidentValidation {
    std::string name;
    const void* host = nullptr;
    size_t bytes = 0;
    uint32_t kinds = 0;
    int64_t index_upper_bound = 0;
};

struct ManorMResidentCache {
    bool enabled = false;
    uint64_t budget = 0;
    uint64_t h2d_bytes = 0;
    uint64_t generated_bytes = 0;
    uint64_t hits = 0;
    uint64_t skipped_h2d_bytes = 0;
    uint64_t reset_d2d_bytes = 0;
    std::vector<std::unique_ptr<ManorMResidentEntry>> entries;
    std::vector<ManorMResidentValidation> validations;

    uint64_t bytes() const {
        uint64_t result = 0;
        for (const auto& e : entries) result += e->bytes;
        return result;
    }
    void* find(const void* host, size_t count, bool count_hit = true) {
        if (!enabled || !host || !count) return nullptr;
        for (const auto& e : entries) {
            if (e->host == host && e->bytes == count) {
                cudaPointerAttributes attributes{};
                const cudaError_t error = cudaPointerGetAttributes(&attributes, e->device);
                if (error != cudaSuccess || attributes.type != cudaMemoryTypeDevice)
                    throw std::runtime_error("Resident device allocation is no longer valid; recreate the session.");
                if (count_hit) ++hits;
                return e->device;
            }
        }
        return nullptr;
    }
    void adopt(const char* name, const void* host, void* device, size_t count) {
        // Caller retains ownership if this throws. Never evict implicitly.
        if (!enabled || !name || !*name || !host || !device || !count)
            throw std::invalid_argument("Invalid resident buffer registration.");
        for (const auto& e : entries)
            if (e->name == name || e->host == host)
                throw std::invalid_argument("Resident buffer already registered; clear its group before replacement.");
        if (count > budget || bytes() > budget - count)
            throw std::invalid_argument("Resident cache byte budget exceeded; clear or explicitly increase the session budget.");
        auto e = std::make_unique<ManorMResidentEntry>();
        e->name = name; e->host = host; e->bytes = count;
        // Allocate vector capacity before taking device ownership.
        entries.reserve(entries.size() + 1);
        e->device = device;
        entries.push_back(std::move(e));
    }
    void record_validation(const char* name, const void* host, size_t bytes,
        uint32_t kind, int64_t index_upper_bound = 0) {
        if (!enabled || !name || !*name || !host || !bytes || !kind)
            throw std::invalid_argument("Invalid resident validation registration.");
        for (auto& validation : validations) {
            if (validation.host == host && validation.bytes == bytes) {
                validation.kinds |= kind;
                if (kind & MANOR_M_RESIDENT_INDEX_RANGE)
                    validation.index_upper_bound = index_upper_bound;
                return;
            }
        }
        validations.push_back(ManorMResidentValidation{
            name, host, bytes, kind, index_upper_bound});
    }
    bool validated(const void* host, size_t bytes, uint32_t kind,
        int64_t index_upper_bound = 0) const {
        if (!enabled || !host || !bytes || !kind) return false;
        for (const auto& validation : validations) {
            if (validation.host != host || validation.bytes != bytes ||
                    (validation.kinds & kind) != kind)
                continue;
            if ((kind & MANOR_M_RESIDENT_INDEX_RANGE) &&
                    validation.index_upper_bound != index_upper_bound)
                continue;
            return true;
        }
        return false;
    }
    void clear(const std::string& group) {
        if (group.empty()) { entries.clear(); validations.clear(); return; }
        const std::string prefix = group + ".";
        for (auto i = entries.begin(); i != entries.end();) {
            if ((*i)->name == group || (*i)->name.compare(0, prefix.size(), prefix) == 0)
                i = entries.erase(i);
            else ++i;
        }
        for (auto i = validations.begin(); i != validations.end();) {
            if (i->name == group || i->name.compare(0, prefix.size(), prefix) == 0)
                i = validations.erase(i);
            else ++i;
        }
    }
};

ManorMResidentCache* manor_m_cuda_internal_resident(ManorMCudaContext* context);
extern thread_local ManorMResidentCache* manor_m_cuda_active_resident;

class ManorMResidentCallScope {
    ManorMResidentCache* previous_;
public:
    explicit ManorMResidentCallScope(ManorMCudaContext* context)
        : previous_(manor_m_cuda_active_resident) {
        manor_m_cuda_active_resident = manor_m_cuda_internal_resident(context);
    }
    ~ManorMResidentCallScope() { manor_m_cuda_active_resident = previous_; }
};

inline void* manor_m_resident_find(const void* host, size_t bytes) {
    return manor_m_cuda_active_resident
        ? manor_m_cuda_active_resident->find(host, bytes) : nullptr;
}

inline bool manor_m_resident_finite_f32(const void* host, size_t bytes) {
    return manor_m_cuda_active_resident &&
        manor_m_cuda_active_resident->validated(
            host, bytes, MANOR_M_RESIDENT_FINITE_F32);
}

inline bool manor_m_resident_index_range_validated(
    const void* host, size_t bytes, int64_t upper_bound) {
    return manor_m_cuda_active_resident &&
        manor_m_cuda_active_resident->validated(
            host, bytes, MANOR_M_RESIDENT_INDEX_RANGE, upper_bound);
}

// Packed owner indices are derived once from the immutable Phase1 mask.
// Keep the full kNN index for Phase1; never compact it destructively.
inline const int32_t* manor_m_resident_compact_index(
    const int32_t* index, const uint8_t* owner, size_t count)
{
    auto* cache = manor_m_cuda_active_resident;
    if (!cache || !cache->enabled) return nullptr;
    for (const auto& e : cache->entries)
        if (e->name == "phase1.compact_index" && e->source_index == index &&
            e->source_owner == owner && e->bytes == count * sizeof(int32_t))
            return static_cast<const int32_t*>(e->host);
    return nullptr;
}

// Only immutable inputs can be borrowed. Mutable Tan/Cen iteration state is
// always a separate allocation, reset from the saved initial state by D2D.
inline cudaError_t manor_m_resident_upload(void* dst, const void* src, size_t bytes) {
    if (!bytes) return cudaSuccess;
    void* cached = manor_m_resident_find(src, bytes);
    if (!cached) {
        const cudaError_t error = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        if (error == cudaSuccess && manor_m_cuda_active_resident &&
            manor_m_cuda_active_resident->enabled)
            manor_m_cuda_active_resident->h2d_bytes += bytes;
        return error;
    }
    if (dst == cached) {
        manor_m_cuda_active_resident->skipped_h2d_bytes += bytes;
        return cudaSuccess;
    }
    manor_m_cuda_active_resident->reset_d2d_bytes += bytes;
    return cudaMemcpy(dst, cached, bytes, cudaMemcpyDeviceToDevice);
}

#endif
