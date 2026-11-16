#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

// Tile whole owner rows, never the ordered k-neighbor reduction. The full
// validated index remains on the host. Shared k-vectors stay resident; K=0
// needs no staging buffers. The default (requested=0) is unchanged residency.
struct OwnerIndexTiles {
    int owners, neighbors, capacity, buffers;
    bool streamed, generated;
    size_t logical_bytes, buffer_bytes, device_bytes;

    static size_t multiply(size_t a, size_t b)
    {
        if (b && a > std::numeric_limits<size_t>::max() / b)
            throw std::overflow_error("Index tile size overflows size_t.");
        return a * b;
    }

    OwnerIndexTiles(int m, int k, int requested, bool shared, int iterations,
                    bool generated_index = false)
        : owners(m), neighbors(k), capacity(0), buffers(0),
          streamed(requested > 0 && !shared && !generated_index),
          generated(generated_index), logical_bytes(0),
          buffer_bytes(0), device_bytes(0)
    {
        if (m <= 0 || k <= 0 || requested < 0 || iterations < 0)
            throw std::invalid_argument("Invalid index tile dimensions.");
        logical_bytes = multiply(multiply(shared ? 1u : size_t(m), size_t(k)), sizeof(int));
        if (generated) {
            capacity = m;
            return;
        }
        capacity = streamed ? std::min(m, requested) : m;
        if (streamed) {
            buffers = iterations ? std::min(2, count()) : 0;
            buffer_bytes = multiply(multiply(size_t(capacity), size_t(k)), sizeof(int));
            device_bytes = multiply(size_t(buffers), buffer_bytes);
        } else {
            device_bytes = logical_bytes;
        }
    }

    int count() const { return 1 + (owners - 1) / capacity; }
    int size_at(int base) const
    {
        if (base < 0 || base >= owners)
            throw std::out_of_range("Index tile base is outside its owner set.");
        return std::min(capacity, owners - base);
    }
    size_t offset(int base) const
    {
        (void)size_at(base);
        return multiply(size_t(base), size_t(neighbors));
    }
    size_t bytes_at(int base) const
    {
        return multiply(multiply(size_t(size_at(base)), size_t(neighbors)), sizeof(int));
    }
    void copy_at(int* destination, const int* source, int base,
                 const int* owner_rows = nullptr) const
    {
        if (!owner_rows) {
            std::memcpy(destination, source + offset(base), bytes_at(base));
            return;
        }
        const size_t row_bytes = multiply(size_t(neighbors), sizeof(int));
        for (int row = 0; row < size_at(base); ++row)
            std::memcpy(destination + size_t(row) * neighbors,
                source + size_t(owner_rows[base + row]) * neighbors, row_bytes);
    }
};
