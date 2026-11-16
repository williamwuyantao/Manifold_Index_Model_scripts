#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

// Independent owner scratch is tiled.  New shared-core builds also use this
// capacity for the exact neighbor-mask work buffer: regression still completes
// globally before heavy/keep selection, then each tile rebuilds its identical
// membership bits immediately before aggregation.  Gradients, owner lookup,
// old/new model state and all stopping decisions retain their global lifetime.
// Zero requests the original resident path. A positive cap may exceed M.
struct OwnerAggregationTiles {
    int owners;
    int capacity;

    OwnerAggregationTiles(int owner_count, int requested)
        : owners(owner_count), capacity(0)
    {
        if (owners < 0 || requested < 0)
            throw std::invalid_argument("Aggregation owner counts must be nonnegative.");
        capacity = requested == 0 ? owners : std::min(owners, requested);
    }

    int count() const { return capacity ? 1 + (owners - 1) / capacity : 0; }

    int size_at(int base) const
    {
        if (base < 0 || base > owners)
            throw std::out_of_range("Aggregation tile base is outside its owner set.");
        return std::min(capacity, owners - base);
    }

    static size_t index_offset(int base, int neighbors, bool shared)
    {
        return shared ? 0 : static_cast<size_t>(base) * neighbors;
    }
};
