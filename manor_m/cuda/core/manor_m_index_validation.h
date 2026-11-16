#ifndef MANOR_M_INDEX_VALIDATION_H
#define MANOR_M_INDEX_VALIDATION_H

#include <cstddef>
#include <cstdint>

// Host-only check for a borrowed zero-based int32 index. Arbitrary inputs use
// it on every call. An explicitly owned immutable resident-session index may
// use it once at registration and bind the result to pointer, byte count,
// upper bound, and generation; clearing the phase1 group revokes that record.
// Moving the error branch outside the loop exposes an integer OR reduction to
// the host compiler's SIMD pass.
inline bool manor_m_index_range_valid(
    const int32_t* values, size_t count, int32_t upper_bound)
{
    if (upper_bound <= 0 || (!values && count)) return false;
    const uint32_t bound = static_cast<uint32_t>(upper_bound);
    uint32_t invalid = 0;
    for (size_t i = 0; i < count; ++i)
        invalid |= static_cast<uint32_t>(values[i]) >= bound;
    // A negative int32 becomes >= 2^31, while the positive bound is <= INT_MAX.
    // Thus this is exactly (value < 0 || value >= upper_bound), without FP math.
    return invalid == 0;
}

#endif
