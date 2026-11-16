#pragma once

// The distance sort is stable, but its input is produced by unordered atomic
// appends. Stability alone therefore does NOT define equal-distance ordering.
// Resolve only exact FP64 ties: self first, then ascending global point ID.
// Non-tied keys/ranks and every distance operation remain unchanged.
#if defined(__CUDACC__)
#define MFM_KNN_TIE_INLINE __host__ __device__ __forceinline__
#else
#define MFM_KNN_TIE_INLINE inline
#endif

MFM_KNN_TIE_INLINE int manor_knn_tie_rank(
    const unsigned long long* keys, const int* ids, int count, int rank,
    int query, int k)
{
    const unsigned long long key = keys[rank];
    // Candidates outside the requested prefix matter only for a boundary tie.
    if (rank >= k && key != keys[k - 1]) return k;
    const bool left = rank && keys[rank - 1] == key;
    const bool right = rank + 1 < count && keys[rank + 1] == key;
    if (!left && !right) return rank;
    int low = 0, high = rank;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (keys[mid] < key) low = mid + 1;
        else high = mid;
    }
    const int first = low, value = ids[rank];
    if (value == query) return first;
    int preceding = 0;
    for (int i = first; i < count && keys[i] == key; ++i)
        if (ids[i] == query || ids[i] < value) ++preceding;
    return first + preceding;
}

#undef MFM_KNN_TIE_INLINE
