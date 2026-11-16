#pragma once

// Production planning queries capacities and workspace sizes only. It never
// runs timing probes or candidate kernels; throughput options are fixed by the
// selected build configuration.

static constexpr size_t p1_allocation_margin = size_t(64) << 20;
static constexpr size_t p1_temporary_selector_bytes_per_row =
    sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint8_t)
    + 6 * sizeof(float);

static void p1_check_capacity_options(const ManorMCudaP1CapacityOptions* o)
{
    phase1_require(o && o->struct_size == sizeof(*o) && o->extension_version == 1,
        "Invalid P1 capacity extension descriptor.");
    phase1_require(std::isfinite(o->memory_fraction) && o->memory_fraction > 0 &&
        o->memory_fraction < 1 && std::isfinite(o->reserve_fraction) &&
        o->reserve_fraction >= 0 && o->reserve_fraction < 1 &&
        o->mode >= 0 && o->mode <= 4 && o->maximum_query_block >= 0 &&
        o->knn_scratch_bytes > 0 && o->retained_limit_bytes > 0,
        "Invalid P1 capacity budget/fractions/mode.");
}

static size_t p1_usable(size_t free, size_t total, const ManorMCudaP1CapacityOptions& o)
{
    const double reserve = std::max(double(o.reserve_bytes), double(total) * o.reserve_fraction);
    size_t usable = static_cast<size_t>(std::max(0.0, double(free) - reserve) * o.memory_fraction);
    if (o.working_bytes) usable = std::min(usable, static_cast<size_t>(o.working_bytes));
    return usable;
}

// Exact simultaneously allocated P1 state in phase1_impl. The original legacy
// estimator double-counted part of the diagnostic records. CUB query storage
// is queried from the actual linked CUDA version, not extrapolated from N.
static size_t p1_global_bytes(
    const Phase1CoreRequest& r, bool query_local_main_mask = false)
{
    const size_t n = r.N, a = r.anchors, d = r.D, mm = r.MM;
    /* The temporary selector record is query-block local in bounded calls.
       The selected record and final scalar/model state remain global. */
    const size_t neighborhood = query_local_main_mask
        ? sizeof(float) * a
        : sizeof(uint32_t) * a * stable_mask_words(r.kSlice);
    return n * (51 + 8 * d * (mm + 1)) + neighborhood
        + 4 * size_t(r.S) * (r.R + 2 + (r.options.h_null_correction ? r.R : 0)) + 32;
}

static size_t p1_query_local_bytes(
    const Phase1CoreRequest& r, int rows, bool fused_hybrid,
    bool query_local_main_mask = false)
{
    size_t per_row = p1_temporary_selector_bytes_per_row;
    if (query_local_main_mask)
        per_row += sizeof(uint32_t) * stable_mask_words(r.kSlice);
    if (fused_hybrid) {
        /* The fallback selector has one more temporary record.  Its bit mask
           is consumed completely by adaptive_final_extract_kernel and can be
           recycled at the next query block; no later kernel reads it. */
        per_row += p1_temporary_selector_bytes_per_row
            + sizeof(uint32_t) * stable_mask_words(r.kSlice);
    }
    return checked_mul_size(
        size_t(rows), per_row, "Phase-1 query-local selector bytes");
}

static size_t p1_staging_bytes(const Phase1CoreRequest& r, int rows)
{
    const int buffers = std::min(2, (r.anchors - 1) / rows + 1);
    return size_t(buffers) * rows * r.kSlice * sizeof(int32_t);
}

static AdaptiveMemoryBudget p1_bounded_budget(const Phase1CoreRequest& r,
    int tmax, size_t fixed, bool streamed, size_t free, size_t total,
    const ManorMCudaP1CapacityOptions& o, bool fused_hybrid,
    bool query_local_main_mask = false,
    size_t generated_knn_bytes_per_query = 0)
{
    AdaptiveMemoryBudget result{};
    auto& b = result.stable;
    b.free_bytes = free; b.total_bytes = total;
    const size_t usable = p1_usable(free, total, o);
    b.usable_bytes = usable > p1_allocation_margin ? usable - p1_allocation_margin : 0;
    b.fixed_bytes = fixed;
    int device = 0;
    STABLE_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    STABLE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    b.grid_y_limit = prop.maxGridSize[1];
    b.upper_bound = std::min(r.anchors, std::min(b.grid_y_limit,
        std::numeric_limits<int>::max() / r.kSlice));
    b.upper_bound = std::min(b.upper_bound, r.options.memory.max_query_block);
    if (o.maximum_query_block) b.upper_bound = std::min(b.upper_bound, o.maximum_query_block);
    auto query = [&](int q) {
        auto value = stable_query_memory(q, r.D, r.kSlice, tmax);
        if (streamed) value.total_bytes += p1_staging_bytes(r, q);
        value.total_bytes = checked_add_size(
            value.total_bytes,
            p1_query_local_bytes(r, q, fused_hybrid, query_local_main_mask),
            "bounded Phase-1 query bytes");
        value.total_bytes = checked_add_size(
            value.total_bytes,
            checked_mul_size(size_t(q), generated_knn_bytes_per_query,
                "generated exact-kNN query bytes"),
            "bounded Phase-1 plus exact-kNN query bytes");
        return value;
    };
    // At q==anchors only one slot is needed. Test that endpoint separately:
    // otherwise the 2->1 slot discontinuity would invalidate binary search.
    int lo = 1, hi = b.upper_bound, best = 0;
    if (hi > 0 && fixed <= b.usable_bytes &&
        query(hi).total_bytes <= b.usable_bytes - fixed &&
        (!streamed || p1_staging_bytes(r, hi) <= o.pinned_bytes)) {
        best = hi; lo = hi + 1;
    }
    while (lo <= hi) {
        const int q = lo + (hi - lo) / 2;
        const auto space = query(q);
        if (fixed <= b.usable_bytes && space.total_bytes <= b.usable_bytes - fixed &&
            (!streamed || p1_staging_bytes(r, q) <= o.pinned_bytes)) {
            best = q; lo = q + 1;
        } else hi = q - 1;
    }
    if (!best) return result;
    b.BQ = best;
    b.selected_query = query(best);
    b.required_bytes = fixed + b.selected_query.total_bytes;
    if (best < b.upper_bound) b.next_required_bytes = fixed + query(best + 1).total_bytes;
    return result;
}
