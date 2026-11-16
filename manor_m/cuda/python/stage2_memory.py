"""Capacity selection for exact P1/kNN and P2 paths.

P1 delegates its CUB workspace query to the native capacity extension; old
libraries retain the conservative resident-only check. P2's required count
mirrors the shared core's current allocation contract.
Device free memory already excludes resident inputs; counting them again is
deliberately conservative and matches the native preallocation check. Sparse
pair storage and an explicit allocation margin are additional, not forgotten.

Production selection is capacity arithmetic only. Throughput options come from
the selected build configuration; this module never probes kernels, warms up
candidate layouts, or times trial cells.
"""
from __future__ import annotations

from dataclasses import dataclass


GIB = 1024 ** 3
I32_MAX = 2 ** 31 - 1
U64_MAX = 2 ** 64 - 1
# Ceiling for variable exact-kNN scratch. Larger neighbor counts leave room
# for fewer rows; a complete request remains unchunked when it fits.
DEFAULT_GENERATED_KNN_VARIABLE_BYTES = 55_821_991_936


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def _ceil_quotient_capacity_breakpoints(total, limit):
    """Yield interval endpoints where ceil(total / capacity) can change."""
    numerator = total - 1
    first = 1
    while first <= limit:
        quotient = numerator // first
        last = limit if quotient == 0 else min(limit, numerator // quotient)
        yield first
        if last != first:
            yield last
        first = last + 1


def _balanced_generated_capacities(*, points, owners, owner_limit,
                                   query_limit, variable_bytes, query_bytes,
                                   owner_bytes):
    """Choose exact, deterministic tile capacities without timing kernels.

    Generated Phase 2 traverses the exact kNN tiles twice per iteration and
    traverses the owner aggregation tiles once.  Among every capacity pair
    that fits the measured byte budget, minimize that predicted tile-launch
    count.  The objective is piecewise constant; checking the endpoints where
    either ceil quotient changes is exact without scanning up to N rows.
    """
    values = (points, owners, owner_limit, query_limit, variable_bytes,
              query_bytes, owner_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in values):
        raise ValueError("positive integer generated-tile terms required")
    owner_limit = min(owners, owner_limit)
    query_limit = min(points, query_limit)
    maximum_query_rows = min(points, query_limit,
        (variable_bytes - owner_bytes) // query_bytes)
    if maximum_query_rows < 1:
        raise MemoryError(
            "No generated Phase2 layout fits one query and one owner")

    candidates = {1, maximum_query_rows}
    candidates.update(_ceil_quotient_capacity_breakpoints(
        points, maximum_query_rows))
    for owner_capacity in _ceil_quotient_capacity_breakpoints(
            owners, owner_limit):
        boundary = (variable_bytes - owner_bytes * owner_capacity) // query_bytes
        for query_rows in (boundary - 1, boundary, boundary + 1):
            if 1 <= query_rows <= maximum_query_rows:
                candidates.add(query_rows)

    best = None
    for query_rows in candidates:
        owner_capacity = min(owner_limit,
            (variable_bytes - query_bytes * query_rows) // owner_bytes)
        if owner_capacity < 1:
            continue
        query_tiles = _ceil_div(points, query_rows)
        owner_tiles = _ceil_div(owners, owner_capacity)
        predicted_launches = 2 * query_tiles + owner_tiles
        # A stable tie-break favors fewer kNN tiles first because each exact
        # response-kNN tile is visited twice, then larger useful blocks.
        key = (predicted_launches, query_tiles, owner_tiles,
               -query_rows, -owner_capacity)
        if best is None or key < best[0]:
            best = (key, query_rows, owner_capacity, query_tiles, owner_tiles)
    if best is None:
        raise MemoryError(
            "No generated Phase2 layout fits one query and one owner")
    _, query_rows, owner_capacity, query_tiles, owner_tiles = best
    return dict(query_rows=query_rows, owner_capacity=owner_capacity,
                query_tiles=query_tiles, owner_tiles=owner_tiles,
                predicted_launches=2 * query_tiles + owner_tiles)


def phase2_storage(n, d, my, mm, k, owners, iterations, queries, rho_count,
                   h_count, response=True, tangent=False, aggregation=0, index=0,
                   generated_index=False, knn_scratch_bytes=0, evaluate=True,
                   validation_queries=0, validation_tangent=False):
    dimensions = (n, d, my, mm, k, owners, iterations)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > I32_MAX
           for v in dimensions):
        raise ValueError("positive int32 dimensions/iterations required")
    if not isinstance(evaluate, bool):
        raise ValueError("evaluate must be boolean")
    evaluation_shape = (queries, rho_count, h_count)
    if evaluate:
        if any(isinstance(v, bool) or not isinstance(v, int)
               or v < 1 or v > I32_MAX for v in evaluation_shape):
            raise ValueError("enabled evaluation requires positive int32 dimensions")
    elif (evaluation_shape != (0, 0, 0) or response or tangent):
        raise ValueError(
            "disabled evaluation requires zero query/grid dimensions and no truth")
    if (isinstance(validation_queries, bool) or not isinstance(validation_queries, int)
            or not 0 <= validation_queries <= I32_MAX or not isinstance(validation_tangent, bool)
            or (validation_tangent and not validation_queries)
            or (validation_queries and not evaluate)):
        raise ValueError("invalid validation evaluation dimensions")
    if owners > n or k > n:
        raise ValueError("invalid owner/neighbor count")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= I32_MAX
           for v in (aggregation, index)):
        raise ValueError("invalid tile capacity")
    if not isinstance(generated_index, bool):
        raise ValueError("generated_index must be boolean")
    if (isinstance(knn_scratch_bytes, bool) or
            not isinstance(knn_scratch_bytes, int) or knn_scratch_bytes < 0):
        raise ValueError("knn_scratch_bytes must be a nonnegative integer")
    if generated_index and (index or not knn_scratch_bytes):
        raise ValueError(
            "generated exact indices require positive scratch and no index tile")
    if not generated_index and knn_scratch_bytes:
        raise ValueError("kNN scratch is only part of generated Phase2")
    acap = min(owners, aggregation) if aggregation else owners
    icap = min(owners, index) if index else owners
    buffers = min(2, (owners + icap - 1) // icap) if index else 0
    index_bytes = 0 if generated_index else 4 * k * icap * (buffers if index else 1)
    slots = rho_count * h_count if evaluate else 0
    maximum_queries = max(queries, validation_queries)
    components = dict(
        fixed_query_scratch=4 * slots * (my + 4) * maximum_queries,
        training=4 * n * (d + my), index=index_bytes,
        generated_knn_scratch=knn_scratch_bytes,
        generated_response_soa=4 * n * my if generated_index else 0,
        generated_owner_row_map=4 * owners if generated_index else 0,
        two_mutable_states=8 * n * d * (mm + 1),
        gradients=4 * owners * my * d, center_pre=4 * owners * d,
        # A bounded aggregation tile also bounds the exact neighbor-mask work
        # buffer. Regression keeps global gradients/counts, the heavy/keep
        # decision remains global, and membership bits are then rebuilt and
        # consumed once per owner tile. A resident request keeps the original
        # full mask and pays no reconstruction pass.
        neighbor_mask=4 * acap * ((k + 31) // 32),
        owner_counts_weights_ids=16 * owners, owner_lookup=4 * n,
        solver_status=owners, owner_heavy_keep=5 * n,
        aggregation=acap * (4 + 4 * d + 4 * d * d), scalars=40,
        # Production calls do not request full iteration histories.  The
        # native core therefore allocates no device history storage.  Debug
        # callers that request histories stream one N-row staging slice to
        # host after each global iteration instead of retaining 6*N*K bytes.
        device_histories=0,
        query_truth=(4 * queries * (d + (my if response else 0)
                                    + (mm * d if tangent else 0)) if evaluate else 0),
        eval_scales=4 * (rho_count + h_count) if evaluate else 0,
        eval_train_stats=4 * n * (mm + 1) if evaluate else 0,
        validation_truth=4 * validation_queries * (d + my + (mm * d if validation_tangent else 0)),
        validation_histories=4 * slots * iterations * (1 + my + int(validation_tangent)) if validation_queries else 0,
        eval_sums=4 * slots * (4 + my) if evaluate else 0,
        eval_histories=(4 * slots * iterations * (4 + my)
                        if evaluate else 0))
    pairs = maximum_queries * slots if evaluate else 0
    sparse = (4 + 4 * max(2 * pairs, pairs + 2 * maximum_queries + 1) + 8 * pairs
              if evaluate else 0)
    return dict(core_required_bytes=sum(components.values()), sparse_bytes=sparse,
                index_device_bytes=index_bytes, pinned_bytes=index_bytes if index else 0,
                aggregation_bytes=components['aggregation'], components=components)


@dataclass(frozen=True)
class MemoryPolicy:
    working_bytes: int = 0  # optional ceiling for this call, excluding retained inputs
    pinned_bytes: int = 2 * GIB
    # Tile ceilings are combined with the available memory budget.
    # A complete request remains unchunked when it fits.
    preferred_tile_owners: int = I32_MAX
    preferred_generated_knn_variable_bytes: int = (
        DEFAULT_GENERATED_KNN_VARIABLE_BYTES)
    reserve_bytes: int = 2_000_000_000
    reserve_fraction: float = .02
    memory_fraction: float = .88
    allocation_margin_bytes: int = 64 * 1024 ** 2
    phase1_working_bytes: int = 0
    phase1_index_heap_bytes: int = 8 * GIB
    # Zero selects the largest exact-kNN tile that fits available memory.
    # A positive value places an additional scratch ceiling.
    phase1_knn_scratch_bytes: int = 0
    phase1_spill_directory: str | None = None
    # 0 automatic; 1 resident; 2 streamed/global mask; 3 streamed/local mask;
    # 4 exact generated kNN tiles with no complete host/device N*k index.
    phase1_mode: int = 0
    phase1_maximum_query_block: int = 0  # 0 uses the native launch/grid limit

    def __post_init__(self):
        for name in ('working_bytes', 'pinned_bytes', 'reserve_bytes', 'allocation_margin_bytes',
                     'phase1_working_bytes', 'phase1_index_heap_bytes',
                     'phase1_knn_scratch_bytes', 'phase1_maximum_query_block'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int) or
                    value < 0 or value > U64_MAX):
                raise ValueError(name + ' must be a nonnegative integer')
        if (not isinstance(self.preferred_tile_owners, int) or isinstance(self.preferred_tile_owners, bool)
                or not 1 <= self.preferred_tile_owners <= I32_MAX
                or not isinstance(self.preferred_generated_knn_variable_bytes, int)
                or isinstance(self.preferred_generated_knn_variable_bytes, bool)
                or not 1 <= self.preferred_generated_knn_variable_bytes <= U64_MAX
                or not 0 < self.memory_fraction < 1 or not 0 <= self.reserve_fraction < 1):
            raise ValueError('invalid memory fraction/tile preference')
        if (type(self.phase1_mode) is not int or self.phase1_mode not in (0, 1, 2, 3, 4)
                or self.phase1_maximum_query_block > I32_MAX):
            raise ValueError('invalid P1 mode/query limit')

    def phase1_options(self, retained_limit):
        return dict(working_bytes=self.phase1_working_bytes, pinned_bytes=self.pinned_bytes,
            retained_limit_bytes=int(retained_limit),
            knn_scratch_bytes=(self.phase1_knn_scratch_bytes or U64_MAX),
            reserve_bytes=self.reserve_bytes, memory_fraction=self.memory_fraction,
            reserve_fraction=max(.08, self.reserve_fraction), mode=self.phase1_mode,
            maximum_query_block=self.phase1_maximum_query_block)

    def usable(self, free, total):
        if not 0 <= free <= total:
            raise ValueError('invalid device memory observation')
        usable = int(max(0, free - max(self.reserve_bytes, total * self.reserve_fraction))
                     * self.memory_fraction)
        return min(usable, self.working_bytes) if self.working_bytes else usable

    def choose(self, *, free, total, reclaim_index_bytes=0, force_index=False,
               aggregation_supported=True, index_supported=True,
               generated_index=False, knn_scratch_bytes=0,
               knn_fixed_bytes=0, knn_bytes_per_query=0, **shape):
        """Choose once from frozen preferences and memory sizes, without trial runs."""
        budget = self.usable(free, total)

        def candidate(mode, aggregation=0, index=0, released=0, scratch=None):
            selected_scratch = (knn_scratch_bytes if scratch is None
                                else scratch)
            storage = phase2_storage(**shape, aggregation=aggregation, index=index,
                generated_index=generated_index,
                knn_scratch_bytes=selected_scratch)
            ceiling = self.usable(min(total, free + released), total)
            required = storage['core_required_bytes'] + storage['sparse_bytes'] + self.allocation_margin_bytes
            return dict(mode=mode, aggregation_tile_owners=aggregation, index_tile_owners=index,
                knn_scratch_bytes=selected_scratch,
                estimated_call_bytes=required, usable_call_bytes=ceiling,
                observed_free_bytes=free, observed_total_bytes=total,
                reclaimed_index_bytes=released, allocation_margin_bytes=self.allocation_margin_bytes,
                **storage)

        def fits(plan):
            return (plan['estimated_call_bytes'] <= plan['usable_call_bytes']
                    and plan['pinned_bytes'] <= self.pinned_bytes)

        if generated_index and force_index:
            raise ValueError("generated Phase2 cannot force host-index streaming")

        # Generated exact-kNN has two independent variable working sets: its
        # query tile and the owner-aggregation tile.  Keep full aggregation if
        # it can coexist with at least one query row.  If both must be tiled,
        # choose their capacities together by minimizing the deterministic
        # launch-count model (two generated-kNN passes plus one aggregation
        # pass).  Production never warms up or times candidate tile sizes.
        if generated_index and not knn_scratch_bytes:
            values = (knn_fixed_bytes, knn_bytes_per_query)
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in values):
                raise ValueError(
                    "automatic generated Phase2 requires positive native "
                    "kNN fixed/per-query byte terms")
            if (knn_fixed_bytes > U64_MAX - knn_bytes_per_query or
                    shape['n'] > (U64_MAX - knn_fixed_bytes) // knn_bytes_per_query):
                raise MemoryError("generated exact-kNN scratch arithmetic overflow")
            minimum_scratch = knn_fixed_bytes + knn_bytes_per_query
            full_scratch = knn_fixed_bytes + shape['n'] * knn_bytes_per_query
            offline_query_limit = max(1, min(
                shape['n'], self.preferred_generated_knn_variable_bytes
                // knn_bytes_per_query))
            full = candidate('generated', scratch=full_scratch)
            if fits(full):
                return dict(full, knn_fixed_bytes=knn_fixed_bytes,
                            knn_bytes_per_query=knn_bytes_per_query,
                            knn_query_rows=shape['n'], knn_query_tiles=1,
                            knn_query_rows_capacity_only=shape['n'],
                            knn_query_rows_offline_ceiling=offline_query_limit,
                            knn_variable_scratch_ceiling_bytes=(
                                self.preferred_generated_knn_variable_bytes),
                            offline_knn_ceiling_applied=False,
                            aggregation_tiles_per_iteration=1,
                            predicted_tile_launches_per_iteration=3,
                            tile_selection_policy=(
                                'capacity_then_offline_ceiling_then_'
                                'launch_count_v2'))

            aggregation = 0
            minimum = candidate('generated_knn_tiled', scratch=minimum_scratch)
            if fits(minimum):
                extra = (minimum['usable_call_bytes'] -
                         minimum['estimated_call_bytes'])
                capacity_rows = min(
                    shape['n'], 1 + extra // knn_bytes_per_query)
                rows = min(capacity_rows, offline_query_limit)
                owner_tiles = 1
                predicted_launches = 2 * _ceil_div(shape['n'], rows) + 1
            else:
                if not aggregation_supported:
                    raise MemoryError(
                        'Generated Phase2 cannot fit one exact-kNN row with '
                        'resident aggregation scratch')
                owner_bytes = (4 * ((shape['k'] + 31) // 32) + 4 +
                               4 * shape['d'] + 4 * shape['d'] * shape['d'])
                one = candidate('generated_knn_and_aggregation_tiled',
                                aggregation=1, scratch=minimum_scratch)
                if not fits(one):
                    raise MemoryError(
                        'No generated Phase2 layout fits even one exact-kNN '
                        'query and one aggregation owner')
                fixed_required = (one['estimated_call_bytes'] -
                                  knn_bytes_per_query - owner_bytes)
                variable_bytes = one['usable_call_bytes'] - fixed_required
                capacity_rows = min(
                    shape['n'],
                    (variable_bytes - owner_bytes) // knn_bytes_per_query)
                balanced = _balanced_generated_capacities(
                    points=shape['n'], owners=shape['owners'],
                    owner_limit=min(shape['owners'],
                                    self.preferred_tile_owners),
                    query_limit=min(capacity_rows, offline_query_limit),
                    variable_bytes=variable_bytes,
                    query_bytes=knn_bytes_per_query,
                    owner_bytes=owner_bytes)
                rows = balanced['query_rows']
                aggregation = balanced['owner_capacity']
                owner_tiles = balanced['owner_tiles']
                predicted_launches = balanced['predicted_launches']
            scratch = knn_fixed_bytes + rows * knn_bytes_per_query
            if aggregation:
                mode = ('generated_aggregation_tiled' if rows == shape['n']
                        else 'generated_knn_and_aggregation_tiled')
            else:
                mode = 'generated_knn_tiled'
            selected = candidate(mode, aggregation=aggregation, scratch=scratch)
            if not fits(selected):
                raise RuntimeError(
                    'automatic generated Phase2 planner exceeded its byte budget')
            return dict(selected, knn_fixed_bytes=knn_fixed_bytes,
                        knn_bytes_per_query=knn_bytes_per_query,
                        knn_query_rows=rows,
                        knn_query_tiles=_ceil_div(shape['n'], rows),
                        knn_query_rows_capacity_only=capacity_rows,
                        knn_query_rows_offline_ceiling=offline_query_limit,
                        knn_variable_scratch_ceiling_bytes=(
                            self.preferred_generated_knn_variable_bytes),
                        offline_knn_ceiling_applied=(rows < capacity_rows),
                        aggregation_tiles_per_iteration=owner_tiles,
                        predicted_tile_launches_per_iteration=predicted_launches,
                        tile_selection_policy=(
                            'capacity_then_offline_ceiling_then_'
                            'launch_count_v2'))

        resident = candidate('generated' if generated_index else 'resident')
        if not force_index and fits(resident):
            return resident
        # Required bytes are monotone in the owner capacity.  Find the exact
        # largest fitting tile with integer arithmetic; this is a capacity
        # calculation, not a production-time throughput trial.  Larger tiles
        # minimize launches/copies while retaining the configured safety
        # reserve and optional offline-qualified ceiling.
        ceiling = min(shape['owners'], self.preferred_tile_owners)

        def largest_fitting(mode, *, index):
            low, high = 1, ceiling
            best = None
            while low <= high:
                cap = low + (high - low) // 2
                plan = candidate(mode, aggregation=cap,
                    index=cap if index else 0,
                    released=reclaim_index_bytes if index else 0)
                if fits(plan):
                    best = plan
                    low = cap + 1
                else:
                    high = cap - 1
            return best

        if not force_index and aggregation_supported:
            plan = largest_fitting(
                'generated_aggregation_tiled' if generated_index
                else 'aggregation_tiled', index=False)
            if plan is not None:
                return plan
        if not generated_index and index_supported and aggregation_supported:
            plan = largest_fitting('index_and_aggregation_tiled', index=True)
            if plan is not None:
                return plan
        raise MemoryError('No supported exact P2 layout fits: resident=%d, usable=%d bytes. '
                          'Index staging cannot remove global model state/history.'
                          % (resident['estimated_call_bytes'], budget))

    def phase1_check(self, *, n, d, my, k, free, total, retained, retained_limit):
        """Reject an impossible full index BEFORE NumPy/CUDA kNN allocation.

        This is a necessary-capacity check, not a replacement for the native
        P1 workspace planner. Full-index P1 is still a large-N limitation.
        """
        if not 1 <= k <= n or n > I32_MAX:
            raise ValueError('Phase1 dimensions outside the exact index API')
        index = 4 * n * k
        state = 4 * n * d * 3 + n
        # Worst-case compact owner-index copy (M<N) may coexist with full index.
        retained_bound = retained + 2 * index + state
        required = index + 7 * GIB + 4 * n * my + state + self.allocation_margin_bytes
        usable = int(max(0, free - max(self.reserve_bytes, total * .08)) * .88)
        if retained_bound > retained_limit or required > usable:
            raise MemoryError('Full-index Phase1 cannot fit this request before kNN allocation: '
                              'index=%d, P1 floor+scratch=%d/%d, retained bound=%d/%d bytes; '
                              'P2 chunking does not make Phase1 out-of-core.'
                              % (index, required, usable, retained_bound, retained_limit))
        return dict(full_index_bytes=index, floor_plus_knn_scratch_bytes=required,
                    usable_bytes=usable, retained_upper_bound_bytes=retained_bound,
                    full_pipeline_out_of_core=False)
