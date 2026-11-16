"""Direct Python estimator over the shared CUDA C ABI (no MATLAB/PyTorch).

One instance owns one N and one exact-k initial model. Query changes retain
training/P1 but replace their own resident group. Every P2 call still resets
mutable state from immutable P1 via D2D; no cell inherits another cell's fit.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import time

import numpy as np

from manor_m_cuda import ManorMCuda
from resident_sweep import SweepSession, _snapshot
from phase1 import _cuts, _hybrid_phase1, _phase2_schedules
from stage2_memory import MemoryPolicy


P1_NAMES = ("Tan0", "Cen0", "Rad0", "owner_mask0", "num_neigh0",
            "selected_cut0", "axial_tan0")


def _checkpoint_array(value, dtype, shape, name):
    """Return a zero-copy immutable checkpoint view with an exact layout."""
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or array.shape != shape:
        raise ValueError("restored %s shape/dtype mismatch" % name)
    if not array.flags.c_contiguous:
        raise ValueError("restored %s must be C contiguous" % name)
    if array.flags.writeable:
        raise ValueError("restored %s must be an immutable checkpoint array" % name)
    return array


def _all_finite_rows(array, mask=None, block_rows=131072):
    """Validate large checkpoint arrays without a full-sized bool temporary."""
    for start in range(0, len(array), block_rows):
        stop = min(len(array), start + block_rows)
        values = array[start:stop]
        if mask is not None:
            values = values[mask[start:stop]]
        if values.size and not np.isfinite(values).all():
            return False
    return True


def _builder(context, x, index, config, *, response=None):
    return _hybrid_phase1(context, x, index,
        np.asarray(config["counts"], dtype=np.int32),
        np.asarray(config["cuts"], dtype=np.int32), config["MM"],
        config["reserve_bytes"], capacity_options=config.get('capacity_options'),
        response=response)[0]


class CudaCoreEstimator:
    def __init__(self, library, x, y, *, dataset_id, device=0,
                 retained_bytes=None, core=None, memory_policy=None,
                 phase1_cache_bytes=1024**3):
        self.core = core if core is not None else ManorMCuda(library)
        self.memory_policy = memory_policy if memory_policy is not None else MemoryPolicy()
        self.device = int(device)
        if retained_bytes is None or retained_bytes == 0:
            info = self.core.device_info(self.device)
            reserve = max(self.memory_policy.reserve_bytes, info.total_global_memory_bytes * .08)
            retained_bytes = int(max(0, info.free_global_memory_bytes - reserve) * .88)
            if retained_bytes <= 0:
                raise MemoryError('No free GPU memory remains after the P1 reserve')
        self.session = SweepSession(self.core, device_ordinal=device,
                                    maximum_retained_bytes=retained_bytes,
                                    maximum_phase1_host_bytes=phase1_cache_bytes)
        self.manager = self.session.sweep(dataset_id, data=dict(x=x, response=y), reuse=False)
        self.sweep = self.manager.__enter__()
        self.k = None
        self.p1 = self.index = None
        self.query = {}
        self.query_key = None
        self.p1_hashes = {}
        self.active_state_hashes = {}
        self.phase2_iteration = None
        self.preparation_count = 0
        self.phase1_restore_count = 0
        self.phase1_active_fast_path_count = 0
        self.index_streamed = False
        self.index_compact = False
        self.index_generated = False
        self.phase1_memory = None
        self.closed = False
        self.builder_module_hash = hashlib.sha256(
            Path(__file__).with_name("phase1.py").read_bytes()).hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        if not self.closed:
            try:
                self.manager.__exit__(kind, error, tb)
            finally:
                try:
                    self.session.close()
                finally:
                    # A profiling callback may keep the closed estimator alive.
                    # Its N*k mmap must still be released at the fit boundary.
                    self.index = self.p1 = None
                    self.query = {}
                    self.closed = True

    def _phase1_configuration(self, k):
        d = self.sweep.data["x"].shape[1]
        return dict(MM=2, counts=[int(k)], cuts=_cuts(d, int(k)).tolist(),
            reserve_bytes=2_000_000_000,
            builder_module_sha256=self.builder_module_hash)

    def prepare(self, k):
        k = int(k)
        if self.k == k and self.p1 is not None:
            # The Stage2 cell order keeps exact k active for 27 consecutive
            # cells.  Avoid rebuilding the configuration/key (including
            # source inspection and full data-hash JSON) on those calls.  The
            # sweep ownership/generation check remains mandatory, and the
            # active session objects must still be exactly the ones held here.
            self.sweep._check()
            session = self.sweep.session
            if session.phase1 is None or session.index is not self.index:
                raise RuntimeError("active Phase1 estimator/session state diverged")
            self.phase1_active_fast_path_count += 1
            return dict(reused=True, wall_seconds=0.0, source="active",
                        index_rebuilt=False)
        config = self._phase1_configuration(k)
        if self.k is not None and self.k != k:
            # Do not keep an unnecessary second N*k host index during kNN.
            self.index = self.p1 = None
        self.index, self.p1, preparation = self.sweep.prepare_phase1(
            k, configuration=config, builder=_builder, preflight=self._phase1_preflight)
        if preparation["source"] != "active":
            self.index_streamed = self.session.index_streamed
            self.index_compact = self.session.index_compact
            self.index_generated = self.session.index_generated
            if preparation["reused"]:
                self.phase1_restore_count += 1
            else:
                self.preparation_count += 1
            self.p1_hashes = {name: hashlib.sha256(np.ascontiguousarray(self.p1[name])).hexdigest()
                              for name in P1_NAMES}
            self.p1_wall = preparation["source_wall_seconds"]
        self.k = k
        self.phase2_iteration = 0
        self.active_state_hashes = dict(self.p1_hashes)
        return preparation

    def restore_generated_phase1(self, k, phase1, *, state_sha256,
                                 initial_state_sha256=None, state_iteration=0,
                                 source_wall_seconds=0.0):
        """Install a verified generated-index P1 snapshot without recomputing P1.

        This is deliberately limited to the exact generated-kNN capacity mode:
        no absent host index is inferred for resident/streamed modes.  The
        caller must have verified the checkpoint hashes and identity before
        this method is entered.  Tan/Cen/owner are registered once on the
        current device; later Phase2 calls reset from them by D2D.
        """
        k = int(k)
        if self.p1 is not None or self.k is not None:
            raise RuntimeError("Phase1 is already active")
        if not np.isfinite(source_wall_seconds) or source_wall_seconds < 0:
            raise ValueError("invalid restored Phase1 source wall time")
        if set(P1_NAMES) - set(phase1):
            raise ValueError("restored Phase1 omitted required arrays")
        if set(P1_NAMES) - set(state_sha256):
            raise ValueError("restored Phase1 omitted array hashes")
        if initial_state_sha256 is None:
            initial_state_sha256 = state_sha256
        if set(P1_NAMES) - set(initial_state_sha256):
            raise ValueError("restored Phase1 omitted initial array hashes")
        if (isinstance(state_iteration, bool) or
                not isinstance(state_iteration, (int, np.integer)) or
                state_iteration < 0):
            raise ValueError("restored Phase2 iteration must be nonnegative")

        data = self.sweep.data
        n, d = data["x"].shape
        layouts = {
            "Tan0": (np.float32, (n, 2, d)),
            "Cen0": (np.float32, (n, d)),
            "Rad0": (np.float32, (n,)),
            "owner_mask0": (np.uint8, (n,)),
            "num_neigh0": (np.int32, (n,)),
            "selected_cut0": (np.uint32, (n,)),
            "axial_tan0": (np.float32, (n,)),
        }
        values = {
            name: _checkpoint_array(phase1[name], dtype, shape, name)
            for name, (dtype, shape) in layouts.items()
        }
        owner = values["owner_mask0"] != 0
        if (not np.array_equal(values["owner_mask0"], owner.astype(np.uint8))
                or not owner.any()
                or not _all_finite_rows(values["Tan0"])
                or not _all_finite_rows(values["Cen0"])
                or not _all_finite_rows(values["Rad0"], owner)
                or not _all_finite_rows(values["axial_tan0"], owner)):
            raise ValueError("restored Phase1 state is nonfinite or ownerless")

        execution = self._phase1_preflight(
            self.sweep.context, data, k)
        capacity = execution.get("capacity_options", {})
        if capacity.get("mode") != 4:
            raise ValueError(
                "disk Phase1 restore is allowed only for exact generated-kNN mode")

        session = self.sweep.session
        session.context._resident_clear("phase1")
        try:
            session.context._resident_put("phase1.Tan0", values["Tan0"])
            session.context._resident_put("phase1.Cen0", values["Cen0"])
            session.context._resident_put("phase1.owner", values["owner_mask0"])
        except BaseException:
            session.context._resident_clear("phase1")
            raise
        session.phase1 = values
        session.phase1_key = "restored-generated:%s:%d" % (
            session.identity, k)
        session.index = None
        session.index_streamed = False
        session.index_compact = False
        session.index_generated = True
        session.generated_neighbor_count = k
        session.generated_knn_scratch_bytes = int(
            capacity["knn_scratch_bytes"])
        session.compact_knn_stats = None

        self.k = k
        self.p1 = values
        self.index = None
        self.index_streamed = False
        self.index_compact = False
        self.index_generated = True
        self.p1_hashes = {
            name: str(initial_state_sha256[name]) for name in P1_NAMES}
        self.active_state_hashes = {
            name: str(state_sha256[name]) for name in P1_NAMES}
        self.phase2_iteration = int(state_iteration)
        self.p1_wall = float(source_wall_seconds)
        self.phase1_restore_count += 1
        return dict(reused=True, wall_seconds=0.0, source="disk_checkpoint",
                    index_rebuilt=False, index_generated=True,
                    generated_neighbor_count=k,
                    generated_knn_scratch_bytes=(
                        session.generated_knn_scratch_bytes))

    def _phase1_preflight(self, context, data, k):
        if hasattr(self.core._library, 'manor_m_cuda_plan_phase1'):
            d = data['x'].shape[1]
            options = self.memory_policy.phase1_options(self.session.maximum_bytes)
            has_fused_execution = hasattr(
                self.core._library, 'manor_m_cuda_phase1_hybrid_precomputed')
            has_fused_plan = hasattr(
                self.core._library, 'manor_m_cuda_plan_phase1_hybrid')
            if has_fused_execution and not has_fused_plan:
                raise RuntimeError(
                    'Fused hybrid Phase1 requires its exact capacity planner')
            plan = context.plan_phase1(data['x'], data['response'], k, [k], _cuts(d, k), 2,
                                      capacity_options=options,
                                      hybrid=has_fused_plan)
            if plan['mode'] == 4:
                required = ('manor_m_cuda_phase2_response_tiled',)
                missing = [name for name in required
                           if not hasattr(self.core._library, name)]
                if missing:
                    raise RuntimeError(
                        'Generated Phase1 requires direct generated-index Phase2 support: '
                        + ', '.join(missing))
            options = dict(options, mode=plan['mode'],
                           maximum_query_block=plan['query_block'],
                           knn_scratch_bytes=plan['knn_scratch_bytes'])
            mode_names = {1: 'resident', 2: 'index_streamed',
                          3: 'query_local_mask', 4: 'generated_exact_tiles'}
            self.phase1_memory = dict(plan, mode_name=mode_names[plan['mode']],
                                     full_pipeline_out_of_core=plan['mode'] == 4)
            return dict(capacity_options=options, knn_scratch_bytes=plan['knn_scratch_bytes'],
                heap_bytes=self.memory_policy.phase1_index_heap_bytes,
                spill_directory=self.memory_policy.phase1_spill_directory)
        if (self.memory_policy.phase1_working_bytes or self.memory_policy.phase1_mode
                or self.memory_policy.phase1_maximum_query_block):
            raise RuntimeError('This core lacks the requested P1 capacity extension')
        info = self.core.device_info(self.device)
        n, d = data['x'].shape
        self.phase1_memory = self.memory_policy.phase1_check(n=n, d=d,
            my=data['response'].shape[1], k=k, free=info.free_global_memory_bytes,
            total=info.total_global_memory_bytes,
            retained=context.resident_stats()['retained_bytes'], retained_limit=self.session.maximum_bytes)

    def _memory_plan(self, owner_count, iterations, rho_count, h_count,
                     *, evaluation=True):
        n, d = self.sweep.data['x'].shape
        if not isinstance(evaluation, bool):
            raise ValueError("evaluation must be boolean")
        if evaluation and not self.query:
            raise RuntimeError("evaluation memory plan requires a query snapshot")
        shape = dict(n=n, d=d, my=self.sweep.data['response'].shape[1], mm=2, k=self.k,
            owners=owner_count, iterations=iterations,
            queries=len(self.query['test_x']) if evaluation else 0,
            rho_count=rho_count if evaluation else 0,
            h_count=h_count if evaluation else 0,
            response=evaluation and 'test_y' in self.query,
            tangent=evaluation and 'test_tangent' in self.query,
            evaluate=evaluation,
            validation_queries=(len(self.query['validation_x'])
                if evaluation and 'validation_x' in self.query else 0),
            validation_tangent=evaluation and 'validation_tangent' in self.query)
        info = self.core.device_info(self.device)
        lib = self.core._library
        if self.index_generated:
            cached_index = 0
        elif self.index_streamed:
            cached_index = 0
        elif self.index_compact:
            cached_index = 4 * self.k * owner_count
        else:
            cached_index = 4 * self.k * (n + (owner_count if owner_count < n else 0))
        selected_knn_scratch = (self.session.generated_knn_scratch_bytes
                                if self.index_generated else 0)
        knn_memory = {}
        if (self.index_generated and
                hasattr(lib, 'manor_m_cuda_response_knn_memory') and
                callable(getattr(self.core, 'response_knn_memory', None))):
            knn_memory = self.core.response_knn_memory(
                n, self.sweep.data['response'].shape[1], self.k,
                tiled_output=True)
            # Zero asks MemoryPolicy to derive the largest safe whole-query
            # tile from current free memory and the core's exact byte terms.
            selected_knn_scratch = 0
        plan = self.memory_policy.choose(free=info.free_global_memory_bytes,
            total=info.total_global_memory_bytes, reclaim_index_bytes=cached_index,
            force_index=self.index_streamed,
            aggregation_supported=hasattr(lib, 'manor_m_cuda_phase2_tiled'),
            index_supported=hasattr(lib, 'manor_m_cuda_phase2_compact_index_tiled'
                if self.index_compact else 'manor_m_cuda_phase2_index_tiled'),
            generated_index=self.index_generated,
            knn_scratch_bytes=selected_knn_scratch,
            knn_fixed_bytes=knn_memory.get('fixed_bytes', 0),
            knn_bytes_per_query=knn_memory.get('bytes_per_query', 0),
            **shape)
        if plan['index_tile_owners'] and not self.index_streamed:
            before = self.sweep.stats()['retained_bytes']
            self.sweep.context._resident_clear('phase1.response_index')
            self.sweep.context._resident_clear('phase1.compact_index')
            self.sweep.context._resident_clear('phase1.compact_index_source')
            released = before - self.sweep.stats()['retained_bytes']
            if released != cached_index:
                raise RuntimeError('resident full/compact index release differs from memory plan')
            self.index_streamed = True
            info = self.core.device_info(self.device)
            plan = self.memory_policy.choose(free=info.free_global_memory_bytes,
                total=info.total_global_memory_bytes, force_index=True, **shape)
            plan['reclaimed_index_bytes'] = released
        return plan

    def advance_phase2(self, tangent, center, *, start_iteration,
                       stop_iteration, h_scale, rho_init, bite):
        """Advance one exact generated-index Phase2 segment without metrics.

        The supplied state is registered once, then copied D2D into the
        mutable buffers.  Schedules are sliced from the same global schedule
        used by an uninterrupted call, so a checkpoint boundary cannot reset
        either h or rho.  Evaluation is intentionally absent; final metrics
        are handled after the requested state has been reached.
        """
        if self.p1 is None or self.k is None:
            raise RuntimeError("prepare or restore Phase1 before advancing Phase2")
        if not self.index_generated or self.index is not None:
            raise RuntimeError(
                "checkpointed Phase2 currently requires exact generated-kNN mode")
        start_iteration = int(start_iteration)
        stop_iteration = int(stop_iteration)
        if start_iteration < 0 or stop_iteration <= start_iteration:
            raise ValueError("Phase2 segment requires 0 <= start < stop")
        if self.phase2_iteration != start_iteration:
            raise ValueError(
                "Phase2 segment start differs from restored checkpoint iteration")

        n, d = self.sweep.data["x"].shape
        tangent = _checkpoint_array(
            tangent, np.float32, (n, 2, d), "Tan_next")
        center = _checkpoint_array(
            center, np.float32, (n, d), "Cen_next")
        owner = self.p1["owner_mask0"]
        owner_count = int(np.count_nonzero(owner))
        if owner_count < 1:
            raise RuntimeError("checkpointed Phase1 has no owners")

        phase1_h, _, full_h, full_rho = _phase2_schedules(
            self.p1["Rad0"], owner.astype(bool), self.p1["axial_tan0"],
            float(h_scale), float(rho_init), int(bite), d, 2,
            stop_iteration)
        h_segment = np.ascontiguousarray(
            full_h[start_iteration:stop_iteration], dtype=np.float32)
        rho_segment = np.ascontiguousarray(
            full_rho[start_iteration:stop_iteration], dtype=np.float32)
        segment_iterations = stop_iteration - start_iteration
        plan = self._memory_plan(
            owner_count, segment_iterations, 0, 0, evaluation=False)

        # The resident registry is keyed to immutable host buffers.  Replace
        # only the Phase2 initial-state group at a checkpoint boundary; X/Y
        # remain registered and are never uploaded again in this sweep.
        session = self.sweep.session
        already_registered = (
            session.phase1 is not None
            and session.phase1["Tan0"] is tangent
            and session.phase1["Cen0"] is center
            and session.phase1["owner_mask0"] is owner)
        if not already_registered:
            session.context._resident_clear("phase1")
            try:
                session.context._resident_put("phase1.Tan0", tangent)
                session.context._resident_put("phase1.Cen0", center)
                session.context._resident_put("phase1.owner", owner)
            except BaseException:
                session.context._resident_clear("phase1")
                raise
            active_p1 = dict(self.p1, Tan0=tangent, Cen0=center,
                             owner_mask0=owner)
            session.phase1 = active_p1
            self.p1 = active_p1

        before = self.sweep.stats()
        started = time.perf_counter()
        result, stats = self.sweep.context.phase2(
            self.sweep.data["x"], self.sweep.data["response"], None,
            tangent, center, owner, float(h_segment[0]), rho_segment, 2,
            h_schedule=h_segment, iteration_count=segment_iterations,
            evaluated_iterations=None,
            gpu_memory_fraction=self.memory_policy.memory_fraction,
            reserve_memory_fraction=self.memory_policy.reserve_fraction,
            reserve_memory_bytes=self.memory_policy.reserve_bytes,
            minimum_query_block=1, maximum_query_block=min(n, 131072),
            return_histories=False, direct_output_buffers=True,
            aggregation_tile_owners=plan["aggregation_tile_owners"],
            index_tile_owners=0,
            generated_neighbor_count=self.k,
            knn_scratch_memory_bytes=plan["knn_scratch_bytes"])
        wall = time.perf_counter() - started
        after = self.sweep.stats()
        delta = {key: after[key] - before[key] for key in
                 ("host_to_device_bytes", "skipped_host_to_device_bytes",
                  "initial_state_device_copy_bytes")}
        if (stats.memory_required_bytes != plan["core_required_bytes"]
                or stats.iteration_count != segment_iterations):
            raise RuntimeError(
                "native checkpoint segment allocation/iteration contract differs")
        generated = result.get("generated_index_stats")
        required_state = {
            "Tan_next", "Cen_next", "num_neigh", "heavy_mask",
            "keep_mask", "solver_failed",
        }
        missing_state = sorted(required_state.difference(result))
        if missing_state:
            raise RuntimeError(
                "checkpoint segment omitted required state outputs: "
                + ", ".join(missing_state))
        unexpected_debug = sorted(set(result).intersection({
            "num_neigh_iter", "heavy_mask_iter", "keep_mask_iter",
            "grad_debug",
        }))
        if unexpected_debug:
            raise RuntimeError(
                "checkpoint segment materialized disabled history/debug outputs: "
                + ", ".join(unexpected_debug))
        if (not generated or not generated["enabled"]
                or generated["materialized_host_index_bytes"] != 0
                or generated["materialized_device_index_bytes"] != 0
                or generated["generation_passes"] != 2 * segment_iterations
                or generated["owner_rows"] !=
                    2 * segment_iterations * owner_count
                or generated["scratch_memory_bytes"] != plan["knn_scratch_bytes"]):
            raise RuntimeError(
                "generated checkpoint segment evidence differs from plan: "
                + str(generated))
        if (delta["host_to_device_bytes"] != 0
                or delta["initial_state_device_copy_bytes"] != n * d * 3 * 4):
            raise RuntimeError(
                "checkpoint segment resident/reset contract changed: " + str(delta))
        diagnostic = dict(
            start_iteration=start_iteration,
            requested_stop_iteration=stop_iteration,
            completed_stop_iteration=start_iteration + int(stats.valid_iterations),
            phase1_h=phase1_h,
            wall_seconds=wall,
            phase2_stats=asdict(stats),
            resident_delta=delta,
            memory_plan=plan,
            generated_index_stats=generated,
        )
        self.phase2_iteration = diagnostic["completed_stop_iteration"]
        return result, stats, diagnostic

    def _set_query_arrays(self, identity, supplied):
        """Own immutable query snapshots; no mutable-pointer pass cache."""
        if len(supplied) == 1:
            raise ValueError("query requires response or tangent truth")
        hashes = {name: (a.shape, a.dtype.str, hashlib.sha256(np.ascontiguousarray(a)).hexdigest())
                  for name, a in supplied.items()}
        key = (identity, hashes)
        if key == self.query_key:
            return False
        snapshots = {name: _snapshot(array) for name, array in supplied.items()}
        self.sweep.context._resident_clear("query")
        self.query = {}
        self.query_key = None
        try:
            for name, array in snapshots.items():
                self.sweep.context._resident_put("query." + name, array)
        except BaseException:
            self.sweep.context._resident_clear("query")
            raise
        self.query, self.query_key = snapshots, key
        return True

    def set_query(self, identity, x, y=None, tangent=None):
        supplied = dict(test_x=x)
        if y is not None:
            supplied["test_y"] = y
        if tangent is not None:
            supplied["test_tangent"] = tangent
        return self._set_query_arrays(identity, supplied)

    def set_evaluation_queries(self, identity, test_x, *, test_y=None,
                               test_tangent=None, validation_x=None,
                               validation_y=None, validation_tangent=None):
        """Register one immutable evaluation family, optionally with validation."""
        if (validation_x is None) != (validation_y is None):
            raise ValueError("validation x/y must be supplied together")
        supplied = dict(test_x=test_x)
        if test_y is not None:
            supplied["test_y"] = test_y
        if test_tangent is not None:
            supplied["test_tangent"] = test_tangent
        if validation_x is not None:
            supplied["validation_x"] = validation_x
            supplied["validation_y"] = validation_y
        if validation_tangent is not None:
            if validation_x is None:
                raise ValueError("validation tangent requires validation X/Y")
            supplied["validation_tangent"] = validation_tangent
        return self._set_query_arrays(identity, supplied)

    def evaluate_checkpoint_state(self, *, state_iteration, h_scale,
                                  rho_init, bite, eval_rhos, eval_hs,
                                  query_identity, test_x, test_y=None,
                                  test_tangent=None, validation_x=None,
                                  validation_y=None):
        """Score the active checkpoint without executing any Phase-2 update.

        ``restore_generated_phase1`` installs both the immutable Phase-1
        diagnostics and the checkpoint's current Tan/Cen state.  This method
        selects the bandwidth belonging to that global state, invokes only the
        existing fixed-order K=0 evaluator, and requires the generated-kNN and
        regression/aggregation paths to remain idle.
        """
        if self.p1 is None or self.k is None:
            raise RuntimeError("restore generated Phase1 before checkpoint evaluation")
        if not self.index_generated or self.index is not None:
            raise RuntimeError(
                "checkpoint evaluation currently requires exact generated-kNN mode")
        if (isinstance(state_iteration, bool) or
                not isinstance(state_iteration, (int, np.integer)) or
                state_iteration < 0):
            raise ValueError("checkpoint evaluation iteration must be nonnegative")
        state_iteration = int(state_iteration)
        if self.phase2_iteration != state_iteration:
            raise ValueError(
                "checkpoint evaluation iteration differs from restored state")

        query_changed = self.set_evaluation_queries(
            query_identity, test_x, test_y=test_y,
            test_tangent=test_tangent, validation_x=validation_x,
            validation_y=validation_y)
        n, d = self.sweep.data["x"].shape
        p1 = self.p1
        owner = p1["owner_mask0"].astype(bool)
        if not owner.any():
            raise RuntimeError("checkpointed Phase1 has no owners")
        phase1_h, _, full_h, full_rho = _phase2_schedules(
            p1["Rad0"], owner, p1["axial_tan0"], float(h_scale),
            float(rho_init), int(bite), d, 2, max(1, state_iteration))
        schedule_index = max(0, state_iteration - 1)
        h_state = np.ascontiguousarray(
            full_h[schedule_index:schedule_index + 1], dtype=np.float32)
        rho_state = np.ascontiguousarray(
            full_rho[schedule_index:schedule_index + 1], dtype=np.float32)

        before = self.sweep.stats()
        started = time.perf_counter()
        result, stats = self.sweep.context.phase2(
            self.sweep.data["x"], self.sweep.data["response"], None,
            p1["Tan0"], p1["Cen0"], p1["owner_mask0"],
            float(h_state[0]), rho_state, 2,
            h_schedule=h_state, iteration_count=1,
            evaluated_iterations=(0,), eval_rho_scales=eval_rhos,
            eval_h_scales=eval_hs,
            gpu_memory_fraction=self.memory_policy.memory_fraction,
            reserve_memory_fraction=self.memory_policy.reserve_fraction,
            reserve_memory_bytes=self.memory_policy.reserve_bytes,
            minimum_query_block=1, maximum_query_block=min(n, 131072),
            return_histories=False, return_debug=False,
            direct_output_buffers=True, evaluation_only=True,
            aggregation_tile_owners=0, index_tile_owners=0,
            generated_neighbor_count=self.k, knn_scratch_memory_bytes=0,
            **self.query)
        wall = time.perf_counter() - started
        after = self.sweep.stats()
        delta = {key: after[key] - before[key] for key in
                 ("host_to_device_bytes", "skipped_host_to_device_bytes",
                  "initial_state_device_copy_bytes")}
        generated = result.get("generated_index_stats")
        expected_generated = {
            "enabled": 1,
            "logical_full_index_bytes": 4 * n * self.k,
            "logical_compact_index_bytes": 4 * int(owner.sum()) * self.k,
            "materialized_host_index_bytes": 0,
            "materialized_device_index_bytes": 0,
            "scratch_memory_bytes": 0,
            "response_soa_bytes": 0,
            "generation_passes": 0,
            "query_tiles": 0,
            "owner_rows": 0,
            "peak_query_rows": 0,
            "peak_owner_rows": 0,
            "peak_device_index_tile_bytes": 0,
            "owner_row_map_h2d_bytes": 0,
        }
        if (not isinstance(generated, dict) or
                any(generated.get(key) != value
                    for key, value in expected_generated.items())):
            raise RuntimeError(
                "checkpoint evaluation generated-index path was not idle: "
                + str(generated))
        zero_timers = (
            "timer_iter_setup_milliseconds",
            "timer_regression_milliseconds",
            "timer_mask_chain_milliseconds",
            "timer_aggregation_accumulate_milliseconds",
            "timer_aggregation_finalize_milliseconds",
            "timer_snapshots_milliseconds",
            "timer_eval_store_milliseconds",
            "timer_valid_store_milliseconds",
        )
        if (stats.valid_iterations != 0 or stats.iteration_count != 1 or
                stats.evaluation_iterations_executed != 1 or
                stats.memory_index_bytes != 0 or
                any(getattr(stats, name) != 0 for name in zero_timers)):
            raise RuntimeError(
                "checkpoint evaluation executed Phase-2 work: "
                + str(asdict(stats)))
        if (delta["host_to_device_bytes"] != 0 or
                delta["initial_state_device_copy_bytes"] != n * d * 3 * 4):
            raise RuntimeError(
                "checkpoint evaluation resident/reset contract changed: "
                + str(delta))
        diagnostic = dict(
            evaluation_only=True,
            state_output_suppressed=True,
            state_output_bytes=0,
            state_iteration=state_iteration,
            schedule_index=schedule_index,
            phase1_h=phase1_h,
            state_h=float(h_state[0]),
            state_rho=float(rho_state[0]),
            query_changed=query_changed,
            wall_seconds=wall,
            phase2_stats=asdict(stats),
            resident_delta=delta,
            generated_index_stats=generated,
        )
        return result, stats, diagnostic

    def evaluate(self, *, h_scale, rho_init, bite, iterations, eval_rhos, eval_hs,
                 fallback_exponent_scale=2.0):
        if self.p1 is None or not self.query:
            raise RuntimeError("prepare exact-k P1 and explicit query view first")
        k_values = [int(k) for k in iterations]
        if not k_values or any(k < 0 for k in k_values):
            raise ValueError("negative/empty K evaluations are not allowed")
        n, d = self.sweep.data["x"].shape
        p1 = self.p1
        owner = p1["owner_mask0"].astype(bool)
        if not owner.any():
            raise RuntimeError("initial model has no owners")
        h0, base, hs, rhos = _phase2_schedules(p1["Rad0"], owner, p1["axial_tan0"],
            float(h_scale), float(rho_init), int(bite), d, 2, max(1, max(k_values)))
        plan = self._memory_plan(int(owner.sum()), len(hs), len(eval_rhos), len(eval_hs))
        before = self.sweep.stats()
        started = time.perf_counter()
        result, stats = self.sweep.context.phase2(
            self.sweep.data["x"], self.sweep.data["response"], self.index,
            p1["Tan0"], p1["Cen0"], p1["owner_mask0"], base, rhos, 2,
            h_schedule=hs, iteration_count=len(hs), evaluated_iterations=k_values,
            eval_rho_scales=eval_rhos, eval_h_scales=eval_hs,
            fallback_exponent_scale=fallback_exponent_scale,
            gpu_memory_fraction=self.memory_policy.memory_fraction,
            reserve_memory_fraction=self.memory_policy.reserve_fraction,
            reserve_memory_bytes=self.memory_policy.reserve_bytes, minimum_query_block=1,
            maximum_query_block=min(n, 131072), return_histories=False,
            direct_output_buffers=True, aggregation_tile_owners=plan['aggregation_tile_owners'],
            index_tile_owners=plan['index_tile_owners'],
            index_is_compact_owner_order=self.index_compact,
            generated_neighbor_count=(self.k if self.index_generated else None),
            knn_scratch_memory_bytes=(plan['knn_scratch_bytes']
                if self.index_generated else 0), **self.query)
        wall = time.perf_counter() - started
        after = self.sweep.stats()
        delta = {key: after[key] - before[key] for key in
                 ("host_to_device_bytes", "skipped_host_to_device_bytes", "initial_state_device_copy_bytes")}
        if stats.memory_required_bytes != plan['core_required_bytes']:
            raise RuntimeError('native P2 allocation contract differs from capacity planner')
        transport = result.get('index_stream_stats')
        generated = result.get('generated_index_stats')
        streamed_bytes = 0
        if plan['index_tile_owners']:
            if (not transport or not transport['enabled']
                    or transport['device_index_bytes'] != plan['index_device_bytes']
                    or transport['pinned_host_bytes'] != plan['pinned_bytes']
                    or transport['host_to_device_bytes'] != 2 * len(hs) * 4 * int(owner.sum()) * self.k):
                raise RuntimeError('index streaming memory/transfer evidence differs from plan')
            streamed_bytes = transport['host_to_device_bytes']
        if self.index_generated:
            expected_full = 4 * n * self.k
            expected_compact = 4 * int(owner.sum()) * self.k
            if (transport is not None or not generated or not generated['enabled']
                    or generated['logical_full_index_bytes'] != expected_full
                    or generated['logical_compact_index_bytes'] != expected_compact
                    or generated['materialized_host_index_bytes'] != 0
                    or generated['materialized_device_index_bytes'] != 0
                    or generated['scratch_memory_bytes'] !=
                        plan['knn_scratch_bytes']
                    or generated['response_soa_bytes'] !=
                        4 * n * self.sweep.data['response'].shape[1]
                    or generated['generation_passes'] != 2 * len(hs)
                    or generated['owner_rows'] !=
                        2 * len(hs) * int(owner.sum())
                    or generated['owner_row_map_h2d_bytes'] !=
                        8 * len(hs) * int(owner.sum())
                    or generated['query_tiles'] < generated['generation_passes']
                    or not 4 * self.k <=
                        generated['peak_device_index_tile_bytes'] <= expected_full
                    or not 1 <= generated['peak_owner_rows'] <= int(owner.sum())
                    or not 1 <= generated['peak_query_rows'] <= n):
                raise RuntimeError(
                    'generated exact-index evidence differs from plan: ' +
                    str(generated))
        # The C ABI includes streamed-index H2D in BOTH the index-stream and
        # resident counters. Keep the raw total, and account for those exact
        # bytes; every other registered input must still have zero upload.
        delta['registered_input_host_to_device_bytes'] = delta['host_to_device_bytes'] - streamed_bytes
        # One D2D reset of Tan[N,2,D] + Cen[N,D], never a CPU initial-state copy.
        if (delta['registered_input_host_to_device_bytes'] != 0
                or delta['initial_state_device_copy_bytes'] != n * d * 3 * 4):
            raise RuntimeError('resident input/reset contract changed: ' + str(delta))
        diagnostics = dict(phase1_h=h0, phase1_owner_fraction=float(owner.mean()),
            phase1_state_sha256=self.p1_hashes, phase1_wall=self.p1_wall,
            cell_wall=wall, phase2_stats=asdict(stats), resident_delta=delta,
            memory_plan=plan, phase1_memory=self.phase1_memory,
            index_stream_stats=transport, generated_index_stats=generated)
        diagnostics['phase1_actual_capacity'] = getattr(self.sweep.context, 'last_hybrid_capacity', None)
        diagnostics['phase1_host_cache'] = self.session.phase1_cache_stats()
        diagnostics['phase1_active_fast_paths'] = self.phase1_active_fast_path_count
        return result, stats, diagnostics
