"""Stage3 fixed-N1M-winner large-N runner over Python + shared CUDA core.

This entry never searches tile sizes or hyperparameters in production.  It
freezes the calibrated N=1M winner, recomputes only the specified iteration
count for N, and lets the already-qualified byte planner prefer resident
execution or choose the largest qualified exact tile that fits observed GPU
memory.  Validation and testing remain independent 5,000-query views.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import re
import sys
import time

CORE_PYTHON = Path(__file__).resolve().parents[1] / "manor_m" / "cuda" / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from calibration_protocol import bind_iteration_count, fixed_n1m_candidate
from cuda_core_data import load_canonical
from estimator import CudaCoreEstimator
from stage2_memory import GIB, MemoryPolicy
from stage2_cuda_core import (CONTRACT, evaluate_candidate, json_value,
                              normalize_calibration as normalize_stage1_calibration, read, save,
                              validate_runtime)


SCHEMA = "mfm-stage3-large-N-fixed-winner.v1"
SETUP_SCHEMA = "mfm-stage3-large-N-setup.v1"
ITERATION_BINDING_CONTRACT = "K=max(0,ceil(2*log(N)+frozen_offset)); offset preserved"
METRIC_PASS_CONTRACT = "separate-response-and-tangent-v1"
P1_CAPACITY_MODES = {
    "resident", "index_streamed", "query_local_mask", "generated_exact_tiles",
}
P2_CAPACITY_MODES = {
    "resident", "aggregation_tiled", "index_and_aggregation_tiled",
    "generated", "generated_knn_tiled", "generated_aggregation_tiled",
    "generated_knn_and_aggregation_tiled",
}


def normalize_calibration(value):
    """Accept an explicitly frozen cross-N selection without relabeling Stage1."""
    if value.get("schema") != "mfm-stage3-fixed-seven-selection.v1":
        return normalize_stage1_calibration(value)
    from fixed_hyperparameter_screen_cuda import candidate_at_n, normalize_parameters
    result = copy.deepcopy(value)
    provenance = result.get("parameter_selection", {})
    if (result.get("status") != "selected_fixed_seven"
            or provenance.get("selection_data") != "validation only"
            or provenance.get("testing_used_for_selection") is not False
            or not re.fullmatch(r"[0-9a-f]{64}", provenance.get("source_sha256", ""))):
        raise ValueError("Stage3 fixed-seven selection lacks frozen provenance")
    parameters = normalize_parameters(result["fixed_parameters"])
    expected, _ = candidate_at_n(parameters, 1_000_000, result["setup"]["D"])
    if result.get("validation_winner") != expected:
        raise ValueError("Stage3 binding anchor differs from the selected seven-tuple")
    # reference_N identifies alpha's exact integer binding only. No new N1M
    # calibration or score is fabricated; the source remains the cross-N screen.
    return result


def selection_metadata(calibration):
    return ({"parameter_selection": calibration["parameter_selection"],
             "fixed_parameters": calibration["fixed_parameters"]}
            if calibration.get("status") == "selected_fixed_seven" else {})


def selection_strategy(calibration):
    return ("validation_selected_cross_N_fixed_seven"
            if selection_metadata(calibration)
            else "exact_frozen_N1M_validation_winner")


def large_candidate(calibration, N):
    N = int(N)
    if N <= 1_000_000:
        raise ValueError("Stage3 requires N > 1M")
    setup = calibration["setup"]
    candidate, clipped = fixed_n1m_candidate(
        calibration["validation_winner"], N, setup["D"])
    effective, raw, bound = bind_iteration_count(
        calibration["validation_winner"]["K_offset"], N)
    if effective != candidate["K"]:
        raise RuntimeError("fixed winner and iteration binding disagree")
    policy = dict(contract=ITERATION_BINDING_CONTRACT, hard_lower_bound=0,
                  raw_K=raw, hard_lower_bound_applied=bound)
    return candidate, clipped, policy


def validate_record(value, calibration, N, runtime, candidate, clipped, policy,
                    identity=None, arrays_sha256=None):
    for key, expected in selection_metadata(calibration).items():
        if value.get(key) != expected:
            raise ValueError("Stage3 fixed-seven provenance changed: " + key)
    if (value.get("schema") != SCHEMA or value.get("N") != N
            or value.get("backend") != "cuda_core"
            or value.get("interface") != "python_direct"
            or value.get("metric_sum_contract") != CONTRACT
            or value.get("cuda_runtime") != runtime
            or value.get("setup", {}).get("id") != calibration["setup"]["id"]
            or value.get("reference_N") != 1_000_000
            or value.get("reference_winner") != calibration["validation_winner"]
            or value.get("candidate") != candidate
            or value.get("k_binding_applied") != clipped
            or value.get("K_binding_policy") != policy
            or value.get("testing_Xclean_loaded") is not False
            or value.get("testing_center_metrics") is not False
            or not value.get("completed_at")):
        raise ValueError("saved Stage3 identity/protocol mismatch")
    if identity is not None and value.get("data_identity") != identity:
        raise ValueError("saved Stage3 data identity mismatch")
    if arrays_sha256 is not None and value.get("input_arrays_sha256") != arrays_sha256:
        raise ValueError("saved Stage3 input hashes mismatch")
    for prefix in ("validation_", "testing_"):
        for metric in ("relMSE_total", "relMSE_coord", "devTan"):
            number = value.get(prefix + metric)
            if not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError("missing/nonfinite Stage3 metric: " + prefix + metric)
    for view in ("validation", "testing"):
        diagnostic = value.get(view + "_diagnostics", {})
        if diagnostic.get("metric_pass_contract") != METRIC_PASS_CONTRACT:
            raise ValueError("Stage3 metric passes were not separated")
        for family in ("response", "tangent"):
            component = diagnostic.get(family, {})
            plan = component.get("memory_plan", {})
            p1 = component.get("phase1_memory", {})
            p1_required = p1.get("phase1_required_bytes")
            p1_usable = p1.get("usable_bytes")
            p1_peak = p1.get("estimated_peak_bytes")
            p1_mode = p1.get("mode_name")
            p2_mode = plan.get("mode")
            generated = component.get("generated_index_stats")
            generated_p1 = p1_mode == "generated_exact_tiles"
            generated_p2 = (isinstance(p2_mode, str)
                            and p2_mode.startswith("generated"))
            if (p2_mode not in P2_CAPACITY_MODES
                    or plan.get("estimated_call_bytes", 1) > plan.get("usable_call_bytes", 0)
                    or p1_mode not in P1_CAPACITY_MODES
                    or generated_p1 != generated_p2
                    or isinstance(p1_required, bool)
                    or not isinstance(p1_required, int) or p1_required <= 0
                    or isinstance(p1_usable, bool)
                    or not isinstance(p1_usable, int) or p1_usable <= 0
                    or p1_required > p1_usable
                    or isinstance(p1_peak, bool)
                    or not isinstance(p1_peak, int) or p1_peak <= 0):
                raise ValueError("missing or impossible Stage3 capacity evidence")
            if generated_p1:
                expected_index_bytes = 4 * N * int(candidate["kSlice"])
                required_generated = {
                    "enabled": 1,
                    "logical_full_index_bytes": expected_index_bytes,
                    "materialized_host_index_bytes": 0,
                    "materialized_device_index_bytes": 0,
                }
                if (p1.get("mode") != 4
                        or p1.get("full_pipeline_out_of_core") is not True
                        or not isinstance(generated, dict)
                        or any(generated.get(key) != expected
                               for key, expected in required_generated.items())
                        or generated.get("scratch_memory_bytes")
                            != plan.get("knn_scratch_bytes")
                        or not isinstance(generated.get("peak_query_rows"), int)
                        or not 1 <= generated["peak_query_rows"] <= N
                        or not isinstance(generated.get("query_tiles"), int)
                        or generated["query_tiles"] < 1):
                    raise ValueError(
                        "missing or impossible Stage3 generated-index evidence")
                phase2 = component.get("phase2_stats", {})
                owners = phase2.get("owner_count")
                tile = plan.get("aggregation_tile_owners", 0)
                query_rows = plan.get("knn_query_rows")
                query_tiles = plan.get("knn_query_tiles")
                capacity_rows = plan.get("knn_query_rows_capacity_only")
                offline_rows = plan.get("knn_query_rows_offline_ceiling")
                offline_bytes = plan.get(
                    "knn_variable_scratch_ceiling_bytes")
                offline_applied = plan.get("offline_knn_ceiling_applied")
                aggregation_tiles = plan.get(
                    "aggregation_tiles_per_iteration")
                components = plan.get("components", {})
                if (isinstance(owners, bool) or not isinstance(owners, int)
                        or owners < 1
                        or isinstance(tile, bool) or not isinstance(tile, int)
                        or tile < 0
                        or (tile and tile > owners)
                        or isinstance(query_rows, bool)
                        or not isinstance(query_rows, int)
                        or not 1 <= query_rows <= N
                        or isinstance(capacity_rows, bool)
                        or not isinstance(capacity_rows, int)
                        or not query_rows <= capacity_rows <= N
                        or isinstance(offline_rows, bool)
                        or not isinstance(offline_rows, int)
                        or not 1 <= offline_rows <= N
                        or isinstance(offline_bytes, bool)
                        or not isinstance(offline_bytes, int)
                        or offline_bytes < 1
                        or offline_applied != (query_rows < capacity_rows)
                        or (query_rows < N and query_rows > offline_rows)
                        or query_tiles != (N + query_rows - 1) // query_rows
                        or aggregation_tiles != (
                            owners + (tile or owners) - 1
                        ) // (tile or owners)
                        or plan.get("predicted_tile_launches_per_iteration")
                            != 2 * query_tiles + aggregation_tiles
                        or plan.get("tile_selection_policy")
                            != ("capacity_then_offline_ceiling_then_"
                                "launch_count_v2")
                        or components.get("neighbor_mask") != 4 * (
                            min(owners, tile) if tile else owners
                        ) * ((int(candidate["kSlice"]) + 31) // 32)):
                    raise ValueError(
                        "missing or impossible Stage3 owner-mask capacity evidence")
    if value.get("testing_phase1_reused") is not True:
        raise ValueError("same-N testing must reuse the validated Phase1")
    return value


@contextmanager
def stage3_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".large_N.running.lock"
    lock.mkdir(exist_ok=False)
    try:
        yield
    finally:
        lock.rmdir()


def run_fixed_setup(calibration, *, dataset, output, library, runtime, n_values,
                    resume=False, device=0, retained_bytes=0,
                    data_loader=load_canonical,
                    estimator_factory=CudaCoreEstimator,
                    memory_policy=None, phase1_cache_bytes=0):
    calibration = normalize_calibration(calibration)
    Ns = list(dict.fromkeys(int(N) for N in n_values))
    plans = {N: large_candidate(calibration, N) for N in Ns}
    root = Path(output) / "large_N_gt_1M"
    if root.exists() and not resume:
        raise ValueError("existing Stage3 output requires explicit --resume")
    with stage3_lock(root):
        setup = copy.deepcopy(calibration["setup"])
        setup.update(dataset=str(Path(dataset).resolve()), result_dir=str(Path(output).resolve()))
        setup_record = dict(schema=SETUP_SCHEMA, setup=setup, reference_N=1_000_000,
            reference_winner=calibration["validation_winner"], N_values=Ns,
            strategy=selection_strategy(calibration), **selection_metadata(calibration), interface="python_direct",
            backend="cuda_core", metric_sum_contract=CONTRACT, cuda_runtime=runtime,
            production_tile_selection="capacity_arithmetic_only_no_trial_runs",
            K_binding_contract=ITERATION_BINDING_CONTRACT)
        setup_path = root / "large_N_setup.json"
        if setup_path.exists():
            existing_setup = read(setup_path)
            for key in ("schema", "reference_N", "reference_winner",
                        "strategy", "backend", "metric_sum_contract", "cuda_runtime"):
                if existing_setup.get(key) != setup_record[key]:
                    raise ValueError("saved Stage3 setup mismatch: " + key)
            existing_Ns = existing_setup.get("N_values")
            if existing_Ns != Ns:
                # A large-N curve is deliberately checkpointed one atomic N at
                # a time.  The only legal evolution of its declared grid is an
                # append: every previously declared N must remain in the same
                # order, and no completed record is rewritten or re-evaluated.
                if (not isinstance(existing_Ns, list)
                        or existing_Ns != Ns[:len(existing_Ns)]):
                    raise ValueError("saved Stage3 setup mismatch: N_values")
                existing_setup["N_values"] = Ns
                existing_setup["N_values_extension"] = dict(
                    previous=existing_Ns,
                    appended=Ns[len(existing_Ns):],
                    extended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    policy="append_only_preserve_completed_N_records.v1")
                save(setup_path, existing_setup)
        else:
            save(setup_path, setup_record)

        complete = {}
        for N, (candidate, clipped, policy) in plans.items():
            path = root / ("N%d_fixed_testing.json" % N)
            if path.exists():
                complete[N] = validate_record(read(path), calibration, N, runtime,
                                               candidate, clipped, policy)
        if len(complete) == len(Ns):
            return dict(state="complete_noop", completed_N=sorted(complete), cuda_contexts=0)

        load_started = time.perf_counter()
        data = data_loader(dataset, setup, maximum_training_count=max(Ns))
        load_seconds = time.perf_counter() - load_started
        contexts = 0
        for N, (candidate, clipped, policy) in plans.items():
            if N in complete:
                continue
            prefix_started = time.perf_counter()
            n_data = data.for_n(N)
            arrays_sha256 = n_data.hashes
            prefix_seconds = time.perf_counter() - prefix_started
            started = time.time()
            try:
                with estimator_factory(library, n_data.arrays["X"], n_data.arrays["Y"],
                        dataset_id=setup["id"] + ":Stage3:N" + str(N), device=device,
                        retained_bytes=retained_bytes, memory_policy=memory_policy,
                        phase1_cache_bytes=phase1_cache_bytes) as estimator:
                    contexts += 1
                    validation = evaluate_candidate(estimator, n_data, candidate, "validation")
                    testing = evaluate_candidate(estimator, n_data, candidate, "testing")
                    record = dict(schema=SCHEMA, N=N, setup=setup, reference_N=1_000_000,
                        reference_winner=calibration["validation_winner"], candidate=candidate,
                        k_binding_applied=clipped, K_binding_policy=policy,
                        strategy=selection_strategy(calibration), **selection_metadata(calibration), backend="cuda_core",
                        interface="python_direct", metric_sum_contract=CONTRACT,
                        cuda_runtime=runtime, data_identity=n_data.identity,
                        input_arrays_sha256=arrays_sha256, testing_Xclean_loaded=False,
                        testing_center_metrics=False,
                        validation_relMSE_total=validation["testing_relMSE_total"],
                        validation_relMSE_coord=validation["testing_relMSE_coord"],
                        validation_devTan=validation["testing_devTan"],
                        testing_relMSE_total=testing["testing_relMSE_total"],
                        testing_relMSE_coord=testing["testing_relMSE_coord"],
                        testing_devTan=testing["testing_devTan"],
                        validation_phase1_reused=validation["phase1_reused"],
                        testing_phase1_reused=testing["phase1_reused"],
                        validation_diagnostics=validation["python_diagnostics"],
                        testing_diagnostics=testing["python_diagnostics"],
                        python_timeline=dict(start_epoch=started, end_epoch=time.time(),
                            dataset_load_seconds=load_seconds,
                            per_N_data_prepare_seconds=prefix_seconds,
                            phase1_preparations=estimator.preparation_count,
                            phase1_restores=getattr(estimator, "phase1_restore_count", 0)),
                        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                record = json_value(record)
                validate_record(record, calibration, N, runtime, candidate, clipped, policy,
                                n_data.identity, arrays_sha256)
                save(root / ("N%d_fixed_testing.json" % N), record)
                complete[N] = record
                print(json.dumps(dict(event="Stage3_N_complete", setup=setup["id"], N=N)),
                      flush=True)
            except Exception as error:
                save(root / "failures" / ("N%d-%d.json" % (N, time.time_ns())),
                     dict(N=N, error=str(error), observed_at=time.time()))
                raise
        status = dict(state="complete" if len(complete) == len(Ns) else "paused",
                      completed_N=sorted(complete), cuda_contexts=contexts,
                      observed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save(root / "python_status.json", status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path,
        help="Explicit large-N canonical HDF5. Required when its filename differs from the N=1M calibration dataset; valid for one setup per invocation.")
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--setups", required=True)
    parser.add_argument("--n-values", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retained-gib", type=float, default=0)
    parser.add_argument("--gpu-working-gib", type=float, default=0)
    parser.add_argument("--pinned-gib", type=float, default=2)
    parser.add_argument("--phase1-working-gib", type=float, default=0)
    parser.add_argument("--phase1-index-heap-gib", type=float, default=8)
    parser.add_argument("--phase1-index-dir", type=Path)
    args = parser.parse_args(argv)
    Ns = list(dict.fromkeys(int(value) for value in args.n_values.split(",")))
    budgets = (args.retained_gib, args.gpu_working_gib, args.pinned_gib,
               args.phase1_working_gib, args.phase1_index_heap_gib)
    if (not Ns or any(N <= 1_000_000 for N in Ns)
            or any(not math.isfinite(v) or v < 0 for v in budgets)):
        parser.error("Stage3 needs N>1M and finite nonnegative memory budgets")
    runtime = validate_runtime(args.library, args.runtime_json)
    setup_ids = args.setups.split(",")
    if args.dataset_file is not None and len(setup_ids) != 1:
        parser.error("--dataset-file supports exactly one setup per invocation")
    for setup_id in setup_ids:
        if not re.fullmatch(r"bumpy_sphere_v[123]_D(?:10|20)_Y[0-4]", setup_id):
            parser.error("invalid setup id")
        calibration = normalize_calibration(read(
            args.calibration_root / (setup_id + "__noise1pct") / "calibration_final.json"))
        dataset = (args.dataset_file if args.dataset_file is not None else
                   args.dataset_root / Path(calibration["setup"]["dataset"]).name)
        if not dataset.is_file():
            raise ValueError("canonical dataset missing on this host: " + str(dataset))
        result = run_fixed_setup(calibration, dataset=dataset,
            output=args.result_root / (setup_id + "__noise1pct"), library=args.library,
            runtime=runtime, n_values=Ns, resume=args.resume, device=args.device,
            retained_bytes=int(args.retained_gib * GIB), phase1_cache_bytes=0,
            memory_policy=MemoryPolicy(working_bytes=int(args.gpu_working_gib * GIB),
                pinned_bytes=int(args.pinned_gib * GIB),
                phase1_working_bytes=int(args.phase1_working_gib * GIB),
                phase1_index_heap_bytes=int(args.phase1_index_heap_gib * GIB),
                phase1_spill_directory=str(args.phase1_index_dir)
                    if args.phase1_index_dir else None))
        print(json.dumps(dict(setup=setup_id, **result)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
