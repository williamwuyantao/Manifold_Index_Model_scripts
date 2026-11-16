"""Bounded, resumable Stage3 state evolution over Python + CUDA core.

This runner exists for large training prefixes whose exact Phase-1/kNN and
Phase-2 work may exceed one scheduler allocation.  It computes Phase 1 once,
publishes immutable checkpoints atomically, and resumes the *global* Phase-2
schedule in fresh processes without materialising the logical ``N*k`` index.

Final validation/testing is deliberately a separate action.  Advancing a
checkpoint never allocates evaluation histories and never reruns an iteration
merely to obtain metrics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


CORE_PYTHON = Path(__file__).resolve().parents[1] / "manor_m" / "cuda" / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))

from estimator import CudaCoreEstimator, P1_NAMES
from phase2_checkpoint import (file_sha256, latest_checkpoint,
                               load_checkpoint, write_checkpoint)
from stage2_memory import GIB, MemoryPolicy
from stage3_cuda_core import (CONTRACT, large_candidate, normalize_calibration,
                              read, validate_runtime, selection_metadata)
from cuda_core_data import load_canonical


SCHEMA = "mfm-stage3-checkpointed-state.v1"
REPORT_SCHEMA = "mfm-stage3-checkpointed-action.v1"
FINAL_SCHEMA = "mfm-stage3-checkpointed-final.v1"
EVALUATION_CONTRACT = "checkpoint-state-fixed-order-k0-three-pass.v1"
STATE_NAMES = (
    "Tan0", "Cen0", "num_neigh", "heavy_mask", "keep_mask",
    "solver_failed",
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-%d-%d" %
                               (os.getpid(), time.time_ns()))
    with temporary.open("x") as stream:
        json.dump(_json_value(value), stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


@contextmanager
def run_lock(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".checkpoint-action.running.lock"
    lock.mkdir()
    try:
        yield
    finally:
        lock.rmdir()


def _checkpoint_hashes(manifest: dict) -> dict[str, str]:
    return {name: row["sha256"]
            for name, row in manifest["arrays"].items()}


def _dataset_record(dataset: Path) -> dict:
    path = Path(dataset).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size}


def _load_prefix(data_loader, dataset, setup, n, *, include_testing=False):
    data = data_loader(dataset, setup, maximum_training_count=n,
                       include_testing=include_testing)
    return data.for_n(n)


def build_identity(calibration: dict, runtime: dict, dataset: Path, n: int,
                   data) -> tuple[dict, dict, bool, dict]:
    calibration = normalize_calibration(calibration)
    candidate, clipped, k_policy = large_candidate(calibration, n)
    hashes = data.hashes
    identity = {
        "schema": SCHEMA,
        "setup": calibration["setup"],
        "reference_N": 1_000_000,
        "reference_winner": calibration["validation_winner"],
        "candidate": candidate,
        "k_binding_applied": clipped,
        "K_binding_policy": k_policy,
        "N": int(n),
        "dataset": _dataset_record(dataset),
        "data_identity": data.identity,
        "input_arrays_sha256": hashes,
        "runtime": runtime,
        "metric_sum_contract": CONTRACT,
        "phase1_mode": "generated_exact_tiles",
        "phase2_index_mode": "generated_exact_tiles",
        "production_tile_selection": "capacity_arithmetic_only_no_trial_runs",
        **selection_metadata(calibration),
    }
    return identity, candidate, clipped, k_policy


def verify_static_identity(saved: dict, calibration: dict, runtime: dict,
                           dataset: Path, n: int) -> None:
    calibration = normalize_calibration(calibration)
    candidate, clipped, k_policy = large_candidate(calibration, n)
    expected = {
        "schema": SCHEMA,
        "setup": calibration["setup"],
        "reference_winner": calibration["validation_winner"],
        "candidate": candidate,
        "k_binding_applied": clipped,
        "K_binding_policy": k_policy,
        "N": int(n),
        "dataset": _dataset_record(dataset),
        "runtime": runtime,
        "metric_sum_contract": CONTRACT,
        **selection_metadata(calibration),
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError("saved Stage3 checkpoint identity differs: " + key)


def _identity_path(root: Path) -> Path:
    return Path(root) / "identity.json"


def _read_identity(root: Path) -> dict:
    path = _identity_path(root)
    before = path.read_bytes()
    value = json.loads(before)
    if path.read_bytes() != before or value.get("schema") != SCHEMA:
        raise ValueError("invalid or changing Stage3 checkpoint identity")
    return value


def _failure(root: Path, action: str, error: BaseException) -> None:
    atomic_json(Path(root) / "failures" /
                ("%s-%d.json" % (action, time.time_ns())), {
        "schema": REPORT_SCHEMA,
        "action": action,
        "error_type": type(error).__name__,
        "error": str(error),
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "process_id": os.getpid(),
    })


def initialize(calibration: dict, *, dataset: Path, output: Path, library: Path,
               runtime: dict, n: int, device: int = 0, retained_bytes: int = 0,
               memory_policy: MemoryPolicy | None = None,
               data_loader=load_canonical,
               estimator_factory=CudaCoreEstimator) -> dict:
    """Compute Phase 1 once and publish the immutable K=0 checkpoint."""
    root = Path(output)
    checkpoints = root / "checkpoints"
    with run_lock(root):
        current = latest_checkpoint(checkpoints)
        if current is not None:
            saved = _read_identity(root)
            verify_static_identity(saved, calibration, runtime, dataset, n)
            return {"state": "initialized_noop", "iteration": int(current.name[1:]),
                    "cuda_contexts": 0, "identity": saved}
        started = time.time()
        try:
            data = _load_prefix(data_loader, dataset, calibration["setup"], n,
                                include_testing=False)
            identity, candidate, _, _ = build_identity(
                calibration, runtime, dataset, n, data)
            identity_path = _identity_path(root)
            if identity_path.exists():
                if _read_identity(root) != identity:
                    raise ValueError("existing Stage3 checkpoint identity differs")
            else:
                atomic_json(identity_path, identity)

            with estimator_factory(
                    library, data.arrays["X"], data.arrays["Y"],
                    dataset_id=calibration["setup"]["id"] + ":Stage3:N" + str(n),
                    device=device, retained_bytes=retained_bytes,
                    memory_policy=memory_policy, phase1_cache_bytes=0) as estimator:
                preparation = estimator.prepare(int(candidate["kSlice"]))
                if (not estimator.index_generated or estimator.index is not None
                        or estimator.phase1_memory.get("mode") != 4):
                    raise RuntimeError(
                        "large-N checkpoint runner requires generated exact Phase1")
                destination = write_checkpoint(
                    checkpoints, 0,
                    {name: estimator.p1[name] for name in P1_NAMES}, identity)
                _, manifest = load_checkpoint(destination, identity)
                report = {
                    "schema": REPORT_SCHEMA,
                    "action": "initialize",
                    "complete": True,
                    "iteration": 0,
                    "target_iteration": int(candidate["K"]),
                    "checkpoint": str(destination),
                    "checkpoint_sha256": _checkpoint_hashes(manifest),
                    "preparation": preparation,
                    "preparation_count": estimator.preparation_count,
                    "phase1_restore_count": estimator.phase1_restore_count,
                    "phase1_memory": estimator.phase1_memory,
                    "phase1_state_sha256": estimator.p1_hashes,
                    "start_epoch": started,
                    "end_epoch": time.time(),
                    "process_id": os.getpid(),
                }
            atomic_json(root / "reports" / "initialize-K000000.json", report)
            return {"state": "initialized", "iteration": 0,
                    "target_iteration": int(candidate["K"]), "cuda_contexts": 1}
        except BaseException as error:
            _failure(root, "initialize", error)
            raise


def _active_state(root: Path, identity: dict, current: Path):
    p1_path = Path(root) / "checkpoints" / "K000000"
    p1, p1_manifest = load_checkpoint(p1_path, identity)
    if current == p1_path:
        state, state_manifest = p1, p1_manifest
    else:
        state, state_manifest = load_checkpoint(current, identity)
    active = dict(p1)
    active["Tan0"] = state["Tan0"]
    active["Cen0"] = state["Cen0"]
    initial_hashes = _checkpoint_hashes(p1_manifest)
    state_hashes = _checkpoint_hashes(state_manifest)
    active_hashes = dict(initial_hashes)
    active_hashes["Tan0"] = state_hashes["Tan0"]
    active_hashes["Cen0"] = state_hashes["Cen0"]
    return active, active_hashes, initial_hashes, p1_manifest, state_manifest


def _unchanged(path: Path, manifest: dict) -> bool:
    return all(file_sha256(path / row["file"]) == row["sha256"]
               for row in manifest["arrays"].values())


def _metric_scalar(result: dict, name: str) -> float:
    value = np.asarray(result[name], dtype=np.float64)
    if value.shape != (1, 1) or not np.isfinite(value[0, 0]):
        raise RuntimeError("invalid checkpoint evaluation metric: " + name)
    return float(value[0, 0])


def _metric_coordinate(result: dict, name: str, count: int) -> np.ndarray:
    value = np.asarray(result[name], dtype=np.float64)
    if value.shape != (1, 1, count) or not np.isfinite(value).all():
        raise RuntimeError("invalid checkpoint coordinate metric: " + name)
    return value[0, 0]


def _validate_final(final: dict, identity: dict, current: Path,
                    checkpoint_hashes: dict) -> dict:
    expected = {
        "schema": FINAL_SCHEMA,
        "complete": True,
        "identity": identity,
        "checkpoint": str(current),
        "checkpoint_sha256": checkpoint_hashes,
        "evaluation_contract": EVALUATION_CONTRACT,
    }
    for key, value in expected.items():
        if final.get(key) != value:
            raise ValueError("saved Stage3 final differs: " + key)
    metrics = final.get("metrics")
    required = (
        "validation_relMSE_total", "validation_relMSE_coord",
        "validation_devTan", "testing_relMSE_total",
        "testing_relMSE_coord", "testing_devTan",
    )
    if (not isinstance(metrics, dict)
            or any(key not in metrics for key in required)
            or any(not math.isfinite(float(metrics[key]))
                   or float(metrics[key]) < 0 for key in required)):
        raise ValueError("saved Stage3 final metrics are invalid")
    return final


def advance(calibration: dict, *, dataset: Path, output: Path, library: Path,
            runtime: dict, n: int, maximum_iterations: int = 1,
            device: int = 0, retained_bytes: int = 0,
            memory_policy: MemoryPolicy | None = None,
            data_loader=load_canonical,
            estimator_factory=CudaCoreEstimator) -> dict:
    """Advance at most ``maximum_iterations`` and publish one new state."""
    if (isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int)
            or maximum_iterations < 1):
        raise ValueError("maximum_iterations must be a positive integer")
    root = Path(output)
    checkpoints = root / "checkpoints"
    with run_lock(root):
        current = latest_checkpoint(checkpoints)
        if current is None:
            raise ValueError("initialize Phase1 before advancing Stage3")
        saved_identity = _read_identity(root)
        verify_static_identity(saved_identity, calibration, runtime, dataset, n)
        target = int(saved_identity["candidate"]["K"])
        start = int(current.name[1:])
        terminal_path = root / "terminal.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            return {"state": "terminal_noop", "iteration": start,
                    "target_iteration": target, "cuda_contexts": 0,
                    "terminal": terminal}
        if start >= target:
            return {"state": "target_reached_noop", "iteration": start,
                    "target_iteration": target, "cuda_contexts": 0}
        stop = min(target, start + maximum_iterations)
        started = time.time()
        try:
            data = _load_prefix(data_loader, dataset, calibration["setup"], n,
                                include_testing=False)
            identity, candidate, _, _ = build_identity(
                calibration, runtime, dataset, n, data)
            if identity != saved_identity:
                raise ValueError("Stage3 resume identity or input arrays differ")
            active, active_hashes, initial_hashes, p1_manifest, state_manifest = (
                _active_state(root, identity, current))
            initialize_report = json.loads(
                (root / "reports" / "initialize-K000000.json").read_text())

            with estimator_factory(
                    library, data.arrays["X"], data.arrays["Y"],
                    dataset_id=calibration["setup"]["id"] + ":Stage3:N" + str(n),
                    device=device, retained_bytes=retained_bytes,
                    memory_policy=memory_policy, phase1_cache_bytes=0) as estimator:
                restore = estimator.restore_generated_phase1(
                    int(candidate["kSlice"]), active,
                    state_sha256=active_hashes,
                    initial_state_sha256=initial_hashes,
                    state_iteration=start,
                    source_wall_seconds=float(
                        initialize_report["preparation"]["wall_seconds"]))
                result, stats, diagnostic = estimator.advance_phase2(
                    active["Tan0"], active["Cen0"],
                    start_iteration=start, stop_iteration=stop,
                    h_scale=float(candidate["h_scale"]),
                    rho_init=float(candidate["rho_init"]),
                    bite=int(candidate["bite"]))
                completed = int(diagnostic["completed_stop_iteration"])
                if not start <= completed <= stop:
                    raise RuntimeError("native completed iteration lies outside segment")
                destination = current
                output_hashes = None
                if completed > start:
                    checkpoint_arrays = {
                        "Tan0": result["Tan_next"],
                        "Cen0": result["Cen_next"],
                        "num_neigh": result["num_neigh"],
                        "heavy_mask": result["heavy_mask"],
                        "keep_mask": result["keep_mask"],
                        "solver_failed": result["solver_failed"],
                    }
                    destination = write_checkpoint(
                        checkpoints, completed, checkpoint_arrays, identity)
                    _, output_manifest = load_checkpoint(destination, identity)
                    output_hashes = _checkpoint_hashes(output_manifest)
                stopped = bool(stats.stopped_no_heavy or completed < stop)
                report = {
                    "schema": REPORT_SCHEMA,
                    "action": "advance",
                    "complete": True,
                    "start_iteration": start,
                    "requested_stop_iteration": stop,
                    "completed_stop_iteration": completed,
                    "target_iteration": target,
                    "stopped_no_heavy": stopped,
                    "checkpoint": str(destination),
                    "checkpoint_sha256": output_hashes,
                    "restore": restore,
                    "preparation_count": estimator.preparation_count,
                    "phase1_restore_count": estimator.phase1_restore_count,
                    "phase2_stats": asdict(stats),
                    "diagnostic": diagnostic,
                    "input_p1_unchanged": _unchanged(
                        checkpoints / "K000000", p1_manifest),
                    "input_state_unchanged": _unchanged(current, state_manifest),
                    "start_epoch": started,
                    "end_epoch": time.time(),
                    "process_id": os.getpid(),
                }
            report_path = root / "reports" / (
                "advance-K%06d-K%06d.json" % (start, completed))
            atomic_json(report_path, report)
            if stopped:
                terminal = {
                    "schema": REPORT_SCHEMA,
                    "state": "stopped_no_heavy",
                    "effective_iteration": completed,
                    "requested_target_iteration": target,
                    "checkpoint": str(destination),
                    "report": str(report_path),
                }
                atomic_json(terminal_path, terminal)
                state = "terminal_no_heavy"
            else:
                state = "target_reached" if completed == target else "advanced"
            return {"state": state, "iteration": completed,
                    "target_iteration": target, "cuda_contexts": 1}
        except BaseException as error:
            _failure(root, "advance", error)
            raise


def evaluate(calibration: dict, *, dataset: Path, output: Path, library: Path,
             runtime: dict, n: int, device: int = 0,
             retained_bytes: int = 0,
             memory_policy: MemoryPolicy | None = None,
             data_loader=load_canonical,
             estimator_factory=CudaCoreEstimator) -> dict:
    """Evaluate the last complete checkpoint without advancing Phase 2."""
    root = Path(output)
    checkpoints = root / "checkpoints"
    final_path = root / "final.json"
    with run_lock(root):
        current = latest_checkpoint(checkpoints)
        if current is None:
            raise ValueError("initialize Phase1 before evaluating Stage3")
        identity = _read_identity(root)
        verify_static_identity(identity, calibration, runtime, dataset, n)
        iteration = int(current.name[1:])
        target = int(identity["candidate"]["K"])
        terminal_path = root / "terminal.json"
        if iteration < target and not terminal_path.exists():
            raise ValueError("Stage3 checkpoint has not reached its terminal state")

        _, current_manifest = load_checkpoint(current, identity)
        current_hashes = _checkpoint_hashes(current_manifest)
        if final_path.exists():
            final = _validate_final(
                json.loads(final_path.read_text()), identity, current,
                current_hashes)
            return {
                "state": "evaluated_noop",
                "iteration": iteration,
                "target_iteration": target,
                "cuda_contexts": 0,
                "metrics": final["metrics"],
            }

        started = time.time()
        try:
            data = _load_prefix(data_loader, dataset, calibration["setup"], n,
                                include_testing=True)
            if data.identity != identity["data_identity"]:
                raise ValueError("Stage3 evaluation data identity differs")
            data_hashes = data.hashes
            for name, expected in identity["input_arrays_sha256"].items():
                if data_hashes.get(name) != expected:
                    raise ValueError(
                        "Stage3 evaluation input array differs: " + name)
            for name in ("Xtest", "Ytest", "Ttest"):
                if name not in data.arrays or name not in data_hashes:
                    raise ValueError("Stage3 evaluation input is missing: " + name)

            active, active_hashes, initial_hashes, p1_manifest, state_manifest = (
                _active_state(root, identity, current))
            initialize_report = json.loads(
                (root / "reports" / "initialize-K000000.json").read_text())
            candidate = identity["candidate"]
            eval_rho = [float(candidate["eval_rho"])]
            eval_h = [float(candidate["eval_h"])]

            with estimator_factory(
                    library, data.arrays["X"], data.arrays["Y"],
                    dataset_id=calibration["setup"]["id"] + ":Stage3:N" + str(n),
                    device=device, retained_bytes=retained_bytes,
                    memory_policy=memory_policy, phase1_cache_bytes=0) as estimator:
                restore = estimator.restore_generated_phase1(
                    int(candidate["kSlice"]), active,
                    state_sha256=active_hashes,
                    initial_state_sha256=initial_hashes,
                    state_iteration=iteration,
                    source_wall_seconds=float(
                        initialize_report["preparation"]["wall_seconds"]))
                common = dict(
                    state_iteration=iteration,
                    h_scale=float(candidate["h_scale"]),
                    rho_init=float(candidate["rho_init"]),
                    bite=int(candidate["bite"]),
                    eval_rhos=eval_rho,
                    eval_hs=eval_h,
                )
                response, _, response_diagnostic = (
                    estimator.evaluate_checkpoint_state(
                        **common, query_identity="response-testing-validation",
                        test_x=data.arrays["Xtest"],
                        test_y=data.arrays["Ytest"],
                        validation_x=data.arrays["Xvalid"],
                        validation_y=data.arrays["Yvalid"]))
                validation_tangent, _, validation_tangent_diagnostic = (
                    estimator.evaluate_checkpoint_state(
                        **common, query_identity="tangent-validation",
                        test_x=data.arrays["Xvalid"],
                        test_tangent=data.arrays["Tvalid"]))
                testing_tangent, _, testing_tangent_diagnostic = (
                    estimator.evaluate_checkpoint_state(
                        **common, query_identity="tangent-testing",
                        test_x=data.arrays["Xtest"],
                        test_tangent=data.arrays["Ttest"]))

                validation_variance, validation_total = data.variance("validation")
                testing_variance, testing_total = data.variance("testing")
                validation_mse = _metric_scalar(response, "EvalValidMSE0")
                validation_coordinate = _metric_coordinate(
                    response, "EvalValidMSECoordinate0", len(validation_variance))
                testing_mse = _metric_scalar(response, "EvalMSE0")
                testing_coordinate = _metric_coordinate(
                    response, "EvalMSECoordinate0", len(testing_variance))
                metrics = {
                    "validation_relMSE_total": validation_mse / validation_total,
                    "validation_relMSE_coord": float(np.mean(
                        validation_coordinate / validation_variance)),
                    "validation_devTan": _metric_scalar(
                        validation_tangent, "EvalDevTan0"),
                    "testing_relMSE_total": testing_mse / testing_total,
                    "testing_relMSE_coord": float(np.mean(
                        testing_coordinate / testing_variance)),
                    "testing_devTan": _metric_scalar(
                        testing_tangent, "EvalDevTan0"),
                }
                if any(not math.isfinite(value) or value < 0
                       for value in metrics.values()):
                    raise RuntimeError("Stage3 evaluation produced invalid metrics")
                final = {
                    "schema": FINAL_SCHEMA,
                    "complete": True,
                    "identity": identity,
                    "candidate": candidate,
                    "requested_iteration": target,
                    "effective_iteration": iteration,
                    "terminal_state_carry_forward": iteration < target,
                    "checkpoint": str(current),
                    "checkpoint_sha256": current_hashes,
                    "evaluation_contract": EVALUATION_CONTRACT,
                    "metrics": metrics,
                    "evaluation_input_arrays_sha256": {
                        name: data_hashes[name] for name in
                        ("Xvalid", "Yvalid", "Tvalid",
                         "Xtest", "Ytest", "Ttest")},
                    "restore": restore,
                    "preparation_count": estimator.preparation_count,
                    "phase1_restore_count": estimator.phase1_restore_count,
                    "evaluation_passes": 3,
                    "response_diagnostic": response_diagnostic,
                    "validation_tangent_diagnostic":
                        validation_tangent_diagnostic,
                    "testing_tangent_diagnostic": testing_tangent_diagnostic,
                    "input_p1_unchanged": _unchanged(
                        checkpoints / "K000000", p1_manifest),
                    "input_state_unchanged": _unchanged(
                        current, state_manifest),
                    "start_epoch": started,
                    "end_epoch": time.time(),
                    "completed_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "process_id": os.getpid(),
                }
            if (not final["input_p1_unchanged"]
                    or not final["input_state_unchanged"]):
                raise RuntimeError("Stage3 evaluation changed checkpoint inputs")
            _validate_final(final, identity, current, current_hashes)
            atomic_json(final_path, final)
            return {
                "state": "evaluated",
                "iteration": iteration,
                "target_iteration": target,
                "cuda_contexts": 1,
                "metrics": metrics,
            }
        except BaseException as error:
            _failure(root, "evaluate", error)
            raise


def status(output: Path) -> dict:
    root = Path(output)
    identity = _read_identity(root)
    current = latest_checkpoint(root / "checkpoints")
    if current is None:
        raise ValueError("no complete Stage3 checkpoint")
    iteration = int(current.name[1:])
    target = int(identity["candidate"]["K"])
    terminal = root / "terminal.json"
    final = root / "final.json"
    return {
        "state": ("evaluated" if final.exists() else
                  "terminal_no_heavy" if terminal.exists() else
                  "target_reached" if iteration >= target else "resumable"),
        "iteration": iteration,
        "target_iteration": target,
        "checkpoint": str(current),
        "final": str(final) if final.exists() else None,
        "cuda_contexts": 0,
    }


def _policy(arguments) -> MemoryPolicy:
    return MemoryPolicy(
        working_bytes=int(arguments.gpu_working_gib * GIB),
        pinned_bytes=int(arguments.pinned_gib * GIB),
        phase1_working_bytes=int(arguments.phase1_working_gib * GIB),
        phase1_index_heap_bytes=int(arguments.phase1_index_heap_gib * GIB),
        phase1_spill_directory=(str(arguments.phase1_index_dir)
                                if arguments.phase1_index_dir else None),
        phase1_mode=4,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",
                        choices=("initialize", "advance", "evaluate", "status"),
                        required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--N", type=int)
    parser.add_argument("--maximum-iterations", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retained-gib", type=float, default=0)
    parser.add_argument("--gpu-working-gib", type=float, default=0)
    parser.add_argument("--pinned-gib", type=float, default=2)
    parser.add_argument("--phase1-working-gib", type=float, default=0)
    parser.add_argument("--phase1-index-heap-gib", type=float, default=0)
    parser.add_argument("--phase1-index-dir", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.action == "status":
        print(json.dumps(status(arguments.output), sort_keys=True))
        return 0
    required = (arguments.library, arguments.runtime_json,
                arguments.dataset_file, arguments.calibration_file, arguments.N)
    if any(value is None for value in required):
        parser.error(
            "initialize/advance/evaluate require library, runtime, dataset, calibration, and N")
    if arguments.N <= 1_000_000:
        parser.error("checkpointed Stage3 requires N > 1M")
    budgets = (arguments.retained_gib, arguments.gpu_working_gib,
               arguments.pinned_gib, arguments.phase1_working_gib,
               arguments.phase1_index_heap_gib)
    if any(not math.isfinite(value) or value < 0 for value in budgets):
        parser.error("memory budgets must be finite and nonnegative")
    runtime = validate_runtime(arguments.library, arguments.runtime_json)
    calibration = normalize_calibration(read(arguments.calibration_file))
    common = dict(calibration=calibration, dataset=arguments.dataset_file,
                  output=arguments.output, library=arguments.library,
                  runtime=runtime, n=arguments.N, device=arguments.device,
                  retained_bytes=int(arguments.retained_gib * GIB),
                  memory_policy=_policy(arguments))
    if arguments.action == "initialize":
        result = initialize(**common)
    elif arguments.action == "advance":
        result = advance(maximum_iterations=arguments.maximum_iterations,
                         **common)
    else:
        result = evaluate(**common)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
