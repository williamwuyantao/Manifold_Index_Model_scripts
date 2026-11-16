#!/usr/bin/env python3
"""Screen one fixed Manor-M seven-tuple on a Monte-Carlo reaction dataset.

The screening stages use validation labels only.  Stage A chooses the
structural parameters ``alpha, h_scale, rho_init, bite`` across several N;
Stage B then chooses ``K_offset, eval_rho, eval_h`` for that single structural
choice.  A final curve evaluates the resulting one fixed seven-tuple on
separate validation and testing views.

This program works directly with ``dataset_generator/reaction_coordinate``
HDF5 output and the public Python + shared-CUDA-core interface.  It does not
use MATLAB, regenerate trajectories, or modify the source dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
for location in (SOURCE_ROOT / "scaling_law_N", SOURCE_ROOT / "manor_m" / "cuda" / "python"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from estimator import CudaCoreEstimator
from manor_m_cuda import ManorMCuda
from stage2_cuda_core import evaluate_candidate
from stage2_memory import GIB, MemoryPolicy


SCHEMA = "mfm-reaction-coordinate-fixed-seven-screen.v1"
STRUCTURAL_GRID = {
    "alpha": (0.005, 0.010, 0.020),
    "h_scale": (2.45, 3.50),
    "rho_init": (0.50, 1.00),
    "bite": (2, 4),
}
STAGE_A_DEFAULTS = {"K_offset": -16, "eval_rho": 0.25, "eval_h": 0.25}
EVALUATION_GRID = {
    "K_offset": (-20, -16, -12),
    "eval_rho": (0.25, 0.50),
    "eval_h": (0.25, 0.50),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text())


def parse_n_values(text: str) -> list[int]:
    values = [int(token) for token in text.split(",") if token]
    if not values or any(value < 1000 for value in values) or len(values) != len(set(values)):
        raise ValueError("N values must be distinct integers >= 1000")
    return values


def minimum_k(dimension: int) -> int:
    return max(9 * dimension, math.ceil((3 * dimension + 127) / 0.8))


def bind_candidate(parameters: dict, n: int, dimension: int) -> dict:
    reference_k = int(math.floor(float(parameters["alpha"]) * 1_000_000 + .5))
    if not 1 <= reference_k <= 1_000_000:
        raise ValueError("alpha must identify a valid reference k")
    raw_k = (reference_k * n + 1_000_000 - 1) // 1_000_000
    floor = minimum_k(dimension)
    k = min(n, max(floor, raw_k))
    raw_iterations = math.ceil(2.0 * math.log(n) + int(parameters["K_offset"]))
    return dict(parameters, reference_kSlice=reference_k, raw_kSlice=raw_k,
                k_floor=floor, k_floor_applied=raw_k < floor,
                N_cap_applied=raw_k > n, kSlice=k,
                raw_K=raw_iterations, K=max(0, raw_iterations))


@dataclass
class ReactionData:
    identity: dict
    arrays: dict

    def variance(self, view: str):
        key = {"validation": "Yvalid", "testing": "Ytest"}[view]
        values = np.asarray(self.arrays[key], dtype=np.float64)
        coordinate = np.var(values, axis=0, ddof=1)
        if not np.isfinite(coordinate).all() or np.any(coordinate <= 0):
            raise ValueError("response variance must be positive and finite")
        return coordinate, float(np.sum(coordinate))


def gather_rows(dataset: h5py.Dataset, rows: np.ndarray, dtype) -> np.ndarray:
    """Read a shuffled nested prefix in chunk-sized HDF5 reads."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0 or np.min(rows) < 0 or np.max(rows) >= dataset.shape[0]:
        raise ValueError("invalid HDF5 row selection")
    trailing = tuple(dataset.shape[1:])
    result = np.empty((rows.size,) + trailing, dtype=dtype)
    chunk = int(dataset.chunks[0] if dataset.chunks else 4096)
    bucket = rows // chunk
    order = np.argsort(bucket, kind="stable")
    cursor = 0
    while cursor < order.size:
        begin = cursor
        group = int(bucket[order[cursor]])
        while cursor < order.size and int(bucket[order[cursor]]) == group:
            cursor += 1
        positions = order[begin:cursor]
        start = group * chunk
        stop = min(start + chunk, dataset.shape[0])
        values = np.asarray(dataset[start:stop], dtype=dtype)
        result[positions] = values[rows[positions] - start]
    return np.ascontiguousarray(result)


def load_data(path: Path, maximum_n: int, include_testing: bool) -> ReactionData:
    with h5py.File(path, "r") as source:
        required = {"X", "Y", "TanTop", "MCDone", "training_indices",
                    "validation_indices", "testing_indices", "global_index"}
        missing = sorted(required - set(source.keys()))
        if missing:
            raise ValueError("reaction HDF5 lacks " + ", ".join(missing))
        attrs = {str(key): plain(value) for key, value in source.attrs.items()}
        d = int(attrs.get("ambient_dimension", 0))
        if (attrs.get("state") != "complete" or d < 3 or int(attrs.get("intrinsic_dimension", 0)) != 2
                or source["X"].shape[1] != d or source["Y"].shape[1] != 4
                or source["TanTop"].shape[1:] != (2, 3)
                or not np.all(np.asarray(source["MCDone"][:], dtype=np.uint8) == 1)):
            raise ValueError("reaction HDF5 identity or completion check failed")
        train = np.asarray(source["training_indices"], dtype=np.int64)
        valid = np.asarray(source["validation_indices"], dtype=np.int64)
        test = np.asarray(source["testing_indices"], dtype=np.int64)
        if (maximum_n > train.size or valid.size != 5000 or test.size != 5000
                or np.intersect1d(valid, test).size
                or np.intersect1d(train[:maximum_n], valid).size
                or np.intersect1d(train[:maximum_n], test).size):
            raise ValueError("reaction HDF5 split contract failed")
        x = gather_rows(source["X"], train[:maximum_n], np.float32)
        y = gather_rows(source["Y"], train[:maximum_n], np.float32)
        xvalid = np.ascontiguousarray(source["X"][valid], dtype=np.float32)
        yvalid = np.ascontiguousarray(source["Y"][valid], dtype=np.float32)
        arrays = dict(X=x, Y=y, Xvalid=xvalid, Yvalid=yvalid)
        if include_testing:
            arrays.update(Xtest=np.ascontiguousarray(source["X"][test], dtype=np.float32),
                          Ytest=np.ascontiguousarray(source["Y"][test], dtype=np.float32))
        tangent_valid = np.zeros((valid.size, 2, d), dtype=np.float32)
        tangent_valid[:, :, :3] = np.asarray(source["TanTop"][valid], dtype=np.float32)
        arrays["Tvalid"] = np.ascontiguousarray(tangent_valid)
        if include_testing:
            tangent_test = np.zeros((test.size, 2, d), dtype=np.float32)
            tangent_test[:, :, :3] = np.asarray(source["TanTop"][test], dtype=np.float32)
            arrays["Ttest"] = np.ascontiguousarray(tangent_test)
        if (not np.isfinite(arrays["X"]).all() or not np.isfinite(arrays["Y"]).all()
                or not np.isfinite(arrays["Xvalid"]).all() or not np.isfinite(arrays["Yvalid"]).all()
                or np.max(np.abs(np.sum(arrays["Yvalid"], axis=1) - 1.0)) > 2e-6
                or (include_testing and np.max(np.abs(np.sum(arrays["Ytest"], axis=1) - 1.0)) > 2e-6)):
            raise ValueError("reaction data finiteness/probability check failed")
    identity = dict(schema="reaction-coordinate-monte-carlo-manor-input.v1",
                    dataset=str(path.resolve()), dataset_sha256=sha256(path),
                    generator=attrs, ambient_dimension=d, manifold_dimension=2,
                    response_dimension=4, maximum_training_n=maximum_n,
                    validation_rows=valid.tolist(),
                    testing_rows=test.tolist() if include_testing else None,
                    training_prefix_sha256=hashlib.sha256(train[:maximum_n].tobytes()).hexdigest())
    return ReactionData(identity, arrays)


def array_hashes(data: ReactionData) -> dict:
    return {name: hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
            for name, value in data.arrays.items()}


def parameters_key(parameters: dict) -> tuple:
    return (parameters["alpha"], parameters["K_offset"], parameters["h_scale"],
            parameters["rho_init"], parameters["bite"], parameters["eval_rho"],
            parameters["eval_h"])


def structural_parameters() -> list[dict]:
    values = []
    for alpha, h_scale, rho_init, bite in itertools.product(
            STRUCTURAL_GRID["alpha"], STRUCTURAL_GRID["h_scale"],
            STRUCTURAL_GRID["rho_init"], STRUCTURAL_GRID["bite"]):
        values.append(dict(alpha=alpha, h_scale=h_scale, rho_init=rho_init,
                           bite=bite, **STAGE_A_DEFAULTS))
    return values


def evaluation_parameters(structural: dict) -> list[dict]:
    values = []
    for k_offset, eval_rho, eval_h in itertools.product(
            EVALUATION_GRID["K_offset"], EVALUATION_GRID["eval_rho"], EVALUATION_GRID["eval_h"]):
        values.append(dict(structural, K_offset=k_offset,
                           eval_rho=eval_rho, eval_h=eval_h))
    return values


def existing_record(path: Path, expected: dict):
    if not path.exists():
        return None
    record = read_json(path)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("existing screen record differs: " + str(path) + ": " + key)
    score = record.get("validation_relMSE")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
        raise ValueError("existing screen record has invalid score")
    return record


def evaluate_validation(data: ReactionData, library: Path, n: int, parameters: list[dict],
                        memory_policy: MemoryPolicy, root: Path, stage: str,
                        manifest_sha: str) -> list[dict]:
    dimension = int(data.identity["ambient_dimension"])
    x = np.ascontiguousarray(data.arrays["X"][:n])
    y = np.ascontiguousarray(data.arrays["Y"][:n])
    n_data = ReactionData(dict(data.identity, selected_N=n),
                          dict(X=x, Y=y, Xvalid=data.arrays["Xvalid"],
                               Yvalid=data.arrays["Yvalid"], Tvalid=data.arrays["Tvalid"]))
    hashes = array_hashes(n_data)
    records = []
    with CudaCoreEstimator(library, x, y,
            dataset_id=f"reaction-coordinate-d20-fixed7:{stage}:N{n}", device=0,
            retained_bytes=0, memory_policy=memory_policy,
            phase1_cache_bytes=2 * GIB) as estimator:
        for ordinal, parameters0 in enumerate(parameters):
            candidate = bind_candidate(parameters0, n, dimension)
            spec = dict(schema=SCHEMA, stage=stage, N=n, ordinal=ordinal,
                        fixed_parameters=parameters0, candidate=candidate,
                        manifest_sha256=manifest_sha,
                        selection_data="validation_only", testing_used=False,
                        data_identity=n_data.identity, input_arrays_sha256=hashes)
            path = root / stage / f"N{n}" / f"candidate-{ordinal:03d}.json"
            record = existing_record(path, spec)
            if record is None:
                measured = evaluate_candidate(estimator, n_data, candidate, "validation",
                                               include_tangent=False)
                record = dict(spec, validation_relMSE=measured["testing_relMSE_total"],
                              requested_K=measured["requested_K"],
                              effective_K=measured["effective_K"],
                              stopped_no_heavy=measured["stopped_no_heavy"],
                              phase1_time=measured["phase1_time"],
                              phase1_reused=measured["phase1_reused"],
                              phase2_time=measured["phase2_time"], wall=measured["wall"],
                              python_diagnostics=measured["python_diagnostics"],
                              completed_at=time.time())
                write_json(path, record)
                print(json.dumps(dict(event="validation_complete", stage=stage, N=n,
                                      ordinal=ordinal, total=len(parameters),
                                      validation_relMSE=record["validation_relMSE"])), flush=True)
            records.append(record)
    return records


def choose(records_by_n: dict[int, list[dict]], stage: str) -> dict:
    n_values = sorted(records_by_n)
    denominator = {n: min(record["validation_relMSE"] for record in records_by_n[n])
                   for n in n_values}
    grouped = {}
    for n in n_values:
        for record in records_by_n[n]:
            key = parameters_key(record["fixed_parameters"])
            grouped.setdefault(key, []).append(record)
    ranked = []
    for key, records in grouped.items():
        if len(records) != len(n_values):
            raise ValueError("screen candidate missing an N")
        by_n = {record["N"]: record["validation_relMSE"] for record in records}
        ratios = [by_n[n] / denominator[n] for n in n_values]
        ranked.append(dict(parameters=records[0]["fixed_parameters"],
                           validation_relMSE_by_N={str(n): by_n[n] for n in n_values},
                           max_regret_fraction=max(ratios) - 1.0,
                           geometric_mean_regret_fraction=(
                               math.exp(sum(math.log(value) for value in ratios) / len(ratios)) - 1.0),
                           mean_validation_relMSE=sum(by_n.values()) / len(by_n)))
    ranked.sort(key=lambda item: (item["max_regret_fraction"],
                                  item["geometric_mean_regret_fraction"],
                                  parameters_key(item["parameters"])))
    return dict(stage=stage, N_values=n_values, per_N_best=denominator,
                candidates=ranked, selected=ranked[0])


def evaluate_curve(data: ReactionData, library: Path, n_values: list[int], parameters: dict,
                   memory_policy: MemoryPolicy, root: Path, manifest_sha: str) -> list[dict]:
    dimension = int(data.identity["ambient_dimension"])
    result = []
    for n in n_values:
        candidate = bind_candidate(parameters, n, dimension)
        n_data = ReactionData(dict(data.identity, selected_N=n), dict(
            X=np.ascontiguousarray(data.arrays["X"][:n]),
            Y=np.ascontiguousarray(data.arrays["Y"][:n]),
            Xvalid=data.arrays["Xvalid"], Yvalid=data.arrays["Yvalid"],
            Tvalid=data.arrays["Tvalid"], Xtest=data.arrays["Xtest"],
            Ytest=data.arrays["Ytest"], Ttest=data.arrays["Ttest"]))
        hashes = array_hashes(n_data)
        spec = dict(schema=SCHEMA, stage="fixed_curve", N=n,
                    fixed_parameters=parameters, candidate=candidate,
                    manifest_sha256=manifest_sha, selection_data="validation_only",
                    testing_used_for_selection=False, data_identity=n_data.identity,
                    input_arrays_sha256=hashes)
        path = root / "fixed_curve" / f"N{n}.json"
        record = existing_record(path, spec)
        if record is None:
            with CudaCoreEstimator(library, n_data.arrays["X"], n_data.arrays["Y"],
                    dataset_id=f"reaction-coordinate-d20-fixed7:curve:N{n}", device=0,
                    retained_bytes=0, memory_policy=memory_policy,
                    phase1_cache_bytes=2 * GIB) as estimator:
                validation = evaluate_candidate(estimator, n_data, candidate, "validation",
                                                include_tangent=True)
                testing = evaluate_candidate(estimator, n_data, candidate, "testing",
                                             include_tangent=True)
            record = dict(spec, validation=validation, testing=testing, completed_at=time.time())
            write_json(path, record)
            print(json.dumps(dict(event="fixed_curve_complete", N=n,
                                  validation_relMSE=validation["testing_relMSE_total"],
                                  testing_relMSE=testing["testing_relMSE_total"],
                                  testing_DevTan=testing["testing_devTan"])), flush=True)
        result.append(record)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen-n", default="20000,80000,200000")
    parser.add_argument("--curve-n", default="5000,20000,80000,200000,500000")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    screen_n = parse_n_values(args.screen_n)
    curve_n = parse_n_values(args.curve_n)
    maximum_n = max(screen_n + curve_n)
    if not args.data.is_file() or not args.library.is_file():
        raise FileNotFoundError("data HDF5 or CUDA library is absent")
    if args.output.exists() and not args.resume:
        raise RuntimeError("output already exists; use --resume to continue this exact run")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / ".running.lock"
    lock.mkdir(exist_ok=False)
    try:
        core = ManorMCuda(args.library)
        info = core.device_info(0)
        # On this host the L20X reports compute capability 9.0 even though
        # older generator notes called it an sm_89-class device.  Validate the
        # actual CUDA capability returned by the loaded core, not a
        # label, and keep the run single-device.
        if core.device_count() != 1 or info.compute_capability not in {"8.9", "9.0"}:
            raise RuntimeError(
                "this run requires one CUDA-visible L20X (reported sm_89 or sm_90)"
            )
        manifest = dict(schema=SCHEMA, program_sha256=sha256(Path(__file__)),
                        library_sha256=sha256(args.library), data=str(args.data.resolve()),
                        data_sha256=sha256(args.data), screen_N=screen_n, curve_N=curve_n,
                        structural_grid=STRUCTURAL_GRID, stage_A_defaults=STAGE_A_DEFAULTS,
                        evaluation_grid=EVALUATION_GRID,
                        selection="validation-only minimax regret across N",
                        testing="independent after one fixed seven-parameter selection",
                        backend="Python + shared CUDA core", metric_sum_contract="fp32-query256-tree-v1",
                        device=dict(name=info.name,
                                    visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                    compute_capability=info.compute_capability,
                                    total_global_memory_bytes=info.total_global_memory_bytes))
        manifest_sha = hashlib.sha256(json.dumps(plain(manifest), sort_keys=True).encode()).hexdigest()
        manifest["manifest_sha256"] = manifest_sha
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != plain(manifest):
            raise RuntimeError("existing run manifest differs")
        write_json(manifest_path, manifest)
        write_json(args.output / "status.json", dict(state="loading", observed_at=time.time()))
        data = load_data(args.data, maximum_n, include_testing=False)
        policy = MemoryPolicy(working_bytes=0, pinned_bytes=16 * GIB,
                              phase1_working_bytes=0, phase1_index_heap_bytes=48 * GIB)
        stage_a_parameters = structural_parameters()
        stage_a = {n: evaluate_validation(data, args.library, n, stage_a_parameters,
                                           policy, args.output, "stage_A", manifest_sha)
                   for n in screen_n}
        stage_a_summary = choose(stage_a, "stage_A")
        write_json(args.output / "stage_A_summary.json", stage_a_summary)
        structural = stage_a_summary["selected"]["parameters"]
        stage_b_parameters = evaluation_parameters(structural)
        stage_b = {n: evaluate_validation(data, args.library, n, stage_b_parameters,
                                           policy, args.output, "stage_B", manifest_sha)
                   for n in screen_n}
        stage_b_summary = choose(stage_b, "stage_B")
        write_json(args.output / "stage_B_summary.json", stage_b_summary)
        fixed_parameters = stage_b_summary["selected"]["parameters"]
        data = load_data(args.data, maximum_n, include_testing=True)
        curve = evaluate_curve(data, args.library, curve_n, fixed_parameters,
                               policy, args.output, manifest_sha)
        summary = dict(schema=SCHEMA, complete=True, manifest_sha256=manifest_sha,
                       fixed_parameters=fixed_parameters, stage_A=stage_a_summary,
                       stage_B=stage_b_summary,
                       fixed_curve=[dict(N=row["N"],
                           validation_relMSE=row["validation"]["testing_relMSE_total"],
                           validation_DevTan=row["validation"]["testing_devTan"],
                           testing_relMSE=row["testing"]["testing_relMSE_total"],
                           testing_DevTan=row["testing"]["testing_devTan"],
                           candidate=row["candidate"],
                           phase1_memory=row["testing"]["python_diagnostics"]["response"].get("phase1_memory"),
                           phase2_memory=row["testing"]["python_diagnostics"]["response"].get("memory_plan"))
                           for row in curve], completed_at=time.time())
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "status.json", dict(state="complete", observed_at=time.time()))
        return 0
    except BaseException as error:
        write_json(args.output / "status.json", dict(state="failed", error=repr(error), observed_at=time.time()))
        raise
    finally:
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
