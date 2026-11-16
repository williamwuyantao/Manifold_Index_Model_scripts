"""Canonical Stage2 runner: Python -> shared CUDA C ABI, never MATLAB.

Example: scaling_law_N.py matrix-submillion --library ... --runtime-json ...
  --dataset-root ... --calibration-root ... --result-root ...
  --setups bumpy_sphere_v1_D10_Y2 --n-values 1000,10000 --resume

The N subset and optional cell budget partition execution, not the scientific
grid. The three N1M k samples represent alpha samples; at small N several may
collapse to the same safety-floor k, so a complete grid can contain 27, 54 or
81 validation cells. Existing records are checked before any CUDA context and
are never silently replaced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

CORE_PYTHON = Path(__file__).resolve().parents[1] / "manor_m" / "cuda" / "python"
if str(CORE_PYTHON) not in sys.path:
    sys.path.insert(0, str(CORE_PYTHON))
from calibration_protocol import (AXES, K_ALPHA_BINDING_CONTRACT, REFERENCE_N,
                                  SUBMILLION_N_VALUES, alpha_k_bindings,
                                  bind_iteration_count, fixed_n1m_candidate,
                                  minimum_k)
from calibration_protocol import file_hash, now, write_atomic, worker_lock, winner as cube_winner
from calibration_protocol import boundary_flags, select_winner, SELECTION_AXES
from cuda_core_data import load_canonical
from estimator import CudaCoreEstimator
from stage2_memory import MemoryPolicy


CONTRACT = "fp32-query256-tree-v1"
CELL_SCHEMA = "mfm-submillion-cuda-validation-cell.v1"
FINAL_SCHEMA = "mfm-setup-submillion-selected-testing.v1"
ITERATION_BINDING_CONTRACT = "K=max(0,ceil(2*log(N)+frozen_offset)); offset preserved"
ORCHESTRATION_SUCCESSOR_SCOPE = (
    "Python orchestration only: exact-k fast path, one-next-N prefix prefetch, "
    "separate response/tangent final metric passes, and explicit K=0 lower-bound metadata; "
    "CUDA arithmetic unchanged")
TERMINAL_STATE_SUCCESSOR_SCOPE = (
    "Python orchestration only: preserve requested K while evaluating a proven "
    "stopped-no-heavy terminal state at its effective K; CUDA arithmetic, scientific "
    "grid, selection, and numerical thresholds unchanged")
ORCHESTRATION_SUCCESSOR_SCOPES = (
    ORCHESTRATION_SUCCESSOR_SCOPE, TERMINAL_STATE_SUCCESSOR_SCOPE)
ORCHESTRATION_UNCHANGED_FIELDS = (
    "backend", "core_sha256", "metric_sum_contract", "compiler", "cuda",
    "architecture", "gpu_uuid", "absolute_tolerance", "relative_tolerance",
    "winner_absolute_tolerance", "legacy_fallback",
    "canonical_preflight_completion_sha256", "canonical_preflight_summary_sha256")


def runtime_compatible(saved, current):
    """Accept only an exact runtime or its explicitly declared Python successor."""
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    return (current.get("compatible_parent_runtime") == saved
            and current.get("adapter_change_scope") in ORCHESTRATION_SUCCESSOR_SCOPES
            and all(key in saved and key in current and current[key] == saved[key]
                    for key in ORCHESTRATION_UNCHANGED_FIELDS))


def normalize_calibration(calibration):
    """Normalize the one explicitly accepted stored D10 reference.

    All newly calibrated setups already carry ``final_profile`` and
    ``validation_winner``.  The frozen v1-D10-Y0 exception predates that
    schema but contains the same scientific objects under explicit reference
    names.  Converting those names here avoids a MATLAB-only special case;
    it does not reselect a winner or change any grid coordinate.
    """
    value = copy.deepcopy(calibration)
    if value.get("status") == "calibrated_bracketed":
        return value
    setup = value.get("setup", {})
    if (value.get("schema") != "mfm-setup-calibration-reference.v1"
            or value.get("status") != "accepted_existing_reference"
            or setup.get("id") != "bumpy_sphere_v1_D10_Y0"
            or setup.get("D") != 10
            or value.get("reuse_requested_by_user") is not True
            or not isinstance(value.get("reference_profile"), dict)
            or not isinstance(value.get("reference_winner"), dict)):
        raise ValueError("a bracketed calibration or the explicit v1-D10-Y0 reference is required")
    profile = copy.deepcopy(value["reference_profile"])
    profile.update(k_min=minimum_k(setup["D"]), k_over_N=0)
    value.update(status="calibrated_bracketed", final_profile=profile,
                 validation_winner=copy.deepcopy(value["reference_winner"]),
                 normalized_from_schema=calibration["schema"],
                 normalized_existing_reference=True)
    return value


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def save(path, value):
    write_atomic(path, json_value(value))


class CellWriter:
    """One outstanding atomic write, overlapped with the next CUDA call.

    Records are detached/validated before submission. Every N, partial-budget
    return, exception and final waits for completion; an I/O failure cannot
    become a successful final or noop. This does not run CUDA on a thread.
    """
    def __init__(self, overlap):
        self.overlap, self.pool, self.pending = overlap, None, None
        self.completed = 0
        self.write_seconds = self.wait_seconds = 0.

    def __enter__(self):
        return self

    @staticmethod
    def _write(path, record):
        start = time.perf_counter()
        save(path, record)
        return time.perf_counter() - start

    def check(self):
        if self.pending is not None and self.pending.done():
            self.flush()

    def submit(self, path, record):
        self.flush()  # capacity is one, never an unbounded queue of results
        if not self.overlap:
            self.write_seconds += self._write(path, record)
            self.completed += 1
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mfm-cell-io')
        self.pending = self.pool.submit(self._write, path, record)

    def flush(self):
        if self.pending is None:
            return
        pending, self.pending = self.pending, None
        start = time.perf_counter()
        try:
            self.write_seconds += pending.result()
            self.completed += 1
        finally:
            self.wait_seconds += time.perf_counter() - start

    def stats(self):
        return dict(mode='overlap' if self.overlap else 'sync', completed_cells=self.completed,
                    full_write_seconds=self.write_seconds, foreground_wait_seconds=self.wait_seconds,
                    max_pending_cells=1 if self.overlap else 0)

    def __exit__(self, *args):
        try:
            self.flush()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=True)


class PrefixPrefetcher:
    """One bounded next-N CPU preparation overlapped with current-N CUDA.

    This never reuses a model across N. It only builds the canonical arrays
    and their hashes for exactly one future prefix. The production policy is
    fixed (one slot, no throughput trials), and any worker failure is raised
    before that N can create a CUDA context or result record.
    """
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.data = self.pool = self.pending = None
        self.pending_N = None
        self.submitted = self.completed = self.synchronous = 0
        self.worker_seconds = self.foreground_wait_seconds = 0.
        self.maximum_pending = 0

    def __enter__(self):
        return self

    def bind(self, data):
        if self.data is not None:
            raise RuntimeError("prefix source is already bound")
        self.data = data

    @staticmethod
    def _build(data, N):
        started = time.perf_counter()
        value = data.for_n(N)
        hashes = value.hashes
        return value, hashes, time.perf_counter() - started

    def submit(self, N):
        if not self.enabled:
            return False
        if self.data is None or self.pending is not None:
            raise RuntimeError("next-N prefetch requires a bound source and one free slot")
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mfm-prefix")
        self.pending_N = int(N)
        self.pending = self.pool.submit(self._build, self.data, self.pending_N)
        self.submitted += 1
        self.maximum_pending = 1
        return True

    def take(self, N):
        if self.data is None:
            raise RuntimeError("prefix source is not bound")
        N = int(N)
        if self.pending is None:
            started = time.perf_counter()
            value, hashes, worker = self._build(self.data, N)
            wait = time.perf_counter() - started
            self.synchronous += 1
            source = "synchronous"
        else:
            if self.pending_N != N:
                raise RuntimeError("prefetched N does not match the requested prefix")
            future, self.pending, self.pending_N = self.pending, None, None
            started = time.perf_counter()
            value, hashes, worker = future.result()
            wait = time.perf_counter() - started
            self.completed += 1
            source = "prefetched"
        self.worker_seconds += worker
        self.foreground_wait_seconds += wait
        return value, hashes, dict(source=source, worker_seconds=worker,
                                   foreground_wait_seconds=wait)

    def stats(self):
        return dict(mode="one_next_N" if self.enabled else "synchronous",
                    submitted=self.submitted, completed=self.completed,
                    synchronous=self.synchronous, maximum_pending=self.maximum_pending,
                    worker_seconds=self.worker_seconds,
                    foreground_wait_seconds=self.foreground_wait_seconds)

    def __exit__(self, kind, error, tb):
        try:
            if self.pending is not None:
                if error is None:
                    # A clean return must not hide a failed prefetched prefix.
                    _, _, worker = self.pending.result()
                    self.worker_seconds += worker
                    self.completed += 1
                else:
                    self.pending.cancel()
        finally:
            self.pending = self.pending_N = None
            if self.pool is not None:
                self.pool.shutdown(wait=True)
                self.pool = None


def read(path):
    before = Path(path).read_bytes()
    value = json.loads(before)
    if Path(path).read_bytes() != before:
        raise RuntimeError("record changed while reading: " + str(path))
    return value


def validate_runtime(library, runtime_path):
    runtime = read(runtime_path)
    if (runtime.get("backend") not in ("cuda_core", "python_cuda_core")
            or runtime.get("core_sha256") != file_hash(library)
            or runtime.get("metric_sum_contract") != CONTRACT
            or runtime.get("legacy_fallback") is not False):
        raise ValueError("runtime must pin the CUDA core, fixed-order sums and no legacy fallback")
    parent = runtime.get("compatible_parent_runtime")
    if ((parent is None) != (runtime.get("adapter_change_scope") is None)
            or (parent is not None and not runtime_compatible(parent, runtime))):
        raise ValueError("runtime successor does not preserve the frozen CUDA/scientific contract")
    return dict(runtime, interface="python_direct", python_entry_sha256=file_hash(__file__),
                python_estimator_sha256=file_hash(CORE_PYTHON / "estimator.py"))


def plan(calibration, N):
    if (calibration.get("status") != "calibrated_bracketed"
            or "final_profile" not in calibration):
        raise ValueError("a bracketed per-setup calibration with final_profile is required")
    setup = calibration["setup"]
    frozen, bound = fixed_n1m_candidate(calibration["validation_winner"], 1_000_000, setup["D"])
    if bound or frozen["K"] != calibration["validation_winner"]["K"]:
        raise ValueError("fixed N1M candidate does not reproduce the calibration winner")
    if N not in SUBMILLION_N_VALUES:
        raise ValueError("N must be one of the 27 canonical submillion values")
    profile = copy.deepcopy(calibration["final_profile"])
    for _, (field, kind) in AXES.items():
        values = profile[field]
        if (len(values) != 3 or len(set(values)) != 3 or sorted(values) != values
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)):
            raise ValueError("each calibrated axis must have three sorted distinct values")
        if kind == "positive" and min(values) <= 0:
            raise ValueError("nonpositive scale in calibrated profile")
        if kind in ("bite", "integer", "k") and any(v != int(v) for v in values):
            raise ValueError("noninteger discrete axis")
        if kind == "bite" and min(values) < 0:
            raise ValueError("negative bite")
    reference_k_grid = [int(value) for value in profile["k_floors"]]
    bindings = alpha_k_bindings(reference_k_grid, N, setup["D"])
    profile["k_floors"] = sorted({item["kSlice"] for item in bindings})
    clipped = any(item["k_floor_applied"] or item["N_cap_applied"]
                  for item in bindings)
    profile.update(k_over_N=None,
                   alpha_grid=[item["alpha"] for item in bindings],
                   reference_kSlice_grid=reference_k_grid,
                   alpha_bindings=bindings,
                   k_alpha_binding_contract=K_ALPHA_BINDING_CONTRACT,
                   k_min=minimum_k(setup["D"]), n_min=N, n_max=N,
                   name="N%d_from_%s" % (N, calibration["final_profile"]["name"]))
    fixed, fixed_clipped = fixed_n1m_candidate(calibration["validation_winner"], N, setup["D"])
    bound = [bind_iteration_count(offset, N)
             for offset in [None] + profile["K_offsets"]]
    K_grid = [item[0] for item in bound]
    policy = dict(contract=ITERATION_BINDING_CONTRACT, hard_lower_bound=0,
                  raw_K_grid=[item[1] for item in bound],
                  hard_lower_bound_applied=[item[2] for item in bound])
    fixed_effective, fixed_raw, fixed_bound = bind_iteration_count(
        calibration["validation_winner"]["K_offset"], N)
    if fixed_effective != fixed["K"]:
        raise RuntimeError("fixed candidate and iteration binding disagree")
    policy.update(fixed_N1M_raw_K=fixed_raw,
                  fixed_N1M_hard_lower_bound_applied=fixed_bound)
    return profile, fixed, clipped, fixed_clipped, K_grid, policy


def cell_name(N, k, h, rho, bite):
    def token(v):
        return format(v, ".12g").replace(".", "p").replace("-", "m")
    return "N%d_k%d_h%s_rho%s_bite%d.json" % (N, k, token(h), token(rho), bite)


def alpha_metadata(profile, k):
    matches = [item for item in profile["alpha_bindings"] if item["kSlice"] == k]
    if not matches:
        raise ValueError("effective k has no N1M alpha provenance")
    alphas = [item["alpha"] for item in matches]
    references = [item["reference_kSlice"] for item in matches]
    return dict(alpha=alphas[0] if len(alphas) == 1 else None,
                alpha_candidates=alphas,
                alpha_identifiable=len(alphas) == 1,
                reference_kSlice_candidates=references,
                raw_kSlice_candidates=[item["raw_kSlice"] for item in matches],
                k_floor=matches[0]["k_floor"],
                k_floor_applied=any(item["k_floor_applied"] for item in matches),
                N_cap_applied=any(item["N_cap_applied"] for item in matches))


def stage2_winner(named, profile):
    winner = select_winner(named)
    winner.update(alpha_metadata(profile, winner["kSlice"]))
    return winner


def expected_grid_size(profile):
    return math.prod(len(profile[field]) for field in
                     ("k_floors", "h_scales", "rho_inits", "bites"))


def cell_specs(N, profile, K_grid, K_policy):
    for k, h, rho, bite in itertools.product(*(profile[f] for f in
            ("k_floors", "h_scales", "rho_inits", "bites"))):
        yield cell_name(N, k, h, rho, bite), dict(N=N, kSlice=k, h_scale=h,
            rho_init=rho, bite=bite, K_grid=K_grid, K_offsets=[None] + profile["K_offsets"],
            K_raw_grid=K_policy["raw_K_grid"],
            K_hard_lower_bound_applied=K_policy["hard_lower_bound_applied"],
            K_binding_contract=K_policy["contract"],
            eval_rho_grid=profile["eval_rhos"], eval_h_grid=profile["eval_hs"],
            **alpha_metadata(profile, k))


def validate_cell(record, spec, setup_id, runtime_or_core, identity=None, arrays_sha256=None):
    current_runtime = runtime_or_core if isinstance(runtime_or_core, dict) else None
    core_sha256 = (current_runtime or {}).get("core_sha256", runtime_or_core)
    observed_runtime = record.get("cuda_runtime", {})
    runtime_ok = (runtime_compatible(observed_runtime, current_runtime)
                  if current_runtime is not None
                  else observed_runtime.get("core_sha256") == core_sha256)
    for key, value in spec.items():
        if key not in record and key in ("K_raw_grid", "K_hard_lower_bound_applied",
                                          "K_binding_contract"):
            if (not any(spec["K_hard_lower_bound_applied"])
                    and min(spec["K_raw_grid"]) >= 0):
                continue
        if record.get(key) != value:
            raise ValueError("saved cell identity/grid mismatch: " + key)
    if (record.get("schema") != CELL_SCHEMA or record.get("backend") != "cuda_core"
            or record.get("metric_sum_contract") != CONTRACT
            or not runtime_ok
            or record.get("testing_used") is not False
            or record.get("testing_Xclean_loaded") is not False
            or record.get("validation_count") != 5000
            or record.get("dataset_identity", {}).get("setup_id") != setup_id
            or record.get("dataset_identity", {}).get("relative_y_noise") != .01
            or not record.get("completed_at")):
        raise ValueError("saved cell backend/data/selection contract mismatch")
    if identity is not None and record["dataset_identity"] != identity:
        raise ValueError("saved cell data identity differs from current inputs")
    if arrays_sha256 is not None and record.get("input_arrays_sha256") != arrays_sha256:
        raise ValueError("saved cell input arrays differ from current per-N inputs")
    base = record["dataset_identity"].get("base_dataset_identity", {})
    if (record.get("D") != base.get("ambient_d") or record.get("MM") != 2
            or record.get("MY") != 6
            or not math.isfinite(record.get("phase1_h", math.nan)) or record["phase1_h"] <= 0
            or not 0 < record.get("phase1_owner_fraction", 0) <= 1):
        raise ValueError("saved cell dimension/initial-model mismatch")
    cube = np.asarray(record["validation_relMSE_total"], dtype=np.float64)
    if cube.shape != (4, 3, 3) or np.isinf(cube).any() or not np.isfinite(cube[0]).all():
        raise ValueError("invalid full validation cube")
    valid = record.get("valid_phase2_iterations")
    # The native Phase-2 schedule has a one-iteration structural minimum even
    # when every requested metric is K=0.  That iteration is not selectable;
    # only the explicit initial-model metrics below are used.
    valid_ceiling = max(1, max(spec["K_grid"]))
    if not isinstance(valid, (int, float)) or valid != int(valid) or not 0 <= valid <= valid_ceiling:
        raise ValueError("invalid completed iteration count")
    for i, K in enumerate(spec["K_grid"]):
        if K <= valid and not np.isfinite(cube[i]).all():
            raise ValueError("missing executed metric slot")
        if K > valid and not np.isnan(cube[i]).all():
            raise ValueError("unexecuted slot cannot participate in selection")
    if np.any(cube[np.isfinite(cube)] < 0):
        raise ValueError("negative squared-error metric")
    for key, value in cube_winner(cube, spec).items():
        if record.get(key) != value:
            raise ValueError("saved cell winner does not match full cube: " + key)
    return record


def existing_cells(root, specs, setup_id, runtime_or_core, identity=None):
    expected = dict(specs)
    records = []
    for path in sorted((root / "cells").glob("*.json")):
        if path.name not in expected:
            raise ValueError("unexpected cell filename: " + path.name)
        record = validate_cell(read(path), expected[path.name], setup_id, runtime_or_core, identity)
        records.append((path.name, record))
    return records


def same_candidate(a, b):
    return all(a[k] == b[k] for k in SELECTION_AXES)


def candidate_score(named, candidate):
    # Exact frozen N1M coordinates need not belong to the new k grid.
    for _, record in named:
        if any(record[k] != candidate[k] for k in ("kSlice", "h_scale", "rho_init", "bite")):
            continue
        try:
            # Several frozen offsets may meet the K=0 hard lower bound.  The
            # effective K alone is then ambiguous: preserve and address the
            # exact frozen offset instead of silently snapping to the first
            # (nullable initial-model) slot.
            k_position = record["K_offsets"].index(candidate["K_offset"])
            if record["K_grid"][k_position] != candidate["K"]:
                continue
            position = (k_position,
                        record["eval_rho_grid"].index(candidate["eval_rho"]),
                        record["eval_h_grid"].index(candidate["eval_h"]))
        except ValueError:
            continue
        score = float(np.asarray(record["validation_relMSE_total"], dtype=float)[position])
        return score if math.isfinite(score) else None
    return None


def validate_final(value, calibration, N, profile, named, runtime_or_core, K_policy):
    current_runtime = runtime_or_core if isinstance(runtime_or_core, dict) else None
    core_sha = (current_runtime or {}).get("core_sha256", runtime_or_core)
    required = expected_grid_size(profile)
    if len(named) != required:
        raise ValueError("testing final requires all %d validation cells" % required)
    winner = stage2_winner(named, profile)
    fixed, clipped = fixed_n1m_candidate(calibration["validation_winner"], N, calibration["setup"]["D"])
    final_runtime = value.get("cuda_runtime", {})
    declared_backend = value.get("backend", "cuda_core" if final_runtime.get("backend") in
                                 ("cuda_core", "python_cuda_core") else None)
    declared_contract = value.get("metric_sum_contract", final_runtime.get("metric_sum_contract"))
    policy_matches = (value.get("K_binding_policy") == K_policy
                      or ("K_binding_policy" not in value
                          and not any(K_policy["hard_lower_bound_applied"])
                          and not K_policy["fixed_N1M_hard_lower_bound_applied"]))
    if (value.get("schema") != FINAL_SCHEMA or value.get("N") != N
            or declared_backend != "cuda_core"
            or declared_contract != CONTRACT
            or value.get("setup", {}).get("id") != calibration["setup"]["id"]
            or value.get("reference_profile") != calibration["final_profile"]
            or value.get("effective_profile") != profile
            or not policy_matches
            or value.get("validation_winner") != winner
            or not same_candidate(value["fixed_N1M_candidate"], fixed)
            or any(value["fixed_N1M_candidate"].get(key) != fixed.get(key)
                   for key in ("alpha", "reference_kSlice", "raw_kSlice",
                               "k_floor", "k_floor_applied", "N_cap_applied"))
            or value.get("fixed_N1M_k_binding_applied") != clipped
            or value.get("testing_Xclean_loaded") is not False
            or value.get("testing_center_metrics") is not False):
        raise ValueError("saved final identity/selection mismatch")
    if value.get("data_identity") != named[0][1]["dataset_identity"]:
        raise ValueError("saved final has a different data identity")
    if current_runtime is None:
        if final_runtime != named[0][1].get("cuda_runtime"):
            raise ValueError("saved final runtime differs from its validation parent cells")
    elif (not runtime_compatible(final_runtime, current_runtime)
          or not all(runtime_compatible(record.get("cuda_runtime", {}), current_runtime)
                     for _, record in named)):
        raise ValueError("saved final/parent runtime is not an explicit compatible successor")
    if final_runtime.get("adapter_change_scope") in ORCHESTRATION_SUCCESSOR_SCOPES:
        if (value.get("selected_validation_metric_pass_contract") !=
                "separate-response-and-tangent-v1"
                or value.get("selected_validation_python_diagnostics", {}).get(
                    "metric_pass_contract") != "separate-response-and-tangent-v1"
                or any(candidate.get("python_diagnostics", {}).get("metric_pass_contract") !=
                       "separate-response-and-tangent-v1"
                       for candidate in value.get("candidates", []))):
            raise ValueError("successor final did not separate response and tangent metrics")
    if final_runtime.get("adapter_change_scope") == TERMINAL_STATE_SUCCESSOR_SCOPE:
        for candidate in value.get("candidates", []):
            requested = candidate.get("requested_K")
            effective = candidate.get("effective_K")
            carried = candidate.get("terminal_state_carry_forward")
            if (requested != candidate.get("K") or not isinstance(effective, int)
                    or effective < 0 or effective > requested
                    or carried is not (effective < requested)
                    or (carried and candidate.get("stopped_no_heavy") is not True)):
                raise ValueError("successor final has invalid terminal-state metadata")
    flags, hard = boundary_flags(profile, winner)
    if value.get("winner_edge_flags") != flags or value.get("winner_hard_boundary_flags") != hard:
        raise ValueError("saved final edge flags mismatch")
    for key in ("validation_relMSE_total", "validation_relMSE_coord", "validation_devTan",
                "testing_relMSE_total", "testing_relMSE_coord", "testing_devTan",
                "fixed_testing_relMSE_total", "fixed_testing_relMSE_coord", "fixed_testing_devTan"):
        if not isinstance(value.get(key), (int, float)) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError("missing/nonfinite final metric: " + key)
    if abs(value["validation_relMSE_total"] - winner["validation_relMSE_total"]) > 5e-8:
        raise ValueError("final validation no longer matches selected model")
    if len(value.get("candidates", [])) != 2:
        raise ValueError("both independent testing strategies are required")
    for candidate, expected, prefix in zip(value["candidates"], (winner, fixed), ("", "fixed_")):
        if not same_candidate(candidate, expected):
            raise ValueError("final candidate tuple mismatch")
        for metric in ("testing_relMSE_total", "testing_relMSE_coord", "testing_devTan"):
            if candidate[metric] != value[prefix + metric]:
                raise ValueError("final candidate metric mismatch")
    # Core provenance is already checked on every validation parent.
    if final_runtime.get("core_sha256", core_sha) != core_sha:
        raise ValueError("mixed runtime in saved final")
    return value


@contextmanager
def setup_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".running.lock"
    # Directory lock is shared with the MATLAB driver; flock additionally
    # protects Python recovery records. Never reclaim another owner's lock.
    lock.mkdir(exist_ok=False)
    try:
        with worker_lock(root):
            yield
    finally:
        lock.rmdir()


@contextmanager
def matrix_lock(root):
    """Same batch lock as the MATLAB driver; do not contend for its GPU."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".submillion.running.lock"
    lock.mkdir(exist_ok=False)
    try:
        yield
    finally:
        lock.rmdir()


def verify_or_create(path, value, keys=None, compatible_runtime=False,
                     legacy_optional=()):
    if path.exists():
        saved = read(path)
        for key in keys or value.keys():
            if key in legacy_optional and key not in saved:
                continue
            if (compatible_runtime and key == "cuda_runtime"
                    and runtime_compatible(saved.get(key), value[key])):
                continue
            if saved.get(key) != value[key]:
                raise ValueError("saved setup differs: %s (%s)" % (path, key))
    else:
        save(path, value)


def metric_cube(result, stats, K_grid, variance):
    cube = np.full((4, 3, 3), np.nan)
    for i, K in enumerate(K_grid):
        if K == 0:
            cube[i] = np.asarray(result["EvalMSE0"], dtype=float) / variance
        elif K <= stats.valid_iterations:
            cube[i] = np.asarray(result["EvalMSE"][K - 1], dtype=float) / variance
    return cube


def evaluate_candidate(estimator, data, candidate, view, grid_record=None,
                       include_tangent=True):
    """Evaluate response and tangent metrics in invariant, separate passes.

    The fused CUDA evaluator chooses its train tile width from all requested
    metric families. Requesting tangent geometry therefore changes the
    response prediction reduction order even though the response truth and
    model are unchanged. Stage2 selection is response-only, so replay that
    exact metric family first and obtain DevTan in a tangent-only pass. This
    preserves every fixed-order sum and avoids coupling response scores to an
    optional diagnostic metric.
    """
    started = time.perf_counter()
    prep = estimator.prepare(candidate["kSlice"])
    suffix = {"validation": "valid", "testing": "test"}[view]
    if grid_record is None:
        iterations = [candidate["K"]]
        eval_rhos = [candidate["eval_rho"]]
        eval_hs = [candidate["eval_h"]]
        position = None
    else:
        for key in ("kSlice", "h_scale", "rho_init", "bite"):
            if grid_record[key] != candidate[key]:
                raise ValueError("validation grid does not own selected candidate: " + key)
        k_position = grid_record["K_offsets"].index(candidate["K_offset"])
        if grid_record["K_grid"][k_position] != candidate["K"]:
            raise ValueError("validation grid K/offset mismatch")
        position = (grid_record["eval_rho_grid"].index(candidate["eval_rho"]),
                    grid_record["eval_h_grid"].index(candidate["eval_h"]))
        iterations = grid_record["K_grid"]
        eval_rhos = grid_record["eval_rho_grid"]
        eval_hs = grid_record["eval_h_grid"]

    def run(metric_family, y=None, tangent=None, requested_iterations=None):
        estimator.set_query(view + "_" + metric_family, data.arrays["X" + suffix],
                            y, tangent)
        return estimator.evaluate(
            h_scale=candidate["h_scale"], rho_init=candidate["rho_init"],
            bite=candidate["bite"],
            iterations=iterations if requested_iterations is None else requested_iterations,
            eval_rhos=eval_rhos, eval_hs=eval_hs)

    def terminal_result(metric_family, result, stats, diagnostic, *, y=None, tangent=None):
        requested = candidate["K"]
        valid = int(stats.valid_iterations)
        if requested <= valid:
            return result, requested, False, bool(getattr(stats, "stopped_no_heavy", 0)), diagnostic, 0.0
        if not bool(getattr(stats, "stopped_no_heavy", 0)):
            raise RuntimeError("evaluation stopped before selected K without no-heavy terminal state")
        # Once no heavy owner remains, subsequent Phase-2 updates are exact
        # no-ops. Re-evaluate the last reachable state explicitly so an
        # unexecuted output slot is never read or silently treated as zero.
        terminal, terminal_stats, terminal_diagnostic = run(
            metric_family, y=y, tangent=tangent, requested_iterations=[valid])
        if valid > int(terminal_stats.valid_iterations):
            raise RuntimeError("terminal-state replay stopped before its effective K")
        return (terminal, valid, True, True, terminal_diagnostic,
                terminal_diagnostic["cell_wall"])

    response, response_stats, response_diagnostic = run(
        "response_only", y=data.arrays["Y" + suffix])
    K = candidate["K"]
    (response, effective_K, terminal_carry, stopped_no_heavy,
     response_effective_diagnostic, response_replay_wall) = terminal_result(
        "response_only", response, response_stats, response_diagnostic,
        y=data.arrays["Y" + suffix])
    def metric(result, name, iteration=effective_K):
        value = np.asarray(result[name + "0"] if iteration == 0 else result[name][iteration - 1],
                           dtype=float)
        if position is not None:
            value = value[position]
        return value.reshape(-1)
    variance, total = data.variance(view)
    mse = metric(response, "EvalMSE")
    coordinate = metric(response, "EvalMSECoordinate")
    if not all(np.isfinite(a).all() for a in (mse, coordinate)):
        raise RuntimeError("selected model has nonfinite independent metrics")
    for name in ("EvalDevCen", "EvalDevCenTan"):
        if name in response and np.isfinite(metric(response, name)).any():
            raise RuntimeError("center truth must not participate in testing")
    tangent = None
    tangent_diagnostic = None
    tangent_effective_diagnostic = None
    tangent_wall = 0.0
    if include_tangent:
        tangent_result, tangent_stats, tangent_diagnostic = run(
            "tangent_only", tangent=data.arrays["T" + suffix])
        (tangent_result, tangent_effective_K, tangent_carry, tangent_stopped,
         tangent_effective_diagnostic, tangent_replay_wall) = terminal_result(
            "tangent_only", tangent_result, tangent_stats, tangent_diagnostic,
            tangent=data.arrays["T" + suffix])
        if (tangent_effective_K != effective_K or tangent_carry != terminal_carry
                or tangent_stopped != stopped_no_heavy):
            raise RuntimeError("response/tangent terminal-state mismatch")
        tangent = metric(tangent_result, "EvalDevTan")
        if not np.isfinite(tangent).all():
            raise RuntimeError("selected model has nonfinite tangent metric")
        for name in ("EvalDevCen", "EvalDevCenTan"):
            if name in tangent_result and np.isfinite(metric(tangent_result, name)).any():
                raise RuntimeError("center truth must not participate in testing")
        tangent_wall = tangent_diagnostic["cell_wall"] + tangent_replay_wall
    diagnostics = dict(metric_pass_contract="separate-response-and-tangent-v1",
                       requested_K=K, effective_K=effective_K,
                       terminal_state_carry_forward=terminal_carry,
                       stopped_no_heavy=stopped_no_heavy,
                       response=response_diagnostic,
                       response_effective=response_effective_diagnostic,
                       tangent=tangent_diagnostic,
                       tangent_effective=tangent_effective_diagnostic)
    return dict(candidate, requested_K=K, effective_K=effective_K,
        terminal_state_carry_forward=terminal_carry,
        stopped_no_heavy=stopped_no_heavy,
        name="", testing_relMSE_total=float(mse[0] / total),
        testing_relMSE_coord=float(np.mean(coordinate / variance)),
        testing_devTan=float(tangent[0]) if tangent is not None else None,
        testing_center_metrics=False, testing_Xclean_loaded=False, wall=time.perf_counter() - started,
        # A host-snapshot restore still regenerates/checks kNN and uploads P1.
        # Only an already active exact-k preparation has zero wall time.
        phase1_time=prep["wall_seconds"],
        phase1_reused=prep["reused"], phase1_source_time=response_diagnostic["phase1_wall"],
        phase1_preparation_wall=prep["wall_seconds"],
        phase2_time=response_diagnostic["cell_wall"] + response_replay_wall + tangent_wall,
        testing_reused_from="", python_diagnostics=diagnostics)


def refresh_csv(root):
    selected, fixed = [], []
    for path in sorted(root.glob("N*_selected_testing.json")):
        value = read(path)
        for rows, key, prefix, label in (
                (selected, "validation_winner", "", "per_N_validation_selected"),
                (fixed, "fixed_N1M_candidate", "fixed_", "fixed_N1M_validation_winner")):
            candidate = value[key]
            rows.append(dict(N=value["N"], candidate=label,
                **{k: candidate[k] for k in SELECTION_AXES},
                alpha=candidate.get("alpha"),
                alpha_candidates=json.dumps(candidate.get("alpha_candidates", [])),
                alpha_identifiable=candidate.get("alpha_identifiable"),
                k_floor=candidate.get("k_floor"),
                k_floor_applied=candidate.get("k_floor_applied"),
                validation_relMSE_total=candidate.get("validation_relMSE_total"),
                validation_devTan=value.get("validation_devTan") if not prefix else None,
                testing_relMSE_total=value[prefix + "testing_relMSE_total"],
                testing_devTan=value[prefix + "testing_devTan"],
                k_binding_applied=value["fixed_N1M_k_binding_applied" if prefix else "k_binding_applied"],
                source_json=path.name))
    for name, rows in (("selected_scaling.csv", selected), ("fixed_N1M_scaling.csv", fixed),
                       ("strategy_comparison.csv", selected + fixed)):
        if not rows:
            continue
        path = root / name
        temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        with temporary.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["N"], row["candidate"])))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def run_setup(calibration, *, dataset, output, library, runtime, n_values,
              resume=False, max_new_cells=None, device=0, retained_bytes=0,
              data_loader=load_canonical, estimator_factory=CudaCoreEstimator,
              overlap_cell_writes=True, memory_policy=None, phase1_cache_bytes=1024**3,
              prefetch_prefixes=True):
    calibration = normalize_calibration(calibration)
    setup = copy.deepcopy(calibration["setup"])
    setup.update(dataset=str(Path(dataset).resolve()), result_dir=str(Path(output).resolve()))
    root = Path(output) / "submillion_N_lt_1M"
    plans = {N: plan(calibration, N) for N in n_values}  # invalid K fails before CUDA/writes
    if root.exists() and not resume:
        raise ValueError("existing output requires explicit --resume")
    with (setup_lock(root), CellWriter(overlap_cell_writes) as writer,
          PrefixPrefetcher(prefetch_prefixes) as prefixes):
        fixed1m, _ = fixed_n1m_candidate(calibration["validation_winner"], 1_000_000, setup["D"])
        verify_or_create(root / "submillion_setup.json", dict(
            schema="mfm-setup-submillion-grid.v1", setup=setup, reference_N=REFERENCE_N,
            reference_profile=calibration["final_profile"], fixed_N1M_validation_winner=fixed1m,
            N_values=list(SUBMILLION_N_VALUES), timeout=[], interface="python_direct",
            backend="cuda_core", metric_sum_contract=CONTRACT, cuda_runtime=runtime,
            K_binding_contract=ITERATION_BINDING_CONTRACT,
            k_alpha_binding_contract=K_ALPHA_BINDING_CONTRACT),
            keys=("schema", "reference_N", "reference_profile", "fixed_N1M_validation_winner",
                  "N_values", "backend", "metric_sum_contract", "cuda_runtime",
                  "K_binding_contract", "k_alpha_binding_contract"), compatible_runtime=True,
            legacy_optional=("backend", "metric_sum_contract", "cuda_runtime",
                             "K_binding_contract"))
        existing, complete = {}, {}
        for N, (profile, _, clipped, _, K_grid, K_policy) in plans.items():
            directory = root / ("N%d" % N)
            legacy_n_optional = ["backend", "metric_sum_contract", "cuda_runtime"]
            if (not any(K_policy["hard_lower_bound_applied"])
                    and not K_policy["fixed_N1M_hard_lower_bound_applied"]):
                legacy_n_optional.append("K_binding_policy")
            verify_or_create(directory / "N_setup.json", dict(
                schema="mfm-setup-submillion-N.v1", setup_id=setup["id"], N=N,
                reference_profile=calibration["final_profile"], effective_profile=profile,
                k_binding_applied=clipped, K_binding_policy=K_policy,
                k_alpha_binding_contract=K_ALPHA_BINDING_CONTRACT,
                testing_used_during_grid=False, timeout=[], backend="cuda_core",
                metric_sum_contract=CONTRACT, cuda_runtime=runtime),
                keys=("schema", "setup_id", "N", "reference_profile", "effective_profile",
                      "k_binding_applied", "K_binding_policy", "k_alpha_binding_contract",
                      "testing_used_during_grid",
                      "timeout", "backend", "metric_sum_contract", "cuda_runtime"),
                compatible_runtime=True,
                legacy_optional=legacy_n_optional)
            specs = list(cell_specs(N, profile, K_grid, K_policy))
            existing[N] = existing_cells(directory, specs, setup["id"], runtime)
            target = root / ("N%d_selected_testing.json" % N)
            if target.exists():
                complete[N] = validate_final(read(target), calibration, N, profile, existing[N],
                                             runtime, K_policy)
        if len(complete) == len(plans):
            return dict(state="complete_noop", new_cells=0, cuda_contexts=0,
                        completed_N=sorted(complete))
        load_started = time.perf_counter()
        data = data_loader(dataset, setup)
        prefixes.bind(data)
        data_manifest = dict(schema="mfm-stage2-python-inputs.v1", identity=data.identity,
                             reference_N=1_000_000,
                             noise_covariance_count_contract="min(N,1000000)",
                             arrays_sha256=data.hashes)
        verify_or_create(root / "python_inputs.json", data_manifest)
        load_wall = time.perf_counter() - load_started
        new_cells = contexts = 0
        for N, (profile, fixed, clipped, fixed_clipped, K_grid, K_policy) in plans.items():
            if N in complete:
                continue
            n_data, n_hashes, prefix_timing = prefixes.take(N)
            prefix_wall = prefix_timing["worker_seconds"]
            directory = root / ("N%d" % N)
            verify_or_create(directory / "python_inputs.json", dict(
                schema="mfm-stage2-python-N-inputs.v1", N=N,
                noise_covariance_count=min(N, 1_000_000),
                identity=n_data.identity, arrays_sha256=n_hashes))
            named = existing[N]
            for name, record in named:
                validate_cell(record, dict(cell_specs(N, profile, K_grid, K_policy))[name], setup["id"],
                              runtime, n_data.identity, n_hashes)
            required = expected_grid_size(profile)
            if max_new_cells is not None and new_cells >= max_new_cells and len(named) < required:
                break
            ordered_N = list(plans)
            next_N = next((candidate for candidate in ordered_N[ordered_N.index(N) + 1:]
                           if candidate not in complete), None)
            remaining_cells = required - len(named)
            if (next_N is not None
                    and (max_new_cells is None
                         or new_cells + remaining_cells < max_new_cells
                         or len(existing[next_N]) == expected_grid_size(plans[next_N][0]))):
                prefixes.submit(next_N)
            before_N = time.time()
            with estimator_factory(library, n_data.arrays["X"], n_data.arrays["Y"],
                    dataset_id=setup["id"] + ":N" + str(N), device=device,
                    retained_bytes=retained_bytes, memory_policy=memory_policy,
                    phase1_cache_bytes=phase1_cache_bytes) as estimator:
                contexts += 1
                _, variance = n_data.variance("validation")
                estimator.set_query("validation_response_only", n_data.arrays["Xvalid"], n_data.arrays["Yvalid"])
                names = {name for name, _ in named}
                for name, spec in cell_specs(N, profile, K_grid, K_policy):
                    writer.check()
                    if name in names:
                        continue
                    if max_new_cells is not None and new_cells >= max_new_cells:
                        break
                    prep = estimator.prepare(spec["kSlice"])
                    begin = time.time()
                    result, stats, diagnostic = estimator.evaluate(
                        h_scale=spec["h_scale"], rho_init=spec["rho_init"], bite=spec["bite"],
                        iterations=K_grid, eval_rhos=profile["eval_rhos"], eval_hs=profile["eval_hs"])
                    cube = metric_cube(result, stats, K_grid, variance)
                    record = dict(spec, schema=CELL_SCHEMA, backend="cuda_core", dataset_identity=n_data.identity,
                        input_arrays_sha256=n_hashes,
                        D=setup["D"], MM=2, MY=6, validation_count=5000, testing_used=False,
                        testing_Xclean_loaded=False, k_over_N=spec["alpha"],
                        validation_relMSE_total=cube.tolist(),
                        **cube_winner(cube, spec), phase1_owner_fraction=diagnostic["phase1_owner_fraction"],
                        phase1_h=diagnostic["phase1_h"], phase1_wall=diagnostic["phase1_wall"],
                        cell_wall=diagnostic["cell_wall"], valid_phase2_iterations=stats.valid_iterations,
                        metric_sum_contract=CONTRACT, cuda_runtime=runtime, completed_at=now(),
                        python_diagnostics=dict(diagnostic, preparation=prep, start_epoch=begin,
                                                end_epoch=time.time()))
                    record = json_value(record)
                    try:
                        validate_cell(record, spec, setup["id"], runtime,
                                      n_data.identity, n_hashes)
                    except Exception as error:
                        # Preserve failed numerical output outside the reusable
                        # cell set. A failed diagnostic never becomes a noop.
                        save(root / "failures" / (name + "." + uuid.uuid4().hex + ".json"),
                             dict(record=record, error=str(error), observed_at=now()))
                        raise
                    writer.submit(root / ("N%d" % N) / "cells" / name, record)
                    named.append((name, record)); names.add(name); new_cells += 1
                    print(json.dumps(dict(event="cell_queued" if overlap_cell_writes else "cell_complete", setup=setup["id"], N=N,
                        cell=name, completed=len(named), total=required)), flush=True)
                writer.flush()
                if len(named) < required:
                    break  # atomic budget boundary; NO selected testing on a partial grid
                selected = stage2_winner(named, profile)
                fixed["validation_relMSE_total"] = candidate_score(named, fixed)
                if fixed["validation_relMSE_total"] is None:
                    measured = evaluate_candidate(estimator, n_data, fixed, "validation",
                                                  include_tangent=False)
                    fixed["validation_relMSE_total"] = measured["testing_relMSE_total"]
                selected_grid = next(record for _, record in named
                    if all(record[key] == selected[key]
                           for key in ("kSlice", "h_scale", "rho_init", "bite")))
                # Re-evaluate validation with the exact parent-cell launch
                # shape.  Compacting 4x3x3 to 1x1 changes floating-point
                # execution geometry and can move a fixed-order metric by an
                # ulp even when P1 state and predictions are otherwise the
                # same.  Testing remains an independent compact call.
                validation = evaluate_candidate(estimator, n_data, selected, "validation",
                                                grid_record=selected_grid)
                testing = evaluate_candidate(estimator, n_data, selected, "testing")
                testing["name"] = "per_N_validation_selected"
                if same_candidate(selected, fixed):
                    fixed_testing = copy.deepcopy(testing)
                    fixed_testing["testing_reused_from"] = testing["name"]
                    for key, item in fixed.items():
                        if key.startswith("alpha") or key in (
                                "reference_kSlice", "raw_kSlice", "k_floor",
                                "k_floor_applied", "N_cap_applied"):
                            fixed_testing[key] = item
                else:
                    fixed_testing = evaluate_candidate(estimator, n_data, fixed, "testing")
                fixed_testing["name"] = "fixed_N1M_validation_winner"
                flags, hard = boundary_flags(profile, selected)
                final = dict(schema=FINAL_SCHEMA, setup=setup, data_identity=n_data.identity,
                    input_arrays_sha256=n_hashes, N=N,
                    backend="cuda_core", metric_sum_contract=CONTRACT,
                    reference_profile=calibration["final_profile"], effective_profile=profile,
                    k_binding_applied=clipped, validation_winner=selected, fixed_N1M_candidate=fixed,
                    fixed_N1M_k_binding_applied=fixed_clipped, K_binding_policy=K_policy,
                    k_alpha_binding_contract=K_ALPHA_BINDING_CONTRACT,
                    winner_edge_flags=flags,
                    winner_hard_boundary_flags=hard, validation_relMSE_total=validation["testing_relMSE_total"],
                    validation_relMSE_coord=validation["testing_relMSE_coord"],
                    validation_devTan=validation["testing_devTan"],
                    selected_validation_phase1_reused=validation["phase1_reused"],
                    selected_validation_phase1_wall=validation["phase1_preparation_wall"],
                    selected_validation_metric_pass_contract=
                        validation["python_diagnostics"]["metric_pass_contract"],
                    selected_validation_python_diagnostics=validation["python_diagnostics"],
                    fixed_validation_relMSE_total=fixed["validation_relMSE_total"],
                    candidates=[testing, fixed_testing], testing_Xclean_loaded=False,
                    testing_center_metrics=False, completed_at=now(), cuda_runtime=runtime,
                    python_timeline=dict(start_epoch=before_N, end_epoch=time.time(),
                        data_load_seconds=load_wall, per_N_data_prepare_seconds=prefix_wall,
                        per_N_data_prepare_source=prefix_timing["source"],
                        per_N_data_prepare_foreground_wait_seconds=
                            prefix_timing["foreground_wait_seconds"],
                        phase1_preparations=estimator.preparation_count,
                        phase1_restores=getattr(estimator, 'phase1_restore_count', 0),
                        phase1_host_cache_budget=phase1_cache_bytes,
                        cell_writer=writer.stats(), prefix_prefetch=prefixes.stats()))
                for prefix, item in (("", testing), ("fixed_", fixed_testing)):
                    for key in ("testing_relMSE_total", "testing_relMSE_coord", "testing_devTan"):
                        final[prefix + key] = item[key]
                final = json_value(final)
                validate_final(final, calibration, N, profile, named,
                               runtime, K_policy)
                save(root / ("N%d_selected_testing.json" % N), final)
                complete[N] = final
                refresh_csv(root)
                print(json.dumps(dict(event="N_complete", setup=setup["id"], N=N)), flush=True)
        report = dict(state="complete" if len(complete) == 27 else "paused", new_cells=new_cells,
                      cuda_contexts=contexts, completed_N=sorted(complete), interface="python_direct",
                      cell_writer=writer.stats(), prefix_prefetch=prefixes.stats())
        save(root / "python_status.json", dict(report, observed_at=now()))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--matrix-lock-root", type=Path,
                        help="optional batch-lock root; use one distinct root per independently "
                             "scheduled setup while setup locks still prevent duplicates")
    parser.add_argument("--setups", required=True, help="comma-separated setup ids")
    parser.add_argument("--n-values", default=",".join(map(str, SUBMILLION_N_VALUES)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retained-gib", type=float, default=0,
                        help="resident cache ceiling; 0 sizes it from actual free GPU memory")
    parser.add_argument("--phase1-cache-gib", type=float, default=1,
                        help="bounded initial P1 host snapshots within N; 0 disables; no N*k index cache")
    parser.add_argument("--gpu-working-gib", type=float, default=0,
                        help="optional P2 working-byte ceiling; 0 uses actual free GPU memory")
    parser.add_argument("--pinned-gib", type=float, default=2,
                        help="maximum two-slot pinned staging capacity")
    parser.add_argument("--phase1-working-gib", type=float, default=0,
                        help="optional P1/kNN allocation ceiling; 0 uses actual free GPU memory")
    parser.add_argument("--phase1-index-heap-gib", type=float, default=8,
                        help="larger exact host indices use file-backed scratch")
    parser.add_argument("--phase1-index-dir", type=Path,
                        help="local scratch filesystem for large host indices; no dataset transfer")
    parser.add_argument("--cell-write-mode", choices=("sync", "overlap"), default="overlap")
    args = parser.parse_args(argv)
    Ns = list(dict.fromkeys(int(v) for v in args.n_values.split(",")))
    if (not Ns or (args.max_new_cells is not None and args.max_new_cells < 1)
            or not all(math.isfinite(v) for v in (args.retained_gib, args.gpu_working_gib, args.pinned_gib,
                args.phase1_cache_gib, args.phase1_working_gib, args.phase1_index_heap_gib))
            or args.retained_gib < 0 or args.gpu_working_gib < 0 or args.pinned_gib < 0
            or args.phase1_cache_gib < 0 or args.phase1_working_gib < 0 or args.phase1_index_heap_gib < 0):
        parser.error("nonempty N list and positive cell/memory budgets required")
    runtime = validate_runtime(args.library, args.runtime_json)
    todo = []
    for setup_id in args.setups.split(","):
        if not re.fullmatch(r"bumpy_sphere_v[123]_D(?:10|20)_Y[0-4]", setup_id):
            parser.error("invalid setup id")
        calibration = normalize_calibration(read(
            args.calibration_root / (setup_id + "__noise1pct") / "calibration_final.json"))
        if calibration["setup"]["id"] != setup_id:
            raise ValueError("calibration setup identity mismatch")
        for N in Ns:
            plan(calibration, N)
        dataset = args.dataset_root / Path(calibration["setup"]["dataset"]).name
        if not dataset.is_file():
            raise ValueError("canonical dataset missing on this host: " + str(dataset))
        todo.append((setup_id, calibration, dataset))
    with matrix_lock(args.matrix_lock_root or args.result_root):
        for setup_id, calibration, dataset in todo:
            result = run_setup(calibration, dataset=dataset,
                output=args.result_root / (setup_id + "__noise1pct"), library=args.library,
                runtime=runtime, n_values=Ns, resume=args.resume, max_new_cells=args.max_new_cells,
                device=args.device, retained_bytes=int(args.retained_gib * 1024**3),
                overlap_cell_writes=args.cell_write_mode == 'overlap',
                phase1_cache_bytes=int(args.phase1_cache_gib * 1024**3),
                memory_policy=MemoryPolicy(working_bytes=int(args.gpu_working_gib * 1024**3),
                    pinned_bytes=int(args.pinned_gib * 1024**3),
                    phase1_working_bytes=int(args.phase1_working_gib * 1024**3),
                    phase1_index_heap_bytes=int(args.phase1_index_heap_gib * 1024**3),
                    phase1_spill_directory=str(args.phase1_index_dir) if args.phase1_index_dir else None))
            print(json.dumps(dict(setup=setup_id, **result)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
