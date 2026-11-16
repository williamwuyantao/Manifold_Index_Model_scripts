"""Validation-only screen for one N-invariant seven-parameter model.

Every named probe fixes ``alpha``, ``K_offset``, ``h_scale``, ``rho_init``,
``bite``, ``eval_rho`` and ``eval_h`` across all requested N.  Only the
algorithmic quantities k(N) and K(N) are derived.  This runner never loads the
testing truths and never selects a different parameter tuple for each N.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

import stage2_cuda_core as stage2
from calibration_protocol import (K_ALPHA_BINDING_CONTRACT,
                                  SUBMILLION_N_VALUES,
                                  fixed_n1m_candidate, matlab_round)
from calibration_protocol import now, write_atomic
from cuda_core_data import load_canonical
from estimator import CudaCoreEstimator
from stage2_memory import MemoryPolicy


PLAN_SCHEMA = "mfm-fixed-seven-boundary-screen.v1"
RECORD_SCHEMA = "mfm-fixed-seven-validation.v1"
SUMMARY_SCHEMA = "mfm-fixed-seven-validation-summary.v1"
PARAMETERS = ("alpha", "K_offset", "h_scale", "rho_init", "bite",
              "eval_rho", "eval_h")
SCREEN_STAGES = ("stage_A", "stage_B", "stage_C", "stage_D", "stage_E")


def read(path):
    return stage2.read(path)


def save(path, value):
    write_atomic(path, stage2.json_value(value))


def normalize_parameters(value):
    if set(value) != set(PARAMETERS):
        raise ValueError("a fixed probe must contain exactly seven parameters")
    result = {}
    for name in PARAMETERS:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("nonfinite fixed parameter: " + name)
        result[name] = float(item)
    if result["alpha"] <= 0 or result["alpha"] > 1:
        raise ValueError("alpha must be in (0,1]")
    for name in ("h_scale", "rho_init", "eval_rho", "eval_h"):
        if result[name] <= 0:
            raise ValueError("positive fixed parameter required: " + name)
    for name in ("K_offset", "bite"):
        if result[name] != int(result[name]):
            raise ValueError("integer fixed parameter required: " + name)
        result[name] = int(result[name])
    if result["bite"] < 0:
        raise ValueError("bite cannot be negative")
    reference_k = matlab_round(result["alpha"] * 1_000_000)
    if reference_k < 1 or reference_k > 1_000_000:
        raise ValueError("alpha does not map to a feasible N1M k")
    exact_alpha = reference_k / 1_000_000
    if abs(result["alpha"] - exact_alpha) > 5e-13:
        raise ValueError("alpha must identify an exact integer N1M k")
    result["alpha"] = exact_alpha
    return result


def candidate_at_n(parameters, N, D):
    parameters = normalize_parameters(parameters)
    reference = dict(parameters, kSlice=matlab_round(parameters["alpha"] * 1_000_000))
    candidate, clipped = fixed_n1m_candidate(reference, int(N), int(D))
    for name in PARAMETERS:
        if candidate[name] != parameters[name]:
            raise RuntimeError("fixed parameter changed while binding N: " + name)
    return candidate, clipped


def load_plan(path, setup_ids, n_values):
    plan = read(path)
    screen_stage = plan.get("screen_stage", "stage_A")
    if (plan.get("schema") != PLAN_SCHEMA
            or plan.get("selection_data") != "validation only"
            or screen_stage not in SCREEN_STAGES
            or plan.get("protocol", {}).get("explicit_K0_as_tunable_axis") is not False
            or plan.get("protocol", {}).get("testing_used_before_final_selection") is not False):
        raise ValueError("incompatible fixed-hyperparameter plan")
    indexed = {}
    for item in plan.get("plans", []):
        setup = item.get("setup")
        if not isinstance(setup, str) or setup in indexed:
            raise ValueError("duplicate or invalid setup in plan")
        if item.get("anchor_N") != plan.get("anchor_N"):
            raise ValueError("setup anchor-N list differs from plan")
        probes = []
        names = set()
        for probe in item.get("probes", []):
            name = probe.get("name")
            if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                    or name in names):
                raise ValueError("duplicate or unsafe probe name")
            names.add(name)
            probes.append(dict(name=name,
                               parameters=normalize_parameters(probe.get("parameters", {}))))
        if not probes:
            raise ValueError("setup has no fixed probes")
        indexed[setup] = dict(item, probes=probes, screen_stage=screen_stage)
    if set(setup_ids) - set(indexed):
        raise ValueError("requested setup is absent from plan")
    if not set(n_values).issubset(set(plan.get("anchor_N", []))):
        raise ValueError("requested N is absent from the fixed screen")
    return plan, {setup: indexed[setup] for setup in setup_ids}


def runtime_for_screen(library, runtime_json):
    runtime = stage2.validate_runtime(library, runtime_json)
    return dict(runtime,
                fixed_screen_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def expected_record(setup_id, N, probe, candidate, clipped, runtime):
    return dict(schema=RECORD_SCHEMA, setup_id=setup_id, N=int(N),
                probe=probe["name"], fixed_parameters=probe["parameters"],
                candidate=candidate, k_binding_applied=bool(clipped),
                k_alpha_binding_contract=K_ALPHA_BINDING_CONTRACT,
                K_binding_contract=stage2.ITERATION_BINDING_CONTRACT,
                backend="cuda_core", metric_sum_contract=stage2.CONTRACT,
                validation_count=5000, testing_used=False,
                testing_Xclean_loaded=False, cuda_runtime=runtime)


def validate_record(record, expected, identity=None, hashes=None):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError("saved fixed-screen record differs: " + key)
    score = record.get("validation_relMSE_total")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or score <= 0):
        raise ValueError("invalid validation score")
    if record.get("validation_relMSE_coord") is not None:
        raise ValueError("screen must not request unused coordinate metrics")
    if record.get("validation_devTan") is not None:
        raise ValueError("screen must not request tangent metrics")
    if not record.get("completed_at"):
        raise ValueError("fixed-screen record lacks completion time")
    if identity is not None and record.get("dataset_identity") != identity:
        raise ValueError("fixed-screen data identity changed")
    if hashes is not None and record.get("input_arrays_sha256") != hashes:
        raise ValueError("fixed-screen input arrays changed")
    diagnostics = record.get("python_diagnostics", {})
    if diagnostics.get("metric_pass_contract") != "separate-response-and-tangent-v1":
        raise ValueError("fixed-screen response metric contract changed")
    return record


@contextmanager
def setup_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".running.lock"
    lock.mkdir(exist_ok=False)
    try:
        yield
    finally:
        lock.rmdir()


def record_path(root, N, probe_name):
    return root / ("N%d" % N) / (probe_name + ".json")


def existing_for_n(root, setup_id, D, N, probes, runtime):
    found = {}
    expected_names = {probe["name"] for probe in probes}
    directory = root / ("N%d" % N)
    if directory.exists():
        unexpected = {path.stem for path in directory.glob("*.json")} - expected_names
        if unexpected:
            raise ValueError("unexpected fixed-screen record: " + sorted(unexpected)[0])
    for probe in probes:
        candidate, clipped = candidate_at_n(probe["parameters"], N, D)
        expected = expected_record(setup_id, N, probe, candidate, clipped, runtime)
        path = record_path(root, N, probe["name"])
        if path.exists():
            found[probe["name"]] = validate_record(read(path), expected)
    return found


def summarize(root, setup_id, probes, n_values, runtime, screen_stage="stage_A"):
    rows = []
    by_n = {}
    for N in n_values:
        values = []
        for probe in probes:
            record = read(record_path(root, N, probe["name"]))
            score = record["validation_relMSE_total"]
            values.append((probe["name"], score))
            rows.append(record)
        by_n[N] = min(score for _, score in values)
    objectives = []
    for probe in probes:
        records = [record for record in rows if record["probe"] == probe["name"]]
        scores = [record["validation_relMSE_total"] for record in records]
        ratios = [score / by_n[record["N"]] for score, record in zip(scores, records)]
        objectives.append(dict(name=probe["name"], parameters=probe["parameters"],
            max_regret_fraction=max(ratios) - 1,
            geometric_mean_regret_fraction=math.exp(sum(math.log(v) for v in ratios) / len(ratios)) - 1,
            arithmetic_mean_regret_fraction=sum(ratios) / len(ratios) - 1,
            mean_validation_relMSE_total=sum(scores) / len(scores),
            worst_N=records[max(range(len(ratios)), key=ratios.__getitem__)]["N"]))
    objectives.sort(key=lambda item: (item["max_regret_fraction"],
                                      item["geometric_mean_regret_fraction"], item["name"]))
    summary = dict(schema=SUMMARY_SCHEMA, setup_id=setup_id, screen_stage=screen_stage,
        selection_data="validation only", testing_used=False,
        fixed_parameters_across_N=list(PARAMETERS), derived_by_N=["kSlice", "K"],
        explicit_K0_as_tunable_axis=False, N_values=list(n_values),
        completed_evaluations=len(rows), objectives=objectives,
        stage_winner=objectives[0], cuda_runtime=runtime, completed_at=now())
    summary[screen_stage + "_winner"] = objectives[0]
    save(root / (screen_stage + "_summary.json"), summary)
    return summary


def run_setup(calibration, plan, *, dataset, output, library, runtime, n_values,
              resume=False, max_new_evaluations=None, device=0, retained_bytes=0,
              phase1_cache_bytes=1024**3, memory_policy=None,
              data_loader=load_canonical, estimator_factory=CudaCoreEstimator,
              prefetch_prefixes=True):
    calibration = stage2.normalize_calibration(calibration)
    setup = calibration["setup"]
    if setup["id"] != plan["setup"] or plan["dimension"] != "D%d" % setup["D"]:
        raise ValueError("fixed-screen setup/dimension mismatch")
    probes = plan["probes"]
    screen_stage = plan.get("screen_stage", "stage_A")
    root = Path(output)
    if root.exists() and not resume:
        raise ValueError("existing fixed-screen output requires --resume")
    existing = {N: existing_for_n(root, setup["id"], setup["D"], N, probes, runtime)
                for N in n_values}
    if all(len(existing[N]) == len(probes) for N in n_values):
        summary = summarize(root, setup["id"], probes, n_values, runtime, screen_stage)
        return dict(state="complete_noop", new_evaluations=0, cuda_contexts=0,
                    completed_N=list(n_values), stage_winner=summary["stage_winner"])

    with setup_lock(root):
        manifest = dict(schema="mfm-fixed-seven-screen-setup.v1", setup=setup,
            plan_schema=PLAN_SCHEMA, plan_setup=plan, N_values=list(n_values),
            selection_data="validation only", testing_used=False,
            fixed_parameters_across_N=list(PARAMETERS), derived_by_N=["kSlice", "K"],
            cuda_runtime=runtime)
        manifest_keys = ("schema", "setup", "plan_schema", "plan_setup", "N_values",
                         "selection_data", "testing_used", "fixed_parameters_across_N",
                         "derived_by_N", "cuda_runtime")
        if screen_stage != "stage_A":
            manifest["screen_stage"] = screen_stage
            manifest_keys += ("screen_stage",)
        stage2.verify_or_create(root / "screen_setup.json", manifest, keys=manifest_keys)
        data = data_loader(dataset, setup, include_testing=False)
        if any(key in data.arrays for key in ("Xtest", "Ytest", "Ttest")):
            raise ValueError("validation-only screen loaded testing arrays")
        new = contexts = 0
        pending_N = [N for N in n_values if len(existing[N]) < len(probes)]
        with stage2.PrefixPrefetcher(prefetch_prefixes) as prefixes:
            prefixes.bind(data)
            for position, N in enumerate(pending_N):
                if max_new_evaluations is not None and new >= max_new_evaluations:
                    break
                n_data, hashes, prefix_timing = prefixes.take(N)
                if position + 1 < len(pending_N):
                    prefixes.submit(pending_N[position + 1])
                todo = []
                for order, probe in enumerate(probes):
                    if probe["name"] in existing[N]:
                        validate_record(existing[N][probe["name"]],
                            expected_record(setup["id"], N, probe,
                                *candidate_at_n(probe["parameters"], N, setup["D"]), runtime),
                            n_data.identity, hashes)
                        continue
                    candidate, clipped = candidate_at_n(probe["parameters"], N, setup["D"])
                    todo.append((candidate["kSlice"], order, probe, candidate, clipped))
                todo.sort(key=lambda item: (item[0], item[1]))
                with estimator_factory(library, n_data.arrays["X"], n_data.arrays["Y"],
                        dataset_id=setup["id"] + ":fixed-screen:N" + str(N),
                        device=device, retained_bytes=retained_bytes,
                        memory_policy=memory_policy,
                        phase1_cache_bytes=phase1_cache_bytes) as estimator:
                    contexts += 1
                    for _, _, probe, candidate, clipped in todo:
                        if max_new_evaluations is not None and new >= max_new_evaluations:
                            break
                        started = time.time()
                        measured = stage2.evaluate_candidate(
                            estimator, n_data, candidate, "validation", include_tangent=False)
                        expected = expected_record(setup["id"], N, probe, candidate, clipped, runtime)
                        record = dict(expected, dataset_identity=n_data.identity,
                            input_arrays_sha256=hashes,
                            validation_relMSE_total=measured["testing_relMSE_total"],
                            validation_relMSE_coord=None, validation_devTan=None,
                            requested_K=measured["requested_K"], effective_K=measured["effective_K"],
                            terminal_state_carry_forward=measured["terminal_state_carry_forward"],
                            stopped_no_heavy=measured["stopped_no_heavy"],
                            phase1_time=measured["phase1_time"],
                            phase1_reused=measured["phase1_reused"],
                            phase2_time=measured["phase2_time"], wall=measured["wall"],
                            prefix_timing=prefix_timing,
                            python_diagnostics=measured["python_diagnostics"],
                            started_at_epoch=started, completed_at=now())
                        validate_record(record, expected, n_data.identity, hashes)
                        save(record_path(root, N, probe["name"]), record)
                        existing[N][probe["name"]] = record
                        new += 1
                        print(json.dumps(dict(event="fixed_validation_complete",
                            setup=setup["id"], N=N, probe=probe["name"],
                            completed=len(existing[N]), total=len(probes))), flush=True)
                if len(existing[N]) < len(probes):
                    break
        complete_N = [N for N in n_values if len(existing[N]) == len(probes)]
        result = dict(state="complete" if len(complete_N) == len(n_values) else "paused",
                      new_evaluations=new, cuda_contexts=contexts,
                      completed_N=complete_N, prefix_prefetch=prefixes.stats())
        if result["state"] == "complete":
            result["stage_winner"] = summarize(
                root, setup["id"], probes, n_values, runtime, screen_stage)["stage_winner"]
        save(root / "screen_status.json", dict(result, observed_at=now()))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--setups", required=True)
    parser.add_argument("--n-values", default="1000,5000,9000,10000,50000,90000,100000,500000,900000")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-evaluations", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retained-gib", type=float, default=0)
    parser.add_argument("--phase1-cache-gib", type=float, default=1)
    parser.add_argument("--gpu-working-gib", type=float, default=0)
    parser.add_argument("--pinned-gib", type=float, default=2)
    parser.add_argument("--phase1-working-gib", type=float, default=0)
    parser.add_argument("--phase1-index-heap-gib", type=float, default=8)
    parser.add_argument("--phase1-index-dir", type=Path)
    args = parser.parse_args(argv)
    setups = [value for value in args.setups.split(",") if value]
    n_values = list(dict.fromkeys(int(value) for value in args.n_values.split(",")))
    if (not setups or not n_values or not set(n_values).issubset(SUBMILLION_N_VALUES)
            or (args.max_new_evaluations is not None and args.max_new_evaluations < 1)):
        parser.error("nonempty setup/canonical-N lists and positive budget required")
    for value in (args.retained_gib, args.phase1_cache_gib, args.gpu_working_gib,
                  args.pinned_gib, args.phase1_working_gib, args.phase1_index_heap_gib):
        if not math.isfinite(value) or value < 0:
            parser.error("memory budgets must be finite and nonnegative")
    _, plans = load_plan(args.plan_json, setups, n_values)
    runtime = runtime_for_screen(args.library, args.runtime_json)
    work = []
    for setup_id in setups:
        calibration = stage2.normalize_calibration(read(
            args.calibration_root / (setup_id + "__noise1pct") / "calibration_final.json"))
        if calibration["setup"]["id"] != setup_id:
            raise ValueError("calibration setup identity mismatch")
        dataset = args.dataset_root / Path(calibration["setup"]["dataset"]).name
        if not dataset.is_file():
            raise ValueError("canonical dataset missing: " + str(dataset))
        work.append((setup_id, calibration, dataset))
    memory = MemoryPolicy(working_bytes=int(args.gpu_working_gib * 1024**3),
        pinned_bytes=int(args.pinned_gib * 1024**3),
        phase1_working_bytes=int(args.phase1_working_gib * 1024**3),
        phase1_index_heap_bytes=int(args.phase1_index_heap_gib * 1024**3),
        phase1_spill_directory=str(args.phase1_index_dir) if args.phase1_index_dir else None)
    remaining = args.max_new_evaluations
    for setup_id, calibration, dataset in work:
        result = run_setup(calibration, plans[setup_id], dataset=dataset,
            output=args.result_root / (setup_id + "__noise1pct"),
            library=args.library, runtime=runtime, n_values=n_values,
            resume=args.resume, max_new_evaluations=remaining, device=args.device,
            retained_bytes=int(args.retained_gib * 1024**3),
            phase1_cache_bytes=int(args.phase1_cache_gib * 1024**3), memory_policy=memory)
        print(json.dumps(dict(setup=setup_id, **result)), flush=True)
        if remaining is not None:
            remaining -= result["new_evaluations"]
            if remaining <= 0:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
