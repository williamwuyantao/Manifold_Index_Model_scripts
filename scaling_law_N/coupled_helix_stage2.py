"""Cross-N fixed-seven selection and testing for the coupled-helix surface.

Reuses the production fixed-parameter screen, selected-curve runner, exact
k/K binding, memory policy and response/tangent evaluator. The application
adapter reads the existing coupled-helix H5 without changing its 1% noise.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

MAP = "natural_exponential3"
SETUP = "coupled_helix_D20_" + MAP
MAPS = ("natural_polynomial3", "natural_rational3", "natural_exponential3")
FIELDS = ("alpha", "K_offset", "h_scale", "rho_init", "bite", "eval_rho", "eval_h")
N_VALUES = [1000, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 30000,
            40000, 50000, 70000, 100000, 200000, 300000, 400000, 500000, 700000]
ANCHORS = [1000, 5000, 10000, 50000, 100000, 300000, 500000, 700000]
PROFILE_FIELDS = dict(alpha="k_floors", K_offset="K_offsets", h_scale="h_scales",
                      rho_init="rho_inits", bite="bites", eval_rho="eval_rhos", eval_h="eval_hs")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(parameters):
    return tuple(parameters[field] for field in FIELDS)


def initial_probes(calibration):
    winner = calibration["validation_winner"]
    baseline = {field: winner[field] for field in FIELDS}
    axes = {field: [value / 1e6 if field == "alpha" else value
                    for value in calibration["final_profile"][column]]
            for field, column in PROFILE_FIELDS.items()}
    probes = [dict(name="n1m_winner", parameters=baseline)]
    for field in FIELDS:
        for index, value in enumerate(axes[field]):
            if value != baseline[field]:
                probes.append(dict(name="axis_%s_%d" % (field, index),
                                   parameters=dict(baseline, **{field: value})))
    return axes, probes


def rank_scores(probes, scores, n_values):
    denominators = {n: min(scores[probe["name"]][n] for probe in probes) for n in n_values}
    if any(not math.isfinite(value) or value <= 0 for row in scores.values() for value in row.values()):
        raise ValueError("invalid validation score")
    ranked = []
    for probe in probes:
        values = scores[probe["name"]]
        ratios = [values[n] / denominators[n] for n in n_values]
        ranked.append(dict(name=probe["name"], parameters=probe["parameters"],
            max_regret_fraction=max(ratios)-1,
            geometric_mean_regret_fraction=math.exp(sum(map(math.log, ratios))/len(ratios))-1,
            mean_validation_relMSE_total=sum(values[n] for n in n_values)/len(n_values),
            validation_relMSE_total_by_N={str(n): values[n] for n in n_values}))
    return sorted(ranked, key=lambda row: (row["max_regret_fraction"],
        row["geometric_mean_regret_fraction"], key(row["parameters"])))


def interaction_probes(axes, probes, ranking):
    seen = {key(probe["parameters"]) for probe in probes}
    result = []
    center = ranking[0]["parameters"]

    def add(name, parameters):
        if key(parameters) not in seen:
            seen.add(key(parameters))
            result.append(dict(name=name, parameters=parameters))

    for field in FIELDS:
        for index, value in enumerate(axes[field]):
            add("refine_%s_%d" % (field, index), dict(center, **{field: value}))
    baseline = probes[0]["parameters"]
    donors = [row for row in ranking if row["name"].startswith("axis_")][:3]
    for count in (2, 3):
        for group in combinations(donors, count):
            candidate = dict(baseline)
            for donor in group:
                for field in FIELDS:
                    if donor["parameters"][field] != baseline[field]:
                        candidate[field] = donor["parameters"][field]
            add("interaction_" + "__".join(row["name"] for row in group), candidate)
    return result


def load_data(dataset, setup, *, include_testing=True):
    from cuda_core_data import Stage2Data
    with h5py.File(dataset, "r") as source:
        identity = json.loads(source.attrs["identity"])
        if (source.attrs["state"] != "complete" or identity["schema"] != "mfm-coupled-helix-data.v1"
                or identity["ambient_dimension"] != 20 or identity["intrinsic_dimension"] != 2
                or identity["relative_y_noise"] != .01 or identity != setup["dataset_identity"]):
            raise ValueError("coupled-helix dataset identity changed")
        arrays = dict(X=source["geometry/train/x_noisy"][:],
                      Y=source["maps/" + MAP + "/train/observed"][:])
        splits = [("validation", "valid", "observed")]
        if include_testing:
            splits.append(("testing", "test", "clean"))
        for split, suffix, truth in splits:
            arrays.update({"X" + suffix: source["geometry/" + split + "/x_noisy"][:],
                "Y" + suffix: source["maps/" + MAP + "/" + split + "/" + truth][:],
                "T" + suffix: source["geometry/" + split + "/tangent"][:]})
    arrays = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in arrays.items()}
    if (arrays["X"].shape != (identity["train_count"], 20)
            or arrays["Y"].shape != (identity["train_count"], 3)
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError("invalid coupled-helix training arrays")
    for _, suffix, _ in splits:
        if (arrays["X" + suffix].shape != (5000, 20)
                or arrays["Y" + suffix].shape != (5000, 3)
                or arrays["T" + suffix].shape != (5000, 2, 20)):
            raise ValueError("expected 5000 independent queries")
    return Stage2Data(dict(schema="mfm-coupled-helix-stage2-data.v1", setup_id=SETUP,
        base_dataset_identity=identity, relative_y_noise=.01,
        response_noise_rule="exact stored 1% noisy responses; fixed covariance across N",
        testing_truth="stored clean response"), arrays)


def collect(screen, root, probes, n_values, runtime):
    scores = {}
    for probe in probes:
        scores[probe["name"]] = {}
        for n in n_values:
            candidate, clipped = screen.candidate_at_n(probe["parameters"], n, 20)
            expected = screen.expected_record(SETUP, n, probe, candidate, clipped, runtime)
            record = screen.validate_record(read(screen.record_path(root, n, probe["name"])), expected)
            scores[probe["name"]][n] = record["validation_relMSE_total"]
    return scores


def main():
    global MAP, SETUP
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "calibration", "library", "runtime-json", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--working-gib", type=int, default=48)
    parser.add_argument("--map", choices=MAPS, default=MAP)
    args = parser.parse_args()
    MAP, SETUP = args.map, "coupled_helix_D20_" + args.map
    if args.working_gib < 1:
        parser.error("working-gib must be positive")
    for directory in (args.source_root / "scaling_law_N", args.source_root / "manor_m/cuda/python"):
        sys.path.insert(0, str(directory))
    import fixed_hyperparameter_screen_cuda as screen
    import fixed_hyperparameter_curve_cuda as curve
    import stage2_cuda_core as stage2
    from calibration_protocol import now
    from stage2_memory import MemoryPolicy

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    original = read(args.calibration)
    if (original["status"] != "calibrated_bracketed" or original["identity"]["map"] != MAP
            or original["identity"]["N"] != 1000000 or original["testing_used_for_selection"] is not False):
        raise ValueError("completed calibration for the requested response map required")
    calibration = copy.deepcopy(original)
    calibration["setup"] = dict(id=SETUP, D=20, version="coupled_helix", map_name=MAP,
        dataset=str(args.dataset), dataset_identity=original["identity"]["dataset_identity"])
    runtime = screen.runtime_for_screen(args.library, args.runtime_json)
    if (runtime["core_sha256"] != original["identity"]["core_sha256"]
            or runtime["production_start_allowed"] is not True
            or runtime["devtan_evaluation_version"] != "orthonormal-fp64-v2"):
        raise ValueError("use the qualified common production core")
    runtime["coupled_stage2_entry_sha256"] = sha(Path(__file__))
    axes, probes = initial_probes(calibration)
    for probe in probes:
        probe["parameters"] = screen.normalize_parameters(probe["parameters"])
    stage2.verify_or_create(root / "plan.json", dict(setup=SETUP, N_values=N_VALUES,
        anchor_N=ANCHORS, calibration_sha256=sha(args.calibration), dataset_sha256=sha(args.dataset),
        axes=axes, initial_probes=probes, selection_data="validation only",
        testing_used_for_selection=False, survivors=6, cuda_runtime=runtime,
        working_gib=args.working_gib, response_noise_rule="reuse exact stored noisy responses"))
    policy = MemoryPolicy(working_bytes=args.working_gib*1024**3,
        phase1_working_bytes=args.working_gib*1024**3, phase1_index_heap_bytes=2*1024**3,
        phase1_spill_directory=str(scratch))

    def run(label, candidates, values):
        stage2.save(root / "status.json", dict(state=label, pid=os.getpid(), utc=now()))
        plan = dict(setup=SETUP, dimension="D20", screen_stage=label, probes=candidates,
                    anchor_N=values, selection_data="validation only")
        stage2.verify_or_create(root / (label + "-plan.json"), plan)
        return screen.run_setup(calibration, plan, dataset=args.dataset, output=root/label,
            library=args.library, runtime=runtime, n_values=values, resume=True,
            memory_policy=policy, phase1_cache_bytes=1024**3, data_loader=load_data)

    run("stage_A", probes, ANCHORS)
    scores = collect(screen, root / "stage_A", probes, ANCHORS, runtime)
    interactions = interaction_probes(axes, probes, rank_scores(probes, scores, ANCHORS))
    if interactions:
        run("stage_B", interactions, ANCHORS)
        scores.update(collect(screen, root / "stage_B", interactions, ANCHORS, runtime))
    all_probes = probes + interactions
    ranking = rank_scores(all_probes, scores, ANCHORS)
    top_names = {row["name"] for row in ranking[:6]}
    survivors = [probe for probe in all_probes if probe["name"] in top_names]
    # Exact candidate/N records retain their original bytes and diagnostics.
    # Seeding Stage C reuses anchor calls instead of calculating them again.
    for probe in survivors:
        origin = "stage_A" if probe in probes else "stage_B"
        for n in ANCHORS:
            source = screen.record_path(root/origin, n, probe["name"])
            target = screen.record_path(root/"stage_C", n, probe["name"])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if sha(source) != sha(target):
                raise ValueError("reused anchor record changed")
    run("stage_C", survivors, N_VALUES)
    final_scores = collect(screen, root/"stage_C", survivors, N_VALUES, runtime)
    objectives = rank_scores(survivors, final_scores, N_VALUES)
    report = dict(setup_id=SETUP, N_values=N_VALUES, selection_data="validation only",
        testing_used=False, fixed_parameters_across_N=list(FIELDS),
        screen_stages=["stage_A", "stage_B", "stage_C"] if interactions else ["stage_A", "stage_C"],
        combined_winner=objectives[0], objectives=objectives)
    selection_path = root / "fixed_selection.json"
    stage2.verify_or_create(selection_path, dict(schema="mfm-fixed-seven-combined-validation-summary.v1",
        selection_data="validation only", testing_used=False, reports=[report]))
    selection_sha, selected = curve.load_selection(selection_path)
    stage2.save(root / "status.json", dict(state="independent_testing", pid=os.getpid(), utc=now()))
    curve.run_setup(calibration, selected[SETUP], selection_sha, dataset=args.dataset,
        output=root/"curve", library=args.library, runtime=runtime, n_values=N_VALUES,
        resume=True, memory_policy=policy, data_loader=load_data)
    stage2.save(root / "status.json", dict(state="complete", pid=os.getpid(), utc=now()))


if __name__ == "__main__":
    main()
