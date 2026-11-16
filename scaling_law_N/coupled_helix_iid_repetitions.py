"""Ten fresh IID fits at each displayed N, with the published A/B/C tuples.

Reuse the existing coupled-helix generator and production estimator. The maps
share each fresh predictor draw; repetitions and N values have independent
train/validation/testing draws. No parameter selection occurs in this entry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

SCHEMA = "mfm-coupled-helix-iid10.v1"
MAPS = ("natural_polynomial3", "natural_rational3", "natural_exponential3")
N_VALUES = [10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000,
            300000, 400000, 500000, 700000, 1000000, 2000000, 3000000,
            4000000, 5000000, 7000000]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temp, path)


def seed(n, repetition):
    return int(np.random.SeedSequence([2026101811, n, repetition]).generate_state(1, dtype=np.uint64)[0])


def worker_tasks(lane, lanes, n_values=N_VALUES, repetitions=range(1, 11)):
    tasks = [(n, r) for n in n_values for r in repetitions]
    return [task for index, task in enumerate(tasks) if index % lanes == lane]


@contextmanager
def claim(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def prepare(args):
    configuration = read(args.screen)
    parameters = configuration['parameters']
    if set(parameters) != set(MAPS):
        raise ValueError('three fixed-seven selections required')
    plan = dict(schema=SCHEMA, N_values=N_VALUES, repetitions=10, maps=list(MAPS),
        parameters=parameters, queries_per_view=5000, response_noise_relative=.01,
        train_validation_testing="fresh IID for every (N,repetition); paired X across A/B/C",
        seed_rule="SeedSequence([2026101811,N,repetition]).generate_state(1,uint64)",
        original_single_runs_in_ensemble=False, hyperparameter_search=False,
        readout_policy_by_map=configuration["readout_policy_by_map"],
        calibration_samples_sha256=digest(args.calibration/"samples.npz"),
        response_screen_sha256=digest(args.screen),
        total_results=3*len(N_VALUES)*10)
    target = args.root/"plan.json"
    if target.exists() and read(target) != plan:
        raise ValueError("an existing IID plan must not change")
    save(target, plan)
    print(json.dumps(dict(stage="prepared", total_results=plan["total_results"])), flush=True)


def load_data(path, name, data_api):
    with h5py.File(path, "r") as f:
        if f.attrs["state"] != "complete":
            raise ValueError("complete fresh dataset required")
        identity = json.loads(f.attrs["identity"])
        arrays = {"X": f["geometry/train/x_noisy"][:],
                  "Y": f["maps/"+name+"/train/observed"][:]}
        for split, suffix, truth in (("validation", "valid", "observed"),
                                     ("testing", "test", "clean")):
            arrays.update({"X"+suffix: f["geometry/"+split+"/x_noisy"][:],
                "Y"+suffix: f["maps/"+name+"/"+split+"/"+truth][:],
                "T"+suffix: f["geometry/"+split+"/tangent"][:]})
        identity = dict(identity, map=name)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite fresh IID array")
    return data_api.Stage2Data(identity, arrays)


def capacity_options(args, n, k):
    """Use a bounded disk index when it fits; otherwise retain generated kNN."""
    mode, spill = 4, None
    if args.stream_index_limit_gib:
        directory=args.root/args.label/"scratch"
        directory.mkdir(parents=True,exist_ok=True)
        index_bytes=4*int(n)*int(k)
        limit=min(args.stream_index_limit_gib*1024**3,
                  max(0,shutil.disk_usage(directory).free-2*1024**3))
        if index_bytes<=limit:
            mode,spill=0,str(directory)
    return dict(working_bytes=args.working_gib*1024**3,
        phase1_working_bytes=args.working_gib*1024**3,
        phase1_mode=mode,phase1_index_heap_bytes=0,
        phase1_spill_directory=spill)


def worker(args):
    import coupled_helix_generator as generator
    import cuda_core_data as data_api
    import stage2_cuda_core as stage2
    from estimator import CudaCoreEstimator
    from fixed_hyperparameter_screen_cuda import candidate_at_n
    from stage2_memory import MemoryPolicy
    from bumpy_global_tube_experiment import evaluate_shared_trajectory

    plan = read(args.root/"plan.json")
    if (plan["schema"] != SCHEMA or plan["repetitions"] != 10
            or plan["N_values"] != N_VALUES or plan["maps"] != list(MAPS)):
        raise ValueError("frozen IID plan differs")
    for path, expected in ((args.calibration/"samples.npz", plan["calibration_samples_sha256"]),
                           (args.screen, plan["response_screen_sha256"])):
        if digest(path) != expected:
            raise ValueError("geometry or response calibration changed")
    runtime = stage2.validate_runtime(args.library, args.runtime_json)
    if (not runtime["production_start_allowed"]
            or runtime["devtan_evaluation_version"] != "orthonormal-fp64-v2"):
        raise ValueError("current production runtime and corrected DevTan required")
    if args.check_only:
        print(json.dumps(dict(stage="inputs_ready", core_sha256=runtime["core_sha256"],
                              requested_pairs=len(worker_tasks(args.lane, args.lanes,
                                  args.n_values, args.repetitions)))), flush=True)
        return
    status = args.root/(args.label+"-status.json")
    (args.root/"locks").mkdir(parents=True, exist_ok=True)
    completed = 0
    for n, repetition in worker_tasks(args.lane, args.lanes, args.n_values, args.repetitions):
        tag = "N%d-r%02d" % (n, repetition)
        with claim(args.root/"locks"/(tag+".lock")) as owned:
            if not owned:
                continue
            common = dict(schema=SCHEMA, N=n, repetition=repetition,
                plan_sha256=digest(args.root/"plan.json"), core_sha256=runtime["core_sha256"])
            targets = {name: args.root/"results"/name/(tag+".json") for name in MAPS}
            pending = []
            for name, target in targets.items():
                spec = dict(common, map=name, fixed_parameters=plan["parameters"][name])
                if target.exists():
                    record = read(target)
                    if any(record.get(k) != v for k, v in spec.items()):
                        raise ValueError("saved IID result identity differs")
                else:
                    pending.append(name)
            if not pending:
                continue
            dataset = args.root/"datasets"/(tag+".h5")
            progress = dict(N=n, repetition=repetition, pid=os.getpid(),
                            completed_results=completed, updated_at=stage2.now())
            save(status, dict(progress, state="generating"))
            generator.generate(SimpleNamespace(calibration=args.calibration,
                screen=args.screen, maps=",".join(MAPS), output=dataset, n=n,
                queries=5000, chunk=65536, seed=seed(n, repetition), geometry_source=None))
            dataset_sha = digest(dataset)
            for name in pending:
                data = load_data(dataset, name, data_api)
                if data.identity["seed"] != seed(n, repetition):
                    raise ValueError("fresh IID seed differs")
                candidate, clipped = candidate_at_n(plan["parameters"][name], n, 20)
                capacity=capacity_options(args,n,candidate["kSlice"])
                policy = MemoryPolicy(**capacity)
                save(status, dict(progress, state="computing", map=name,
                    capacity=capacity,updated_at=stage2.now()))
                started = time.perf_counter()
                with CudaCoreEstimator(args.library, data.arrays["X"], data.arrays["Y"],
                        dataset_id="coupled-iid:"+tag+":"+name, device=0,
                        memory_policy=policy, retained_bytes=args.retained_gib*1024**3,
                        phase1_cache_bytes=1024**3) as estimator:
                    measured = evaluate_shared_trajectory(estimator, data, candidate,
                        all_iterations=False, fallback_exponent_scale=
                        plan['readout_policy_by_map'][name]['fallback_exponent_scale'])
                    validation, testing = measured['validation'], measured['testing']
                for view in (validation, testing):
                    for key in ("testing_relMSE_total", "testing_devTan"):
                        if not math.isfinite(view[key]) or view[key] < 0:
                            raise ValueError("invalid independent metric")
                save(targets[name], dict(common, map=name,
                    fixed_parameters=plan["parameters"][name], candidate=candidate,
                    k_clipped=clipped, dataset_identity=data.identity,
                    dataset_sha256=dataset_sha, input_arrays_sha256=data.hashes,
                    validation=validation, testing=testing,
                    testing_used_for_selection=False, testing_queries=5000,
                    capacity=capacity,wall_seconds=time.perf_counter()-started, completed_at=stage2.now()))
                completed += 1
                print(json.dumps(dict(stage="completed", N=n, repetition=repetition,
                    map=name, relMSE=testing["testing_relMSE_total"],
                    devTan=testing["testing_devTan"])), flush=True)
                del data
    save(status, dict(state="complete", completed_results=completed, pid=os.getpid(),
                     updated_at=stage2.now()))


def summary(args):
    rows = []
    for name in MAPS:
        for n in N_VALUES:
            records = [read(p) for p in sorted((args.root/"results"/name).glob("N%d-r*.json" % n))]
            if not records:
                continue
            row = dict(map=name, N=n, count=len(records))
            for key in ("testing_relMSE_total", "testing_devTan"):
                values = np.array([r["testing"][key] for r in records])
                sd = float(values.std(ddof=1)) if len(values) > 1 else None
                row[key] = dict(mean=float(values.mean()), sample_standard_deviation=sd,
                    standard_error=sd/math.sqrt(len(values)) if sd is not None else None)
            rows.append(row)
    save(args.root/"ensemble_summary.json", dict(schema=SCHEMA, rows=rows,
        completed_results=sum(row["count"] for row in rows), expected_results=len(MAPS)*len(N_VALUES)*10,
        error_bar="sample standard deviation across ten independent fits"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "worker", "summary"))
    repository = Path(__file__).resolve().parents[1]
    defaults = {'source-root': repository, 'production-source': repository,
        'calibration': repository/'data/coupled_helix',
        'screen': repository/'config/coupled_experiment.json',
        'root': repository/'reproduced/coupled_iid',
        'library': repository/'build/libmanor_m_cuda.so'}
    for field in ('root', 'source-root', 'calibration', 'screen', 'library',
                  'runtime-json', 'production-source'):
        p.add_argument('--'+field, type=Path, default=defaults.get(field))
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--label", default="lane0")
    p.add_argument("--working-gib", type=int, default=24)
    p.add_argument("--retained-gib", type=int, default=0)
    p.add_argument("--stream-index-limit-gib", type=int, default=0)
    p.add_argument('--n-values', type=int, nargs='+', default=N_VALUES)
    p.add_argument('--repetitions', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--check-only', action='store_true', help='Check worker inputs without generating data or fitting')
    args = p.parse_args()
    args.runtime_json = args.runtime_json or args.library.parent/'runtime.json'
    if not set(args.n_values) <= set(N_VALUES) or not set(args.repetitions) <= set(range(1, 11)):
        p.error('select sample sizes and repetitions from the paper grid')
    if (not args.root or not 0 <= args.lane < args.lanes or args.working_gib < 1
            or args.retained_gib<0 or args.stream_index_limit_gib<0):
        p.error("root, valid lane, and positive memory budget required")
    if args.command == "worker":
        for base, relative in ((args.source_root, "dataset_generator"),
                               (args.source_root, "scaling_law_N/analysis"),
                               (args.production_source, "scaling_law_N"),
                               (args.production_source, "manor_m/cuda/python")):
            sys.path.insert(0, str(base/relative))
        worker(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
