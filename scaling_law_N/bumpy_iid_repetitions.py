"""Independent repetitions of Figure 9 with its frozen seven parameters.

Each (N, repetition) has fresh IID train/validation/testing samples and noise.
Y0/Y2 share those predictor draws as a paired comparison. Ten independent repetitions form the reported ensemble.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
from contextlib import contextmanager

import h5py
import numpy as np

N_VALUES = [10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000,
            300000, 400000, 500000, 700000, 1000000, 2000000, 3000000,
            4000000, 5000000, 7000000]
PARAMETERS = {
    "bumpy_sphere_v2_D20_Y0": dict(alpha=.00152, K_offset=12, h_scale=2.45,
        rho_init=2.6241338308253725, bite=6, eval_rho=1.125,
        eval_h=.31685994165138015),
    "bumpy_sphere_v2_D20_Y2": dict(alpha=.00152, K_offset=0,
        h_scale=6.999999999999999, rho_init=.3448275862068966, bite=4,
        eval_rho=1.5396007178390019, eval_h=.14814814814814814),
}
SEED = 2026100809
SCHEMA = "mfm-bumpy-figure9-iid-repetitions.v1"
ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def claim_pair(path, skip_busy=False):
    """Share the per-repetition lock with additive workers."""
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not skip_busy:
                raise
            yield False
            return
        yield True


def worker_tasks(args):
    tasks = [(n, repetition) for n in args.n_values if n >= args.minimum_n
             for repetition in args.repetitions]
    if args.reverse:
        tasks.reverse()
    return [task for index, task in enumerate(tasks) if index % args.lanes == args.lane]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8*1024**2), b""):
            result.update(block)
    return result.hexdigest()


def rng(n, repetition, split, stream, start=0):
    return np.random.default_rng(np.random.SeedSequence(
        [SEED, int(n), int(repetition), int(split), int(stream), int(start)]))


def predictor_chunk(n, repetition, split, start, count, identity, calibration, generator):
    unit = rng(n, repetition, split, 1, start).standard_normal((count, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    geometry = generator._family_geometry(unit, generator._family_geometry_parameters("v2"))
    distance, index = calibration["tree"].query(unit, k=32, workers=4)
    weight = 1/np.maximum(distance, 1e-12)**2
    raw_reach = np.sum(weight*calibration["local_reach"][index], axis=1)/weight.sum(axis=1)
    exact = distance[:, 0] <= 1e-14
    raw_reach[exact] = calibration["local_reach"][index[exact, 0]]
    raw_reach = np.minimum(raw_reach, geometry["focal_reach"])
    scale, sigma = identity["geometry_scale"], identity["x_noise_sigma"]
    reach = (scale*raw_reach).astype("f4")
    clean = np.zeros((count, 20), dtype="f4")
    clean[:, :3] = scale*geometry["radius"][:, None]*unit
    tangent = np.zeros((count, 20, 2), dtype="f4")
    tangent[:, :3, 0] = geometry["tangent_first"]
    tangent[:, :3, 1] = geometry["tangent_second"]
    direction = rng(n, repetition, split, 2, start).standard_normal((count, 20), dtype="f4")
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), np.finfo("f4").tiny)
    radius = generator._family_truncated_gamma(rng(n, repetition, split, 3, start),
        count, 10., .81*identity["target_global_reach"]**2/sigma**2)
    radius = (np.float32(sigma)*(np.float32(.98)*reach)
              /np.float32(identity["target_global_reach"])*np.sqrt(radius))
    noise = direction*radius[:, None]
    coefficient = np.einsum("ndm,nd->nm", tangent, noise, optimize=True)
    noise -= np.einsum("ndm,nm->nd", tangent, coefficient, optimize=True)
    ratio = np.linalg.norm(noise, axis=1)/reach
    if not np.isfinite(ratio).all() or float(ratio.max()) > .88202:
        raise ValueError("normal-noise support differs from the Figure 9 model")
    if float(np.max(np.abs(np.einsum("ndm,nd->nm", tangent, noise)))) > 2e-5:
        raise ValueError("noise is not normal to the clean surface")
    return dict(X=clean+noise, clean3=clean[:, :3], latent=unit,
        local_reach=reach, T=np.ascontiguousarray(tangent.transpose(0, 2, 1)),
        standard_Y=rng(n, repetition, split, 4, start).standard_normal((count, 6), dtype="f4"))


def prepare(args, generator):
    """Initialize a fresh run from the paper's included sampling recipe."""
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT/"config/bumpy_experiment.json").read_text())
    identity = spec["canonical_identity"]
    formula = generator._family_formula_hash(generator._family_geometry_parameters("v2"))
    if formula != identity["geometry_formula_sha256"] or spec["parameters"] != PARAMETERS:
        raise ValueError("bundled geometry or fixed parameters differ")
    if args.canonical is not None:
        with h5py.File(args.canonical, "r") as source:
            if json.loads(source.attrs["identity_json"])["geometry_formula_sha256"] != formula:
                raise ValueError("canonical data use a different geometry")
    plan_path = root/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != spec:
        raise ValueError("existing repetition plan differs")
    save(plan_path, spec)
    bundled = ROOT/"config/geometry_recipe.npz"
    recipe_path = root/"geometry_recipe.npz"
    if recipe_path.exists():
        if digest(recipe_path) != digest(bundled):
            raise ValueError("existing geometry recipe differs")
    else:
        shutil.copyfile(bundled, recipe_path)
    print(json.dumps(dict(stage="plan_ready", experiments=360, predictor_datasets=180)), flush=True)


def generate(path, n, repetition, plan, calibration, generator, data_api):
    encoded = json.dumps(dict(plan_sha256=digest(path.parents[1]/"plan.json"),
        N=n, repetition=repetition, queries=5000, seed=SEED), sort_keys=True)
    if path.exists():
        with h5py.File(path, "r") as source:
            if source.attrs["identity"] != encoded or source.attrs["state"] != "complete":
                raise ValueError("completed repetition dataset differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.h5")
    with h5py.File(temporary, "a") as output:
        if "identity" in output.attrs and output.attrs["identity"] != encoded:
            raise ValueError("partial repetition dataset differs")
        output.attrs.update(identity=encoded, state="generating")
        for split_code, split in enumerate(("train", "validation", "testing"), 1):
            count = n if split == "train" else 5000
            group = output.require_group(split)
            fields = dict(X=(20,), clean3=(3,), latent=(3,), local_reach=(), standard_Y=(6,))
            if split != "train":
                fields["T"] = (2, 20)
            for name, shape in fields.items():
                if name not in group:
                    group.create_dataset(name, (count,)+shape, dtype="f8" if name == "latent" else "f4",
                        chunks=(min(count, 16384),)+shape)
            for start in range(int(group.attrs.get("completed", 0)), count, 65536):
                end = min(start+65536, count)
                values = predictor_chunk(n, repetition, split_code, start, end-start,
                    plan["canonical_identity"], calibration, generator)
                for name in fields:
                    group[name][start:end] = values[name]
                output.flush()
                group.attrs["completed"] = end
                output.flush()
        rotation = np.asarray(plan["response_rotation"])
        scale = plan["canonical_identity"]["geometry_scale"]
        for map_id in (0, 2):
            reference = output["train/clean3"][:min(n, 1000000)].astype(float)/scale
            clean_reference = data_api.response_map(reference@rotation.T, map_id)
            noise_root = data_api.covariance_root(clean_reference).astype("f4")
            for split in ("train", "validation", "testing"):
                count = n if split == "train" else 5000
                group = output.require_group("Y%d/"%map_id+split)
                for name in ("clean", "observed"):
                    if name not in group:
                        group.create_dataset(name, (count, 6), dtype="f4", chunks=(min(count, 16384), 6))
                for start in range(int(group.attrs.get("completed", 0)), count, 65536):
                    end = min(start+65536, count)
                    clean = data_api.response_map((output[split+"/clean3"][start:end].astype(float)/scale)@rotation.T, map_id)
                    group["clean"][start:end] = clean
                    group["observed"][start:end] = clean+data_api.noise_product_fixed_order(
                        noise_root, output[split+"/standard_Y"][start:end])
                    output.flush()
                    group.attrs["completed"] = end
                    output.flush()
        output.attrs["state"] = "complete"
    os.replace(temporary, path)


def load_data(path, map_id, data_api):
    with h5py.File(path, "r") as source:
        arrays = dict(X=source["train/X"][:], Y=source["Y%d/train/observed"%map_id][:])
        for split, suffix, truth in (("validation", "valid", "observed"), ("testing", "test", "clean")):
            arrays.update({"X"+suffix:source[split+"/X"][:],
                "Y"+suffix:source["Y%d/%s/%s"%(map_id, split, truth)][:],
                "T"+suffix:source[split+"/T"][:]})
        identity = json.loads(source.attrs["identity"])
        identity.update(schema=SCHEMA, map_id=map_id, response_noise_relative=.01)
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("nonfinite fresh IID data")
    return data_api.Stage2Data(identity, arrays)


def worker(args, generator, data_api):
    from scipy.spatial import cKDTree
    import stage2_cuda_core as stage2
    from estimator import CudaCoreEstimator
    from fixed_hyperparameter_curve_cuda import load_selection
    from fixed_hyperparameter_screen_cuda import candidate_at_n
    from stage2_memory import MemoryPolicy
    plan = json.loads((args.root/"plan.json").read_text())
    if plan["parameters"] != PARAMETERS or plan["N_values"] != N_VALUES or plan["repetitions"] != 10:
        raise ValueError("repetition configuration changed")
    library = args.library
    runtime = stage2.validate_runtime(library, args.runtime_json)
    if not runtime["production_start_allowed"] or runtime["devtan_evaluation_version"] != "orthonormal-fp64-v2":
        raise ValueError("qualified production core required")
    selection_sha, selected = load_selection(args.selection)
    for setup, parameters in PARAMETERS.items():
        if selected[setup]["parameters"] != parameters:
            raise ValueError("frozen Figure 9 tuple differs from production selection")
    with np.load(args.root/"geometry_recipe.npz") as recipe:
        unit, reach = recipe["unit"].copy(), recipe["local_reach"].copy()
    calibration = dict(tree=cKDTree(unit), local_reach=reach)
    tasks = worker_tasks(args)
    if args.check_only:
        print(json.dumps(dict(stage="inputs_ready", requested_pairs=len(tasks),
                              core_sha256=runtime["core_sha256"])), flush=True)
        return
    lock_dir = args.root/"locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.root / ((args.status_label or "lane%d" % args.lane) + "-status.json")
    completed = 0
    for n, repetition in tasks:
        name = "N%d-r%02d"%(n, repetition)
        with claim_pair(lock_dir/(name+".lock"), args.skip_busy) as claimed:
            if not claimed:
                continue
            dataset = args.root/"datasets"/(name+".h5")
            save(status_path, dict(
                state="generating", N=n, repetition=repetition, pid=os.getpid(), completed_pairs=completed))
            generate(dataset, n, repetition, plan, calibration, generator, data_api)
            for map_id, setup in ((0, "bumpy_sphere_v2_D20_Y0"), (2, "bumpy_sphere_v2_D20_Y2")):
                target = args.root/"results"/setup/(name+".json")
                spec = dict(schema=SCHEMA, setup_id=setup, N=n, repetition=repetition,
                    fixed_parameters=PARAMETERS[setup], plan_sha256=digest(args.root/"plan.json"),
                    selection_sha256=selection_sha, core_sha256=runtime["core_sha256"])
                if target.exists():
                    record = json.loads(target.read_text())
                    if any(record.get(key) != value for key, value in spec.items()):
                        raise ValueError("saved repetition result differs")
                    continue
                data = load_data(dataset, map_id, data_api)
                candidate, clipped = candidate_at_n(PARAMETERS[setup], n, 20)
                scratch = args.root/"scratch"/name/setup
                scratch.mkdir(parents=True, exist_ok=True)
                policy = MemoryPolicy(working_bytes=args.working_gib*1024**3,
                    phase1_working_bytes=args.working_gib*1024**3,
                    phase1_mode=args.phase1_mode,
                    phase1_index_heap_bytes=args.index_heap_gib*1024**3,
                    phase1_spill_directory=str(scratch))
                save(status_path, dict(
                    state="computing", N=n, repetition=repetition, setup=setup,
                    pid=os.getpid(), completed_pairs=completed))
                started = time.perf_counter()
                with CudaCoreEstimator(library, data.arrays["X"], data.arrays["Y"],
                        dataset_id="figure9-iid:"+setup+":"+name, device=0,
                        memory_policy=policy, retained_bytes=0, phase1_cache_bytes=1024**3) as estimator:
                    validation = stage2.evaluate_candidate(estimator, data, candidate, "validation")
                    testing = stage2.evaluate_candidate(estimator, data, candidate, "testing")
                for view in (validation, testing):
                    for key in ("testing_relMSE_total", "testing_devTan"):
                        if not math.isfinite(view[key]) or view[key] < 0:
                            raise ValueError("invalid independently evaluated metric")
                save(target, dict(spec, candidate=candidate, k_clipped=clipped,
                    dataset_identity=data.identity, input_arrays_sha256=data.hashes,
                    dataset_sha256=digest(dataset), validation=validation, testing=testing,
                    testing_used_for_selection=False, testing_queries=5000,
                    wall_seconds=time.perf_counter()-started, completed_at=stage2.now()))
                print(json.dumps(dict(stage="completed_repetition", setup=setup, N=n,
                    repetition=repetition, relMSE=testing["testing_relMSE_total"],
                    devTan=testing["testing_devTan"], seconds=time.perf_counter()-started)), flush=True)
                del data
            completed += 1
            if args.max_pairs and completed >= args.max_pairs:
                break
    save(status_path, dict(
        state="batch_complete" if args.max_pairs else "complete", completed_pairs=completed, pid=os.getpid()))


def summarize(root):
    rows = []
    for setup in PARAMETERS:
        for n in N_VALUES:
            records = [json.loads(path.read_text()) for path in sorted((root/"results"/setup).glob("N%d-r*.json"%n))]
            if not records:
                continue
            result = dict(setup=setup, N=n, count=len(records), complete=len(records) == 10)
            for metric in ("testing_relMSE_total", "testing_devTan"):
                values = np.array([record["testing"][metric] for record in records])
                sd = float(values.std(ddof=1)) if len(values) > 1 else None
                result[metric] = dict(mean=float(values.mean()), standard_deviation=sd,
                    standard_error=sd/np.sqrt(len(values)) if sd is not None else None)
            rows.append(result)
    save(root/"ensemble_summary.json", dict(schema=SCHEMA, rows=rows,
        complete=len(rows) == 36 and all(row["complete"] for row in rows), error_bar="sample standard deviation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "worker", "summary"))
    parser.add_argument("--root", type=Path, default=ROOT/"reproduced/bumpy_iid")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--production", type=Path, default=ROOT)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--selection", type=Path, default=ROOT/"config/bumpy_parameter_selection.json")
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--repetitions", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--check-only", action="store_true", help="Check worker inputs without generating data or fitting")
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--working-gib", type=int, default=24)
    parser.add_argument("--index-heap-gib", type=int, default=2,
                        help="Host-RAM ceiling for the exact kNN index before disk spill")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--phase1-mode", type=int, choices=(0, 4), default=0)
    parser.add_argument("--minimum-n", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--skip-busy", action="store_true")
    parser.add_argument("--status-label", default="")
    args = parser.parse_args()
    args.library = args.library or args.production/"build/libmanor_m_cuda.so"
    args.runtime_json = args.runtime_json or args.library.parent/"runtime.json"
    if not set(args.n_values) <= set(N_VALUES) or not set(args.repetitions) <= set(range(1, 11)):
        parser.error("select sample sizes and repetitions from the paper grid")
    if (not 0 <= args.lane < args.lanes or args.working_gib < 1
            or args.index_heap_gib < 0):
        parser.error("invalid lane or memory ceiling")
    if "/" in args.status_label or "\\" in args.status_label:
        parser.error("status-label must be a filename label")
    for relative in ("dataset_generator", "scaling_law_N"):
        sys.path.insert(0, str(args.source_root/relative))
    if args.production:
        for relative in ("scaling_law_N", "manor_m/cuda/python"):
            sys.path.insert(0, str(args.production/relative))
    import bumpy_sphere_generator as generator
    import cuda_core_data as data_api
    if args.command == "prepare":
        prepare(args, generator)
    elif args.command == "worker":
        worker(args, generator, data_api)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
