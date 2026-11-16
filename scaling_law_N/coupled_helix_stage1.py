#!/usr/bin/env python3
"""N1M validation-only seven-axis calibration for the coupled Meyer surface."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import fcntl
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


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8*1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def event(**fields):
    print(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), **fields)), flush=True)


@dataclass
class Data:
    arrays: dict

    def variance(self, view):
        y = np.asarray(self.arrays[{"validation":"Yvalid", "testing":"Ytest"}[view]], dtype=float)
        values = np.var(y, axis=0, ddof=1)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("nonpositive response variance")
        return values, float(values.sum())


def load(path, name, n):
    with h5py.File(path, "r") as source:
        if source.attrs["schema"] != "mfm-coupled-helix-data.v1" or source.attrs["state"] != "complete":
            raise ValueError("completed coupled-helix data required")
        identity = json.loads(source.attrs["identity"])
        if n > identity["train_count"] or identity["validation_count"] != 5000 or identity["testing_count"] != 5000:
            raise ValueError("training prefix and independent queries do not match protocol")
        arrays = dict(X=source["geometry/train/x_noisy"][:n],
                      Y=source["maps/"+name+"/train/observed"][:n],
                      Xvalid=source["geometry/validation/x_noisy"][:],
                      Yvalid=source["maps/"+name+"/validation/observed"][:],
                      Tvalid=source["geometry/validation/tangent"][:])
    return Data({key: np.ascontiguousarray(value, dtype=np.float32) for key,value in arrays.items()}), identity


def run_map(args, name, api):
    CudaCoreEstimator, ManorMCuda, evaluate_candidate, metric_cube, cell_name, initial_profile, next_profile, select_winner, MemoryPolicy = api
    root = args.output/name
    root.mkdir(parents=True, exist_ok=True)
    with (root/"stage1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/"calibration_final.json").exists():
            event(stage="map_already_complete", map=name)
            return
        data, dataset_identity = load(args.dataset, name, args.n)
        identity = dict(dataset_identity=dataset_identity, map=name, N=args.n,
                        core_sha256=digest(args.library), runtime_sha256=digest(args.runtime),
                        python_entry_sha256=digest(Path(__file__)),
                        selection_view="5000 independent noisy validation queries",
                        testing_view="5000 independent clean-response queries", testing_used_for_selection=False,
                        input_hashes={key:hashlib.sha256(value).hexdigest() for key,value in data.arrays.items()})
        saved_identity = root/"identity.json"
        if saved_identity.exists() and json.loads(saved_identity.read_text()) != identity:
            raise ValueError("existing calibration identity differs")
        save(saved_identity, identity)
        profile = initial_profile(20)
        core = ManorMCuda(args.library)
        with CudaCoreEstimator(args.library, data.arrays["X"], data.arrays["Y"],
                               dataset_id="coupled_helix_"+name, core=core,
                               memory_policy=MemoryPolicy(), phase1_cache_bytes=2*1024**3) as estimator:
            estimator.set_query("validation_response_only", data.arrays["Xvalid"], data.arrays["Yvalid"])
            for number in range(1, args.rounds+1):
                round_root = root/("round_%02d"%number)
                round_root.mkdir(parents=True, exist_ok=True)
                records = []
                offsets = [None]+profile["K_offsets"]
                iterations = [0]+[max(0, math.ceil(2*math.log(args.n)+value)) for value in offsets[1:]]
                specs = itertools.product(*(profile[field] for field in ("k_floors","h_scales","rho_inits","bites")))
                for k,h,rho,bite in specs:
                    filename = cell_name(args.n,k,h,rho,bite)
                    path = round_root/"cells"/filename
                    spec = dict(N=args.n,kSlice=k,h_scale=h,rho_init=rho,bite=bite,
                                K_grid=iterations,K_offsets=offsets,
                                eval_rho_grid=profile["eval_rhos"],eval_h_grid=profile["eval_hs"])
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record["identity"] != identity or any(record[key] != value for key,value in spec.items()):
                            raise ValueError("stale existing cell")
                    else:
                        preparation = estimator.prepare(k)
                        result, stats, diagnostic = estimator.evaluate(h_scale=h, rho_init=rho, bite=bite,
                            iterations=iterations,eval_rhos=profile["eval_rhos"],eval_hs=profile["eval_hs"])
                        cube = metric_cube(result,stats,iterations,data.variance("validation")[1])
                        if not np.isfinite(cube).any():
                            raise RuntimeError("no valid model in complete cell cube")
                        # Match MATLAB first tie in column-major order, with K0 retained.
                        best = int(np.nanargmin(cube.ravel(order="F")))
                        ik,ir,ih = np.unravel_index(best,cube.shape,order="F")
                        record = dict(spec,identity=identity,alpha=k/args.n,validation_count=5000,
                                      testing_used=False,completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                                      best_validation_relMSE_total=float(cube[ik,ir,ih]),
                                      validation_selected_K_offset=offsets[ik],validation_selected_K=iterations[ik],
                                      validation_selected_eval_rho=profile["eval_rhos"][ir],
                                      validation_selected_eval_h=profile["eval_hs"][ih],
                                      validation_cube_relMSE=cube,valid_iterations=int(stats.valid_iterations),
                                      preparation=preparation,worker_diagnostics=diagnostic)
                        save(path,record)
                        event(stage="stage1_cell",map=name,round=number,cell=filename,
                              completed=len(records)+1,total=81,best_validation=record["best_validation_relMSE_total"])
                    records.append((filename,record))
                if len(records) != 81 or len({row[0] for row in records}) != 81:
                    raise RuntimeError("round is not a complete 81-cell Cartesian grid")
                winner = select_winner(records)
                following, edges, hard = next_profile(profile,winner,number+1)
                save(round_root/"round_summary.json",dict(profile=profile,validation_winner=winner,
                    expandable_edge_flags=edges,hard_lower_boundary_flags=hard,
                    optimal_region_covered=following is None,testing_used=False,cell_count=81))
                event(stage="stage1_round",map=name,round=number,winner=winner,edges=edges)
                if following is None:
                    winner["alpha"] = winner["kSlice"]/args.n
                    # Testing is loaded only after the validation grid is bracketed.
                    with h5py.File(args.dataset,"r") as source:
                        data.arrays.update(Xtest=source["geometry/testing/x_noisy"][:],
                                           Ytest=source["maps/"+name+"/testing/clean"][:],
                                           Ttest=source["geometry/testing/tangent"][:])
                    validation = evaluate_candidate(estimator,data,winner,"validation")
                    testing = evaluate_candidate(estimator,data,winner,"testing")
                    save(root/"calibration_final.json",dict(identity=identity,rounds=number,
                        status="calibrated_bracketed",final_profile=profile,validation_winner=winner,
                        validation=validation,testing=testing,testing_used_for_selection=False))
                    return
                profile = following
        save(root/"pending_extension.json",dict(status="needs_more_boundary_rounds",
            completed_rounds=args.rounds,next_profile=profile,last_validation_winner=winner))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--maps",default="polynomial6,bernstein8,sigmoid6,gaussian6")
    parser.add_argument("--n",type=int,default=1000000)
    parser.add_argument("--rounds",type=int,default=3)
    args = parser.parse_args()
    if args.n != 1000000 or args.rounds < 1:
        parser.error("Stage1 uses N1M and a positive finite boundary-round budget")
    runtime = json.loads(args.runtime.read_text())
    if (runtime["core_sha256"] != digest(args.library) or runtime["metric_sum_contract"] != "fp32-query256-tree-v1"
            or runtime["legacy_fallback"] is not False):
        raise ValueError("use the pinned public production CUDA core")
    for directory in (args.source_root/"scaling_law_N",args.source_root/"manor_m/cuda/python"):
        sys.path.insert(0,str(directory))
    from estimator import CudaCoreEstimator
    from manor_m_cuda import ManorMCuda
    from stage2_cuda_core import evaluate_candidate,metric_cube,cell_name
    from calibration_protocol import initial_profile,next_profile
    from calibration_protocol import select_winner
    from stage2_memory import MemoryPolicy
    api = (CudaCoreEstimator,ManorMCuda,evaluate_candidate,metric_cube,cell_name,
           initial_profile,next_profile,select_winner,MemoryPolicy)
    for name in args.maps.split(","):
        run_map(args,name,api)
    event(stage="stage1_batch_complete",maps=args.maps)


if __name__ == "__main__":
    main()
