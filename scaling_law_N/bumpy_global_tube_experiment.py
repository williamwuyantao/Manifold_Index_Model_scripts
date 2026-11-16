"""Parallel Stage1/Stage2 for the three fixed-global-width bumpy experiments.

Workers claim scientific cells or complete fixed-N screens with file locks.
All arithmetic, schedules, selection ties and boundary shifts use production.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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

import numpy as np

FIELDS = ("alpha", "K_offset", "h_scale", "rho_init", "bite", "eval_rho", "eval_h")
N_VALUES = [1000,2000,3000,4000,5000,7000,10000,20000,30000,40000,50000,70000,
            100000,200000,300000,400000,500000,700000]
TRAJECTORY_CONTRACT = "all-phase2-iterations-separate-metrics.v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    from stage2_cuda_core import save as atomic
    atomic(Path(path), value)


@contextmanager
def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True


def report(args, **value):
    value.update(pid=os.getpid(), utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 iteration_contract=TRAJECTORY_CONTRACT)
    save(args.root / ("worker-"+args.label+".json"), value)
    print(json.dumps(value), flush=True)


def estimator_for(args, data, identity):
    from estimator import CudaCoreEstimator
    from stage2_memory import MemoryPolicy
    class SharedCardPolicy(MemoryPolicy):
        def phase1_options(self, retained_limit):
            value = super().phase1_options(retained_limit)
            # Explicit byte reserve for an operator-budgeted shared GPU.
            value["reserve_fraction"] = self.reserve_fraction
            return value
    class SharedCardEstimator(CudaCoreEstimator):
        def _phase1_configuration(self, k):
            value = super()._phase1_configuration(k)
            value["reserve_bytes"] = 128*1024**2
            return value
    policy = SharedCardPolicy(working_bytes=int(args.working_gib*1024**3),
        phase1_working_bytes=int(args.working_gib*1024**3),
        reserve_fraction=0, reserve_bytes=128*1024**2,
        phase1_index_heap_bytes=2*1024**3, phase1_mode=0,
        phase1_spill_directory=str(args.root/"scratch"/args.label))
    return SharedCardEstimator(args.library, data.arrays["X"], data.arrays["Y"],
        dataset_id=identity, memory_policy=policy,
        retained_bytes=int((.5 if args.low_memory else 8)*1024**3),
        phase1_cache_bytes=512*1024**2)


def evaluate_trajectory(estimator, data, candidate, view):
    """One refinement pass per metric, evaluating K=0 and every update.

    Selection uses only the prescribed endpoint, never the best point of a
    testing trajectory. Unexecuted no-heavy iterations are labelled no-ops.
    """
    started = time.perf_counter()
    preparation = estimator.prepare(candidate["kSlice"])
    suffix = {"validation":"valid", "testing":"test"}[view]
    requested = int(candidate["K"])
    if requested < 0:
        raise ValueError("negative iteration count")
    values, states, statistics, diagnostics = {}, {}, {}, {}
    variance, total = data.variance(view)
    for mode, family in (("response", "EvalMSE"), ("tangent", "EvalDevTan")):
        estimator.set_query(view+"_"+mode+"_only", data.arrays["X"+suffix],
            y=data.arrays["Y"+suffix] if mode=="response" else None,
            tangent=data.arrays["T"+suffix] if mode=="tangent" else None)
        result, stats, diagnostic = estimator.evaluate(
            h_scale=candidate["h_scale"], rho_init=candidate["rho_init"],
            bite=candidate["bite"], iterations=list(range(requested+1)),
            eval_rhos=[candidate["eval_rho"]], eval_hs=[candidate["eval_h"]],
        fallback_exponent_scale=fallback_exponent_scale)
        valid = min(requested, int(stats.valid_iterations))
        stopped = bool(stats.stopped_no_heavy)
        if valid < requested and not stopped:
            raise RuntimeError("refinement stopped without a terminal no-heavy state")
        statistics[mode] = dict(effective_K=valid, stopped_no_heavy=stopped)
        states[mode] = {key:hashlib.sha256(memoryview(np.ascontiguousarray(result[key])).cast("B")).hexdigest()
                       for key in ("Tan_next", "Cen_next", "num_neigh", "heavy_mask", "keep_mask")}
        def series(name):
            initial = np.asarray(result[name+"0"],dtype=float).reshape(-1)
            following = np.asarray(result[name],dtype=float)
            out = [initial]+[following[k].reshape(-1) for k in range(valid)]
            if not all(np.isfinite(v).all() and (v>=0).all() for v in out):
                raise RuntimeError("nonfinite/negative executed iteration metric")
            return out
        values[mode] = series(family)
        if mode=="response":
            values["coordinate"] = series("EvalMSECoordinate")
        diagnostics[mode] = diagnostic
        del result
    if states["response"] != states["tangent"] or statistics["response"] != statistics["tangent"]:
        raise RuntimeError("response/tangent passes produced different model states")
    valid = statistics["response"]["effective_K"]
    rows=[]
    for k in range(requested+1):
        effective=min(k,valid)
        rows.append(dict(iteration=k,state_iteration=effective,terminal_noop=k>valid,
            relMSE_total=float(values["response"][effective][0]/total),
            relMSE_coord=float(np.mean(values["coordinate"][effective]/variance)),
            devTan=float(values["tangent"][effective][0])))
    endpoint=rows[-1]
    return dict(candidate, testing_relMSE_total=endpoint["relMSE_total"],
        testing_relMSE_coord=endpoint["relMSE_coord"],testing_devTan=endpoint["devTan"],
        requested_K=requested,effective_K=valid,terminal_state_carry_forward=valid<requested,
        stopped_no_heavy=statistics["response"]["stopped_no_heavy"],
        iteration_contract=TRAJECTORY_CONTRACT,metric_view=view,iteration_rows=rows,
        phase1_is_iteration_zero=True,selection_iteration=requested,
        readout_bandwidth="production h/rho schedule at each iteration",
        testing_used_for_selection=False,final_state_sha256=states,
        preparation=preparation,python_diagnostics=diagnostics,wall=time.perf_counter()-started)


def evaluate_shared_trajectory(estimator, data, candidate, *, all_iterations=True,
                               fallback_exponent_scale=2.0):
    """One model trajectory; independent validation/testing metric reductions."""
    started = time.perf_counter()
    requested = int(candidate["K"])
    if requested < 0:
        raise ValueError("negative iteration count")
    preparation = estimator.prepare(candidate["kSlice"])
    estimator.set_evaluation_queries("validation_and_testing", data.arrays["Xtest"],
        test_y=data.arrays["Ytest"], test_tangent=data.arrays["Ttest"],
        validation_x=data.arrays["Xvalid"], validation_y=data.arrays["Yvalid"],
        validation_tangent=data.arrays["Tvalid"])
    selected = list(range(requested + 1)) if all_iterations else [requested]
    result, stats, diagnostic = estimator.evaluate(
        h_scale=candidate["h_scale"], rho_init=candidate["rho_init"],
        bite=candidate["bite"], iterations=selected,
        eval_rhos=[candidate["eval_rho"]], eval_hs=[candidate["eval_h"]])
    valid = min(requested, int(stats.valid_iterations))
    stopped = bool(stats.stopped_no_heavy)
    if valid < requested and not stopped:
        raise RuntimeError("refinement stopped without a terminal no-heavy state")
    states = {key: hashlib.sha256(memoryview(np.ascontiguousarray(result[key])).cast("B")).hexdigest()
              for key in ("Tan_next", "Cen_next", "num_neigh", "heavy_mask", "keep_mask")}
    terminal = None
    if not all_iterations and valid < requested:
        # Score the terminal state directly. Do not replay its training updates.
        from phase1 import _phase2_schedules
        p1 = estimator.p1
        _, _, hs, rhos = _phase2_schedules(p1["Rad0"], p1["owner_mask0"].astype(bool),
            p1["axial_tan0"], float(candidate["h_scale"]), float(candidate["rho_init"]),
            int(candidate["bite"]), data.arrays["X"].shape[1], 2, max(1, requested))
        position = max(0, valid - 1)
        terminal, terminal_stats = estimator.sweep.context.phase2(
            estimator.sweep.data["x"], estimator.sweep.data["response"], estimator.index,
            result["Tan_next"], result["Cen_next"], p1["owner_mask0"],
            float(hs[position]), rhos[position:position + 1], 2,
            h_schedule=hs[position:position + 1], iteration_count=1,
            evaluated_iterations=[0], eval_rho_scales=[candidate["eval_rho"]],
            eval_h_scales=[candidate["eval_h"]], evaluation_only=True,
            return_histories=False, direct_output_buffers=True,
            fallback_exponent_scale=fallback_exponent_scale,
            generated_neighbor_count=estimator.k if estimator.index_generated else None,
            knn_scratch_memory_bytes=0, **estimator.query)
        if terminal_stats.valid_iterations != 0:
            raise RuntimeError("terminal readout executed model updates")
        diagnostic["terminal_readout_only"] = dict(
            state_iteration=valid, model_update_count=0,
            milliseconds=terminal_stats.total_milliseconds)
    output = {}
    for view, prefix in (("validation", "EvalValid"), ("testing", "Eval")):
        variance, total = data.variance(view)
        def at(family, k):
            name = prefix + family
            value = np.asarray(terminal[name + "0"] if terminal is not None else
                result[name + "0"] if k == 0 else result[name][k - 1], dtype=float).reshape(-1)
            if not np.isfinite(value).all() or (value < 0).any():
                raise RuntimeError("nonfinite/negative executed iteration metric")
            return value
        rows = []
        for k in selected:
            effective = min(k, valid)
            rows.append(dict(iteration=k, state_iteration=effective, terminal_noop=k > valid,
                relMSE_total=float(at("MSE", effective)[0] / total),
                relMSE_coord=float(np.mean(at("MSECoordinate", effective) / variance)),
                devTan=float(at("DevTan", effective)[0])))
        endpoint = rows[-1]
        output[view] = dict(candidate,
            testing_relMSE_total=endpoint["relMSE_total"],
            testing_relMSE_coord=endpoint["relMSE_coord"], testing_devTan=endpoint["devTan"],
            requested_K=requested, effective_K=valid, stopped_no_heavy=stopped,
            terminal_state_carry_forward=valid < requested,
            iteration_contract=TRAJECTORY_CONTRACT, metric_view=view, iteration_rows=rows,
            phase1_is_iteration_zero=True, selection_iteration=requested,
            readout_bandwidth="production h/rho schedule at each iteration",
            testing_used_for_selection=False,
            final_state_sha256={"response": states, "tangent": states},
            preparation=preparation if view == "validation" else
                dict(reused=True, source="active", wall_seconds=0.0, index_rebuilt=False),
            python_diagnostics={"shared_trajectory": diagnostic},
            model_update_passes=1, shared_evaluation_groups=["validation", "testing"])
    return dict(output, diagnostic=diagnostic, wall=time.perf_counter() - started)


def trajectory_complete(path, expected_c=None):
    if not path.exists():
        return False
    value=read(path)
    if expected_c is not None and value.get("dataset_identity",{}).get("c",value.get("c")) != expected_c:
        raise ValueError("cached Stage2 result belongs to a different tube width")
    measurements=[row["measurement"] for row in value["records"]] if "records" in value else [value["testing"]]
    return all(m.get("iteration_contract")==TRAJECTORY_CONTRACT and
               len(m.get("iteration_rows",[]))==m["requested_K"]+1 for m in measurements)


def preserve_endpoint_record(path):
    """Keep already completed endpoint-only batches beside the richer result."""
    if path.exists() and not trajectory_complete(path):
        old=path.with_suffix(".endpoint-v1.json")
        if old.exists():
            if old.read_bytes()!=path.read_bytes():
                raise ValueError("different endpoint-only backup already exists")
        else:
            with old.open("xb") as stream:
                stream.write(path.read_bytes())


def check_previous_endpoints(path, records):
    old=path.with_suffix(".endpoint-v1.json")
    if not old.exists():
        return
    previous={row["probe"]:row for row in read(old)["records"]}
    for row in records:
        ref=previous[row["probe"]]
        if ref["candidate"] != row["candidate"]:
            raise ValueError("iteration record changed the candidate")
        baseline=ref["validation_relMSE_total"]
        delta=row["validation_relMSE_total"]-baseline
        if abs(delta)>5e-7 or abs(delta)>5e-4*abs(baseline):
            raise ValueError("all-iteration endpoint differs from the saved endpoint")
        row["difference_from_endpoint_only_relMSE"]=delta


def stage1(args, c, dataset):
    from bumpy_global_tube import label, load_data
    from calibration_protocol import initial_profile, next_profile
    from calibration_protocol import select_winner
    from stage2_cuda_core import cell_name, metric_cube
    root = args.root/label(c)/"stage1"
    if (root/"calibration_final.json").exists():
        return False
    profile = initial_profile(20)
    number = 1
    while (root/("round_%02d"%number)/"round_summary.json").exists():
        summary = read(root/("round_%02d"%number)/"round_summary.json")
        following, _, _ = next_profile(profile, summary["validation_winner"], number+1)
        if following is None:
            break
        profile, number = following, number+1
    round_root = root/("round_%02d"%number)
    specs = list(itertools.product(*(profile[key] for key in ("k_floors","h_scales","rho_inits","bites"))))
    paths = [round_root/"cells"/cell_name(1_000_000,*spec) for spec in specs]
    did_work = False
    estimator = None
    try:
        for (k,h,rho,bite), path in zip(specs, paths):
            if path.exists():
                continue
            with claim(path.with_suffix(".lock")) as owned:
                if not owned or path.exists():
                    continue
                if estimator is None:
                    data = load_data(dataset, include_testing=False, expected_c=c)
                    estimator = estimator_for(args, data, "global-tube:"+label(c))
                    estimator.__enter__()
                    estimator.set_query("validation", data.arrays["Xvalid"], data.arrays["Yvalid"])
                report(args, state="computing_stage1", c=c, round=number, cell=path.name)
                preparation = estimator.prepare(k)
                offsets = [None]+profile["K_offsets"]
                iterations = [0]+[max(0,math.ceil(2*math.log(1_000_000)+v)) for v in offsets[1:]]
                result, stats, diagnostic = estimator.evaluate(h_scale=h, rho_init=rho,
                    bite=bite, iterations=iterations, eval_rhos=profile["eval_rhos"],
                    eval_hs=profile["eval_hs"])
                cube = metric_cube(result, stats, iterations, data.variance("validation")[1])
                assert np.isfinite(cube).any()
                position = np.unravel_index(np.nanargmin(cube.ravel(order="F")), cube.shape, order="F")
                ik,ir,ih = position
                save(path, dict(N=1_000_000,kSlice=k,h_scale=h,rho_init=rho,bite=bite,
                    alpha=k/1_000_000, K_grid=iterations, K_offsets=offsets,
                    eval_rho_grid=profile["eval_rhos"],eval_h_grid=profile["eval_hs"],
                    dataset_identity=data.identity,core_sha256=sha(args.library),
                    validation_count=5000,testing_used=False,validation_cube_relMSE=cube,
                    best_validation_relMSE_total=float(cube[position]),
                    validation_selected_K_offset=offsets[ik],validation_selected_K=iterations[ik],
                    validation_selected_eval_rho=profile["eval_rhos"][ir],
                    validation_selected_eval_h=profile["eval_hs"][ih],
                    valid_iterations=int(stats.valid_iterations),preparation=preparation,
                    worker_diagnostics=diagnostic))
                did_work = True
                del result
    finally:
        if estimator is not None:
            estimator.__exit__(None,None,None)
    if all(p.exists() for p in paths):
        records = [(p.name,read(p)) for p in paths]
        assert len(records) == len(set(p.name for p in paths)) == 81
        if any(row["dataset_identity"]["c"]!=c for _,row in records):
            raise ValueError("Stage1 cell belongs to a different tube width")
        winner = select_winner(records)
        winner["alpha"] = winner["kSlice"]/1_000_000
        following, edges, hard = next_profile(profile,winner,number+1)
        with claim(root/"summary.lock") as owned:
            if owned:
                save(round_root/"round_summary.json",dict(profile=profile,validation_winner=winner,
                    expandable_edge_flags=edges,hard_lower_boundary_flags=hard,
                    optimal_region_covered=following is None,testing_used=False,cell_count=81))
                if following is None:
                    save(root/"calibration_final.json",dict(status="calibrated_bracketed",
                        rounds=number,final_profile=profile,validation_winner=winner,
                        testing_used_for_selection=False,setup=dict(id=label(c),D=20)))
    return did_work


def stage2(args,c,dataset):
    from bumpy_global_tube import label, load_data
    from calibration_protocol import fixed_n1m_candidate
    from coupled_helix_stage2 import initial_probes, interaction_probes
    root = args.root/label(c)/"stage2"
    calibration = read(args.root/label(c)/"stage1/calibration_final.json")
    axes, probes = initial_probes(calibration)
    stages = [("stage_A",probes)]
    if (root/"stage_B_plan.json").exists():
        stages.append(("stage_B",read(root/"stage_B_plan.json")))
    for stage_name,candidates in stages:
        for n in N_VALUES:
            target = root/stage_name/("N%d.json"%n)
            if trajectory_complete(target,c):
                continue
            with claim(target.with_suffix(".lock")) as owned:
                if not owned or trajectory_complete(target,c):
                    continue
                preserve_endpoint_record(target)
                data = load_data(dataset,include_testing=False,expected_c=c).for_n(n)
                records=[]
                with estimator_for(args,data,label(c)+":N%d"%n) as estimator:
                    for probe in candidates:
                        parameters=probe["parameters"]
                        reference=dict(parameters,kSlice=round(parameters["alpha"]*1_000_000))
                        candidate,_=fixed_n1m_candidate(reference,n,20)
                        report(args,state="computing_stage2",c=c,N=n,probe=probe["name"],stage=stage_name)
                        measured=evaluate_trajectory(estimator,data,candidate,"validation")
                        records.append(dict(probe=probe["name"],parameters=parameters,
                            validation_relMSE_total=measured["testing_relMSE_total"],
                            candidate=candidate,measurement=measured,testing_used=False))
                check_previous_endpoints(target,records)
                save(target,dict(N=n,records=records,dataset_identity=data.identity,
                                 input_arrays_sha256=data.hashes,
                                 core_sha256=sha(args.library),testing_used_for_selection=False))
                return True
        if not all(trajectory_complete(root/stage_name/("N%d.json"%n),c) for n in N_VALUES):
            return False
    all_probes=[p for _,ps in stages for p in ps]
    scores={p["name"]:{} for p in all_probes}
    for stage_name,candidates in stages:
        for n in N_VALUES:
            for row in read(root/stage_name/("N%d.json"%n))["records"]:
                scores[row["probe"]][n]=row["validation_relMSE_total"]
    denominators={n:min(row[n] for row in scores.values()) for n in N_VALUES}
    ranking=[]
    for probe in all_probes:
        ratios=[scores[probe["name"]][n]/denominators[n] for n in N_VALUES]
        assert all(math.isfinite(x) and x>0 for x in ratios)
        ranking.append(dict(name=probe["name"],parameters=probe["parameters"],
            max_regret_fraction=max(ratios)-1,
            geometric_mean_regret_fraction=math.exp(sum(map(math.log,ratios))/len(ratios))-1))
    ranking.sort(key=lambda r:(r["max_regret_fraction"],r["geometric_mean_regret_fraction"],r["name"]))
    if len(stages)==1:
        interactions=interaction_probes(axes,probes,ranking)
        if interactions:
            with claim(root/"selection.lock") as owned:
                if owned: save(root/"stage_B_plan.json",interactions)
            return True
    with claim(root/"selection.lock") as owned:
        if owned: save(root/"fixed_selection.json",dict(winner=ranking[0],ranking=ranking,
            N_values=N_VALUES,selection_data="validation only",testing_used=False))
    selected=ranking[0]["parameters"]
    for n in N_VALUES:
        target=root/"curve"/("N%d.json"%n)
        if trajectory_complete(target,c): continue
        with claim(target.with_suffix(".lock")) as owned:
            if not owned or trajectory_complete(target,c): continue
            preserve_endpoint_record(target)
            data=load_data(dataset,include_testing=True,expected_c=c).for_n(n)
            candidate,_=fixed_n1m_candidate(dict(selected,kSlice=round(selected["alpha"]*1_000_000)),n,20)
            report(args,state="independent_testing",c=c,N=n)
            with estimator_for(args,data,label(c)+":final:N%d"%n) as estimator:
                testing=evaluate_trajectory(estimator,data,candidate,"testing")
            save(target,dict(N=n,fixed_parameters=selected,testing=testing,
                             dataset_identity=data.identity,testing_used_for_selection=False,
                             core_sha256=sha(args.library)))
            return True
    save(args.root/label(c)/"complete.json",dict(stage1=True,stage2=True,fixed_parameters=selected))
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","source-root","application-source","plan","library","runtime"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--cs",default="0.25,0.5")
    parser.add_argument("--label",required=True)
    parser.add_argument("--working-gib",type=float,default=8)
    parser.add_argument("--low-memory",action="store_true")
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    for directory in (args.source_root/"scaling_law_N",args.source_root/"manor_m/cuda/python",
                      args.application_source/"scaling_law_N",args.application_source/"dataset_generator"):
        sys.path.insert(0,str(directory))
    import bumpy_global_tube as tube
    import bumpy_sphere_generator as generator
    import cuda_core_data as data_api
    from stage2_cuda_core import validate_runtime
    runtime=validate_runtime(args.library,args.runtime)
    assert runtime["production_start_allowed"] and runtime["legacy_fallback"] is False
    cs=[float(value) for value in args.cs.split(",")]
    for c in cs: tube.label(c)
    with claim(args.root/("worker-"+args.label+".lock")) as owned:
        if not owned: raise RuntimeError("worker already running")
        while True:
            worked=False
            for c in cs:
                if (args.root/tube.label(c)/"complete.json").exists(): continue
                dataset=args.root/"datasets"/(tube.label(c)+".h5")
                if not dataset.exists():
                    with claim(dataset.with_suffix(".lock")) as generator_owned:
                        if not generator_owned: continue
                        report(args,state="generating",c=c)
                        tube.generate(dataset,c,read(args.plan),generator,data_api)
                if not (args.root/tube.label(c)/"stage1/calibration_final.json").exists():
                    worked=stage1(args,c,dataset) or worked
                else:
                    worked=stage2(args,c,dataset) or worked
            if all((args.root/tube.label(c)/"complete.json").exists() for c in cs):
                report(args,state="complete",cs=cs)
                break
            if not worked: time.sleep(5)


if __name__=="__main__":
    main()
