"""Small-memory validation pilots for fixed global tubes; no testing access.

These evaluated tuples are retained as candidates for cross-N selection after
Stage1 has set its ranges. They are not a completed Stage1 or final selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","source-root","application-source","plan","library","runtime"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--label",default="small-n")
    args=parser.parse_args()
    for folder in (args.source_root/"scaling_law_N",args.source_root/"manor_m/cuda/python",
                   args.application_source/"scaling_law_N",args.application_source/"dataset_generator"):
        sys.path.insert(0,str(folder))
    import bumpy_global_tube as tube
    from bumpy_global_tube_experiment import (estimator_for, claim, report, save, read,
        evaluate_trajectory, trajectory_complete, preserve_endpoint_record,check_previous_endpoints)
    from calibration_protocol import fixed_n1m_candidate
    import stage2_cuda_core as production
    production.validate_runtime(args.library,args.runtime)
    args.working_gib=2
    args.low_memory=True
    reference=read(args.plan)["parameters"]["bumpy_sphere_v2_D20_Y2"]
    probes=[dict(name="existing_Y2_reference",parameters=reference)]
    for field in ("alpha","K_offset","h_scale","rho_init","bite","eval_rho","eval_h"):
        values=([reference[field]-4,reference[field]+4] if field=="K_offset" else
                [max(0,reference[field]-2),reference[field]+2] if field=="bite" else
                [reference[field]/1.4,reference[field]*1.4])
        if field=="alpha": values=[round(v*1_000_000)/1_000_000 for v in values]
        for index,value in enumerate(values):
            probes.append(dict(name="pilot_%s_%d"%(field,index),parameters=dict(reference,**{field:value})))
    with claim(args.root/(args.label+".lock")) as owned:
        if not owned: raise RuntimeError("small-N worker already running")
        save(args.root/"smallN-pilot-plan.json",dict(probes=probes,
            N_values=[1000,2000,3000,4000,5000,7000,10000,20000,30000,40000,50000,70000,100000],
            selection_data="validation only",testing_used=False,final_selection=False))
        for c in (.25,.5):
            stage_b=args.root/tube.label(c)/"stage2/stage_B_plan.json"
            if stage_b.exists():
                assert read(stage_b)==probes
            else:
                save(stage_b,probes)
            dataset=args.root/"datasets"/(tube.label(c)+".h5")
            data_all=tube.load_data(dataset,include_testing=False,expected_c=c)
            for n in [100000,70000,50000,40000,30000,20000,10000,7000,5000,4000,3000,2000,1000]:
                # Write the exact Stage2 batch layout. The main Stage2 worker
                # reuses these finished N/probe batches after Stage1 completes.
                target=args.root/tube.label(c)/"stage2/stage_B"/("N%d.json"%n)
                if trajectory_complete(target,c):continue
                with claim(target.with_suffix(".lock")) as task:
                    if not task or trajectory_complete(target,c):continue
                    preserve_endpoint_record(target)
                    data=data_all.for_n(n)
                    records=[];cache={}
                    candidates=[]
                    for probe in probes:
                        p=probe["parameters"]
                        candidate,_=fixed_n1m_candidate(dict(p,kSlice=round(p["alpha"]*1_000_000)),n,20)
                        candidates.append((candidate["kSlice"],probe,candidate))
                    with estimator_for(args,data,"global-smallN:"+tube.label(c)+":N%d"%n) as estimator:
                        for _,probe,candidate in sorted(candidates,key=lambda x:x[0]):
                            effective=tuple(candidate[k] for k in ("kSlice","K","h_scale","rho_init","bite","eval_rho","eval_h"))
                            report(args,state="computing_smallN_validation_pilot",c=c,N=n,probe=probe["name"])
                            if effective not in cache:
                                cache[effective]=evaluate_trajectory(estimator,data,candidate,"validation")
                            records.append(dict(probe=probe["name"],parameters=probe["parameters"],
                                candidate=candidate,measurement=cache[effective],
                                validation_relMSE_total=cache[effective]["testing_relMSE_total"],
                                testing_used=False))
                    check_previous_endpoints(target,records)
                    save(target,dict(N=n,c=c,records=records,dataset_identity=data.identity,
                                     input_arrays_sha256=data.hashes,testing_used=False,final_selection=False))
        report(args,state="smallN_pilot_complete")


if __name__=="__main__":main()
