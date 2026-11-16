"""Per-iteration tangent measurements and paired projection readouts."""
from dataclasses import replace
import hashlib
import numpy as np

STATE_KEYS = ("Tan_next", "Cen_next", "num_neigh", "heavy_mask", "keep_mask")


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def trajectory(result, family, valid_iterations):
    initial = np.asarray(result[family + "0"], dtype=float).reshape(-1)
    later = np.asarray(result[family], dtype=float)
    if initial.size != 1 or later.shape[0] < valid_iterations:
        raise ValueError("one fixed readout pair and complete iteration storage required")
    values = [float(initial[0])]
    values.extend(float(later[k].reshape(-1)[0]) for k in range(valid_iterations))
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError("invalid executed iteration metric")
    return values


def true_basis_from_latent(unit, generator):
    """Reconstruct the generator's basis at its stored, normalized sphere point."""
    unit=np.asarray(unit,dtype=np.float64)
    radius,gradient,_=generator._family_radius_derivatives(
        unit,generator._family_geometry_parameters("v2"))
    first,second=generator.tangent_frame(unit)
    jac_first=radius[:,None]*first+np.sum(gradient*first,axis=1)[:,None]*unit
    jac_second=radius[:,None]*second+np.sum(gradient*second,axis=1)[:,None]*unit
    tangent_first=jac_first/np.maximum(np.linalg.norm(jac_first,axis=1,keepdims=True),
                                      np.finfo(np.float64).tiny)
    residual=jac_second-np.sum(tangent_first*jac_second,axis=1,keepdims=True)*tangent_first
    tangent_second=residual/np.maximum(np.linalg.norm(residual,axis=1,keepdims=True),
                                      np.finfo(np.float64).tiny)
    tangent=np.zeros((len(unit),2,20),dtype=np.float32)
    tangent[:,0,:3],tangent[:,1,:3]=tangent_first,tangent_second
    return tangent


def projection_states(initial, final_tangent, final_center, true_tangent):
    """Three tangent choices on exactly the same training-row owner support."""
    names=("Tan0","Cen0","Rad0","owner_mask0","num_neigh0","selected_cut0","axial_tan0")
    common={name:np.array(initial[name],copy=True,order="C") for name in names}
    center=np.array(final_center,dtype=np.float32,copy=True,order="C")
    if center.shape!=common["Cen0"].shape:
        raise ValueError("final center layout differs")
    common["Cen0"]=center
    for a in common.values():a.setflags(write=False)
    states={}
    for label,tangent in (("initial_projection",initial["Tan0"]),
                          ("final_projection",final_tangent),("true_projection",true_tangent)):
        basis=np.array(tangent,dtype=np.float32,copy=True,order="C")
        if basis.shape!=common["Tan0"].shape or not np.isfinite(basis).all():
            raise ValueError("projection basis layout/nonfinite mismatch")
        basis.setflags(write=False)
        states[label]=dict(common,Tan0=basis)
    return states


def projection_readout(args,data,candidate,initial,final_tangent,final_center,
                       true_tangent,effective_K,estimator_class,policy):
    """Use the production evaluation-only entry, with the final bandwidth."""
    states=projection_states(initial,final_tangent,final_center,true_tangent)
    initial_hashes={name:array_sha(value) for name,value in initial.items()}
    controls={}; bandwidths=[]
    for label,state in states.items():
        hashes={name:array_sha(value) for name,value in state.items()}
        values={};diagnostics={}
        with estimator_class(getattr(args, "library", args.production/"build/libmanor_m_cuda.so"),
                data.arrays["X"],data.arrays["Y"],dataset_id="projection:"+label,
                memory_policy=replace(policy,phase1_mode=4),
                retained_bytes=512*1024**2 if args.low_memory else 0,
                phase1_cache_bytes=0) as estimator:
            estimator.restore_generated_phase1(candidate["kSlice"],state,
                state_sha256=hashes,initial_state_sha256=initial_hashes,
                state_iteration=effective_K)
            for mode in ("response","tangent"):
                result,stats,diagnostic=estimator.evaluate_checkpoint_state(
                    state_iteration=effective_K,h_scale=candidate["h_scale"],
                    rho_init=candidate["rho_init"],bite=candidate["bite"],
                    eval_rhos=[candidate["eval_rho"]],eval_hs=[candidate["eval_h"]],
                    query_identity=label+":"+mode,test_x=data.arrays["Xtest"],
                    test_y=data.arrays["Ytest"] if mode=="response" else None,
                    test_tangent=data.arrays["Ttest"] if mode=="tangent" else None)
                family="EvalMSE0" if mode=="response" else "EvalDevTan0"
                value=float(np.asarray(result[family]).reshape(-1)[0])
                if not np.isfinite(value) or value<0:
                    raise ValueError("invalid projection-control metric")
                values[mode]=value
                diagnostics[mode]=diagnostic
                bandwidths.append((diagnostic["state_h"],diagnostic["state_rho"]))
                del result
            if estimator.preparation_count!=0 or estimator.phase1_restore_count!=1:
                raise RuntimeError("readout control recomputed Phase1")
        if hashes!={name:array_sha(value) for name,value in state.items()}:
            raise RuntimeError("readout control changed its model")
        controls[label]=dict(testing_relMSE_total=values["response"]/data.variance("testing")[1],
            testing_devTan=values["tangent"],state_sha256=hashes,diagnostics=diagnostics)
    if len(set(bandwidths))!=1:
        raise RuntimeError("projection controls used different bandwidths")
    return dict(controls=controls,state_iteration=effective_K,
        h_base=bandwidths[0][0],rho_base=bandwidths[0][1],
        eval_h=candidate["eval_h"],eval_rho=candidate["eval_rho"],
        held_fixed=["training X/Y","testing X/Y/tangent truth","owner support",
                    "final centers","P1 radius/axial schedule inputs","final readout bandwidth"],
        true_tangent_source="analytic tangent at each training row's stored generative clean footpoint",
        true_tangent_used_in_fitting=False,testing_used_for_selection=False,
        phase2_updates_during_controls=0)


def direct_error(predicted, truth):
    errors, largest_angles, orthogonal = [], [], []
    for start in range(0, len(predicted), 32768):
        q = np.array(predicted[start:start+32768], dtype=np.float64, copy=True)
        gram = np.einsum('nmd,njd->nmj', q, q)
        orthogonal.append(float(np.max(np.abs(gram-np.eye(2)))))
        q[:, 0] /= np.linalg.norm(q[:, 0], axis=1)[:, None]
        for unused in range(2):
            q[:, 1] -= np.einsum('nd,nd->n', q[:, 0], q[:, 1])[:, None]*q[:, 0]
        q[:, 1] /= np.linalg.norm(q[:, 1], axis=1)[:, None]
        cross = np.einsum('nmd,njd->nmj', truth[start:start+32768].astype(np.float64), q)
        errors.append(np.sqrt(np.maximum(0, 2-np.square(cross).sum(axis=(1, 2)))))
        minimum = np.linalg.eigvalsh(cross@np.swapaxes(cross, 1, 2))[:, 0]
        largest_angles.append(np.sqrt(np.maximum(0, 1-minimum)))
    values = np.concatenate(errors)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
        q90=float(np.quantile(values, .9)),
        projector_operator_norm_mean=float(np.concatenate(largest_angles).mean()),
        raw_basis_max_orthogonality_error=max(orthogonal))
