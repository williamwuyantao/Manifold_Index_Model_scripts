"""Same-data multi-GPU qualification; writes only under --workdir."""
from pathlib import Path
import argparse
import ctypes as C
from dataclasses import asdict
import hashlib
import json
import sys
import tempfile
import time
import shutil
import subprocess

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from multi_gpu_estimator import MultiGpuEstimator, _Stats, _hash
from manor_m_cuda import ManorMCuda

STATE = ('Tan_next', 'Cen_next', 'num_neigh', 'heavy_mask', 'keep_mask', 'solver_failed')


def same(left, right, label):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        finite = np.isfinite(a) & np.isfinite(b)
        error = float(np.abs(a[finite].astype(float)-b[finite].astype(float)).max(initial=0))
        raise AssertionError((label, a.shape, b.shape, error))


def phase1(args, d=20, my=6):
    from phase1 import _hybrid_phase1, _cuts
    n, k = 257, 234
    rng = np.random.default_rng(8301031)
    x = rng.normal(0, .15, (n, d)).astype('f4')
    y = rng.normal(size=(n, my)).astype('f4')
    index = ((np.arange(n)[:, None]+np.arange(k)[None, :]) % n).astype('i4')
    index.flags.writeable = False
    started = time.perf_counter()
    with ManorMCuda(args.baseline).context(0) as context:
        reference, _ = _hybrid_phase1(context, x, index, np.array([k], 'i4'),
                                     _cuts(d, k), 2, 128*1024**2)
    with tempfile.TemporaryDirectory(dir=args.workdir) as directory:
        with MultiGpuEstimator(args.library,x,y,devices=args.devices,workdir=directory,
                reserve_bytes=128*1024**2,pinned_bytes_per_device=8*1024**2) as model:
            model.index = index
            value, stats = model._phase1(k, _cuts(d, k))
            for name in value:
                same(reference[name],value[name],('Phase1',d,my,len(args.devices),name))
    result = dict(case='parallel_phase1',D=d,MY=my,devices=len(args.devices),
                  seven_phase1_arrays_bitwise=True,global_owner_thresholds=True,
                  owners=stats['owner_count'],wall_seconds=time.perf_counter()-started)
    print(json.dumps(result),flush=True)
    return result


def functional(args, case, d=20, my=6):
    if case == 'sparse_count':
        return sparse_count(args, d, my)
    n, k, steps = 257, 80, 3
    rng = np.random.default_rng(8301031)
    x = rng.normal(0, .15, (n, d)).astype('f4')
    y = rng.normal(size=(n, my)).astype('f4')
    tangent = np.zeros((n, 2, d), 'f4')
    tangent[:, 0, 0] = tangent[:, 1, 1] = 1
    owner = (np.arange(n) % 3 != 1).astype('u1')
    index = ((np.arange(n)[:, None]+np.arange(k)[None, :]) % n).astype('i4')
    index.flags.writeable = False
    views = {name: dict(x=rng.normal(0, .15, (count, d)).astype('f4'),
                        y=rng.normal(size=(count, my)).astype('f4'),
                        tangent=tangent[:count].copy())
             for name, count in [('testing', 17), ('validation', 13)]}
    if case == 'query_shards':
        views = {'testing': {key: value for key, value in views['testing'].items()
                             if key != 'tangent'}}
    hs, rhos = np.array([1, .9, .8], 'f4'), np.array([1, .7, .5], 'f4')
    heavy = 1e20 if case == 'stopped' else .1
    sparse = 1e30 if case in ('sparse', 'query_shards') else 10
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=args.workdir) as directory:
        with MultiGpuEstimator(args.library, x, y, devices=args.devices, workdir=directory,
                working_bytes=12*1024**3, reserve_bytes=128*1024**2,
                host_bytes=16*1024**3, maximum_tile_owners=31,
                pinned_bytes_per_device=17*k*8,
                force_index_stream=case != 'resident', force_host_relay=case == 'relay') as model:
            # Explicit functional fixture, not a scientific P1/kNN result.
            model.index, model.k = index, k
            model.p1 = dict(Tan0=tangent, Cen0=x, owner_mask0=owner)
            model._create_model(heavy_factor=heavy)
            for iteration in range(1, steps+1):
                stats = _Stats()
                model._check(model.lib.manor_m_multi_step(model._model, hs[iteration-1], rhos[iteration-1], C.byref(stats)))
                state = model.snapshot()
                with ManorMCuda(args.baseline).context(0) as context:
                    reference, control = context.phase2(x, y, index, tangent, x, owner,
                        1., rhos[:iteration], 2, iteration_count=iteration,
                        h_schedule=hs[:iteration], return_histories=False,
                        direct_output_buffers=True, heavy_factor=heavy,
                        reserve_memory_fraction=0, reserve_memory_bytes=128*1024**2,
                        aggregation_tile_owners=31, index_tile_owners=17)
                for name in STATE:
                    same(reference[name], state[name], (case, d, my, iteration, name))
                assert (stats.valid_iterations, stats.stopped_no_heavy) == (control.valid_iterations, control.stopped_no_heavy)
                position = max(0, stats.valid_iterations-1)
                metrics = model.evaluate_state(h=hs[position], rho=rhos[position],
                    views=views, eval_rhos=[.5, 1.], eval_hs=[.5, 1., 1.5],
                    sum_sparse_threshold=sparse, initial_normalization=True)
                for view, query in views.items():
                    for mode in ('response', 'tangent'):
                        if ('y' if mode == 'response' else 'tangent') not in query:
                            continue
                        with ManorMCuda(args.baseline).context(0) as context:
                            value, _ = context.phase2(x, y, None, reference['Tan_next'], reference['Cen_next'], owner,
                                float(hs[position]), rhos[position:position+1], 2,
                                iteration_count=1, h_schedule=hs[position:position+1],
                                test_x=query['x'], test_y=query['y'] if mode == 'response' else None,
                                test_tangent=query['tangent'] if mode == 'tangent' else None,
                                evaluated_iterations=[0], eval_rho_scales=[.5, 1.], eval_h_scales=[.5, 1., 1.5],
                                evaluation_only=True, generated_neighbor_count=k,
                                return_histories=False, direct_output_buffers=True,
                                sum_sparse_threshold=sparse, reserve_memory_fraction=0,
                                reserve_memory_bytes=128*1024**2)
                        for family in (('MSE', 'MSECoordinate') if mode == 'response' else ('DevTan',)):
                            same(value['Eval'+family+'0'], metrics[(view, family)], (case, d, my, view, family))
            if case == 'relay' and len(args.devices) > 1:
                assert stats.relayed_bytes > 0 and stats.peer_bytes == 0
            row = dict(case=case, D=d, MY=my, iterations=steps,
                       model_states_bitwise=True, metrics_bitwise=True,
                       stats={key: getattr(stats, key) for key, _ in stats._fields_},
                       wall_seconds=time.perf_counter()-started)
            print(json.dumps(row), flush=True)
            return row


def sparse_count(args, d=20, my=6):
    """A finite readout with positive weights but fewer than 2*D owners.

    Disabling the weight threshold isolates the automatic count threshold.
    The explicit plane state avoids extrapolating an unrelated fitted state.
    """
    n, k = 257, 80
    rng = np.random.default_rng(8301031)
    x = rng.normal(0, .04, (n,d)).astype('f4')
    y = rng.normal(size=(n,my)).astype('f4')
    tangent = np.zeros((n,2,d),'f4')
    tangent[:,0,0] = tangent[:,1,1] = 1
    owner = np.zeros(n,'u1'); owner[:2*d-1] = 1
    index = ((np.arange(n)[:,None]+np.arange(k)[None,:])%n).astype('i4')
    views = {'testing':dict(x=x[:17].copy(),y=y[:17].copy(),tangent=tangent[:17].copy())}
    with tempfile.TemporaryDirectory(dir=args.workdir) as directory:
        with MultiGpuEstimator(args.library,x,y,devices=args.devices,workdir=directory,
                reserve_bytes=128*1024**2) as model:
            model.index, model.k = index, k
            model.p1 = dict(Tan0=tangent,Cen0=x,owner_mask0=owner)
            model._create_model()
            actual = model.evaluate_state(h=1.,rho=1.,views=views,eval_rhos=[1.],
                eval_hs=[1.],sum_sparse_threshold=0,initial_normalization=True)
            for mode in ('response','tangent'):
                q=views['testing']
                with ManorMCuda(args.baseline).context(0) as context:
                    expected,stats=context.phase2(x,y,None,tangent,x,owner,1.,[1.],2,
                        iteration_count=1,h_schedule=[1.],test_x=q['x'],
                        test_y=q['y'] if mode=='response' else None,
                        test_tangent=q['tangent'] if mode=='tangent' else None,
                        evaluated_iterations=[0],eval_rho_scales=[1.],eval_h_scales=[1.],
                        evaluation_only=True,generated_neighbor_count=k,
                        return_histories=False,direct_output_buffers=True,
                        sum_sparse_threshold=0,reserve_memory_fraction=0,
                        reserve_memory_bytes=128*1024**2)
                assert stats.sparse_pair_count==len(q['x'])
                for family in (('MSE','MSECoordinate') if mode=='response' else ('DevTan',)):
                    assert np.isfinite(expected['Eval'+family+'0']).all()
                    same(expected['Eval'+family+'0'],actual[('testing',family)],('sparse_count',family))
    row=dict(case='sparse_count',D=d,MY=my,devices=len(args.devices),
             active_owners=int(owner.sum()),automatic_count_threshold=2*d,
             weight_threshold=0,model_states_bitwise=True,metrics_bitwise=True)
    print(json.dumps(row),flush=True)
    return row


def real(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scaling_law_N'))
    from cuda_core_data import Stage2Data
    import h5py
    with h5py.File(args.dataset, 'r') as f:
        print('DATASET_KEYS', list(f.keys()), flush=True)
    # The application adapter loads its canonical layout, not an inferred H5 layout.
    import importlib.util
    spec = importlib.util.spec_from_file_location('tube', args.tube_module)
    tube = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tube)
    data = tube.load_data(Path(args.dataset), expected_c=args.c)
    assert len(data.arrays['Xvalid']) == len(data.arrays['Xtest']) == 5000
    from fixed_hyperparameter_screen_cuda import candidate_at_n
    parameters = json.loads(Path(args.parameters).read_text())
    candidate, _ = candidate_at_n(parameters, len(data.arrays['X']), data.arrays['X'].shape[1])
    from estimator import CudaCoreEstimator
    from stage2_memory import MemoryPolicy
    from bumpy_global_tube_experiment import evaluate_shared_trajectory
    before = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    started = time.perf_counter()
    with CudaCoreEstimator(args.baseline, data.arrays['X'], data.arrays['Y'], dataset_id='multi-gpu-control',
            memory_policy=MemoryPolicy(working_bytes=12*1024**3, phase1_working_bytes=12*1024**3),
            retained_bytes=8*1024**3) as control:
        reference = evaluate_shared_trajectory(control, data, candidate)
        reference_p1 = {name: value.copy() for name, value in control.p1.items()}
        reference_index = control.index.copy()
    baseline_seconds = time.perf_counter()-started
    started = time.perf_counter()
    with MultiGpuEstimator(args.library, data.arrays['X'], data.arrays['Y'], devices=args.devices,
            workdir=args.workdir/'fit', reserve_bytes=128*1024**2, working_bytes=12*1024**3) as model:
        value = model.fit(candidate, data)
        same(reference_index, model.index, 'parallel exact kNN')
        for name in model.p1:
            same(reference_p1[name], model.p1[name], 'P1 '+name)
    multi_seconds = time.perf_counter()-started
    for view in ('validation', 'testing'):
        a, b = reference[view], value[view]
        for family in ('response', 'tangent'):
            for state, digest in a['final_state_sha256'][family].items():
                assert b['final_state_sha256'][family][state] == digest, (view, state)
        for left, right in zip(a['iteration_rows'], b['iteration_rows']):
            for field in ('iteration', 'state_iteration', 'terminal_noop', 'relMSE_total', 'relMSE_coord', 'devTan'):
                assert left[field] == right[field], (view, left['iteration'], field, left[field], right[field])
    assert hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest() == before
    row = dict(canonical_data_sha256=before, N=len(data.arrays['X']),
               query_count=5000, all_iterations_bitwise=True, phase1_bitwise=True,
               exact_index_bitwise=True, baseline_seconds=baseline_seconds,
               multi_seconds=multi_seconds, measurements=value)
    (args.workdir/'real-result.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in row.items() if k != 'measurements'}), flush=True)
    return row


def qualify(args):
    if not args.dataset or not args.parameters or not args.tube_module:
        raise ValueError('Qualification requires the canonical dataset, parameters and dataset adapter')
    sanitizer = shutil.which('compute-sanitizer')
    if sanitizer is None:
        raise FileNotFoundError('compute-sanitizer')
    import h5py
    with h5py.File(args.dataset, 'r') as source:
        if len(source['geometry/train/x_noisy']) > 100000:
            raise ValueError('Use a canonical qualification case of at most 100k points')
    devices = args.devices[:]
    for count in range(1, len(devices)+1):
        args.devices = devices[:count]
        phase1(args)
        functional(args, 'streamed')
    args.devices = devices
    for case in ('resident','sparse','sparse_count','stopped','relay','query_shards'):
        functional(args, case)
    real(args)
    for mode in ('--phase1','--case'):
        for tool in ('memcheck','synccheck','racecheck'):
            command = [sanitizer,'--tool',tool,'--error-exitcode','88',sys.executable,
                str(Path(__file__)), '--library',args.library,'--baseline',args.baseline,
                '--devices',','.join(map(str,devices[:2])),
                '--workdir',str(args.workdir/('sanitizer-'+mode[2:]+'-'+tool)),mode]
            if mode == '--case':command.append('query_shards')
            with (args.workdir/('sanitizer-'+mode[2:]+'-'+tool+'.log')).open('x') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    manifest = dict(status='qualified',
        core_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        metric_sum_contract='fp32-query256-tree-v1',tested_device_counts=list(range(1,len(devices)+1)),
        absolute_tolerance=5e-7,relative_tolerance=5e-4,winner_absolute_tolerance=5e-8,
        phase1_and_model_states_and_metrics_bitwise=True,
        memcheck='0 errors',synccheck='0 errors',racecheck='0 hazards')
    args.qualification_output.write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--devices', default='0,1')
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--case', choices=['resident', 'streamed', 'sparse', 'sparse_count', 'stopped', 'relay', 'query_shards'])
    p.add_argument('--phase1', action='store_true')
    p.add_argument('--qualification-output', type=Path)
    p.add_argument('--dataset')
    p.add_argument('--tube-module')
    p.add_argument('--parameters')
    p.add_argument('--c', type=float, default=.75)
    args = p.parse_args()
    args.devices = [int(s) for s in args.devices.split(',')]
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.qualification_output:
        qualify(args)
    elif args.phase1:
        proof=phase1(args)
        (args.workdir/'phase1-complete.json').write_text(json.dumps(proof,indent=2)+'\n')
    elif args.dataset:
        real(args)
    else:
        proof = [functional(args, case) for case in ([args.case] if args.case else
                 ['resident', 'streamed', 'sparse', 'stopped', 'relay'])]
        (args.workdir/'functional-complete.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    main()
