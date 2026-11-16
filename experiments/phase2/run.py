"""Reproduce the fixed-parameter iteration and projection experiment.

Use the supplied IID datasets, or regenerate missing data from the scientific
recipe. Outputs contain measurements and parameters only.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / relative) for relative in (
    'scaling_law_N/analysis', 'scaling_law_N', 'dataset_generator', 'manor_m/cuda/python')]


def read(path):
    return json.loads(path.read_text())


def check_dataset(path, n, repetition):
    with h5py.File(path, 'r') as data:
        assert data.attrs['state'] == 'complete'
        identity = json.loads(data.attrs['identity'])
        assert identity['N'] == n and identity['repetition'] == repetition
        assert data['train/X'].shape == (n, 20)
        assert data['Y2/train/observed'].shape == (n, 6)
        for view in ('validation', 'testing'):
            assert data[view + '/X'].shape == (5000, 20)
            assert data[view + '/T'].shape == (5000, 2, 20)
            assert data['Y2/' + view + '/clean'].shape == (5000, 6)


def true_basis(path):
    import bumpy_sphere_generator as geometry
    from phase2_measurements import true_basis_from_latent
    with h5py.File(path, 'r') as data:
        for view in ('validation', 'testing'):
            rebuilt = true_basis_from_latent(data[view + '/latent'][:], geometry)
            np.testing.assert_array_equal(rebuilt, data[view + '/T'][:])
        truth = np.empty((len(data['train/X']), 2, 20), dtype=np.float32)
        for start in range(0, len(truth), 65536):
            truth[start:start + 65536] = true_basis_from_latent(
                data['train/latent'][start:start + 65536], geometry)
    return truth


def run_case(args, dataset, n, repetition, parameters):
    import bumpy_iid_repetitions as iid
    import phase2_measurements as diagnostic
    from phase2_measurements import direct_error, STATE_KEYS
    import cuda_core_data as data_api
    from estimator import CudaCoreEstimator
    from fixed_hyperparameter_screen_cuda import candidate_at_n
    from phase1 import _phase2_schedules
    from stage2_memory import MemoryPolicy

    name = 'N%d-r%02d' % (n, repetition)
    readout_target = args.output / 'readout' / (name + '.json')
    direct_target = args.output / 'direct' / (name + '.json')
    if readout_target.exists() and direct_target.exists():
        for target in (readout_target, direct_target):
            old = read(target)
            assert old['N'] == n and old['repetition'] == repetition
            assert old['fixed_parameters'] == parameters
        print('Already complete: ' + name, flush=True)
        return
    data = iid.load_data(dataset, 2, data_api)
    truth = true_basis(dataset)
    candidate, clipped = candidate_at_n(parameters, n, 20)
    common = dict(setup_id='bumpy_sphere_v2_D20_Y2', N=n, repetition=repetition,
        fixed_parameters=parameters, candidate=candidate, k_clipped=clipped,
        testing_used_for_selection=False)
    policy = MemoryPolicy(working_bytes=args.working_gib * 1024**3,
        phase1_working_bytes=args.working_gib * 1024**3, phase1_mode=2,
        phase1_index_heap_bytes=2 * 1024**3,
        phase1_spill_directory=str(args.output / 'scratch'))
    requested = list(range(candidate['K'] + 1))
    with CudaCoreEstimator(args.library, data.arrays['X'], data.arrays['Y'],
            dataset_id='phase2:' + name, memory_policy=policy, retained_bytes=512 * 1024**2,
            phase1_cache_bytes=1024**3) as estimator:
        estimator.prepare(candidate['kSlice'])
        initial = {key:np.array(value, copy=True, order='C')
                   for key,value in estimator.p1.items() if key in estimator.p1_hashes}
        trajectories, endpoints = {}, {}
        for mode, family in (('tangent', 'EvalDevTan'), ('response', 'EvalMSE')):
            estimator.set_query('iteration-' + mode, data.arrays['Xtest'],
                y=data.arrays['Ytest'] if mode == 'response' else None,
                tangent=data.arrays['Ttest'] if mode == 'tangent' else None)
            result, stats, unused = estimator.evaluate(h_scale=candidate['h_scale'],
                rho_init=candidate['rho_init'], bite=candidate['bite'], iterations=requested,
                eval_rhos=[candidate['eval_rho']], eval_hs=[candidate['eval_h']])
            assert stats.valid_iterations == candidate['K'] and not stats.stopped_no_heavy
            trajectories[mode] = diagnostic.trajectory(result, family, stats.valid_iterations)
            endpoints[mode] = {key:diagnostic.array_sha(result[key]) for key in STATE_KEYS}
            if mode == 'response':
                final_tangent = np.array(result['Tan_next'], copy=True, order='C')
                final_center = np.array(result['Cen_next'], copy=True, order='C')
            del result
        assert endpoints['tangent'] == endpoints['response']
        rows = [dict(iteration=k, testing_relMSE_total=mse / data.variance('testing')[1],
                     testing_devTan=trajectories['tangent'][k])
                for k,mse in enumerate(trajectories['response'])]
        tangent, center = initial['Tan0'], initial['Cen0']
        owner = initial['owner_mask0']
        mask = owner.astype(bool)
        index = estimator.index
        assert index is not None and estimator.index_streamed
        index_hash = diagnostic.array_sha(index)
        unused1, unused2, hs, rhos = _phase2_schedules(initial['Rad0'], mask,
            initial['axial_tan0'], candidate['h_scale'], candidate['rho_init'],
            candidate['bite'], 20, 2, candidate['K'])
        direct_rows = [dict(iteration=0, direct_tangent_error=direct_error(tangent[mask], truth[mask]))]
        for step in range(candidate['K']):
            context = estimator.sweep.context
            context._resident_clear('phase1')
            context._resident_put('phase1.Tan0', tangent)
            context._resident_put('phase1.Cen0', center)
            context._resident_put('phase1.owner', owner)
            result, stats = context.phase2(data.arrays['X'], data.arrays['Y'], index,
                tangent, center, owner, float(hs[step]), rhos[step:step+1], 2,
                iteration_count=1, h_schedule=hs[step:step+1], evaluated_iterations=None,
                reserve_memory_fraction=0, reserve_memory_bytes=128 * 1024**2,
                maximum_query_block=8192, return_histories=False, direct_output_buffers=True,
                aggregation_tile_owners=8192, index_tile_owners=8192)
            assert stats.valid_iterations == 1 and not stats.stopped_no_heavy
            tangent, center = result['Tan_next'], result['Cen_next']
            tangent.setflags(write=False)
            center.setflags(write=False)
            direct_rows.append(dict(iteration=step+1,
                direct_tangent_error=direct_error(tangent[mask], truth[mask]),
                heavy_count=int(np.count_nonzero(result['heavy_mask'])),
                keep_count=int(np.count_nonzero(result['keep_mask']))))
        assert {key:diagnostic.array_sha(result[key]) for key in STATE_KEYS} == endpoints['response']
        assert diagnostic.array_sha(index) == index_hash
        del result
    controls = diagnostic.projection_readout(
        argparse.Namespace(production=ROOT, library=args.library, low_memory=False), data, candidate,
        initial, final_tangent, final_center, truth, candidate['K'], CudaCoreEstimator, policy)
    for metric in ('testing_relMSE_total', 'testing_devTan'):
        actual = controls['controls']['final_projection'][metric]
        expected = rows[-1][metric]
        assert abs(actual - expected) <= 5e-7
        assert abs(actual - expected) <= 5e-4 * abs(expected)
    public_controls = {role:{key:value[key] for key in ('testing_relMSE_total', 'testing_devTan')}
                       for role,value in controls['controls'].items()}
    iid.save(readout_target, dict(common, rows=rows,
        projection_readout=public_controls, projection_control=dict(
            state_iteration=candidate['K'], h_base=controls['h_base'], rho_base=controls['rho_base'],
            eval_h=candidate['eval_h'], eval_rho=candidate['eval_rho'],
            true_tangent_used_in_fitting=False)))
    iid.save(direct_target, dict(common, rows=direct_rows, owner_count=int(mask.sum()),
                                endpoint_byte_exact=True))
    print('Complete: ' + name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'dataset/bumpy_sphere/iid')
    parser.add_argument('--library', type=Path, default=ROOT / 'build/libmanor_m_cuda.so')
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced/phase2_experiment')
    parser.add_argument('--n-values', nargs='+', type=int, default=[500000, 700000, 1000000])
    parser.add_argument('--repetitions', nargs='+', type=int, default=list(range(1, 11)))
    parser.add_argument('--working-gib', type=int, default=24)
    parser.add_argument('--generate-data', action='store_true')
    parser.add_argument('--check-datasets', action='store_true')
    args = parser.parse_args()
    assert set(args.n_values) <= {500000, 700000, 1000000}
    assert set(args.repetitions) <= set(range(1, 11))
    parameters = read(ROOT / 'config/phase2_experiment.json')['fixed_parameters']
    for n in args.n_values:
        for repetition in args.repetitions:
            dataset = args.dataset_dir / ('N%d-r%02d.h5' % (n, repetition))
            if not dataset.exists() and args.generate_data:
                from scipy.spatial import cKDTree
                import bumpy_iid_repetitions as iid
                import bumpy_sphere_generator as generator
                import cuda_core_data as data_api
                generated = args.output / 'generated'
                plan = read(ROOT / 'config/bumpy_experiment.json')
                iid.save(generated / 'plan.json', plan)
                with np.load(ROOT / 'config/geometry_recipe.npz') as recipe:
                    calibration = dict(tree=cKDTree(recipe['unit']), local_reach=recipe['local_reach'])
                dataset = generated / 'datasets' / dataset.name
                iid.generate(dataset, n, repetition, plan, calibration, generator, data_api)
            check_dataset(dataset, n, repetition)
            if not args.check_datasets:
                args.output.mkdir(parents=True, exist_ok=True)
                run_case(args, dataset, n, repetition, parameters)
    print('All requested datasets verified.' if args.check_datasets else 'All requested fits complete.')


if __name__ == '__main__':
    main()
