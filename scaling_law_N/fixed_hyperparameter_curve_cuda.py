"""Evaluate a validation-selected, N-invariant seven-tuple on all Stage-2 N.

Uses the existing production data loader, capacity policy and separate response/
tangent evaluator. Selection is frozen before any testing data is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import stage2_cuda_core as stage2
from calibration_protocol import SUBMILLION_N_VALUES
from calibration_protocol import now
from cuda_core_data import load_canonical
from estimator import CudaCoreEstimator
from fixed_hyperparameter_screen_cuda import (PARAMETERS, candidate_at_n,
                                              normalize_parameters, setup_lock)
from stage2_memory import MemoryPolicy

SCHEMA = 'mfm-fixed-seven-selected-curve.v1'


def load_selection(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if (plan.get('schema') != 'mfm-fixed-seven-combined-validation-summary.v1'
            or plan.get('selection_data') != 'validation only'
            or plan.get('testing_used') is not False):
        raise ValueError('curve selection must be frozen from validation only')
    selected = {}
    for report in plan['reports']:
        setup = report['setup_id']
        if (setup in selected or report.get('testing_used') is not False
                or report.get('selection_data') != 'validation only'
                or report.get('fixed_parameters_across_N') != list(PARAMETERS)):
            raise ValueError('invalid or duplicate selected setup')
        winner = report['combined_winner']
        ranked = sorted(report['objectives'], key=lambda item: (
            item['max_regret_fraction'], item['geometric_mean_regret_fraction'],
            tuple(item['parameters'][field] for field in PARAMETERS)))
        if not ranked or winner != ranked[0]:
            raise ValueError('winner differs from validation-only minimax selection')
        parameters = normalize_parameters(winner['parameters'])
        anchors = winner['validation_relMSE_total_by_N']
        if (set(anchors) != {str(n) for n in report['N_values']}
                or any(not math.isfinite(v) or v <= 0 for v in anchors.values())):
            raise ValueError('selection anchor scores are invalid')
        selected[setup] = dict(parameters=parameters, anchor_validation=anchors,
                               screen_stages=report['screen_stages'])
    return hashlib.sha256(raw).hexdigest(), selected


def expected(setup, N, selection, selection_sha, runtime):
    candidate, _ = candidate_at_n(selection['parameters'], N, setup['D'])
    return dict(schema=SCHEMA, setup_id=setup['id'], N=N,
                fixed_parameters=selection['parameters'], candidate=candidate,
                selection_sha256=selection_sha, selection_data='validation only',
                testing_used_for_selection=False, testing_count=5000,
                validation_count=5000, cuda_runtime=runtime)


def validate_record(record, spec, identity=None, hashes=None):
    for key, value in spec.items():
        if record.get(key) != value:
            raise ValueError('saved selected curve differs: ' + key)
    for view in ('validation', 'testing'):
        result = record.get(view, {})
        if any(result.get(k) != v for k, v in spec['candidate'].items()):
            raise ValueError('evaluated candidate changed in ' + view)
        for key in ('testing_relMSE_total', 'testing_relMSE_coord', 'testing_devTan'):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid independent metric: ' + view + '/' + key)
        if result.get('python_diagnostics', {}).get('metric_pass_contract') != 'separate-response-and-tangent-v1':
            raise ValueError('response/tangent evaluation contract changed')
    if (not record.get('completed_at') or record.get('testing_Xclean_loaded') is not False
            or record.get('anchor_validation_matches') is False):
        raise ValueError('incomplete or incompatible final curve record')
    if identity is not None and record.get('dataset_identity') != identity:
        raise ValueError('curve dataset identity changed')
    if hashes is not None and record.get('input_arrays_sha256') != hashes:
        raise ValueError('curve input arrays changed')


def summarize(root, setup, selection, selection_sha, runtime, n_values, found):
    stage2.save(root / 'curve_summary.json', dict(schema=SCHEMA,
        setup_id=setup['id'], selection_sha256=selection_sha,
        fixed_parameters=selection['parameters'], N_values=list(n_values),
        testing_used_for_selection=False, independent_testing=True,
        completed_at=now(), cuda_runtime=runtime, rows=[dict(N=N,
            validation_relMSE=found[N]['validation']['testing_relMSE_total'],
            validation_devTan=found[N]['validation']['testing_devTan'],
            testing_relMSE=found[N]['testing']['testing_relMSE_total'],
            testing_devTan=found[N]['testing']['testing_devTan']) for N in n_values]))


def run_setup(calibration, selection, selection_sha, *, dataset, output, library,
              runtime, n_values, resume=False, max_new_n=None, memory_policy=None,
              data_loader=load_canonical, estimator_factory=CudaCoreEstimator):
    setup = stage2.normalize_calibration(calibration)['setup']
    root = Path(output)
    if root.exists() and not resume:
        raise ValueError('existing curve requires --resume')
    specs = {N: expected(setup, N, selection, selection_sha, runtime) for N in n_values}
    found = {}
    for N, spec in specs.items():
        path = root / ('N%d.json' % N)
        if path.exists():
            found[N] = stage2.read(path)
            validate_record(found[N], spec)
    if len(found) == len(n_values):
        with setup_lock(root):
            if not (root / 'curve_summary.json').exists():
                summarize(root, setup, selection, selection_sha, runtime, n_values, found)
        return dict(state='complete_noop', completed_N=list(n_values), new_N=0, cuda_contexts=0)
    with setup_lock(root):
        stage2.verify_or_create(root / 'selection.json', dict(
            setup=setup, selection=selection, selection_sha256=selection_sha,
            N_values=list(n_values), cuda_runtime=runtime))
        data = data_loader(dataset, setup, include_testing=True)
        todo = [N for N in n_values if N not in found]
        new = contexts = 0
        with stage2.PrefixPrefetcher(True) as prefixes:
            prefixes.bind(data)
            for position, N in enumerate(todo):
                if max_new_n is not None and new >= max_new_n:
                    break
                n_data, hashes, timing = prefixes.take(N)
                if position + 1 < len(todo) and (max_new_n is None or new + 1 < max_new_n):
                    prefixes.submit(todo[position + 1])
                with estimator_factory(library, n_data.arrays['X'], n_data.arrays['Y'],
                        dataset_id=setup['id'] + ':fixed-seven-curve:N' + str(N), device=0,
                        retained_bytes=0, memory_policy=memory_policy,
                        phase1_cache_bytes=1024**3) as estimator:
                    contexts += 1
                    candidate = specs[N]['candidate']
                    validation = stage2.evaluate_candidate(estimator, n_data, candidate, 'validation')
                    anchor = selection['anchor_validation'].get(str(N))
                    matches = None
                    if anchor is not None:
                        diff = abs(validation['testing_relMSE_total'] - anchor)
                        matches = (diff <= runtime['absolute_tolerance'] and
                                   diff / abs(anchor) <= runtime['relative_tolerance'])
                        if not matches:
                            stage2.save(root / ('N%d-validation-mismatch.json' % N),
                                        dict(expected=anchor, measured=validation))
                            raise ValueError('selected anchor validation changed before testing')
                    testing = stage2.evaluate_candidate(estimator, n_data, candidate, 'testing')
                    record = dict(specs[N], validation=validation, testing=testing,
                        dataset_identity=n_data.identity, input_arrays_sha256=hashes,
                        testing_Xclean_loaded=False, anchor_validation_matches=matches,
                        prefix_timing=timing, completed_at=now())
                    validate_record(record, specs[N], n_data.identity, hashes)
                    stage2.save(root / ('N%d.json' % N), record)
                    found[N] = record
                    new += 1
                    print(json.dumps(dict(event='fixed_curve_complete', setup=setup['id'],
                                          N=N, completed=len(found), total=len(n_values))), flush=True)
        result = dict(state='complete' if len(found) == len(n_values) else 'paused',
                      completed_N=sorted(found), new_N=new, cuda_contexts=contexts)
        if result['state'] == 'complete':
            summarize(root, setup, selection, selection_sha, runtime, n_values, found)
        stage2.save(root / 'curve_status.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'runtime-json', 'dataset-root', 'calibration-root',
                 'selection-json', 'result-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setups', required=True)
    parser.add_argument('--n-values', default=','.join(map(str, SUBMILLION_N_VALUES)))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-new-n', type=int)
    parser.add_argument('--phase1-index-dir', type=Path, required=True)
    args = parser.parse_args()
    n_values = list(map(int, args.n_values.split(',')))
    if not n_values or len(set(n_values)) != len(n_values) or any(N not in SUBMILLION_N_VALUES for N in n_values):
        parser.error('N values must be distinct members of the frozen 27-N schedule')
    if args.max_new_n is not None and args.max_new_n < 1:
        parser.error('max-new-n must be positive')
    selection_sha, selected = load_selection(args.selection_json)
    setups = args.setups.split(',')
    if len(set(setups)) != len(setups) or set(setups) - set(selected):
        parser.error('unknown or duplicated selected setup')
    runtime = stage2.validate_runtime(args.library, args.runtime_json)
    runtime = dict(runtime, fixed_curve_entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    policy = MemoryPolicy(working_bytes=0, pinned_bytes=2 * 1024**3,
        phase1_working_bytes=0, phase1_index_heap_bytes=8 * 1024**3,
        phase1_spill_directory=str(args.phase1_index_dir))
    for setup_id in setups:
        calibration = stage2.read(args.calibration_root / (setup_id + '__noise1pct') / 'calibration_final.json')
        if calibration['setup']['id'] != setup_id:
            raise ValueError('calibration setup differs')
        dataset = args.dataset_root / Path(calibration['setup']['dataset']).name
        if not dataset.is_file():
            raise ValueError('missing canonical dataset: ' + str(dataset))
        result = run_setup(calibration, selected[setup_id], selection_sha,
            dataset=dataset, output=args.result_root / (setup_id + '__noise1pct'),
            library=args.library, runtime=runtime, n_values=n_values,
            resume=args.resume, max_new_n=args.max_new_n, memory_policy=policy)
        print(json.dumps(dict(setup=setup_id, **result)), flush=True)


if __name__ == '__main__':
    main()
