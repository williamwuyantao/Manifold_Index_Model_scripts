"""Extend an immutable validation-selected seven-tuple beyond the Stage2 N range.

This is a selection adapter, not a CUDA implementation. It uses the production
curve evaluator for the new N1M bridge and the production capacity-routed Stage3
entry for larger N. Each host/setup/N has one owner and resumable atomic output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import stage2_cuda_core as stage2
import stage3_cuda_core as stage3
from fixed_hyperparameter_curve_cuda import load_selection, run_setup as run_bridge
from fixed_hyperparameter_screen_cuda import candidate_at_n
from stage2_memory import MemoryPolicy


def selected_configuration(calibration, selected, selection_sha):
    original = stage2.normalize_calibration(calibration)
    parameters = selected['parameters']
    candidate, _ = candidate_at_n(parameters, 1_000_000, original['setup']['D'])
    result = dict(schema='mfm-stage3-fixed-seven-selection.v1',
        status='selected_fixed_seven', setup=original['setup'],
        fixed_parameters=parameters, validation_winner=candidate,
        parameter_selection=dict(source_sha256=selection_sha,
            selection_data='validation only', testing_used_for_selection=False,
            origin='combined_cross_N_fixed_seven', binding_reference_N=1_000_000,
            Stage1_recalibrated=False))
    return stage3.normalize_calibration(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'runtime-json', 'dataset-root', 'calibration-root',
                 'selection-json', 'result-root', 'phase1-index-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--setups', required=True)
    parser.add_argument('--n-values', default='1000000,2000000')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    ns = list(map(int, args.n_values.split(',')))
    if (not ns or ns != sorted(set(ns))
            or any(n < 1_000_000 or n > 16_000_000 for n in ns)):
        parser.error('distinct increasing N in [1M,16M] required')
    digest, selections = load_selection(args.selection_json)
    setups = args.setups.split(',')
    if len(setups) != len(set(setups)) or set(setups) - set(selections):
        parser.error('unknown/duplicate fixed-seven setup')
    runtime = stage2.validate_runtime(args.library, args.runtime_json)
    if runtime.get('production_start_allowed') is not True:
        raise ValueError('host qualification must complete before Stage3 dispatch')
    runtime = dict(runtime, fixed7_stage3_entry_sha256=hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest())
    args.phase1_index_dir.mkdir(parents=True, exist_ok=True)
    # Large-N production must never materialize the logical N*k Phase-1
    # index.  The old automatic policy could fall back to a disk-backed
    # mmap; touching that mmap made the whole exact index resident in host
    # RSS.  Mode 4 is the native generated exact-kNN path: it keeps no full
    # host/device index and derives the largest useful query/owner tiles from
    # the observed GPU capacity without timing candidate sizes.
    policy = MemoryPolicy(
        phase1_mode=4,
        phase1_knn_scratch_bytes=0,
        phase1_index_heap_bytes=0,
        phase1_spill_directory=None,
    )
    for setup in setups:
        calibration = stage2.read(args.calibration_root / (setup + '__noise1pct') /
                                  'calibration_final.json')
        if calibration['setup']['id'] != setup:
            raise ValueError('wrong calibration identity')
        selected = selections[setup]
        configuration = selected_configuration(calibration, selected, digest)
        dataset = args.dataset_root / Path(configuration['setup']['dataset']).name
        root = args.result_root / (setup + '__noise1pct')
        stage2.verify_or_create(root / 'fixed7-selection.json', configuration)
        if 1_000_000 in ns:
            result = run_bridge(calibration, selected, digest, dataset=dataset,
                output=root / 'N1M_bridge', library=args.library, runtime=runtime,
                n_values=[1_000_000], resume=args.resume, memory_policy=policy)
            print(json.dumps(dict(event='fixed7_N1M_bridge', setup=setup, **result)), flush=True)
        larger = [n for n in ns if n > 1_000_000]
        if larger:
            result = stage3.run_fixed_setup(configuration, dataset=dataset,
                output=root, library=args.library, runtime=runtime,
                n_values=larger, resume=args.resume, memory_policy=policy)
            print(json.dumps(dict(event='fixed7_Stage3', setup=setup, **result)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
