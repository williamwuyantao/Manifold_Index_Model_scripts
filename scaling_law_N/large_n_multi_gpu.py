"""Optional large-N backend: one fixed-parameter experiment across CUDA GPUs.

The existing single-GPU runners remain unchanged. Call run_fixed with their
already prepared Stage2Data, or use the CLI for global-tube/coupled-helix H5s.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'manor_m/cuda/python'))
from multi_gpu_estimator import MultiGpuEstimator
from cuda_core_data import Stage2Data
from fixed_hyperparameter_screen_cuda import candidate_at_n


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_dataset(path, response_map=None):
    import h5py
    with h5py.File(path, 'r') as f:
        identity = json.loads(f.attrs['identity'])
        if f.attrs.get('state') != 'complete':
            raise ValueError('Dataset generation is incomplete')
        maps = list(f['maps'])
        if response_map is None:
            if len(maps) != 1:
                raise ValueError('Select one response map explicitly')
            response_map = maps[0]
        if response_map not in maps:
            raise ValueError('Response map is absent from this dataset')
        arrays = dict(X=f['geometry/train/x_noisy'][:],
                      Y=f['maps/'+response_map+'/train/observed'][:])
        for split, suffix, truth in (('validation', 'valid', 'observed'),
                                      ('testing', 'test', 'clean')):
            arrays['X'+suffix] = f['geometry/'+split+'/x_noisy'][:]
            arrays['Y'+suffix] = f['maps/'+response_map+'/'+split+'/'+truth][:]
            arrays['T'+suffix] = f['geometry/'+split+'/tangent'][:]
    if response_map != 'quadratic6':
        identity = dict(identity, map=response_map)
    return Stage2Data(identity, arrays)


def validate_release(library, qualification, device_count):
    proof = json.loads(Path(qualification).read_text())
    if (proof.get('status') != 'qualified' or proof.get('core_sha256') != file_hash(library)
            or proof.get('metric_sum_contract') != 'fp32-query256-tree-v1'
            or device_count not in proof.get('tested_device_counts', [])
            or proof.get('absolute_tolerance') != 5e-7
            or proof.get('relative_tolerance') != 5e-4
            or proof.get('winner_absolute_tolerance') != 5e-8):
        raise ValueError('Library/device count does not match the qualified release')
    return proof


def run_fixed(library, data, parameters, *, devices, workdir, qualification,
              all_iterations=True, on_iteration=None, **memory):
    """Callable successor for fixed-parameter Stage2/Stage3 experiments."""
    validate_release(library, qualification, len(devices))
    n, d = data.arrays['X'].shape
    candidate, clipped = candidate_at_n(parameters, n, d)
    with MultiGpuEstimator(library, data.arrays['X'], data.arrays['Y'],
                          devices=devices, workdir=workdir, **memory) as estimator:
        measured = estimator.fit(candidate, data, all_iterations=all_iterations,
                                 on_iteration=on_iteration)
    return dict(schema='mfm-fixed-parameter-multi-gpu.v1', N=n,
                fixed_parameters=dict(parameters), candidate=candidate, k_clipped=clipped,
                dataset_identity=data.identity, input_arrays_sha256=data.hashes,
                testing_queries=len(data.arrays['Xtest']), testing_used_for_selection=False,
                metric_sum_contract='fp32-query256-tree-v1', core_sha256=file_hash(library),
                validation=measured['validation'], testing=measured['testing'],
                diagnostic=measured['diagnostic'], wall_seconds=measured['wall'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--qualification', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--response-map')
    p.add_argument('--parameters', required=True, help='JSON object containing the fixed seven parameters')
    p.add_argument('--devices', required=True, help='Visible CUDA ordinals, e.g. 0,1,2,3')
    p.add_argument('--workdir', type=Path, required=True, help='Shared multi-process large-fit lock/scratch directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint-only', action='store_true')
    p.add_argument('--working-gib', type=float, default=0)
    p.add_argument('--host-gib', type=float)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Completed output already exists')
    devices = [int(s) for s in args.devices.split(',')]
    data = load_dataset(args.dataset, args.response_map)
    parameters = json.loads(Path(args.parameters).read_text())
    value = run_fixed(args.library, data, parameters, devices=devices,
        workdir=args.workdir, qualification=args.qualification,
        all_iterations=not args.endpoint_only,
        working_bytes=int(args.working_gib*1024**3),
        host_bytes=None if args.host_gib is None else int(args.host_gib*1024**3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
