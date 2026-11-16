"""Record the loaded CUDA ABI, numerical conventions and library checksum."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'manor_m/cuda/python'))
from manor_m_cuda import ManorMCuda


def inspect_library(path):
    path = path.resolve(strict=True)
    core = ManorMCuda(path)
    values = {}
    for key, symbol in (
            ('metric_sum_contract', 'manor_m_cuda_metric_sum_contract'),
            ('devtan_evaluation_version', 'manor_m_cuda_devtan_evaluation_version')):
        try:
            function = getattr(core._library, symbol)
        except AttributeError as error:
            raise ValueError('Rebuild the library with scripts/build_core.sh.') from error
        function.argtypes = []
        function.restype = ctypes.c_char_p
        values[key] = function().decode('ascii')
    if (values['metric_sum_contract'] != 'fp32-query256-tree-v1'
            or values['devtan_evaluation_version'] != 'orthonormal-fp64-v2'):
        raise ValueError('This build does not use the paper numerical conventions: ' + str(values))
    return dict(schema='manor-runtime.v1', backend='python_cuda_core',
                core_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                core_version=core.version, abi_version=core.abi_version,
                legacy_fallback=False, production_start_allowed=True,
                validation_basis='loaded ABI and compiled feature identifiers',
                numerical_comparison_performed=False, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT/'build/libmanor_m_cuda.so')
    parser.add_argument('--output', type=Path, default=ROOT/'build/runtime.json')
    # Tolerances used by paired-readout comparisons.
    parser.add_argument('--absolute-tolerance', type=float, default=5e-7)
    parser.add_argument('--relative-tolerance', type=float, default=5e-4)
    args = parser.parse_args()
    if not (0 < args.absolute_tolerance < 1 and 0 < args.relative_tolerance < 1):
        parser.error('comparison tolerances must be positive and below one')
    record = inspect_library(args.library)
    record.update(absolute_tolerance=args.absolute_tolerance,
                  relative_tolerance=args.relative_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    temporary.replace(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
