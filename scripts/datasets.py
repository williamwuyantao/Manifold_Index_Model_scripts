"""List, check, or retrieve full datasets using the published array manifest."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check(path, expected, hashes=False):
    if not path.is_file():
        raise ValueError('missing file')
    with h5py.File(path, 'r') as source:
        for name, spec in expected['arrays'].items():
            if name not in source:
                raise ValueError('missing array '+name)
            array = source[name]
            if list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']:
                raise ValueError('shape or dtype differs: '+name)
            if hashes:
                digest = hashlib.sha256()
                if array.shape:
                    row_bytes = max(1, int(np.prod(array.shape[1:]))*array.dtype.itemsize)
                    step = max(1, (8 << 20)//row_bytes)
                    for start in range(0, array.shape[0], step):
                        digest.update(np.ascontiguousarray(array[start:start+step]).tobytes())
                else:
                    digest.update(np.asarray(array[()]).tobytes())
                if digest.hexdigest() != spec['sha256']:
                    raise ValueError('array checksum differs: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'check', 'fetch'))
    parser.add_argument('--root', type=Path, default=ROOT/'dataset')
    parser.add_argument('--match', default='*', help='Dataset path glob; quote * in the shell')
    parser.add_argument('--hash', action='store_true', help='Check every array byte; fetch always does this')
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--source', type=Path, help='Root of an existing full dataset copy')
    sources.add_argument('--base-url', help='HTTP(S) root preserving the manifest directory layout')
    args = parser.parse_args()
    if args.command == 'fetch' and args.source is None and args.base_url is None:
        parser.error('fetch requires --source or --base-url')
    if args.base_url and urlparse(args.base_url).scheme not in ('http', 'https'):
        parser.error('--base-url must be HTTP(S)')
    inventory = json.loads((ROOT/'config/dataset_inventory.json').read_text())['datasets']
    selected = {k: v for k, v in inventory.items() if fnmatch.fnmatch(k, args.match)}
    if not selected:
        parser.error('no dataset matches '+args.match)
    failed = 0
    for name, spec in sorted(selected.items()):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid manifest path')
        target = args.root/name
        if args.command == 'list':
            print(f'{name}\t{spec["bytes"]}\t{len(spec["arrays"])} arrays')
            continue
        try:
            if args.command == 'fetch' and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix+'.partial')
                if args.source is not None:
                    with (args.source/name).open('rb') as source, temporary.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 8 << 20)
                else:
                    url = args.base_url.rstrip('/')+'/'+quote(name, safe='/')
                    with urlopen(url, timeout=120) as source, temporary.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 8 << 20)
                check(temporary, spec, hashes=True)
                temporary.replace(target)
            else:
                check(target, spec, hashes=args.hash or args.command == 'fetch')
            print('OK '+name)
        except (OSError, ValueError, KeyError) as error:
            failed += 1
            print('FAILED '+name+': '+str(error))
    if failed:
        raise SystemExit(f'{failed}/{len(selected)} datasets failed')


if __name__ == '__main__':
    main()
