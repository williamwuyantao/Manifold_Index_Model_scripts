"""Check repository files and recompute the independent-repetition statistics."""
import ast
import hashlib
import json
import math
import importlib.util
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def verify_curves():
    data=json.loads((ROOT/'results/curves/published_curves.json').read_text())
    config=json.loads((ROOT/'config/curve_experiments.json').read_text())
    assert len(data['experiments'])==6 and not data['individual_repetitions_available']
    count=0
    for experiment in data['experiments']:
        p=config[experiment['model']][f"D{experiment['ambient_dimension']}"]
        for curves in experiment['metrics'].values():
            assert len(curves)==3
            for curve in curves:
                assert curve['N']==p['N_values']
                assert all(b>a for a,b in zip(curve['N'],curve['N'][1:]))
                assert len(curve['mean'])==len(curve['N'])==len(curve['displayed_error_endpoints'])
                assert all(math.isfinite(v) and v>0 for v in curve['mean'])
                assert abs(curve['loglog_slope']-curve['printed_slope'])<.006
                for mean,bar in zip(curve['mean'],curve['displayed_error_endpoints']):
                    assert bar['lower'] is None or 0<bar['lower']<=mean*(1+1e-5)
                    assert bar['upper'] is None or bar['upper']>=mean*(1-1e-5)
                count+=len(curve['N'])
    print(f'One-dimensional curves: six figures and {count} displayed means verified.')


def shared_files():
    excluded={'.git','__pycache__','dataset','reproduced','build','.venv','artifacts'}
    return {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
            and not excluded.intersection(p.relative_to(ROOT).parts)
            and p.name!='SHA256SUMS'}


def verify_imports():
    folders=['manor_m/cuda/python','scaling_law_N','scaling_law_N/analysis',
             'dataset_generator','figures']
    sys.path[:0]=[str(ROOT/p) for p in folders]
    count=0
    for folder in folders:
        for path in sorted((ROOT/folder).glob('*.py')):
            name='verify_import_'+path.stem
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module
            spec.loader.exec_module(module)
            count+=1
    print(f'Imported {count} runtime, experiment, generator and plotting modules.')


def verify_scientific_records():
    private=re.compile(r'/Users/|/home/|/projects/|/anvil/|\b20\d\d-\d\d-\d\d[T ]\d\d:',re.I)
    for folder in ('config','results','data'):
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in ('.json','.jsonl'):
                assert not private.search(path.read_text()),str(path.relative_to(ROOT))
    counts=json.loads((ROOT/'config/record_counts.json').read_text())
    for label,relative in [('bumpy_iid','results/bumpy_sphere/iid.jsonl'),
                           ('coupled_iid','results/coupled_helix/iid.jsonl')]+[
        (c+'_'+kind,'results/global_tube/'+c+'/'+('selection' if kind=='selection' else 'iid')+'.jsonl')
        for c in ('c025','c050','c075') for kind in ('selection','iid')]:
        rows=[json.loads(line) for line in (ROOT/relative).read_text().splitlines()]
        assert len(rows)==counts[label],label
        if label.endswith('iid'):
            unique={(x.get('map',x.get('setup_id',x.get('c'))),x['N'],x['repetition']) for x in rows}
            assert len(unique)==len(rows),label
            for row in rows:
                assert row['testing_used_for_selection'] is False
                for metric in ('testing_relMSE_total','testing_devTan'):
                    assert math.isfinite(row['testing'][metric]) and row['testing'][metric]>=0
    print('Scientific record counts, identities, metrics and privacy verified.')


def main():
    count=0
    named=set()
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);p=ROOT/name
        assert name not in named
        named.add(name)
        assert p.is_file(),name
        digest=hashlib.sha256()
        with p.open('rb') as source:
            for block in iter(lambda:source.read(8<<20),b''):digest.update(block)
        assert digest.hexdigest()==expected,name
        if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
        count+=1
    assert named==shared_files(),(sorted(named-shared_files()),sorted(shared_files()-named))
    guide=(ROOT/'FILES.md').read_text()
    assert all('`'+name+'`' in guide for name in named|{'SHA256SUMS'})
    verify_imports()
    verify_scientific_records()
    subprocess.run([sys.executable,str(ROOT/'figures/reproduce.py'),'all','--verify-only'],check=True)
    verify_curves()
    print(f'Verified {count} source and scientific data files.')


if __name__=='__main__':main()
