"""Regenerate paper figures from the scientific observations and result records."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def records(name):
    return [json.loads(line) for line in (ROOT/name).read_text().splitlines() if line]


def bumpy_statistics():
    from bumpy_rates import fit_mean
    rows=records('results/bumpy_sphere/iid.jsonl')
    assert len(rows)==360
    output=[]
    for setup,label in [('bumpy_sphere_v2_D20_Y0','linear'),('bumpy_sphere_v2_D20_Y2','quadratic')]:
        group=[r for r in rows if r['setup_id']==setup]
        ns=sorted({r['N'] for r in group});points=[]
        for n in ns:
            batch=sorted((r for r in group if r['N']==n),key=lambda r:r['repetition'])
            assert [r['repetition'] for r in batch]==list(range(1,11))
            assert all(r['fixed_parameters']==group[0]['fixed_parameters'] for r in batch)
            point={'N':n}
            for short,key in [('relMSE','testing_relMSE_total'),('devTan','testing_devTan')]:
                values=np.array([r['testing'][key] for r in batch])
                assert np.isfinite(values).all()
                point[short]=float(values.mean());point[short+'_std']=float(values.std(ddof=1))
            points.append(point)
        fits={key:fit_mean(np.array(ns),np.array([r[key] for r in points])) for key in ('relMSE','devTan')}
        legend_label = r'$f^{\rm lin}$' if label == 'linear' else r'$f^{\rm quad}$'
        output.append(dict(name=f'bumpy_{label}_scaling', setup=setup, rows=points,
                           fits=fits, legend_label=legend_label))
    return output



def ensemble_statistics(rows, label, lower=500000, partial=False):
    from bumpy_rates import fit_mean
    points=[]
    for n in sorted({r['N'] for r in rows}):
        batch=sorted((r for r in rows if r['N']==n),key=lambda r:r['repetition'])
        repetitions=[r['repetition'] for r in batch]
        assert len(set(repetitions))==len(repetitions)
        if partial and repetitions!=list(range(1,11)):
            continue
        assert repetitions==list(range(1,11))
        assert all(r['fixed_parameters']==rows[0]['fixed_parameters'] for r in batch)
        point={'N':n}
        for short,key in [('relMSE','testing_relMSE_total'),('devTan','testing_devTan')]:
            values=np.array([r['testing'][key] for r in batch])
            assert np.isfinite(values).all() and (values>0).all()
            point[short]=float(values.mean());point[short+'_std']=float(values.std(ddof=1))
        points.append(point)
    ns=np.array([p['N'] for p in points])
    return dict(name=label,rows=points,fit_lower_N=lower,
        fits={key:fit_mean(ns,np.array([r[key] for r in points]),lower=lower)
              for key in ('relMSE','devTan')})


def coupled_statistics():
    rows=records('results/coupled_helix/iid.jsonl')
    assert len(rows)==540
    output=[]
    for label, name, legend_label in [
            ('A', 'natural_polynomial3', r'$f^{\rm poly}$'),
            ('B', 'natural_rational3', r'$f^{\rm rat}$'),
            ('C', 'natural_exponential3', r'$f^{\rm exp}$')]:
        item = ensemble_statistics([r for r in rows if r['map']==name],
                                   'coupled_helix_'+label+'_scaling')
        item['legend_label'] = legend_label
        output.append(item)
    return output


def global_statistics():
    output=[]
    for c in ('c025','c050','c075'):
        rows=records('results/global_tube/'+c+'/iid.jsonl')
        if c=='c075':assert len(rows)==180
        item=ensemble_statistics(rows,'global_tube_'+c+'_scaling',partial=c!='c075')
        item['legend_label'] = rf'$c={c[1:]}$'
        item['legend_location']='lower left'
        output.append(item)
    return output


def render_scaling(items, out, *, name, reference_line=None, keep=None):
    import bumpy_rates
    selected = [item for item in items if keep is None or item.get('name') in keep]
    output = out/'output/pdf'/(name+'.pdf')
    bumpy_rates.render_overlay(selected, output, reference_line=reference_line)
    return selected


def verify_phase2(data):
    readout=records('results/phase2/readout.jsonl')
    direct=records('results/phase2/direct_states.jsonl')
    assert len(readout)==len(direct)==30
    for group in data['groups']:
        a=sorted((r for r in readout if r['N']==group['N']),key=lambda r:r['repetition'])
        b=sorted((r for r in direct if r['N']==group['N']),key=lambda r:r['repetition'])
        assert len(a)==len(b)==10
        for metric in ('testing_relMSE_total',):
            values=np.array([[v[metric] for v in r['rows']] for r in a])
            np.testing.assert_allclose(values.mean(axis=0),group['trajectory'][metric]['mean'],rtol=2e-14)
            np.testing.assert_allclose(values.std(axis=0,ddof=1),group['trajectory'][metric]['sample_standard_deviation'],rtol=2e-14)
        values=np.array([[v['direct_tangent_error']['mean'] for v in r['rows']] for r in b])
        np.testing.assert_allclose(values.mean(axis=0),group['trajectory']['direct_tangent_error']['mean'],rtol=2e-14)
        np.testing.assert_allclose(values.std(axis=0,ddof=1),group['trajectory']['direct_tangent_error']['sample_standard_deviation'],rtol=2e-14)
    return len(readout),sum(len(r['rows']) for r in direct)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure',choices=('bumpy','phase2','coupled','bumpy-maps','bumpy-geometry','reaction','global-tube','diagrams','curves','all'))
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    for folder in (out/'output/pdf',out/'output/figures',out/'previews'):
        folder.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'cache'))
    sys.path[:0]=[str(ROOT/'scaling_law_N'),str(ROOT/'scaling_law_N/analysis'),str(ROOT/'dataset_generator')]
    if args.figure in ('diagrams','all') and not args.verify_only:
        import diagrams
        diagrams.render(out/'output/pdf')
    if args.figure in ('curves','all') and not args.verify_only:
        import curves
        curves.render(out/'output/pdf')
    if args.figure in ('bumpy','all'):
        items=bumpy_statistics()
        if not args.verify_only:
            render_scaling(items, out, name='Bumpy_Sphere_D20_scaling_IID10',
                          reference_line=1_000_000)
            (out/'bumpy_statistics.json').write_text(json.dumps(items,indent=2)+'\n')
        print('Bumpy sphere: 360 observations, 18 sample sizes, 10 repetitions per point.')
    if args.figure in ('phase2','all'):
        data=json.loads((ROOT/'results/phase2/plot_data.json').read_text())
        cases,states=verify_phase2(data)
        if not args.verify_only:
            import phase2
            phase2.PDFS=out/'output/pdf';phase2.FIGURES=out/'previews'
            phase2.paper_style();report=[]
            phase2.iteration_figure(data,report,tangent_label='Mean tangent-space error at training points',
                tangent_metric='direct_tangent_error',tangent_legend='lower left',name='phase2_iterations')
            phase2.projection_figure(data,report)
        print(f'Phase 2: {cases} cases and {states} direct states verified.')
    if args.figure in ('coupled','all'):
        items=coupled_statistics()
        if not args.verify_only:
            import coupled
            coupled.OUT=out/'output/pdf';coupled.PREVIEW=out/'previews'
            coupled.style();coupled.geometry_figure();coupled.response_figures()
            render_scaling(items, out, name='Coupled_Meyer_Helix_D20_AB_scaling_IID10',
                          reference_line=500_000,
                          keep={'coupled_helix_A_scaling', 'coupled_helix_B_scaling'})
            (out/'coupled_statistics.json').write_text(json.dumps(items,indent=2)+'\n')
        print('Coupled helix: 540 IID observations; 180 each for A, B and C.')
    if args.figure in ('global-tube','all'):
        items=global_statistics()
        if not args.verify_only:
            for item in items:
                render_scaling([item], out, name=item['name'],
                               reference_line=item.get('fit_lower_N'))
            (out/'global_tube_statistics.json').write_text(json.dumps(items,indent=2)+'\n')
        print('Global tubes: IID means and SD verified; partial groups remain in the raw records.')
    if args.figure in ('bumpy-maps','all') and not args.verify_only:
        import bumpy_maps
        sys.argv=['bumpy_maps','--paper-notation','--output',str(out/'output')];bumpy_maps.main()
    if args.figure in ('reaction','all') and not args.verify_only:
        import reaction
        reaction.OUT=out/'output/pdf';reaction.main()
    if args.figure in ('bumpy-geometry','all') and not args.verify_only:
        from geometry import radius,radial_parameters,render_bumpy_geometry
        context=json.loads((ROOT/'config/bumpy_geometry.json').read_text())
        identity=context['Y0']['data_identity']['base_dataset_identity']
        theta,phi=np.meshgrid(np.linspace(0,np.pi,161),np.linspace(0,2*np.pi,321))
        unit=np.stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)),axis=-1)
        radial=radius(unit.reshape(-1,3),radial_parameters(identity)).reshape(theta.shape)
        xyz=identity['geometry_scale']*radial[:,:,None]*unit
        geometry=dict(name='bumpy_geometry',X=xyz[:,:,0],Y=xyz[:,:,1],Z=xyz[:,:,2],
            scale=identity['geometry_scale'],geometry_formula_sha256=identity['geometry_formula_sha256'])
        render_bumpy_geometry(geometry,out/'output/pdf/bumpy_geometry.pdf',view=(100,24),paper_clean=True)


if __name__=='__main__':main()
