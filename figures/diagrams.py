"""Compile the manuscript's geometry and TikZ diagrams outside the source tree."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT/'figures/paper_diagrams'
TIKZ = ('phase1_scan', 'phase2_refinement', 'local_cylinder', 'theory_roadmap', 'curve_link')


def render(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latex = shutil.which('pdflatex')
    if latex is None:
        raise RuntimeError('Install pdfLaTeX with TikZ, AMS fonts and graphicx to build diagrams.')
    with tempfile.TemporaryDirectory(prefix='mim-diagrams-') as directory:
        work = Path(directory)
        subprocess.run([sys.executable, str(SOURCES/'update_m2_geometry_tube.py'),
                        '--output', str(work/'MFM_m2_geometry.pdf')], check=True)
        subprocess.run([sys.executable, str(SOURCES/'model_geometry.py'),
                        '--geometry', str(work/'MFM_m2_geometry.pdf'),
                        '--output', str(output/'MFM_model_geometry.pdf')], check=True)
        subprocess.run([sys.executable, str(SOURCES/'response_preimage_slabs.py'),
                        str(work/'MFM_m2_preimage_slabs.pdf')], check=True, cwd=work)
        for name in TIKZ:
            shutil.copyfile(SOURCES/(name+'.tex'), work/(name+'.tex'))
            command = [latex, '-interaction=nonstopmode', '-halt-on-error', name+'.tex']
            result = subprocess.run(command, cwd=work, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError('Could not compile '+name+':\n'+result.stdout[-6000:])
            shutil.copyfile(work/(name+'.pdf'), output/(name+'.pdf'))
    print('Geometry and five TikZ diagrams:', output)
