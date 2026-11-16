"""Compose the paper's six retained curve plots without changing their data."""
from pathlib import Path
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

ROOT = Path(__file__).resolve().parents[1]


def render(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The submitted manuscript clips each row into prediction/tangent panels.
    # The error bars are retained; the third (base-point) panel is omitted.
    for model, right_edges in (('single_index', (521, 520, 519)),
                               ('meyer_helix', (521, 524, 521))):
        writer = PdfWriter()
        canvas = writer.add_blank_page(width=812, height=698)
        for column, (dimension, right) in enumerate(zip((18, 24, 30), right_edges)):
            source = ROOT/'figures/pdf/curves'/f'{model}_D{dimension}.pdf'
            page = PdfReader(source).pages[0]
            for left, edge, y in ((0, 260, 339), (262, right, 0)):
                page.cropbox = RectangleObject([left, 2, edge, 335])
                canvas.merge_transformed_page(page, Transformation().translate(
                    column*275-left, y-2))
        header = io.BytesIO()
        with plt.rc_context({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'}):
            figure = plt.figure(figsize=(812/72, 698/72))
            for column, dimension in enumerate((18, 24, 30)):
                figure.text((column*275+130)/812, 687/698, rf'$D={dimension}$',
                            ha='center', va='center', fontsize=15)
            figure.savefig(header, format='pdf', transparent=True)
            plt.close(figure)
        header.seek(0)
        canvas.merge_page(PdfReader(header).pages[0])
        writer.add_metadata({'/Title': model.replace('_', ' ') + ': prediction and tangent-space errors'})
        writer.write(output/(model+'_paper_panels.pdf'))
    print('Curve figures: retained vector plots, prediction and tangent panels only.')
