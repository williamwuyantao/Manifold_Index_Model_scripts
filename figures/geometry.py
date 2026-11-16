from pathlib import Path
import numpy as np
def radial_parameters(identity):
    f=identity["geometry_formula"]
    i=np.arange(32,dtype=float); z=1-2*(i+.5)/32
    angle=i*np.pi*(3-np.sqrt(5))+f["micro_fibonacci_phase"]
    centers=np.column_stack((np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z))
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    w=.26+.055*np.remainder(5*i+1,8)/7
    raw=.72*np.sin(1.73*i+.21)+.28*np.cos(2.91*i+.47)
    a=raw-raw.mean();a=f["micro_amplitude"]*a/np.max(np.abs(a))
    return dict(centers=np.concatenate((np.array(f["v0_centers"]),centers)),
        width=np.r_[f["v0_width"],w], amplitude=np.r_[.75*np.array(f["v0_amplitude"]),a],
        offset=np.r_[np.zeros(6),.5*w*w*(1-np.exp(-2/(w*w)))],linear=.75*np.array(f["v0_linear"]))


def radius(unit,p):
    r=1+unit@p["linear"]
    for c,w,a,o in zip(p["centers"],p["width"],p["amplitude"],p["offset"]):
        r+=a*(np.exp((unit@c-1)/(w*w))-o)
    assert r.min()>.35 and np.isfinite(r).all()
    return r




def centered_cm_text(page, center, text, font, path, name, size, rotate=0):
    # MATLAB's CM fonts use explicit word positioning rather than a
    # Unicode space glyph. Preserve that layout in the PDF text layer.
    words = text.split()
    widths = [font.text_length(word, fontsize=size) for word in words]
    space = size / 3
    length = sum(widths) + space * (len(words) - 1)
    advance = 0
    for word, width in zip(words, widths):
        origin = ((center[0], center[1] + length / 2 - advance) if rotate == 90
                  else (center[0] - length / 2 + advance, center[1]))
        page.insert_text(origin, word, fontname=name, fontfile=str(path),
                         fontsize=size, rotate=rotate)
        advance += width + space


def render_bumpy_geometry(g, target: Path, view=(100, 24), axis_limit=None, label='', paper_clean=False) -> dict:
    """Render the same clean geometry and style from an orthographic view."""
    from io import BytesIO
    import numpy as np
    import pymupdf as pdf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    cm = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    bold_path = cm / 'cmb10.ttf'
    bold_font = pdf.Font(fontfile=str(bold_path))
    # An orthographic projection retains the geometry, with the paper's flat
    # axis frame instead of a Matplotlib three-dimensional pane/cube.
    xyz = np.stack([np.array(g[k]) for k in ('X', 'Y', 'Z')], axis=-1)[::4, ::4]
    az, el = np.deg2rad(view)
    eye = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.cross(eye, right)
    quads = np.stack([xyz[:-1, :-1], xyz[1:, :-1], xyz[1:, 1:], xyz[:-1, 1:]], axis=-2).reshape(-1, 4, 3)
    order = np.argsort(quads.mean(axis=1) @ eye)
    quads = quads[order]
    projected = np.stack([quads @ right, quads @ up], axis=-1)
    normal = (np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0])
              + np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 0]))
    # Outward normals and a single upper-left light make the radial bumps
    # legible in grayscale, without glossy highlights or changing the shape.
    normal *= np.where(np.sum(normal * quads.mean(axis=1), axis=1) < 0, -1, 1)[:, None]
    norm = np.linalg.norm(normal, axis=1)
    light = .75 * eye - .45 * right + .60 * up
    light /= np.linalg.norm(light)
    cosine = np.divide(normal @ light, norm, out=np.zeros_like(norm), where=norm > 0)
    gray = .44 + .44 * np.clip(cosine, 0, 1)
    facecolors = np.column_stack([gray, gray, gray, np.ones_like(gray)])
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman'],
                         'font.size':12, 'axes.linewidth':.5, 'pdf.fonttype':42,
                         'mathtext.fontset':'cm'})
    fig, ax = plt.subplots(figsize=(612/72, 510/72))
    fig.subplots_adjust(left=.085, right=.97, bottom=.13, top=.92)
    surface = PolyCollection(projected, facecolors=facecolors, edgecolors=(.22, .22, .22, .45),
                             linewidths=.22, rasterized=True, zorder=3)
    ax.add_collection(surface)
    limit = axis_limit if axis_limit is not None else 1.09 * np.max(np.abs(projected))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
    ax.set_xticks([-50, 0, 50]); ax.set_yticks([-50, 0, 50])
    ax.tick_params(direction='in', width=.5, length=3)
    ax.set_axisbelow(True); ax.grid(color='.65', alpha=.25, linewidth=.5)
    if not paper_clean:
        ax.legend([Line2D([], [], color='.35', linewidth=2)],
                  [r'$\mathcal{M}$: the underlying manifold'], frameon=True, fancybox=False,
                  edgecolor='black', framealpha=1, loc='upper left',
                  bbox_to_anchor=(-.01, -.095), borderaxespad=0,
                  prop=FontProperties(family='serif', size=13))
    geometry_stream = BytesIO()
    fig.savefig(geometry_stream, format='pdf', dpi=300)
    plt.close(fig)
    geometry_doc = pdf.open(stream=geometry_stream.getvalue(), filetype='pdf')
    title = '20-dimensional bumpy sphere v2: clean manifold'
    if not paper_clean:
        centered_cm_text(geometry_doc[0], (306, 23), f'{label}: {title}' if label else title,
                         bold_font, bold_path, 'MFMBold', 13)
    geometry_doc.save(target, garbage=4, deflate=True)
    geometry_doc.close()
    return {
        'name':g['name'], 'scale':g['scale'], 'geometry_formula_sha256':g['geometry_formula_sha256'],
        'page_size_points':[612, 510], 'orthographic_view_degrees':list(view),
        'observation_points_added':False, 'projection':'first three clean coordinates, orthographic',
        'surface_gray_range':[.44, .88], 'light_direction':light.tolist(),
        'mesh_sample_stride':4, 'mesh_line_width_points':.22, 'axis_limit':float(limit),
        'title_and_legend_removed':paper_clean,
    }

