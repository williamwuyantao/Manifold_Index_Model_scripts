"""Rebuild Figure 1's cylindrical patch and visible tube, preserving annotations.

Run with a Python environment containing pypdf. The original figure's drawing
source is unavailable; the retained base PDF supplies its exact geometry, fonts,
and labels. All additions remain vector paths with PDF transparency.
"""
import argparse
import json
import math
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, DictionaryObject, FloatObject, NameObject

HERE = Path(__file__).resolve().parent
# Dimension marker for the tube diameter, drawn beside the back radial edge
# of the right end face; its label is typeset with the figure's STIX fonts.
DIMENSION_COLOR = (.25, .30, .33)
DIMENSION_LABEL = r'$O(\sigma_N\sqrt{D-m})$'


def points(block):
    return [(float(x), float(y)) for x, y in
            re.findall(rb'([-\d.]+) ([-\d.]+) [ml]', block)]


def shift(curve, dx, dy):
    return [(x + dx, y + dy) for x, y in curve]


def path(curve, closed=False):
    lines = [f'{x:.6f} {y:.6f} {"m" if i == 0 else "l"}'
             for i, (x, y) in enumerate(curve)]
    if closed:
        lines.append('h')
    return '\n'.join(lines) + '\n'


def fill(curve, state, color):
    return (f'q /{state} gs {color} rg\n' + path(curve, True) + 'f Q\n').encode()


def stroke(curve, state, width=.65, dashed=False, closed=False):
    dash = '[3 2.2] 0 d' if dashed else '[] 0 d'
    return (f'q /{state} gs 1 J 1 j {width} w {dash} '
            '0.43 0.51 0.56 RG\n' + path(curve, closed) + 'S Q\n').encode()


def replace_path(block, curve, closed=False, painting=b'S'):
    """Retain the original graphics style and replace only its path geometry."""
    start = re.search(rb'[-\d.]+ [-\d.]+ m\n', block).start()
    return block[:start]+path(curve, closed).encode()+b'\n'+painting+b'\n'


def make_label(path, text=DIMENSION_LABEL, fontsize=11):
    """Typeset the label with Matplotlib's STIX mathtext, as in the base figure."""
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'mathtext.fontset': 'stix',
                                'font.family': 'STIXGeneral', 'pdf.fonttype': 42})
    figure = plt.figure(figsize=(2, .5))
    figure.text(.05, .3, text, fontsize=fontsize, color=DIMENSION_COLOR)
    figure.savefig(path, bbox_inches='tight', pad_inches=.02, transparent=True)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE/'source/MFM_m2_geometry_base.pdf')
    parser.add_argument('--output', type=Path, default=HERE/'MFM_m2_geometry.pdf')
    parser.add_argument('--label', type=Path, default=HERE/'source/MFM_m2_diameter_label.pdf',
                        help='typeset diameter label; created with Matplotlib if missing')
    args = parser.parse_args()
    reader = PdfReader(args.source)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    original = page.get_contents().get_data()
    blocks = original.split(b'Q q')
    assert len(blocks) == 58, 'The base figure structure has changed.'
    # Recover the original cylinder projection. Shorten its right end before
    # the projection folds, then rotate the entire geometric illustration
    # together below. Changing just the axial direction would leave the
    # original fibers and right-angle marker inconsistent with the surface.
    original_front = points(blocks[2])[:160]
    old_outer = points(blocks[0])[4:]
    front_corner = points(blocks[2])[0]
    back_corner = points(blocks[34])[-1]
    old_dx, old_dy = (back_corner[i]-front_corner[i] for i in range(2))
    assert abs(old_dx-50.738123) < 1e-5 and abs(old_dy-44.799103) < 1e-5
    front = original_front[:143]
    dx, dy = old_dx, old_dy
    # A fixed sign of this projected area prevents the curved and straight
    # boundaries from folding across one another at the right-hand corner.
    assert all(dy*(x1-x0)-dx*(y1-y0)>0
               for (x0,y0),(x1,y1) in zip(front,front[1:]))
    back = shift(front, dx, dy)
    # The two original radius offsets determine the projected circle center
    # and the tube-to-radius ratio. Preserve that ratio for the shortened arc.
    d0 = [old_outer[0][i]-original_front[0][i] for i in range(2)]
    d1 = [old_outer[-1][i]-original_front[-1][i] for i in range(2)]
    dd = [d0[i]-d1[i] for i in range(2)]
    mm = [original_front[0][i]-original_front[-1][i] for i in range(2)]
    radius_ratio = sum(mm[i]*dd[i] for i in range(2))/sum(v*v for v in dd)
    center = [original_front[0][i]-radius_ratio*d0[i] for i in range(2)]
    tube_ratio = 1/radius_ratio
    outer = [(center[0]+(1+tube_ratio)*(x-center[0]),
              center[1]+(1+tube_ratio)*(y-center[1])) for x,y in front]
    inner = [(center[0]+(1-tube_ratio)*(x-center[0]),
              center[1]+(1-tube_ratio)*(y-center[1])) for x,y in front]
    front_cap = outer + inner[::-1]
    outer_back, inner_back = shift(outer,dx,dy), shift(inner,dx,dy)
    back_cap = outer_back + inner_back[::-1]
    # The union's outer silhouette removes the artificial wireframe made by
    # overlapping projected faces, especially at the left end of the tube.
    outline_data = json.loads((HERE/'source/MFM_m2_tube_outline.json').read_text())
    assert outline_data['front_samples']==len(front)
    assert all(abs(a-b)<1e-8 for a,b in zip(outline_data['axial_displacement'],[dx,dy]))
    assert abs(outline_data['tube_ratio']-tube_ratio)<1e-10
    tube_outline = outline_data['vertices']
    states = page['/Resources']['/ExtGState']
    for name, fill_alpha, stroke_alpha in [
        ('TubeBack', .12, .4), ('TubeFront', .08, .8),
        ('TubeEdge', 1, .66), ('TubeFarEdge', 1, .42),
        ('TangentSpace', .32, .8), ('Dimension', 1, 1),
    ]:
        states[NameObject('/'+name)] = DictionaryObject({
            NameObject('/Type'): NameObject('/ExtGState'),
            NameObject('/ca'): FloatObject(fill_alpha),
            NameObject('/CA'): FloatObject(stroke_alpha),
        })
    # Draw the volume behind the original blue surface, then its translucent
    # front section over the surface. Tangent plane, points and arrows follow.
    backdrop = b'q 29.0 34.776 330 195 re W n\n'
    backdrop += fill(tube_outline, 'TubeBack', '0.55 0.65 0.71')
    backdrop += b'Q\n'
    foreground = b'q 29.0 34.776 330 195 re W n\n'
    foreground += fill(front_cap, 'TubeFront', '0.55 0.65 0.71')
    # Two complete curved end sections and their four connecting edges show
    # all twelve edges of the curved three-dimensional box, each only once.
    foreground += stroke(back_cap, 'TubeFarEdge', width=.65, dashed=True, closed=True)
    foreground += stroke(front_cap, 'TubeEdge', width=.7, dashed=True, closed=True)
    for edge in [[outer[0],outer_back[0]],[outer[-1],outer_back[-1]],
                 [inner[0],inner_back[0]],[inner[-1],inner_back[-1]]]:
        foreground += stroke(edge,'TubeEdge',width=.65,dashed=True)
    foreground += b'Q\n'
    # Tube diameter marker: a solid line parallel to the back radial edge of
    # the right end face, offset outward, with perpendicular end ticks. It is
    # drawn inside the rotated panel; the upright label is merged at the end.
    dim_a, dim_b = inner_back[-1], outer_back[-1]
    dim_len = math.dist(dim_a, dim_b)
    dim_u = [(dim_b[j]-dim_a[j])/dim_len for j in range(2)]
    dim_n = [-dim_u[1], dim_u[0]]
    if dim_n[0]*dx+dim_n[1]*dy < 0:
        dim_n = [-dim_n[0], -dim_n[1]]
    dim_offset, dim_tick = 7.0, 2.5
    dim_ends = [[p[j]+dim_offset*dim_n[j] for j in range(2)] for p in (dim_a, dim_b)]
    dim_rg = ' '.join(f'{v:.2f}' for v in DIMENSION_COLOR)
    dimension = f'q /Dimension gs 0 J 0 j 0.6 w [] 0 d {dim_rg} RG\n'.encode()
    dimension += path(dim_ends).encode()
    for p in dim_ends:
        dimension += path([[p[j]-dim_tick*dim_n[j] for j in range(2)],
                           [p[j]+dim_tick*dim_n[j] for j in range(2)]]).encode()
    dimension += b'S Q\n'
    foreground += dimension
    # Keep the accepted upper edge and its top-right corner fixed. Extend
    # only the lower curve to the NW-pointing near edge of the right flat face,
    # then connect that endpoint to the fixed upper corner by a straight side.
    # Start the continuation while the lower arc still bends downward, so
    # the lower-right boundary keeps a visible bow and meets the straight
    # right side at a clear angle. Preserve its endpoint and join tangent.
    surface_front=list(original_front[:76])
    p0=surface_front[-1]
    p3=tuple(.7*outer[-1][j]+.3*inner[-1][j] for j in range(2))
    # This is the first edge encountered on the near side of that face,
    # running NW from outer[-1] to inner[-1], not the farther axial edge.
    edge=[inner[-1][j]-outer[-1][j] for j in range(2)]
    view_angle=math.radians(11)
    assert math.cos(view_angle)*edge[0]-math.sin(view_angle)*edge[1]<0
    assert math.sin(view_angle)*edge[0]+math.cos(view_angle)*edge[1]>0
    chord=[p3[j]-p0[j] for j in range(2)]
    length=math.hypot(*chord)
    tangent=[surface_front[-1][j]-surface_front[-2][j] for j in range(2)]
    theta0=math.atan2(tangent[1],tangent[0])
    theta1=2*math.atan2(chord[1],chord[0])-theta0
    p1=(p0[0]+length/3*math.cos(theta0),p0[1]+length/3*math.sin(theta0))
    p2=(p3[0]-length/3*math.cos(theta1),p3[1]-length/3*math.sin(theta1))
    for i in range(1,65):
        t=i/64
        surface_front.append(tuple((1-t)**3*p0[j]+3*(1-t)**2*t*p1[j]
                                   +3*(1-t)*t*t*p2[j]+t**3*p3[j] for j in range(2)))
    surface_back=back
    assert surface_front[-1]==p3
    assert surface_back[-1]==back[-1]
    surface_boundary=surface_front+surface_back[::-1]
    def resample(curve,count):
        result=[]
        for i in range(count):
            t=i*(len(curve)-1)/(count-1)
            k=min(int(t),len(curve)-2);w=t-k
            result.append(tuple((1-w)*curve[k][j]+w*curve[k+1][j] for j in range(2)))
        return result
    near=resample(surface_front,160)
    far=resample(surface_back,160)
    def surface_section(fraction):
        return [tuple((1-fraction)*a[j]+fraction*b[j] for j in range(2))
                for a,b in zip(near,far)]
    for i in range(33):
        strip=surface_section(i/33)+surface_section((i+1)/33)[::-1]
        original=original.replace(blocks[i+2],replace_path(blocks[i+2],strip,True,b'f'),1)
    original=original.replace(blocks[35],replace_path(blocks[35],surface_boundary,True),1)
    fraction=(points(blocks[36])[0][0]-front_corner[0])/old_dx
    original=original.replace(blocks[36],replace_path(blocks[36],surface_section(fraction)),1)
    # Shorten each solid normal fiber about its base point, preserving its
    # direction. Keep the central fiber long enough to contain X and X'.
    first_fiber=blocks[37]
    for index,factor in [(37,.65),(38,.65),(43,.86)]:
        endpoints=points(blocks[index])
        assert len(endpoints)==2
        if index in [37,38]:
            old_base=[sum(p[j] for p in endpoints)/2 for j in range(2)]
            nearest=min(range(len(front)),key=lambda i:sum((front[i][j]-old_base[j])**2 for j in range(2)))
            new_base=front[nearest+(14 if index==37 else -14)]
            # Advance along the circular section and recompute its radial
            # normal, rather than translating the old fiber direction.
            endpoints=[(new_base[0]+sign*tube_ratio*(new_base[0]-center[0]),
                        new_base[1]+sign*tube_ratio*(new_base[1]-center[1])) for sign in [1,-1]]
            shift_x,shift_y=[new_base[j]-old_base[j] for j in range(2)]
            marker=blocks[39 if index==37 else 40]
            moved_marker=(f'\nq 1 0 0 1 {shift_x:.9f} {shift_y:.9f} cm\n'.encode()
                          +marker+b'Q\n')
            original=original.replace(marker,moved_marker,1)
        midpoint=[sum(p[j] for p in endpoints)/2 for j in range(2)]
        short=[(midpoint[0]+factor*(x-midpoint[0]),
                midpoint[1]+factor*(y-midpoint[1])) for x,y in endpoints]
        replacement=replace_path(blocks[index],short)
        original=original.replace(blocks[index],replacement,1)
        if index==37:first_fiber=replacement
    # Share the same translucent orange fill in the tangent patch and the
    # tangent-space diagram, so their relationship is explicit across panels.
    tangent_color='1 0.72 0.38'
    tangent_patch=(blocks[41][:blocks[41].index(b'/A4 gs')]
                   +f'/TangentSpace gs .75 w 0.86 0.49 0.17 RG {tangent_color} rg\n'.encode()
                   +path(points(blocks[41]),True).encode()+b'B\n')
    original=original.replace(blocks[41],tangent_patch,1)
    panel_background=(blocks[51][:blocks[51].index(b'/A2 gs')]
                      +f'/TangentSpace gs {tangent_color} rg\n'.encode()
                      +path(points(blocks[51]),True).encode()+b'f\n')
    original=original.replace(blocks[51],panel_background,1)
    # Preserve the original tangent plane, all three normal fibers, projection
    # arrows and the right-angle marker as one geometric configuration.
    # Labels move with their anchors but are counter-rotated to remain upright.
    angle = math.radians(11)
    cosine, sine = math.cos(angle), math.sin(angle)
    # Place M in the open interior of the visible blue surface. Coordinates
    # are pulled back through the common rotation, leaving the label upright.
    label_x, label_y = 185.0, 119.0
    qx, qy = 148.95688, 98.547582
    mx = qx+cosine*(label_x-qx)+sine*(label_y-qy)
    my = qy-sine*(label_x-qx)+cosine*(label_y-qy)
    assert blocks[50].count(b'74.2333661208 184.4674109416 cm')==1
    label_source = blocks[50].replace(b'74.2333661208 184.4674109416 cm',
                                    f'{mx:.12f} {my:.12f} cm'.encode(),1)
    labels = re.sub(rb'1 0 -0 1 ([-\d.]+) ([-\d.]+) cm\nBT',
                    lambda m: (f'{cosine:.12f} {-sine:.12f} {sine:.12f} {cosine:.12f} '.encode()
                               +m[1]+b' '+m[2]+b' cm\nBT'),label_source)
    original = original.replace(blocks[50],labels,1)
    # Remove the old dotted bounds; matching bounds are drawn with the tube.
    # Retain the original first block's white page background.
    first_path = blocks[0].index(b'q 29.7082278499')
    original = original.replace(blocks[0],blocks[0][:first_path]+b'q\n',1)
    original = original.replace(blocks[1],b'\n',1)
    # Insert before the first existing clipped object and before the normal
    # fibers respectively; all labels and other drawing commands are retained.
    start = original.index(b'q\n')
    front_insert = original.index(b'Q q'+first_fiber)
    changed = original[:start]+backdrop+original[start:front_insert+1]+b'\n'+foreground+original[front_insert+1:]
    # Rotate the complete left-panel projection about its base point Q. A
    # rigid rotation preserves normal/tangent relations and the complete upper
    # curved boundary; it also tilts the right straight edge as requested.
    qx, qy = 148.95688, 98.547582
    tx, ty = qx-cosine*qx+sine*qy, qy-sine*qx-cosine*qy
    rotation = (f'q\n{cosine:.12f} {sine:.12f} {-sine:.12f} {cosine:.12f} '
                f'{tx:.12f} {ty:.12f} cm\n').encode()
    panel_start = changed.index(backdrop)
    panel_end = changed.index(b'Q q'+panel_background)+1
    changed = changed[:panel_start]+rotation+changed[panel_start:panel_end]+b'\nQ\n'+changed[panel_end:]
    assert original.count(b'/F3 17 Tf') == changed.count(b'/F3 17 Tf')
    stream = DecodedStreamObject()
    stream.set_data(changed)
    page.replace_contents(stream)
    # Merge the diameter label upright in page coordinates, centred on the
    # outward normal of the marker and clear of it by a few points.
    if not args.label.exists():
        make_label(args.label)
    label_page = PdfReader(args.label).pages[0]
    box = label_page.mediabox
    width, height = float(box.width), float(box.height)
    def rotate(p):
        return (cosine*p[0]-sine*p[1]+tx, sine*p[0]+cosine*p[1]+ty)
    mid = rotate([(dim_ends[0][j]+dim_ends[1][j])/2 for j in range(2)])
    n_rot = (cosine*dim_n[0]-sine*dim_n[1], sine*dim_n[0]+cosine*dim_n[1])
    clearance = 3+width/2*abs(n_rot[0])+height/2*abs(n_rot[1])
    center = [mid[j]+clearance*n_rot[j] for j in range(2)]
    page.merge_transformed_page(label_page, Transformation().translate(
        center[0]-width/2-float(box.left), center[1]-height/2-float(box.bottom)))
    if reader.metadata:
        writer.add_metadata(dict(reader.metadata))
    with args.output.open('wb') as f:
        writer.write(f)
    print(args.output)


if __name__ == '__main__':
    main()
