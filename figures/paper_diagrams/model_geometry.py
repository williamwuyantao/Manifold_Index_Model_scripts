"""Figure 1: one cylindrical surface and link in all panels.

The accepted panel-(a) vector geometry is retained from MFM_m2_geometry.pdf;
its text is retypeset for the smaller panel. Panels (b) and (c) are generated
from explicit coordinates. Outputs remain vector PDFs. Run with a Python
environment containing numpy, matplotlib, and pypdf.

Panel (b): q(s,a)=(sin(s),a,1-cos(s)), normal n(s)=(sin(s),0,-cos(s)),
X=q(s,a)+r*n(s). The link is f(s,a)=((s+a)/sqrt(2),a), as in panel (c).
The sheets F1=0 and F2=0 intersect in the normal fiber through q(0,0).
grad F1=(tau/(1+r)+e_a)/sqrt(2), grad F2=e_a. Panel (c) translates
the same three gradients at r=-.35,0,.35 to a common tangent-plane origin.
"""
from pathlib import Path
import argparse
import io

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as PlotPath
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, RectangleObject

HERE = Path(__file__).resolve().parent
BLUE = '#286f94'
ORANGE = '#c56829'
TEAL = '#27828b'
GRADIENT_XPRIME = '#b51f35'
INK = '#25343c'
WIDTH, HEIGHT = 720., 285.
PANEL_A_SCALE, PANEL_A_ORIGIN = .94, (-3.,69.)
FONT = 14
matplotlib.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix',
                           'pdf.fonttype':42, 'ps.fonttype':42})
LEVEL1, LEVEL2 = '#8868a8', '#348fa4'
# Exact colors and opacities of the retained panel-(a) geometry.
MANIFOLD_EDGE = '#286c96'
MANIFOLD_DARK = np.array([.85,.92,.96])
MANIFOLD_LIGHT = np.array([.95666667,.97818182,.98909091])
TUBE_FILL, TUBE_EDGE = (.55,.65,.71), (.43,.51,.56)


def arrow(ax, a, b, color=INK, lw=1.9, head=10, **kw):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=head,
                                linewidth=lw,color=color,shrinkA=0,shrinkB=0,
                                zorder=6,**kw))


def label(ax,x,y,text,size=FONT,**kw):
    ax.text(x,y,text,fontsize=size,va='center',**kw)


def panel_a(ax):
    # Same transform as the retained PDF geometry in compose().
    def place(x,y):return (PANEL_A_ORIGIN[0]+PANEL_A_SCALE*(x-15),
                          PANEL_A_ORIGIN[1]+PANEL_A_SCALE*(y-40))
    label(ax,142,267,'(a) Normal fibers',16,ha='center')
    for x,y,t,s,col in [
        (159.4,136.,r'$X$',15,ORANGE), (161.4,65.,r"$X'$",15,TEAL),
        (107.2,91.,r'$Q$',14,INK), (119.1,164.,'normal fiber',14,INK),
        (177.,85.,r'$T_Q\mathcal{M}$',14,BLUE), (185.,122.,r'$\mathcal{M}$',18,BLUE),
        (259.,214.,r'$O(\sigma_N\sqrt{D-m})$',13,INK)]:
        label(ax,*place(x,y),t,s,color=col)
    label(ax,142,47,r"$\Pi_{\mathcal{M}}(X)=\Pi_{\mathcal{M}}(X')=Q$",14,ha='center')
    label(ax,142,25,r"$F(X)=F(X')=f(Q)$",14,ha='center')


def panel_c(ax):
    label(ax,662,267,'(c) Gradients',16,ha='center')
    # The background extends below the grad F2 label.
    ax.add_patch(Rectangle((610,95),105,120,fc='#ffe9cd',ec='none'))
    label(ax,618,204,r'$T_Q\mathcal{M}$',14,color=BLUE)
    origin=np.array([619.,123.])
    # The axial component of grad F1 is constant; the circumferential one
    # changes with normal offset. grad F2 is the axial unit vector.
    for tip,col,txt,dy in [([654,123+35/.65],ORANGE,r'$\nabla F_1(X)$',2),
                          ([654,158],INK,r'$\nabla F_1(Q)$',3),
                          ([654,123+35/1.35],GRADIENT_XPRIME,r"$\nabla F_1(X')$",-4)]:
        arrow(ax,origin,tip,col,2.3,11)
        label(ax,660,tip[1]+dy,txt,13,color=col)
    arrow(ax,origin,origin+[35*np.sqrt(2),0],color='#667b7c',lw=2,head=10)
    ax.plot(*origin,'o',color=INK,ms=3)
    label(ax,680,109,r'$\nabla F_2$',14,ha='center',color='#667b7c')


def panel_b(ax):
    label(ax,451,267,'(b) Component level sets',16,ha='center')
    az,el=np.deg2rad([30.,50.])
    screen=np.array([[np.cos(az),np.sin(az),0.],
                     [-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)]])
    eye=np.cross(screen[0],screen[1])
    center=np.array([450.,151.]);scale=95.;radius=.35
    arc_limit,axial_limit=.9,.82
    # Clip each level surface to the same base patch as the manifold.
    # For F1=0, a=-s reaches an axial edge first; F2=0 reaches s=+-arc_limit.
    level_limits={1:min(arc_limit,axial_limit),2:arc_limit}
    axial=np.array([0.,1.,0.])
    def normal(s):return np.array([np.sin(s),0.,-np.cos(s)])
    def tangent(s):return np.array([np.cos(s),0.,np.sin(s)])
    def x(s,a,r=0.):return np.array([np.sin(s),a,1-np.cos(s)])+r*normal(s)
    def project(points):return center+scale*np.asarray(points)@screen.T
    def depth(points):return float(np.mean(np.asarray(points)@eye))
    along=np.linspace(-arc_limit,arc_limit,150)
    # Fill the projected tube once, so overlapping faces do not darken it.
    # Consistently oriented subpaths give their union under nonzero winding.
    tube_faces=[]
    for r in (-radius,radius):
        tube_faces.append([x(s,-axial_limit,r) for s in along]+
                          [x(s,axial_limit,r) for s in along[::-1]])
    for a in (-axial_limit,axial_limit):
        tube_faces.append([x(s,a,-radius) for s in along]+
                          [x(s,a,radius) for s in along[::-1]])
    for s in (-arc_limit,arc_limit):
        tube_faces.append([x(s,a,r) for a,r in
                           [(-axial_limit,-radius),(axial_limit,-radius),
                            (axial_limit,radius),(-axial_limit,radius)]])
    paths=[]
    for face in tube_faces:
        p=project(face)
        area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
        if area<0:p=p[::-1]
        paths.append(PlotPath(np.vstack([p,p[0]]),closed=True))
    ax.add_patch(PathPatch(PlotPath.make_compound_path(*paths),
                           fc=TUBE_FILL,ec='none',alpha=.12,zorder=.5))
    surfaces=[]
    # The same curved surface as (a), with the two response components.
    ss=np.linspace(-arc_limit,arc_limit,23);aa=np.linspace(-axial_limit,axial_limit,19)
    for s0,s1 in zip(ss[:-1],ss[1:]):
        for a0,a1 in zip(aa[:-1],aa[1:]):
            verts=np.array([x(s,a) for s,a in [(s0,a0),(s1,a0),(s1,a1),(s0,a1)]])
            shade=(.5*(s0+s1)+arc_limit)/(2*arc_limit)
            color=(1-shade)*MANIFOLD_DARK+shade*MANIFOLD_LIGHT
            surfaces.append((depth(verts),verts,color,1.))
    # F1=0 is a=-s; F2=0 is a=0. Both extend along complete normals.
    for component,color in [(1,LEVEL1),(2,LEVEL2)]:
        extent=level_limits[component]
        along=np.linspace(-extent,extent,65)
        for s0,s1 in zip(along[:-1],along[1:]):
            for r0,r1 in [(-radius,0.),(0.,radius)]:
                verts=np.array([x(s,-s if component==1 else 0.,r)
                                for s,r in [(s0,r0),(s1,r0),(s1,r1),(s0,r1)]])
                surfaces.append((depth(verts),verts,color,.28))
    for _,verts,color,alpha in sorted(surfaces,key=lambda a:a[0]):
        ax.add_patch(Polygon(project(verts),closed=True,fc=color,ec='none',
                             alpha=alpha,antialiased=False))
    # Outline the manifold and both level surfaces with smooth vector paths.
    def line(points,color,lw=1.,alpha=1.,**kw):
        p=project(points);ax.plot(p[:,0],p[:,1],color=color,lw=lw,alpha=alpha,**kw)
    # The same faint grey front cap and dashed tube edges as in panel (a).
    ax.add_patch(Polygon(project(tube_faces[2]),closed=True,fc=TUBE_FILL,
                         ec='none',alpha=.08,zorder=1.1))
    along=np.linspace(-arc_limit,arc_limit,150)
    for a in (-axial_limit,axial_limit):
        for r in (-radius,radius):
            line([x(s,a,r) for s in along],TUBE_EDGE,.65,.66,ls=(0,(3,2.2)))
        for s in (-arc_limit,arc_limit):
            line([x(s,a,r) for r in (-radius,radius)],TUBE_EDGE,.65,.66,ls=(0,(3,2.2)))
    for s in (-arc_limit,arc_limit):
        for r in (-radius,radius):
            line([x(s,a,r) for a in (-axial_limit,axial_limit)],TUBE_EDGE,.65,.66,ls=(0,(3,2.2)))
    for a in (-axial_limit,axial_limit):line([x(s,a) for s in along],MANIFOLD_EDGE,1.8)
    for s in (-arc_limit,arc_limit):line([x(s,a) for a in (-axial_limit,axial_limit)],MANIFOLD_EDGE,1.8)
    for component,color in [(1,LEVEL1),(2,LEVEL2)]:
        extent=level_limits[component]
        along=np.linspace(-extent,extent,120)
        for r in (-radius,radius):
            line([x(s,-s if component==1 else 0.,r) for s in along],color,1.,.85)
        for s in (-extent,extent):
            line([x(s,-s if component==1 else 0.,r) for r in (-radius,radius)],color,1.,.85)
        line([x(s,-s if component==1 else 0.) for s in along],color,1.8)
    # Their common normal fiber and the exact arrows compared in panel (c).
    line([x(0,0,r) for r in (-radius,radius)],INK,1.1)
    for r,color,gradient_color,text,gradient_text in [
            (-radius,ORANGE,ORANGE,r'$X$',r'$\nabla F_1(X)$'),
            (0.,INK,INK,r'$Q$',r'$\nabla F_1(Q)$'),
            (radius,TEAL,GRADIENT_XPRIME,r"$X'$",r"$\nabla F_1(X')$")]:
        start=x(0,0,r);gradient=(tangent(0)/(1+r)+axial)/np.sqrt(2)
        p=project(start);ax.plot(*p,'o',ms=4.1,color=color,zorder=8)
        label(ax,p[0]-10,p[1]+4,text,14,ha='right',color=color,zorder=9)
        tip=project(start+.5*gradient)
        arrow(ax,p,tip,gradient_color,lw=2.1,head=10)
        label(ax,tip[0]+5,tip[1],gradient_text,13,color=gradient_color,zorder=9,
              bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
    p=project(x(0,0));arrow(ax,p,project(x(0,0)+.5*axial),'#667b7c',lw=1.8,head=10)
    label(ax,480,191,r'$\nabla F_2$',13,color='#53646c',zorder=9,
          bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
    label(ax,330,117,r'$\mathcal{M}$',16,color=BLUE,zorder=9)
    for left,color,text in [(371,LEVEL1,r'$F_1=0$'),(469,LEVEL2,r'$F_2=0$')]:
        ax.add_patch(Rectangle((left,43),13,9,fc=color,alpha=.35,ec=color,lw=.8))
        label(ax,left+19,47,text,14,color=color)
    label(ax,451,25,r'$f(s,a)=((s+a)/\sqrt{2},a)$',14,ha='center')
    # Verify ambient gradients, level-set equations, and surface normals.
    rng=np.random.default_rng(143);worst=0.
    def response(pt):
        s=np.arctan2(pt[0],1-pt[2]);a=pt[1]
        return np.array([(s+a)/np.sqrt(2),a])
    for component,extent in level_limits.items():
        for s in (-extent,extent):
            a=-s if component==1 else 0.
            assert abs(s)<=arc_limit and abs(a)<=axial_limit
            assert min(abs(abs(s)-arc_limit),abs(abs(a)-axial_limit))<1e-12
            assert abs(response(x(s,a))[component-1])<1e-12
    for _ in range(50):
        s,a,r=rng.uniform(-.7,.7),rng.uniform(-.7,.7),rng.uniform(-radius,radius)
        pt=x(s,a,r);eps=1e-6
        finite=np.column_stack([(response(pt+eps*np.eye(3)[j])-response(pt-eps*np.eye(3)[j]))/(2*eps)
                                for j in range(3)])
        exact=np.array([(tangent(s)/(1+r)+axial)/np.sqrt(2),axial])
        worst=max(worst,float(np.max(abs(finite-exact))))
        assert abs(response(x(s,-s,r))[0])<1e-12
        assert abs(response(x(s,0.,r))[1])<1e-12
        assert abs(exact[0]@((1+r)*tangent(s)-axial))<1e-12
        assert np.max(abs(exact@normal(s)))<1e-12
    assert worst<1e-8
    print(f'Cylindrical link: 4 contour endpoints on the boundary; 50 gradient and level-set checks; maximum derivative error {worst:.3g}.')


def compose(output, base_geometry=None):
    fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,WIDTH),ylim=(0,HEIGHT),aspect='equal')
    ax.axis('off')
    panel_a(ax);panel_b(ax);panel_c(ax)
    overlay=io.BytesIO();fig.savefig(overlay,format='pdf',transparent=True)
    plt.close(fig);overlay.seek(0)
    original=PdfWriter(clone_from=base_geometry or HERE/'MFM_m2_geometry.pdf')
    geometry=original.pages[0]
    stream=ContentStream(geometry.get_contents(),original)
    kept=[];inside=False;text_rules=set()
    for i,(operands,operator) in enumerate(stream.operations):
        if operator==b'BT':inside=True
        if not inside and i not in text_rules:kept.append((operands,operator))
        if operator==b'ET':
            inside=False
            # Mathtext draws the square-root bar just after the text object.
            # Remove that old annotation rule along with its replaced glyphs.
            following=stream.operations[i+1:i+5]
            if [op for _,op in following]==[b'q',b're',b'f',b'Q']:
                text_rules.update(range(i+1,i+5))
    stream.operations=kept;geometry.replace_contents(stream)
    geometry.cropbox=RectangleObject([15,40,342,225])
    writer=PdfWriter();page=writer.add_blank_page(WIDTH,HEIGHT)
    page.merge_transformed_page(geometry,Transformation().translate(-15,-40)
                                .scale(PANEL_A_SCALE).translate(*PANEL_A_ORIGIN))
    page.merge_page(PdfReader(overlay).pages[0])
    writer.add_metadata({'/Title':'Manifold-Index Model: geometry, response values and gradients',
                         '/Creator':'model_geometry.py'})
    with output.open('wb') as f:writer.write(f)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'MFM_model_geometry.pdf')
    parser.add_argument('--geometry',type=Path,help='Rebuilt cylindrical-patch PDF')
    args=parser.parse_args()
    compose(args.output,args.geometry)
