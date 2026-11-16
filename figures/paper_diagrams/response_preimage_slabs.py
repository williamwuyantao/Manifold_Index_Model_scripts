"""Draw the response-preimage panel of Figure 2 for a two-dimensional M.

M is a bowl in R^3 with a tube of normal half-length R around it. For p=2,
the slab {x: |F_k(x)-Y_{i,k}| <= delta} is the union of the normal fibers
over a thin strip of M around the level set of f_k: two curves for k=1, one
curve for k=2. The slabs meet in two cylinders over the patches around the
two points of f^{-1}(Y_i), the packets C_1 and C_2 of panels (a)-(c).

The scene is projected orthographically and drawn back to front: the parts
below M, then M (translucent), then the parts above M. Slabs are translucent
and cylinders opaque. Labels are typeset in the TeX figure; this script
prints their anchors as fractions of the image width and height.

Usage: python response_preimage_slabs.py [output.pdf]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'MFM_m2_preimage_slabs.pdf'

R = 0.34            # normal half-length of the tube
W = 0.075           # half-width of each strip (response window)
AZ, EL = np.radians(24), np.radians(38)
U0 = 0.5            # level curves of f_1: u - A1 v^2 = +-U0
A1 = 0.15
V0, A2 = 0.12, 0.25  # level curve of f_2: v + A2 u^2 = V0

rgb = lambda *c: tuple(x / 255 for x in c)
M_FILL, M_EDGE = rgb(214, 226, 233), rgb(142, 171, 188)
S1_FILL, S1_EDGE = rgb(104, 160, 104), rgb(62, 118, 62)
S2_FILL, S2_EDGE = rgb(128, 116, 190), rgb(86, 74, 150)
C_FILL, C_EDGE = rgb(230, 164, 123), rgb(184, 80, 37)
BLACK = rgb(34, 43, 49)


def height(u, v):
    return 0.22 * (u**2 + 0.6 * v**2)


def point(u, v, t):
    n = np.array([-0.44 * u, -0.264 * v, 1.0])
    return np.array([u, v, height(u, v)]) + t * n / np.linalg.norm(n)


ca, sa, ce, se = np.cos(AZ), np.sin(AZ), np.cos(EL), np.sin(EL)


def project(p):
    x, y, z = p
    xr, yr = x * ca + y * sa, -x * sa + y * ca
    return np.array([xr, z * ce + yr * se]), -yr * ce + z * se  # screen, depth


def poly(points3):
    pr = [project(p) for p in points3]
    return np.array([q[0] for q in pr]), float(np.mean([q[1] for q in pr]))


def f1_dev(u, v):   # signed distance-like offset from the f_1 level curves
    return abs(u - A1 * v**2) - U0


def f2_dev(u, v):
    return v + A2 * u**2 - V0


def strip1(sign, s, v):     # point of the f_1 strip boundary at offset s
    return sign * (U0 + s) + A1 * v**2, v


def strip2(s, u):
    return u, V0 - A2 * u**2 + s


def corner(sign, s1, s2):
    v = V0
    for _ in range(50):
        u, _ = strip1(sign, s1, v)
        v = V0 - A2 * u**2 + s2
    return strip1(sign, s1, v)


def patch_boundary(sign, n=8):
    """Boundary of the patch {|f1_dev|<=W, |f2_dev|<=W} near u=sign*U0."""
    c = {(a, b): corner(sign, a, b) for a in (-W, W) for b in (-W, W)}
    pts = []
    for (a0, b0), (a1, b1) in [((-W, -W), (W, -W)), ((W, -W), (W, W)),
                               ((W, W), (-W, W)), ((-W, W), (-W, -W))]:
        (ua, va), (ub, vb) = c[(a0, b0)], c[(a1, b1)]
        for k in range(n):
            lam = k / n
            if b0 == b1:   # along a strip-2 boundary: parametrize by u
                u = ua + lam * (ub - ua)
                pts.append(strip2(b0, u))
            else:          # along a strip-1 boundary: parametrize by v
                v = va + lam * (vb - va)
                pts.append(strip1(sign, a0, v))
    return pts


def pieces(curve_uv, keep):
    """Split a parametrized curve into runs where keep(u,v) holds."""
    runs, cur = [], []
    for u, v in curve_uv:
        if keep(u, v):
            cur.append((u, v))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) > 1]


def ribbon(run, t0, t1):
    top = [point(u, v, t1) for u, v in run]
    bot = [point(u, v, t0) for u, v in reversed(run)]
    return top + bot


def slab_items(t0, t1):
    """Translucent faces of both slabs between normal offsets t0 < t1."""
    items = []
    vs = np.linspace(-1, 1, 400)
    us = np.linspace(-1, 1, 400)
    out2 = lambda u, v: abs(f2_dev(u, v)) > W
    out1 = lambda u, v: abs(f1_dev(u, v)) > W
    cap_t = t1 if t1 > 0 else t0
    for sign in (-1, 1):
        for s in (-W, W):
            for run in pieces([strip1(sign, s, v) for v in vs], out2):
                items.append(('slab', ribbon(run, t0, t1), S1_FILL, S1_EDGE))
        # cap: band between the two boundaries at the outer offset
        for run in pieces([strip1(sign, 0, v) for v in vs], out2):
            a = [point(*strip1(sign, -W, v), cap_t) for _, v in run]
            b = [point(*strip1(sign, W, v), cap_t) for _, v in reversed(run)]
            items.append(('cap', a + b, S1_FILL, S1_EDGE))
    for s in (-W, W):
        for run in pieces([strip2(s, u) for u in us], out1):
            items.append(('slab', ribbon(run, t0, t1), S2_FILL, S2_EDGE))
    for run in pieces([strip2(0, u) for u in us], out1):
        a = [point(*strip2(-W, u), cap_t) for u, _ in run]
        b = [point(*strip2(W, u), cap_t) for u, _ in reversed(run)]
        items.append(('cap', a + b, S2_FILL, S2_EDGE))
    return items


def cylinder_items(t0, t1):
    items = []
    for sign in (-1, 1):
        bd = patch_boundary(sign)
        m = len(bd)
        for k in range(m):
            (ua, va), (ub, vb) = bd[k], bd[(k + 1) % m]
            q = [point(ua, va, t0), point(ub, vb, t0),
                 point(ub, vb, t1), point(ua, va, t1)]
            items.append(('cyl', q, C_FILL, C_EDGE))
        cap = t1 if t1 > 0 else t0
        items.append(('cylcap', [point(u, v, cap) for u, v in bd], C_FILL, C_EDGE))
        allp = [point(u, v, t) for u, v in bd for t in (t0, t1)]
        items.append(('cylhull', allp, C_FILL, C_EDGE))
    return items


def shade(color, pts):
    p = np.array(pts)
    nrm = np.cross(p[1] - p[0], p[2] - p[0])
    if np.linalg.norm(nrm) == 0:
        return color
    nrm /= np.linalg.norm(nrm)
    light = np.array([-0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    k = 0.82 + 0.18 * abs(nrm @ light)
    return tuple(min(1, c * k + (1 - k) * 0.15) for c in color)


def hull(xy):
    """Convex hull of planar points (monotone chain)."""
    pts = sorted(map(tuple, xy))
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def draw_layer(ax, items):
    drawn = []
    for kind, pts, fc, ec in items:
        xy, depth = poly(pts)
        if kind == 'cylhull':   # outline after all faces of its cylinder
            depth = max(project(p)[1] for p in pts) + 1e-3
        drawn.append((depth, kind, xy, fc, ec, pts))
    drawn.sort(key=lambda d: d[0])
    for depth, kind, xy, fc, ec, pts in drawn:
        if kind == 'slab':
            ax.add_patch(Polygon(xy, closed=True, facecolor=fc, alpha=.22, edgecolor='none'))
            ax.add_patch(Polygon(xy, closed=True, facecolor='none', edgecolor=ec,
                                 linewidth=.3, alpha=.55))
        elif kind == 'cap':
            ax.add_patch(Polygon(xy, closed=True, facecolor=fc, alpha=.5, edgecolor=ec,
                                 linewidth=.3))
        elif kind == 'cyl':
            c = shade(fc, pts)
            ax.add_patch(Polygon(xy, closed=True, facecolor=c, edgecolor=c, linewidth=.25))
        elif kind == 'cylcap':
            ax.add_patch(Polygon(xy, closed=True, facecolor=fc, edgecolor=ec, linewidth=.4))
        elif kind == 'cylhull':
            ax.add_patch(Polygon(hull(xy), closed=True, facecolor='none', edgecolor=ec,
                                 linewidth=.5))


def boundary_uv(n=60):
    t = np.linspace(-1, 1, n)
    return ([(x, -1) for x in t] + [(1, y) for y in t] +
            [(x, 1) for x in t[::-1]] + [(-1, y) for y in t[::-1]])


def main():
    fig = plt.figure(figsize=(4.6 / 2.54, 4.0 / 2.54))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    bd = boundary_uv()
    draw_layer(ax, slab_items(-R, 0) + cylinder_items(-R, 0))
    # M, translucent, with a light grid
    xy, _ = poly([point(u, v, 0) for u, v in bd])
    ax.add_patch(Polygon(xy, closed=True, facecolor=M_FILL, alpha=.78, edgecolor=M_EDGE, lw=.6))
    g = np.linspace(-1, 1, 5)
    for c in g[1:-1]:
        for line in ([(c, y) for y in np.linspace(-1, 1, 40)], [(x, c) for x in np.linspace(-1, 1, 40)]):
            xy, _ = poly([point(u, v, 0) for u, v in line])
            ax.plot(xy[:, 0], xy[:, 1], color=M_EDGE, lw=.2, alpha=.45)
    # strips on M around the level curves
    vs, us = np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)
    for sign in (-1, 1):
        a = [point(*strip1(sign, -W, v), 0) for v in vs]
        b = [point(*strip1(sign, W, v), 0) for v in vs[::-1]]
        xy, _ = poly(a + b)
        ax.add_patch(Polygon(xy, closed=True, facecolor=S1_FILL, alpha=.55, edgecolor='none'))
    a = [point(*strip2(-W, u), 0) for u in us]
    b = [point(*strip2(W, u), 0) for u in us[::-1]]
    xy, _ = poly(a + b)
    ax.add_patch(Polygon(xy, closed=True, facecolor=S2_FILL, alpha=.55, edgecolor='none'))
    # normal section of panels (a)-(c), through both packets
    vsec = corner(1, 0, 0)[1]
    xy, _ = poly([point(u, vsec, 0) for u in np.linspace(-1, 1, 80)])
    ax.plot(xy[:, 0], xy[:, 1], color=BLACK, lw=.5, linestyle=(0, (3, 2)))
    draw_layer(ax, slab_items(0, R) + cylinder_items(0, R))
    # frame: fit all drawn content
    allpts = np.array([project(point(u, v, t))[0] for u, v in bd for t in (-R, R)])
    lo, hi = allpts.min(0) - .03, allpts.max(0) + .03
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    w, h = hi - lo
    fig.set_size_inches(4.6 / 2.54, 4.6 / 2.54 * h / w)
    fig.savefig(OUT, transparent=True)
    frac = lambda p: tuple(np.round((project(p)[0] - lo) / (hi - lo), 3))
    anchors = {
        'M (front-right corner)': point(1, -1, 0),
        'C1 cylinder top': point(*corner(-1, 0, 0), R),
        'C2 cylinder top': point(*corner(1, 0, 0), R),
        'slab1 left curve top, v=0.85': point(*strip1(-1, 0, .85), R),
        'slab1 right curve top, v=0.85': point(*strip1(1, 0, .85), R),
        'slab2 top, u=-0.95': point(*strip2(0, -.95), R),
        'slab2 top, u=0.95': point(*strip2(0, .95), R),
        'section start u=-1': point(-1, vsec, 0),
        'section end u=1': point(1, vsec, 0),
        'tube corner fiber (1,-1) top': point(1, -1, R),
        'tube corner fiber (1,-1) bottom': point(1, -1, -R),
    }
    print('size_cm', 4.6, round(4.6 * h / w, 3))
    for k, p in anchors.items():
        print(k, frac(p))


if __name__ == '__main__':
    main()
