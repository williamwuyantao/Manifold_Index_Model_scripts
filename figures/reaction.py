#!/usr/bin/env python3
"""Paper figures for the existing D20 sphere reaction-coordinate dataset."""
from pathlib import Path
import json

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from bumpy_maps import paper_colormap, REFERENCE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/pdf"
CM = None
FONT = FontProperties(family="cmr10", size=10)
COLORS = ["#0072BD", "#D95319", "#77AC30", "#7E2F8E"]
A = np.array([0.595, 0.765, 0.935, 1.105])
Q = np.array([[-np.sqrt(3)/4, -.75, -1/3],
              [np.sqrt(3)/2, 0, -1/3],
              [-np.sqrt(3)/4, .75, -1/3], [0, 0, 1]])


def sphere(phi, theta):
    return np.stack((np.cos(theta)*np.cos(phi),
                     np.cos(theta)*np.sin(phi), np.sin(theta)), axis=-1)


def target_boundaries():
    boundaries = []
    low, high, halfwidth = np.arctan(-.5), 0., np.arccos(.8660254)
    for center in (-2*np.pi/3, 0., 2*np.pi/3):
        edge = np.linspace(-halfwidth, halfwidth, 80)
        vertical = np.linspace(low, high, 40)
        phi = np.r_[center+edge, np.full(40, center+halfwidth),
                    center+edge[::-1], np.full(40, center-halfwidth)]
        theta = np.r_[np.full(80, low), vertical,
                      np.full(80, high), vertical[::-1]]
        boundaries.append((phi, theta))
    boundaries.append((np.linspace(-np.pi, np.pi, 300),
                       np.full(300, np.arctan(1/.202710))))
    return boundaries


def angle_axes(ax):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  ["-180", "-90", "0", "90", "180"])
    ax.set_yticks([-np.pi/2, 0, np.pi/2],
                  ["-90", "0", "90"])
    ax.set_xlabel(r"$\lambda$: longitude (degrees)")
    ax.set_ylabel(r"$\varphi$: latitude (degrees)")
    ax.set_aspect(1)
    ax.tick_params(direction="in", top=True, right=True, width=.5)
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontproperties(FONT)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman"],
                         "mathtext.fontset":"cm", "font.size":10,
                         "axes.linewidth":.5, "pdf.fonttype":42,
                         "axes.titlesize":11, "xtick.labelsize":9,
                         "ytick.labelsize":9})
    OUT.mkdir(parents=True, exist_ok=True)
    cmap, _ = paper_colormap(REFERENCE)
    phi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, 501),
                             np.linspace(-np.pi/2, np.pi/2, 251))
    u = sphere(phi, theta)
    potential = np.sum(np.exp((u @ Q.T)*A), axis=-1)/10
    borders = target_boundaries()

    fig = plt.figure(figsize=(8.0, 3.1))
    ax = fig.add_axes([.015, .16, .48, .76])
    for sign, center in ((1., -1.12), (-1., 1.12)):
        ax.add_patch(Circle((center, 0), 1, facecolor=".97", edgecolor=".25", lw=.65))
        for lat in np.linspace(-np.pi/3, np.pi/3, 5):
            points = sphere(np.linspace(-np.pi, np.pi, 360), np.full(360, lat))
            visible = sign*points[:, 0] >= 0
            ax.plot(np.where(visible, center+sign*points[:, 1], np.nan),
                    np.where(visible, points[:, 2], np.nan), color=".75", lw=.35)
        for lon in np.linspace(-np.pi, np.pi, 13):
            points = sphere(np.full(200, lon), np.linspace(-np.pi/2, np.pi/2, 200))
            visible = sign*points[:, 0] >= -1e-12
            ax.plot(np.where(visible, center+sign*points[:, 1], np.nan),
                    np.where(visible, points[:, 2], np.nan), color=".75", lw=.35)
        for j, (p, t) in enumerate(borders):
            points = sphere(p, t)
            visible = sign*points[:, 0] >= -1e-12
            ax.plot(np.where(visible, center+sign*points[:, 1], np.nan),
                    np.where(visible, points[:, 2], np.nan), color=COLORS[j], lw=1.5)
        ax.text(center, -1.14, r"$u_1\geq0$" if sign>0 else r"$u_1\leq0$",
                ha="center", va="top")
    for label, position in ((r"$A_2$",(-1.12,-.22)), (r"$A_1$",(1.88,-.22)),
                            (r"$A_3$",(.36,-.22)), (r"$A_4$",(-1.12,.88)),
                            (r"$A_4$",(1.12,.88))):
        ax.text(*position, label, ha="center", va="center", fontsize=10)
    ax.set(xlim=(-2.22,2.22), ylim=(-1.30,1.08), aspect="equal")
    ax.set_axis_off()
    ax.set_title(r"(a) $\mathbb{S}^2$ and target regions", pad=8)
    ax = fig.add_axes([.57,.19,.34,.70])
    im = ax.pcolormesh(phi,theta,potential, cmap=cmap,shading="auto",rasterized=True)
    for p,t in borders:
        ax.plot(p,t,color="white",lw=1)
    angle_axes(ax)
    ax.set_aspect(2)
    ax.set_title(r"(b) $U_{\rm tan}$", pad=7)
    bar = fig.colorbar(im,cax=fig.add_axes([.935,.23,.012,.62]))
    bar.ax.tick_params(labelsize=9,width=.5)
    save(fig,"reaction_geometry")

    with np.load(ROOT / "data/reaction_samples.npz") as values:
        rows = values["source_rows"]
        p, t = values["phi"], values["theta"]
        clean, y = values["XcleanTop"], values["Y"]
        assert np.max(np.abs(sphere(p,t)-clean)) < 5e-7
    fig, axes = plt.subplots(2,2,figsize=(7.8,4.8))
    fig.subplots_adjust(left=.09,right=.88,bottom=.11,top=.95,wspace=.32,hspace=.36)
    for j, ax in enumerate(axes.flat):
        im = ax.scatter(p,t,c=y[:,j],s=1.0,cmap=cmap,norm=Normalize(0,1),
                        edgecolors="none",rasterized=True)
        angle_axes(ax)
        ax.set_title(rf"$Y_{j+1}$",pad=5)
    bar = fig.colorbar(im,cax=fig.add_axes([.925,.14,.015,.75]),ticks=[0,.25,.5,.75,1])
    bar.set_label("First-hit probability",fontsize=10)
    for label in bar.ax.get_yticklabels():
        label.set_fontproperties(FONT)
    save(fig,"reaction_components")
    print(json.dumps({"response_sample_count":len(rows),
                      "directional_potential_range":[float(potential.min()),float(potential.max())],
                      "figure_files":["reaction_geometry.pdf",
                                      "reaction_components.pdf"]}))


if __name__ == "__main__":
    main()
