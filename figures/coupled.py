"""Geometry, response components and learning curves for the coupled-helix experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures/pdf"
PREVIEW = ROOT / "reproduced/previews"
REPORTS = ROOT / "reproduced"
SCREEN = ROOT / "data/coupled_helix"
MAPS = (("A", "natural_polynomial3", "Polynomial", "poly", "#0072BD"),
        ("B", "natural_rational3", "Rational", "rat", "#D95319"),
        ("C", "natural_exponential3", "Exponential", "exp", "#22845C"))
FIELDS = ("alpha", "K_offset", "h_scale", "rho_init", "bite", "eval_rho", "eval_h")
FIT_LOWER = 1_000_000
PARAMETER_DIVISIONS = 10
PARAMETER_DASHES = 64
SELECTED_GEOMETRY_SEED = 202610210936  # Seed for the stored projection.


def read(path):
    return json.loads(path.read_text())


def source(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())








def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": 13, "axes.labelsize": 13, "axes.titlesize": 13,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": .55, "lines.linewidth": 1.8, "pdf.fonttype": 42,
        "axes.unicode_minus": False, "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / (name+".pdf"), dpi=450, bbox_inches="tight", pad_inches=.045)
    fig.savefig(PREVIEW / (name+".png"), dpi=180, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)
    return OUT / (name+".pdf")




def response_figures():
    summary = read(SCREEN / "response_summary.json")
    with np.load(SCREEN / "response_fields.npz") as data:
        uv = data["grid_uv"]
        side = int(np.sqrt(len(uv)))
        responses = {k: data[k+"_response"].copy() for k in ("A", "B")}
    outputs = []
    for label, symbol, name in (
            ("A", "poly", "Coupled_Meyer_Helix_D20_response_components"),
            ("B", "rat", "Coupled_Meyer_Helix_D20_rational_response_components")):
        values = responses[label].reshape(side, side, 3)
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45))
        for j in range(3):
            ax = axes[j]
            limit = float(np.max(np.abs(values[:, :, :2] if j < 2 else values[:, :, 2])))
            image = ax.imshow(values[:, :, j], origin="lower", extent=(0, 1, 0, 1),
                              cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                              interpolation="bilinear", rasterized=True)
            ax.set(xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.set_xlabel("$s$"); ax.set_ylabel("$t$")
            ax.set_title(rf"$f_{{{j+1}}}^{{\mathrm{{{symbol}}}}}$", fontsize=16, pad=8)
            ax.tick_params(direction="in", top=True, right=True, width=.5, length=3)
            cb = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
            cb.ax.tick_params(labelsize=10, width=.5, length=2)
        fig.subplots_adjust(left=.055, right=.985, bottom=.16, top=.86, wspace=.48)
        outputs.append(save(fig, name))
    return outputs, {k: summary["maps"][k] for k in ("A", "B")}


def geometry_data(side=141):
    """Project the surface onto the stored predictor PCA basis."""
    from coupled_helix_generator import square_surface
    directory = ROOT / "data/coupled_helix"
    report = read(directory / "summary.json")
    with np.load(directory / "samples.npz") as values:
        n = int(values["training_rows"])
        latent = values["latent"][:n].copy()
        basis = values["pca_basis"][:, :3].copy()
        pca_mean = values["pca_mean"].copy()
        noisy = values["x_noisy"][:n].astype(float)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-14)
    centered = noisy-noisy.mean(axis=0)
    covariance = centered.T @ centered/(n-1)
    eig, vectors = np.linalg.eigh(covariance)
    np.testing.assert_allclose(basis @ basis.T, vectors[:, -3:] @ vectors[:, -3:].T, atol=1e-6)
    center = np.zeros(20)
    for lo in range(0, n, 4096):
        center += square_surface(latent[lo:lo+4096], "square_coupled_helix")[0].sum(axis=0)
    center /= n
    scale = report["geometric_parameters"]["scale_factor"]
    axis = np.linspace(0, 1, side)
    s, t = np.meshgrid(axis, axis)
    grid = np.column_stack((s.ravel(), t.ravel()))
    xyz = (square_surface(grid, "square_coupled_helix")[0]-center)*scale-pca_mean
    return dict(xyz=xyz, s=s, t=t, basis=basis, pca_mean=pca_mean,
                source_directory=directory, scale=scale,
                variance_fraction=float(eig[-3:].sum()/eig.sum()))


def random_basis(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((20, 3)))
    q *= np.sign(np.diag(r))
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=2e-14)
    return q


def coordinate_colors(s, t, mode):
    # Place primary and intermediate hues at the centers of three equal
    # bands on each axis. C1-smooth interpolation makes orange/purple as
    # broad as their adjacent primaries, not a narrow boundary mixture.
    s, t = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(t, dtype=float))
    color_s = np.full_like(s, .5) if mode == "t" else s
    color_t = np.full_like(t, .5) if mode == "s" else t
    def band_weights(value):
        left = np.clip(3*(value-1/6), 0, 1)
        right = np.clip(3*(value-.5), 0, 1)
        left = left**2*(3-2*left)
        right = right**2*(3-2*right)
        return np.stack((1-left, left-right, right), axis=-1)
    # Rows run from t=0 to t=1; columns run from s=0 to s=1.
    anchors = np.array([
        [[.12, .33, .82], [.08, .66, .72], [.10, .65, .27]],
        [[.62, .20, .70], [.62, .62, .62], [.55, .78, .15]],
        [[.88, .16, .15], [.97, .47, .10], [.98, .82, .10]],
    ])
    rgb = np.einsum("...i,...j,jic->...c", band_weights(color_s),
                    band_weights(color_t), anchors)
    distance = np.hypot(color_s-.5, color_t-.5)
    fade = np.clip((distance-.025)/(.09-.025), 0, 1)
    gray_weight = 1-fade**2*(3-2*fade)
    rgb = rgb*(1-gray_weight[..., None])+.62*gray_weight[..., None]
    return np.concatenate((rgb, np.ones(s.shape+(1,))), axis=-1)


def draw_surface(ax, data, basis, mode, view=(20, 35), *, detail=True):
    shape = data["s"].shape
    mesh = (data["xyz"] @ basis).reshape(*shape, 3)
    colors = coordinate_colors(data["s"], data["t"], mode)
    if mode == "st" and detail:
        # Put the coordinate curves in the surface texture so hidden portions
        # are occluded correctly, rather than drawing 3D lines through folds.
        for j in range(1, PARAMETER_DIVISIONS):
            column = round(j*(shape[1]-1)/PARAMETER_DIVISIONS)
            row = round(j*(shape[0]-1)/PARAMETER_DIVISIONS)
            colors[:, column, :3] = .75+.25*colors[:, column, :3]
            dash = (np.floor(data["s"][row]*2*PARAMETER_DASHES).astype(int) % 2) == 0
            colors[row, dash, :3] *= .42
    ax.plot_surface(*np.moveaxis(mesh, 2, 0), rstride=1, cstride=1,
        facecolors=colors, linewidth=0,
        antialiased=False, shade=False, rasterized=True)
    if detail:
        for line in (mesh[0], mesh[-1], mesh[:, 0], mesh[:, -1]):
            ax.plot(*line.T, color=".15", alpha=.4, lw=.45)
    lo, hi = mesh.reshape(-1, 3).min(axis=0), mesh.reshape(-1, 3).max(axis=0)
    span = float((hi-lo).max()*1.09)
    middle = (hi+lo)/2
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
        setter(c-span/2, c+span/2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set(xticks=[], yticks=[], zticks=[])
    if detail:
        ax.set(xlabel="$p_1$", ylabel="$p_2$", zlabel="$p_3$")
    ax.grid(False)
    for coordinate in (ax.xaxis, ax.yaxis, ax.zaxis):
        coordinate.pane.set_facecolor((.98, .98, .98, 1))
        coordinate.pane.set_edgecolor((.80, .80, .80, 1))


def color_key(fig, mode):
    if mode in ("s", "t"):
        ax = fig.add_axes([.385, .045, .23, .025])
        axis = np.linspace(0, 1, 256)
        cmap = ListedColormap(coordinate_colors(axis, axis, mode))
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
                          cax=ax, orientation="horizontal", ticks=[0, .5, 1])
        cb.set_label("$"+mode+"$", fontsize=12, labelpad=0)
        cb.ax.tick_params(labelsize=10, length=2)
    else:
        ax = fig.add_axes([.448, .008, .104, .208])
        s, t = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
        ax.imshow(coordinate_colors(s, t, mode), origin="lower", extent=(0, 1, 0, 1))
        for value in np.arange(1, PARAMETER_DIVISIONS)/PARAMETER_DIVISIONS:
            ax.axvline(value, color="white", lw=.5, alpha=.85)
            ax.axhline(value, color=".15", lw=.5, ls=(0, (1.2, 1.2)), alpha=.8)
        ax.set(xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel="$s$", ylabel="$t$")
        ax.tick_params(labelsize=9, length=2, pad=1)
        ax.xaxis.labelpad = 0
        ax.yaxis.labelpad = 0


def projection_pair(data, seed, mode, view=(20, 35), label=None, *, random_view=None):
    fig = plt.figure(figsize=(10.4, 4.7))
    ax1 = fig.add_axes([0, .005, .495, .95], projection="3d")
    ax2 = fig.add_axes([.505, .005, .495, .95], projection="3d")
    draw_surface(ax1, data, data["basis"], mode, view)
    draw_surface(ax2, data, random_basis(seed), mode, view if random_view is None else random_view)
    for ax in (ax1, ax2):
        ax.set_box_aspect((1, 1, 1), zoom=1.18)
    ax2.set(xlabel="$r_1$", ylabel="$r_2$", zlabel="$r_3$")
    ax1.set_title("First three principal components", fontsize=14, pad=1)
    ax2.set_title("Random three-dimensional subspace", fontsize=14, pad=1)
    color_key(fig, mode)
    if mode == "st":
        key = fig.axes[-1]
        key.set_position([.471, .015, .058, .058*10.4/4.7])
        key.tick_params(labelsize=8, length=1.7, pad=1)
        key.xaxis.label.set_size(10)
        key.yaxis.label.set_size(10)
    if label:
        fig.text(.03, .025, label, fontsize=14)
    return fig




def geometry_figure():
    data = geometry_data(side=401)
    seed, mode, view = SELECTED_GEOMETRY_SEED, "st", (20, 35)
    geometry_pdf = save(projection_pair(data, seed, mode, view),
                        "coupled_helix_geometry")
    records = [dict(option="A", random_seed=seed, color=mode,
                    camera_elevation=view[0], camera_azimuth=view[1],
                    random_basis=random_basis(seed).tolist())]
    return geometry_pdf, dict(
        source_directory=str(data["source_directory"].relative_to(ROOT)),
        projection_fit="PCA of the stored 100000 noisy training predictors",
        projection_basis=data["basis"].tolist(), projection_mean=data["pca_mean"].tolist(),
        random_projection_rule="Gaussian 20x3 matrix followed by QR, with orthonormal columns",
        geometry_coloring="3x3 color anchors at coordinate centers (1/6,1/2,5/6), with C1-smooth adjacent interpolation. Top: red (.88,.16,.15), orange (.97,.47,.10), yellow (.98,.82,.10). Middle: purple (.62,.20,.70), gray (.62,.62,.62), yellow-green (.55,.78,.15). Bottom: blue (.12,.33,.82), cyan (.08,.66,.72), green (.10,.65,.27). Small exact-gray center radius .025, local gray blend vanishing by radius .09. No lighting or mesh-antialias whitening",
        edge_color_band_target_width=1/3,
        color_anchor_coordinates=[1/6, .5, 5/6],
        parameter_curves="s=j/10 (j=1..9): white solid; t=j/10 (j=1..9): dark dashed, 64 dash cycles along s; embedded surface texture with correct fold occlusion",
        options=records, projection_id="A",
        layout=dict(figure_inches=[10.4, 4.7], isotropic_camera_zoom=1.18,
                    color_key_side_inches=.058*10.4),
        first_three_variance_fraction=data["variance_fraction"],
        intrinsic_dimension=2, ambient_dimension=20, parameter_domain="[0,1]^2",
        parameter_sampling="uniform parameter area", geometric_scale=data["scale"])






