#!/usr/bin/env python3
"""Render clean Y0/Y2 components on a full-sphere longitude/latitude chart.

The colors come directly from the previous paper's PDF colorbar. Geometry,
rotation and SINGLE response arithmetic are the experimental definitions.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
from pypdf import PdfReader

from geometry import radial_parameters, radius

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scaling_law_N"))
from cuda_core_data import response_map

REFERENCE = None
REFERENCE_PDF = "Parula color lookup table"


def paper_colormap(archive=None):
    colors = np.asarray(json.loads((REPO / "config/parula.json").read_text()))
    return ListedColormap(colors, name="paper_parula"), len(colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=REPO / "reproduced")
    parser.add_argument("--paper-notation", action="store_true",
                        help="Use f_1,...,f_6 titles and new f_lin/f_quad output names")
    args = parser.parse_args()
    context = json.loads((REPO/"config/bumpy_geometry.json").read_text())
    identity = context["Y0"]["data_identity"]["base_dataset_identity"]
    rotation = np.array(context["Y0"]["data_identity"]["response_rotation"])
    assert identity["geometry_formula"] == context["Y2"]["data_identity"]["base_dataset_identity"]["geometry_formula"]
    assert np.array_equal(rotation, context["Y2"]["data_identity"]["response_rotation"])
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12, rtol=0)

    longitude, latitude = np.linspace(-180, 180, 721), np.linspace(-90, 90, 361)
    lon, lat = np.deg2rad(np.meshgrid(longitude, latitude))
    unit = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1).reshape(-1, 3)
    radial = radius(unit, radial_parameters(identity))
    # Physical geometry_scale cancels in z = X_clean[:3] / scale @ R.T;
    # the actual radial deformation r(u) remains in the response input.
    z = (unit * radial[:, None]) @ rotation.T
    cmap, color_count = paper_colormap(args.reference)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "font.size": 14, "mathtext.fontset": "cm", "axes.linewidth": .5,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
        "xtick.direction": "in", "ytick.direction": "in"})
    pdf_dir, png_dir = args.output / "pdf", args.output / "previews"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    report = dict(projection="equirectangular longitude/latitude", grid_shape=list(lon.shape),
        definition="Y_map(r(u) u R^T), six components", response_noise_added=False,
        experiment_relative_Y_noise=.01, color_limits=[0, 1],
        colormap_source_pdf=REFERENCE_PDF, colormap_color_count=color_count,
        geometry_formula_sha256=identity["geometry_formula_sha256"],
        font="Times New Roman, Computer Modern math", axis_line_width_points=.5)
    for map_id in (0, 2):
        fields = response_map(z, map_id).reshape(*lon.shape, 6)
        assert fields.dtype == np.float32 and np.isfinite(fields).all()
        assert fields.min() >= 0 and fields.max() <= 1
        seam = float(np.max(np.abs(fields[:, 0] - fields[:, -1])))
        poles = float(max(np.ptp(fields[0], axis=0).max(), np.ptp(fields[-1], axis=0).max()))
        assert seam < 2e-6 and poles < 2e-6
        fig, axes = plt.subplots(2, 3, figsize=(13, 5))
        fig.subplots_adjust(left=.061, right=.904, bottom=.13, top=.91,
                            wspace=.31, hspace=.53)
        for index, ax in enumerate(axes.flat):
            im = ax.imshow(fields[:, :, index], origin="lower", cmap=cmap,
                vmin=0, vmax=1, extent=(-180, 180, -90, 90),
                interpolation="nearest", aspect="equal", rasterized=True)
            ax.set(xlim=(-180, 180), ylim=(-90, 90),
                   xticks=[-180, -90, 0, 90, 180], yticks=[-90, 0, 90])
            ax.xaxis.set_minor_locator(MultipleLocator(30))
            ax.yaxis.set_minor_locator(MultipleLocator(30))
            ax.grid(which="major", color="black", alpha=.18, linewidth=.5)
            ax.grid(which="minor", color="black", alpha=.08, linewidth=.5, linestyle=":")
            ax.tick_params(which="major", width=.5, length=3, labelsize=12)
            ax.tick_params(which="minor", width=.5, length=1.5)
            title = (r"$f_{%d}$" % (index + 1) if args.paper_notation
                     else r"$\mathbf{Y}_{%d}^{(%d)}$" % (map_id, index + 1))
            ax.set_title(title, fontsize=15, pad=9)
            ax.set_xlabel(r"$\lambda$: longitude (degrees)", fontsize=13, labelpad=5)
            ax.set_ylabel(r"$\varphi$: latitude (degrees)", fontsize=13, labelpad=4)
        fig.canvas.draw()
        bottom, top = axes[1, 2].get_position().y0, axes[0, 2].get_position().y1
        colorbar = fig.colorbar(im, cax=fig.add_axes([.926, bottom, .012, top - bottom]),
                               ticks=[0, .25, .5, .75, 1])
        colorbar.ax.tick_params(width=.5, length=3, labelsize=12)
        colorbar.outline.set_linewidth(.5)
        stem = f"bumpy_v2_D20_Y{map_id}_components_2d"
        if args.paper_notation:
            stem = "bumpy_%s_components" % ("linear" if map_id == 0 else "quadratic")
        fig.savefig(pdf_dir / f"{stem}.pdf", dpi=300)
        fig.savefig(png_dir / f"{stem}.png", dpi=160)
        plt.close(fig)
        report[f"Y{map_id}"] = dict(component_count=6, seam_max_abs=seam,
            pole_max_abs=poles, minimum=float(fields.min()), maximum=float(fields.max()))
        output = PdfReader(pdf_dir / f"{stem}.pdf")
        assert len(output.pages) == 1
        assert "longitude" in output.pages[0].extract_text()
        print(f"Y{map_id}: six components; seam={seam:g}, poles={poles:g}; {pdf_dir / (stem + '.pdf')}")
    report_path = REPO / "reproduced/response_maps.json"
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
