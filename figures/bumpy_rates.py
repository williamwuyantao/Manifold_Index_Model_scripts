"""Paper-style learning curves with one common two-panel layout.

Each panel reports one metric and overlays the response links.  The layout is
shared by the bumpy-sphere and coupled-helix scaling figures so the reader can
compare links within a metric without switching between four small panels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = (("relMSE", "testing_relMSE_total"), ("devTan", "testing_devTan"))
COLORS = ("#0072BD", "#D95319", "#22845C", "#7E2F8E")
MARKERS = ("o", "s", "^", "D")


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
        "legend.fontsize": 8.3, "xtick.labelsize": 8.7, "ytick.labelsize": 8.7,
        "axes.linewidth": .55, "lines.linewidth": 1.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False, "savefig.facecolor": "white",
        "legend.framealpha": 1, "legend.fancybox": False,
    })


def _configure_axis(ax, xlim, ylim):
    ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim)
    ax.set_axisbelow(True)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=20))
        axis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
        axis.set_major_formatter(LogFormatterMathtext())
        axis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="black", alpha=.16, linewidth=.45)
    ax.grid(which="minor", color="black", alpha=.14,
            linewidth=.4, linestyle=(0, (1, 4)))
    ax.tick_params(which="both", direction="in", top=True, right=True, width=.5)
    ax.tick_params(which="major", length=3.0, pad=2.6)
    ax.tick_params(which="minor", length=1.4)


def render_overlay(items: list[dict], output: Path, *,
                   labels: list[str] | None = None,
                   colors: tuple[str, ...] = COLORS,
                   reference_line: float | None = None,
                   subject: str = "Ten IID repetitions; mean +/- sample SD") -> dict:
    """Render two metric panels, overlaying one curve per item.

    ``items`` contain ``rows`` and ``fits`` as returned by the statistics
    builders.  The same routine is used for the bumpy and coupled figures.
    """
    _style()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = labels or ["Relative prediction MSE", "Mean tangent-space error"]
    fig = plt.figure(figsize=(6.17, 3.07), facecolor="white")
    axes = [fig.add_axes([.085, .16, .385, .755]),
            fig.add_axes([.575, .16, .385, .755])]
    all_ns = np.concatenate([np.asarray([r["N"] for r in item["rows"]], float)
                             for item in items])
    xlim = (float(all_ns.min()) / 1.04, float(all_ns.max()) * 1.015)
    report = []
    for j, (short, _) in enumerate(METRICS):
        values = []
        for item in items:
            mean = np.asarray([r[short] for r in item["rows"]], float)
            sd = np.asarray([r[short + "_std"] for r in item["rows"]], float)
            values.extend((mean - sd).tolist())
            values.extend((mean + sd).tolist())
        low = max(float(np.min(values)) * .84, np.finfo(float).tiny)
        high = float(np.max(values)) * 1.18
        ax = axes[j]
        _configure_axis(ax, xlim, (low, high))
        if reference_line is not None:
            ax.axvline(reference_line, color=".20", linewidth=.55,
                       linestyle="-.", zorder=1)
        curves = []
        for i, item in enumerate(items):
            n = np.asarray([r["N"] for r in item["rows"]], float)
            mean = np.asarray([r[short] for r in item["rows"]], float)
            sd = np.asarray([r[short + "_std"] for r in item["rows"]], float)
            color = colors[i % len(colors)]
            marker = MARKERS[i % len(MARKERS)]
            ax.errorbar(n, mean, yerr=sd, fmt="none", ecolor=color,
                        elinewidth=.7, capsize=2.0, capthick=.7,
                        clip_on=False, zorder=3)
            line, = ax.plot(n, mean, marker=marker, linestyle="-" if i == 0 else "--",
                            color=color, linewidth=1.7, markersize=3.2,
                            markeredgewidth=.45, markerfacecolor="white" if i else color,
                            clip_on=False, zorder=4)
            label = item.get("legend_label", item.get("name", ""))
            slope = item["fits"][short]["slope"]
            curves.append((line, rf"{label}: slope {slope:.3f}"))
        legend = ax.legend([line for line, _ in curves], [text for _, text in curves],
                           loc="upper right", borderpad=.28, handlelength=1.8,
                           handletextpad=.35, borderaxespad=.38, labelspacing=.25)
        legend.get_frame().set(edgecolor="black", linewidth=.45)
        ax.set_xlabel(r"$n$: sample size")
        ax.set_ylabel(labels[j])
        report.append({"metric": short, "axes_position": list(ax.get_position().bounds),
                       "curves": [{"name": item.get("name"),
                                   "slope": item["fits"][short]["slope"]}
                                  for item in items],
                       "y_limits": list(ax.get_ylim())})
    fig.savefig(output, dpi=450, bbox_inches="tight", pad_inches=.045,
                metadata={"Title": "", "Subject": subject})
    preview = output.with_suffix(".png")
    fig.savefig(preview, dpi=180, bbox_inches="tight", pad_inches=.045)
    plt.close(fig)
    print(output)
    return {"name": output.stem, "page_size_points": [float(v * 72) for v in fig.get_size_inches()],
            "axes": report}


def render_figure(item: dict, stats: dict, reference: dict | None = None) -> dict:
    """Compatibility wrapper for a single curve (global-tube figures)."""
    output = ROOT / "output/pdf" / (item["name"] + ".pdf")
    return render_overlay([item], output, reference_line=item.get("fit_lower_N", 1_000_000))


def fit_mean(n: np.ndarray, means: np.ndarray, lower=1_000_000) -> dict:
    mask = n >= lower
    x, y = np.log(n[mask]), np.log(means[mask])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {
        "N_values": n[mask].tolist(), "slope": float(slope),
        "decay_exponent_a": float(-slope), "C": float(np.exp(intercept)),
        "R_squared": float(1 - residual @ residual / ((y - y.mean()) @ (y - y.mean()))),
        "method": "unweighted OLS of log(arithmetic mean testing error) on log(N)",
    }
