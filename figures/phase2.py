import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter,LogFormatterMathtext,LogLocator,NullFormatter
ROOT=Path(__file__).resolve().parents[1]
PDFS=ROOT/"figures/pdf"
FIGURES=ROOT/"reproduced/previews"
COLORS=((0,114/255,189/255),(217/255,83/255,25/255),(119/255,172/255,48/255))
ROLES=("initial_projection","final_projection","true_projection")
def paper_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
        "font.size": 14, "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.linewidth": .5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.framealpha": 1, "legend.fancybox": False})


def axes_style(ax):
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="black", alpha=.15, linewidth=.5)
    ax.grid(which="minor", color="black", alpha=.15, linewidth=.5, linestyle=(0, (1, 4)))
    ax.tick_params(which="both", direction="in", top=True, right=True, width=.5)
    ax.tick_params(which="major", length=3.5, pad=4.65)
    ax.tick_params(which="minor", length=1.6)


def legend(ax, loc):
    leg = ax.legend(loc=loc, title=r"$\sigma_Y=1\%$", fontsize=13,
        title_fontsize=13, borderpad=.35, handlelength=1.7,
        handletextpad=.5, borderaxespad=.55)
    leg.get_frame().set(edgecolor="black", linewidth=.5)


def export(fig, name, report):
    pdf, png = PDFS/(name+".pdf"), FIGURES/(name+".png")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= bounds.x1+1 and box.y1 <= bounds.y1+1
    fig.savefig(pdf, metadata={"Title": "", "Subject": "Ten paired IID repetitions; arithmetic mean and sample SD",
        "CreationDate": None, "ModDate": None, "Creator": "", "Producer": ""})
    fig.savefig(png, dpi=180)
    plt.close(fig)
    report.append(dict(name=name, pdf=str(pdf), png=str(png)))
    print(pdf)


def iteration_figure(data, report, *, tangent_label="Mean tangent-space error on test data",
                     tangent_metric="testing_devTan",
                     tangent_legend="upper center",
                     name="phase2_iterations"):
    reference = json.loads((ROOT/"config/figure_style.json").read_text())[0]
    width, height = reference["page_size_points"]
    px, py, pw, ph = reference["paper_position_points"]
    fig = plt.figure(figsize=(width/72, height/72), facecolor="white")
    labels = (tangent_label, r"$L^2$-relative MSE on test data")
    for j, metric in enumerate((tangent_metric, "testing_relMSE_total")):
        x, y, w, h = reference["axes_positions"][j]
        ax = fig.add_axes([(px+pw*x)/width, (py+ph*y)/height, pw*w/width, ph*h/height])
        axes_style(ax)
        lows, highs = [], []
        for group, color, marker in zip(data["groups"], COLORS, ("o", "s", "^")):
            s = group["trajectory"][metric]
            mean, sd = np.array(s["mean"]), np.array(s["sample_standard_deviation"])
            ax.errorbar(group["iterations"], mean, yerr=sd, color=color,
                marker=marker, markersize=3.6, markeredgewidth=.6, linewidth=1.8,
                elinewidth=.65, capsize=1.8, capthick=.65,
                label=rf"$n={group['N']//100000}\times10^5$", clip_on=False)
            lows.extend(mean-sd); highs.extend(mean+sd)
        last = max(group["iterations"][-1] for group in data["groups"])
        ax.set(xlim=(-.4, last+.4), ylim=(min(lows)/1.10, max(highs)*1.10))
        ax.xaxis.set_major_locator(FixedLocator(list(range(0, last+1, 5))))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        if j == 0:
            ax.yaxis.set_major_locator(FixedLocator([.002, .005, .01, .02, .05, .1, .2, .5]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}$"))
        ax.set_xlabel(r"$k$: Phase 2 iteration", fontsize=14)
        ax.xaxis.set_label_coords(.5, -.080674846626)
        ax.xaxis.label.set_verticalalignment("top")
        ax.set_ylabel(labels[j], fontsize=14)
        ax.yaxis.set_label_coords(-.12, .5)
        ax.yaxis.label.set_verticalalignment("bottom")
        legend(ax, tangent_legend if j == 0 else "lower left")
    export(fig, name, report)


def projection_figure(data, report):
    fig = plt.figure(figsize=(520/72, 400/72), facecolor="white")
    ax = fig.add_axes([.155, .15, .80, .80])
    axes_style(ax)
    x = np.arange(3)
    all_low, all_high = [], []
    for group, color, marker in zip(data["groups"], COLORS, ("o", "s", "^")):
        values = [group["projection_readout"][role]["testing_relMSE_total"] for role in ROLES]
        mean = np.array([v["mean"] for v in values])
        sd = np.array([v["sample_standard_deviation"] for v in values])
        ax.errorbar(x, mean, yerr=sd, color=color, marker=marker, linestyle="-",
            linewidth=2, markersize=5, markeredgewidth=.6, elinewidth=.9,
            capsize=3.5, capthick=.9, label=rf"$n={group['N']//100000}\times10^5$", clip_on=False)
        all_low.extend(mean-sd); all_high.extend(mean+sd)
    ax.set(xlim=(-.13, 2.13), ylim=(min(all_low)/1.13, max(all_high)*1.16))
    ax.set_xticks(x, ("Initial", "Final", "True tangent"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("Projection used for final prediction", fontsize=14, labelpad=8)
    ax.set_ylabel(r"$L^2$-relative MSE on test data", fontsize=14, labelpad=10)
    legend(ax, "upper right")
    export(fig, "phase2_projections", report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render paired Phase-2 IID results.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'reproduced')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    assert all(g['repetitions'] == 10 for g in data['groups'])
    PDFS, FIGURES = args.output/'pdf', args.output/'previews'
    PDFS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paper_style()
    report = []
    iteration_figure(data, report, tangent_label='Mean tangent-space error at training points',
        tangent_metric='direct_tangent_error', tangent_legend='lower left', name='phase2_iterations')
    projection_figure(data, report)


