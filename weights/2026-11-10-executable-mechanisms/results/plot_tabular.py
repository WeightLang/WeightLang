"""Reproduce tabular_tradeoff.{png,svg} from recorded experiment results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


HERE = Path(__file__).resolve().parent
report = json.loads((HERE / "tabular_results.json").read_text())
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#777777", "axes.linewidth": 0.7,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "svg.fonttype": "none", "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), sharex=True, sharey=True)
colors = {"cost_tree": "#205A91", "cart": "#777777", "random_forest": "#171717"}
labels = {"cost_tree": "CostTree", "cart": "CART", "random_forest": "Random forest\n100 trees"}
markers = {"cost_tree": "o", "cart": "s", "random_forest": "^"}

for ax, dataset, title in zip(
    axes,
    ["wine", "breast_cancer_wisconsin_diagnostic"],
    ["Wine · 178 observations", "WDBC · 569 observations"],
):
    for model in ["cart", "cost_tree", "random_forest"]:
        row = next(s for s in report["summary"] if s["dataset"] == dataset and s["model"] == model)
        x = 1000 * row["total_training_seconds_mean"]
        y = 100 * row["accuracy_mean"]
        dy = 100 * row["accuracy_sd"]
        ax.errorbar(x, y, yerr=dy, fmt=markers[model], markersize=6.3,
                    color=colors[model], elinewidth=1.0, capsize=3, capthick=0.8,
                    markeredgewidth=0.8, zorder=3)
        offset, align = ((9, 4), "left")
        if model == "cart":
            offset = (9, -12)
        if model == "random_forest":
            offset, align = ((-9, -25), "right")
            if dataset == "breast_cancer_wisconsin_diagnostic":
                offset = (-9, 32)
        ax.annotate(labels[model], (x, y), xytext=offset, textcoords="offset points",
                    ha=align, va="center", color=colors[model], fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, pad=14)
    ax.set_xscale("log")
    ax.set_xlim(10, 6000)
    ax.set_ylim(80, 100)
    ax.set_xticks([10, 100, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.tick_params(axis="both", length=3)
    ax.set_xlabel("Total training time (ms, logarithmic scale)", labelpad=9)
axes[0].set_ylabel("Held-out accuracy (%)", labelpad=9)
fig.suptitle("Accuracy and CPU training cost", x=0.08, y=0.985, ha="left", fontsize=15)
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.30, wspace=0.16)
fig.text(0.08, 0.115,
         "Five fixed stratified holdouts; points show means, bars show one standard deviation.\n"
         "Training includes validation search and final fit. Timings compare these implementations on a shared CPU.\n"
         "CostTree is a conventional symbolic-tree probe; these datasets do not establish general intelligence.",
         fontsize=8.3, color="#555555", ha="left", va="center", linespacing=1.6)
fig.savefig(HERE / "tabular_tradeoff.png", dpi=220)
fig.savefig(HERE / "tabular_tradeoff.svg")
plt.close(fig)
