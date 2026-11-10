"""Plot frozen primary string outcomes; all counts/timings come from JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "string_summary.json").read_text())
cold = data["summary"]["cold_indexed"]
cached = data["summary"]["whole_program_cache"]
counts = [cold["all_test_correct"], cold["fit_but_wrong"], cold["no_fit"]]
assert sum(counts) == cold["runs"]
discovery = data["library_discovery_s"]
cache_total = cached["train_total_s"] + discovery
assert abs(cache_total - data["cache_including_discovery_s"]) < 1e-9

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#777777",
    "axes.linewidth": 0.7, "svg.fonttype": "none",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "savefig.facecolor": "white",
})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})
left, right = axes
blue, gray, ink = "#205A91", "#B9B9B9", "#555555"

offset = 0
for value, face, edge, textcolor in zip(counts, [blue, gray, "white"], [blue, gray, "#777777"], ["white", "#222222", "#222222"]):
    left.barh(0.66, value, left=offset, height=0.24, color=face, edgecolor=edge, linewidth=0.8)
    left.text(offset + value / 2, 0.66, str(value), ha="center", va="center", color=textcolor, fontsize=11)
    offset += value
left.set_xlim(0, cold["runs"])
left.set_ylim(0, 1)
left.set_xticks([0, cold["runs"] / 4, cold["runs"] / 2, 3 * cold["runs"] / 4, cold["runs"]])
left.set_yticks([])
left.set_xlabel("Task–split evaluations", labelpad=9)
left.set_title(f"Cold search · {data['protocol']['test_tasks']} tasks × {len(data['protocol']['seeds'])} splits",
               loc="left", fontsize=11, pad=16)
legend_labels = [
    f"All held-out outputs correct   {counts[0] / cold['runs']:.0%}",
    f"Training fit, held-out error   {counts[1] / cold['runs']:.0%}",
    f"No program found   {counts[2] / cold['runs']:.0%}",
]
left.legend([Patch(facecolor=blue), Patch(facecolor=gray), Patch(facecolor="white", edgecolor="#777777")],
            legend_labels, loc="lower left", bbox_to_anchor=(0, 0.06), frameon=False,
            handlelength=1.0, handleheight=0.85, fontsize=9, borderaxespad=0, labelspacing=0.7)

cold_time, cache_time = cold["train_total_s"], cached["train_total_s"]
right.barh(0.75, cold_time, height=0.21, color=ink)
right.barh(0.31, cache_time, height=0.21, color=ink)
right.barh(0.31, discovery, left=cache_time, height=0.21,
           color="white", edgecolor=ink, linewidth=0.8, hatch="////")
for y, total in [(0.75, cold_time), (0.31, cache_total)]:
    right.text(total + 0.12, y, f"{total:.3f} s", va="center", fontsize=10, color="#222222")
right.set_yticks([0.75, 0.31], ["Cold", "Program\ncache"])
right.tick_params(axis="y", length=0, pad=9)
right.set_xlim(0, max(cold_time, cache_total) * 1.20)
right.set_ylim(0, 1)
right.set_xticks([0, 2, 4, 6, 8])
right.set_xlabel("Cumulative learning time (seconds)", labelpad=9)
right.set_title("Acquisition cost counts", loc="left", fontsize=11, pad=16)

fig.suptitle("Executable string learning: successes and costs", x=0.065, y=0.99, ha="left", fontsize=15)
fig.subplots_adjust(left=0.065, right=0.98, top=0.81, bottom=0.38, wspace=0.38)
fig.text(0.065, 0.23,
         f"Cache learning: {cache_time:.3f} s search/checking + {discovery:.3f} s library discovery = {cache_total:.3f} s (hatched segment).",
         ha="left", fontsize=8.7, color="#444444")
fig.text(0.065, 0.14,
         f"Learning totals are not end-to-end runtime. The complete experiment across all modes took {data['total_wall_s']:.3f} s.\n"
         "Task–split outcomes are correlated. A training-consistent program can still fail on unseen examples.\n"
         "Source: frozen primary benchmark summary; custom PBE generalization experiment, not an official SyGuS score.",
         ha="left", va="center", fontsize=8.4, color="#555555", linespacing=1.6)
fig.savefig(HERE / "string_outcomes.png", dpi=220)
fig.savefig(HERE / "string_outcomes.svg")
plt.close(fig)
