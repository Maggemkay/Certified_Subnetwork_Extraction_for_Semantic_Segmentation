"""Mean pairwise overlap of the per-class circuits vs. top-k ratio (not part of the codebase pipeline).

The line-plot summary of make_overlap_matrices.py: the "mean IoU" and "mean IoMin" values from each panel title
(mean over all class pairs), with one line per circuit type. Left panel IoU, right panel IoMin.
Thin lines in each circuit type's colour and line style mark the min and the max over all class pairs at each
top-k (no shaded fill: three overlapping fills blend into an unreadable grey).
The grey dotted line is the expected overlap of two random circuits that each keep a share k of every layer of U:
    IoU = k / (2 − k),   IoMin = k
It shows how much of the overlap follows from circuit size alone. The certified circuit is a bit smaller than k
(see the matrix panel titles), so its random-chance line would lie slightly lower.

  subnetwork_overlap/mean_overlap_curves.png                   all circuit types in one panel per metric
  subnetwork_overlap/mean_overlap_curves_small_multiples.png   rows IoU / IoMin, one column per circuit type, with
                                                               the min-max spread as a shaded band

Run:  python src/Extra_Material/Extra_images/make_overlap_curves.py
"""

import sys
from pathlib import Path

import matplotlib.patches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_overlap_matrices as M  # noqa: E402  (loading, overlap(), constants and figure style)

plt = M.plt
OUT = M.OUT
# Hue per circuit type, as in s06 and make_extra_images.py.
COLORS = {"baseline_circuit": "#a8431a", "majority": "#0b6b48", "certified": "#174a8c"}
MARKERS = {"baseline_circuit": "s", "majority": "D", "certified": "o"}
# Baseline and majority vote are almost identical: majority is dashed with smaller markers and drawn on top, so the
# baseline line stays visible underneath.
STYLES = {"baseline_circuit": dict(linestyle="-", linewidth=2.6, markersize=5.5),
          "majority": dict(linestyle=(0, (3, 2)), linewidth=1.3, markersize=3.5),
          "certified": dict(linestyle="-", linewidth=1.6, markersize=4.5)}
GRID = "#e1e0d9"


def off_diagonal(matrix):
    return matrix[~np.eye(len(matrix), dtype=bool)]


def mean_overlaps(masks, grid):
    """{circuit type: {"IoU" | "IoMin": {"mean" | "min" | "max": [|grid|]}}}, over all class pairs."""
    result = {}
    for name in M.CIRCUIT_NAMES:
        result[name] = {metric: {stat: [] for stat in ("mean", "min", "max")} for metric in ("IoU", "IoMin")}
        for level in range(len(grid)):
            for metric, matrix in zip(("IoU", "IoMin"), M.overlap(masks[name][level])):
                values = off_diagonal(matrix)
                for stat, function in (("mean", np.nanmean), ("min", np.nanmin), ("max", np.nanmax)):
                    result[name][metric][stat].append(function(values))
        result[name] = {metric: {stat: np.array(v) for stat, v in stats.items()}
                        for metric, stats in result[name].items()}
    return result


def main():
    OUT.mkdir(exist_ok=True)
    masks, classes, grid, sizes = M.load_circuits()
    means = mean_overlaps(masks, grid)
    k = np.linspace(0, 1, 201)
    random_chance = {"IoU": k / (2 - k), "IoMin": k}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, metric in zip(axes, ["IoU", "IoMin"]):
        ax.plot(k, random_chance[metric], color=M.MUTED, linestyle=":", linewidth=1.2, zorder=1)
        for name in M.CIRCUIT_NAMES:  # min and max lines first, under the mean lines
            for stat in ("min", "max"):
                ax.plot(grid, means[name][metric][stat], color=COLORS[name], linestyle=STYLES[name]["linestyle"],
                        linewidth=0.8, alpha=0.7, zorder=2)
        for name in M.CIRCUIT_NAMES:
            ax.plot(grid, means[name][metric]["mean"], color=COLORS[name], marker=MARKERS[name], markerfacecolor=COLORS[name],
                    markeredgecolor=M.SURFACE, markeredgewidth=0.6, zorder=3, **STYLES[name])
        ax.set_title(f"{metric}: mean and min–max over class pairs", fontsize=10, color=M.INK)
        ax.set_xlabel("top-k ratio (channels kept per layer of U)", fontsize=9, color=M.INK_2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, width=0.5, color=M.INK_2, labelsize=8)
    axes[0].set_ylabel("overlap between class circuits", fontsize=9, color=M.INK_2)

    fig.tight_layout()
    handles = [plt.Line2D([], [], color=COLORS[n], marker=MARKERS[n], markeredgecolor=M.SURFACE, **STYLES[n])
               for n in M.CIRCUIT_NAMES]
    handles.append(plt.Line2D([], [], color=M.MUTED, linestyle=":", linewidth=1.2))
    handles.append(plt.Line2D([], [], color=M.INK_2, linewidth=0.8, alpha=0.7))
    labels = list(M.CIRCUIT_NAMES.values()) + ["random circuits of size k (expected)",
                                                 "thin lines: min and max over class pairs"]
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False,
                        fontsize=8.5, handlelength=3.5)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"Mean pairwise overlap of the per-class circuits vs. top-k ratio\n{M.SUBTITLE}",
                 y=top + 0.05 / fig.get_figheight(), va="bottom", fontsize=10.5, color=M.INK)
    path = OUT / "mean_overlap_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    small_multiples(means, grid, k, random_chance)
    for name in M.CIRCUIT_NAMES:  # numbers for the log / a table
        for metric in ("IoU", "IoMin"):
            for stat in ("mean", "min", "max"):
                print(f"{M.CIRCUIT_NAMES[name]:>18}  {metric:<5} {stat:<4}",
                      " ".join(f"{v:.2f}" for v in means[name][metric][stat]))


def style_axes(ax):
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, color=M.INK_2, labelsize=8)


def small_multiples(means, grid, k, random_chance):
    """Rows IoU / IoMin, one column per circuit type: mean line + min-max band + random-chance line."""
    names = list(M.CIRCUIT_NAMES)
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names) + 0.6, 6.6), sharex=True, sharey=True)
    for row, metric in enumerate(["IoU", "IoMin"]):
        for col, name in enumerate(names):
            ax = axes[row, col]
            stats = means[name][metric]
            ax.plot(k, random_chance[metric], color=M.MUTED, linestyle=":", linewidth=1.2, zorder=1)
            ax.fill_between(grid, stats["min"], stats["max"], color=COLORS[name], alpha=0.18, linewidth=0, zorder=2)
            ax.plot(grid, stats["mean"], color=COLORS[name], marker=MARKERS[name], markersize=4.5,
                    markerfacecolor=COLORS[name], markeredgecolor=M.SURFACE, markeredgewidth=0.6, linewidth=1.6,
                    zorder=3)
            style_axes(ax)
            if row == 0:
                ax.set_title(M.CIRCUIT_NAMES[name], fontsize=10, color=M.INK)
            else:
                ax.set_xlabel("top-k ratio\n(channels kept per layer of U)", fontsize=9, color=M.INK_2)
            if col == 0:
                ax.set_ylabel(f"{metric}\nbetween class circuits", fontsize=9, color=M.INK_2)
    fig.tight_layout()
    handles = [plt.Line2D([], [], color=M.INK_2, marker="o", markersize=4.5, markeredgecolor=M.SURFACE,
                          linewidth=1.6),
               matplotlib.patches.Patch(facecolor=M.INK_2, alpha=0.18, linewidth=0),
               plt.Line2D([], [], color=M.MUTED, linestyle=":", linewidth=1.2)]
    labels = ["mean over class pairs", "min–max over class pairs", "random circuits of size k (expected)"]
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False,
                        fontsize=8.5, handlelength=3)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"Pairwise overlap of the per-class circuits vs. top-k ratio\n{M.SUBTITLE}",
                 y=top + 0.05 / fig.get_figheight(), va="bottom", fontsize=10.5, color=M.INK)
    path = OUT / "mean_overlap_curves_small_multiples.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
