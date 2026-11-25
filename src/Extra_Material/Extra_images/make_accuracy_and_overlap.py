"""Accuracy of the certified circuits together with their pairwise overlap, voc_val (not part of the pipeline).

The rows of s06's group-1 "curves" (MEAN_ROWS in make_extra_images.py, mean over classes), certified circuit only,
raw and logit-standardized (same colours, line styles and markers as make_extra_images.py), plus the full model.
Two columns: left with the IoU, right with the IoMin between the certified circuits of all class pairs, drawn in
faded grey on a second y-axis on the right (0–1): the mean as a line, the min–max over class pairs as a band
(as in make_overlap_curves.py). Each overlap point sits at the certified circuit's mean size for that top-k, so both
share the x-axis.
Note: in the "share of pixels predicted as c" row the two y-axes have different scales (left about 0–0.1, right
0–1), so crossings of the accuracy and overlap lines there mean nothing. In the other rows both axes run 0–1.

  accuracy_and_overlap_certified_voc_val.png

Run:  python src/Extra_Material/Extra_images/make_accuracy_and_overlap.py
"""

import sys
from pathlib import Path

import matplotlib.patches
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_overlap_curves as O  # noqa: E402  (overlap over class pairs)
import make_overlap_matrices as M  # noqa: E402
import make_extra_images as E  # noqa: E402  (imported last: its figure style is the one used here)

plt = E.plt
EVAL_DATASET = "voc_val"
CIRCUIT = "certified"
OVERLAP_METRICS = ["IoU", "IoMin"]  # one column each
OVERLAP_COLOR = E.INK_2
OVERLAP_LINE_ALPHA = 0.55
OVERLAP_BAND_ALPHA = 0.12


def overlap_axis(ax, sizes, stats, show_labels, metric):
    """Second y-axis on the right (0–1) with the mean overlap line and the min–max band, drawn under the accuracy."""
    right = ax.twinx()
    right.fill_between(sizes, stats["min"], stats["max"], color=OVERLAP_COLOR, alpha=OVERLAP_BAND_ALPHA, linewidth=0)
    right.plot(sizes, stats["mean"], color=OVERLAP_COLOR, alpha=OVERLAP_LINE_ALPHA, linewidth=1.2, marker="^",
               markersize=3.5, markerfacecolor="none", markeredgewidth=0.8)
    right.set_ylim(0, 1.02)
    right.grid(False)
    right.spines["right"].set_visible(True)
    right.spines["top"].set_visible(False)
    right.tick_params(axis="y", labelright=True, labelsize=8)
    if show_labels:
        right.set_ylabel(f"{metric} between class circuits\n(grey, right axis)", color=E.INK_2)
    ax.set_zorder(right.get_zorder() + 1)  # accuracy lines on top of the overlap
    ax.patch.set_visible(False)


def main():
    summary = pd.read_csv(E.SUMMARY)
    keys = ["eval_dataset", "image_set", "logit_standardization_variant", "circuit", "topk_layer_normalized_ratio"]
    metrics = sorted({m for _, m, _, _ in E.MEAN_ROWS} | {"subnet_size_layer_normalized_ratio_mean"})
    means = summary.groupby(keys, dropna=False)[metrics].mean().reset_index()  # full model: top-k is NaN
    means = means[(means.eval_dataset == EVAL_DATASET) & means.circuit.isin([CIRCUIT, "full"])]

    masks, classes, grid, sizes = M.load_circuits()
    overlaps = O.mean_overlaps(masks, grid)[CIRCUIT]  # {"IoU" | "IoMin": {"mean" | "min" | "max": [|grid|]}}

    circuit_label, colors = E.CIRCUITS[CIRCUIT]
    rows = len(E.MEAN_ROWS)
    fig, axes = plt.subplots(rows, len(OVERLAP_METRICS), figsize=(6.4 * len(OVERLAP_METRICS) + 0.8, 3.3 * rows),
                             sharex=True, squeeze=False)
    for col, overlap_metric in enumerate(OVERLAP_METRICS):
        for row, (image_set, metric, label, ylim) in enumerate(E.MEAN_ROWS):
            ax = axes[row, col]
            d = means[means.image_set == image_set]
            overlap_axis(ax, sizes[CIRCUIT], overlaps[overlap_metric], show_labels=True,
                         metric=overlap_metric)
            for variant, (_, style, marker) in E.VARIANTS.items():
                sub = d[(d.circuit == CIRCUIT) & (d.logit_standardization_variant == variant)].sort_values(
                    "topk_layer_normalized_ratio")
                ax.plot(sub.subnet_size_layer_normalized_ratio_mean, sub[metric], zorder=3,
                        **E.look(colors[variant], style, marker))
            full = d[(d.circuit == "full") & (d.logit_standardization_variant == "raw")][metric]
            if len(full):
                ax.axhline(full.iloc[0], color=E.MUTED, linewidth=0.8, zorder=2)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(*ylim) if ylim else ax.set_ylim(bottom=0)
            # Every panel gets both axis labels (and x tick labels, which sharex would hide).
            ax.set_ylabel(f"{label}\n[{E.IMAGE_SET_SHORT[image_set]}]")
            ax.set_xlabel(E.SIZE_LABEL, fontsize=8)
            ax.tick_params(axis="x", labelbottom=True)
        axes[0, col].set_title(f"accuracy and overlap ({overlap_metric})")
    fig.tight_layout()

    handles, labels = [], []
    for variant, (vname, style, marker) in E.VARIANTS.items():
        handles.append(plt.Line2D([], [], **E.look(colors[variant], style, marker)))
        labels.append(f"{circuit_label}, {vname} (left axis)")
    handles.append(plt.Line2D([], [], color=E.MUTED, linewidth=0.8))
    labels.append("full model (left axis)")
    handles.append(plt.Line2D([], [], color=OVERLAP_COLOR, alpha=OVERLAP_LINE_ALPHA, linewidth=1.2, marker="^",
                              markersize=3.5, markerfacecolor="none", markeredgewidth=0.8))
    labels.append("overlap between the certified circuits, mean over class pairs (right axis)")
    handles.append(matplotlib.patches.Patch(facecolor=OVERLAP_COLOR, alpha=OVERLAP_BAND_ALPHA * 1.5, linewidth=0))
    labels.append("overlap, min–max over class pairs (right axis)")
    legend = fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=3.5)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"Certified circuits on {EVAL_DATASET}: accuracy (mean over classes) and overlap between classes "
                 f"vs. subnetwork size\n{E.SUBTITLE}", y=top + 0.08 / fig.get_figheight(), va="bottom",
                 fontsize=10, color=E.INK)
    path = E.OUT / f"accuracy_and_overlap_{CIRCUIT}_{EVAL_DATASET}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
