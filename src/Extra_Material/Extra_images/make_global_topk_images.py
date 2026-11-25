"""Extra figures for the GLOBAL top-K run (TOPK_SELECTION_MODE = "global_raw"), from its s05 summary.csv.
Not part of the codebase pipeline; same look as make_extra_images.py, but only the CERTIFIED circuit (raw and
logit-standardized) and the grey full-model line.

  global_topk_certified_raw_and_standardized_curves_per_class_voc_val.png   cIoU per class (images with c)
  global_topk_certified_raw_and_standardized_curves.png                     mean over classes; one column each for
                                                                            voc_val and discovery_samples

The global run's s05 evaluated only "images_with_class_c", so the mean figure has the two rows of s06's group-1
"curves" for that image set (cIoU, mIoU); the images-without-c rows of raw_and_standardized_curves.png are absent.

x axis: the POOLED share of U's channels kept (subnet_size_group_pooled_ratio), the quantity K fixes in global mode
(set X_COLUMN to "subnet_size_layer_normalized_ratio_mean" for the per-layer mean used in make_extra_images.py).

Run:  python src/Extra_Material/Extra_images/make_global_topk_images.py
"""

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from make_extra_images import (CIRCUITS, INK, INK_2, MUTED, STANDARDIZED, VARIANTS, VOC, look)  # also sets rcParams

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / ("results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores/"
                  "s03_pred_actgrad_pdel0.60_n100-1000_sseed0_topk_global_raw_backbone/s04_backbone_tau0.95_a0.001/"
                  "s04b_logit_statistics_discovery_samples_minpx1000/s05_eval_CapAll/summary.csv")
OUT = Path(__file__).resolve().parent
PREFIX = "global_topk_certified_"
EVAL_DATASET = "voc_val"                            # per-class figure
MEAN_DATASETS = ["voc_val", "discovery_samples"]    # mean figure, one column each
CIRCUIT = "certified"
X_COLUMN = "subnet_size_group_pooled_ratio"
SIZE_LABEL = {"subnet_size_group_pooled_ratio": "Subnetwork size, pooled ratio\n(share of all channels of U kept)",
              "subnet_size_layer_normalized_ratio_mean": "Subnetwork size, layer-normalized ratio\n"
                                                         "(channels kept per layer of U, mean over layers)"}[X_COLUMN]
SUBTITLE = ("run Dbottleneck262 · GLOBAL top-K (global_raw over U) · 20 classes · U = backbone · τ = 0.95 · "
            "p_del = 0.6 · N = 1000 · target pred · actgrad\nstandardized = " + STANDARDIZED)
MEAN_ROWS = [  # s06 GROUP_CURVES[1], images-with-c rows only
    ("images_with_class_c", "cIoU", "cIoU (vs GT)\n[images with c]"),
    ("images_with_class_c", "mIoU_vs_ground_truth", "mIoU, all 21 classes (vs GT)\n[images with c]")]


def draw(ax, d, metric):
    """Certified raw and standardized, plus the grey full-model line (raw)."""
    for variant, (_, style, marker) in VARIANTS.items():
        sub = d[(d.circuit == CIRCUIT) & (d.logit_standardization_variant == variant)].sort_values(
            "topk_layer_normalized_ratio")
        ax.plot(sub[X_COLUMN], sub[metric], **look(CIRCUITS[CIRCUIT][1][variant], style, marker))
    full = d[(d.circuit == "full") & (d.logit_standardization_variant == "raw")][metric]
    if len(full):
        ax.axhline(full.iloc[0], color=MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)


def legend_and_title(fig, title):
    handles = [plt.Line2D([], [], **look(CIRCUITS[CIRCUIT][1][v], style, marker))
               for v, (_, style, marker) in VARIANTS.items()] + [plt.Line2D([], [], color=MUTED, linewidth=0.8)]
    labels = [f"{CIRCUITS[CIRCUIT][0]}, {name}" for name, _, _ in VARIANTS.values()] + ["full model"]
    legend = fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=3.5)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"{title}\n{SUBTITLE}", y=top + 0.08 / fig.get_figheight(), va="bottom", fontsize=10, color=INK)


def save(fig, filename):
    path = OUT / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def per_class(df):
    d = df[(df.eval_dataset == EVAL_DATASET) & (df.image_set == "images_with_class_c")]
    classes = sorted(d["class"].unique())
    cols = 5
    rows = math.ceil(len(classes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 0.6, 2.2 * rows + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for ax, c in zip(axes.flat, classes):
        draw(ax, d[d["class"] == c], "cIoU")
        ax.set_title(VOC[c], fontsize=9)
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    fig.supxlabel(SIZE_LABEL, fontsize=9, color=INK_2)
    fig.supylabel("cIoU (vs GT)", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend_and_title(fig, f"Global top-K: cIoU per class on {EVAL_DATASET}, images with c: "
                          f"certified raw vs. standardized")
    save(fig, f"{PREFIX}raw_and_standardized_curves_per_class_{EVAL_DATASET}.png")


def mean_over_classes(df):
    keys = ["eval_dataset", "image_set", "logit_standardization_variant", "circuit", "topk_layer_normalized_ratio"]
    metrics = sorted({m for _, m, _ in MEAN_ROWS} | {X_COLUMN})
    means = df.groupby(keys, dropna=False)[metrics].mean().reset_index()  # full model: K is NaN
    fig, axes = plt.subplots(len(MEAN_ROWS), len(MEAN_DATASETS), figsize=(3.4 * len(MEAN_DATASETS) + 1.0,
                             2.4 * len(MEAN_ROWS)), sharex=True, squeeze=False)
    for col, name in enumerate(MEAN_DATASETS):
        for row, (image_set, metric, label) in enumerate(MEAN_ROWS):
            ax = axes[row, col]
            draw(ax, means[(means.eval_dataset == name) & (means.image_set == image_set)], metric)
            if col == 0:
                ax.set_ylabel(label)
        axes[0, col].set_title(name)
        axes[-1, col].set_xlabel(SIZE_LABEL)
    fig.tight_layout()
    legend_and_title(fig, "Global top-K: accuracy against the ground truth vs. subnetwork size (mean over classes):\n"
                          "certified raw vs. standardized")
    save(fig, f"{PREFIX}raw_and_standardized_curves.png")


if __name__ == "__main__":
    summary = pd.read_csv(SUMMARY)
    per_class(summary)
    mean_over_classes(summary)
