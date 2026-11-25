"""Extra figures from an existing s05 summary.csv (not part of the codebase pipeline).

Raw and logit-standardized circuits in the same figure, voc_val:
  raw_and_standardized_curves_per_class_voc_val.png   cIoU per class (images with c), all circuit types
  raw_and_standardized_curves.png                     mean over classes, the rows of s06's group-1 "curves";
                                                      one column each for voc_val and discovery_samples
  raw_and_standardized_faithfulness_curves.png        the same for s06's group-2 "curves" (faithfulness)

Hue = circuit type (as in s06); raw = darker shade, solid line, squares; standardized = brighter shade, dashed
line, filled circles.
The matched variant leaves the full model unchanged, so there is a single grey full-model line.

Run:  python src/Extra_Material/Extra_images/make_extra_images.py
"""

import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Input: the per-layer run (TOPK_SELECTION_MODE = "per_layer"), evaluated by s05 with MAX_EVAL_IMAGES = 200 and
# EVALUATE_IMAGES_WITHOUT_CLASS = True (the figures have rows for the images without c).
ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / ("results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores/"
                  "s03_pred_actgrad_pdel0.60_n100-1000_sseed0/s04_backbone_tau0.95_a0.001/"
                  "s04b_logit_statistics_discovery_samples_minpx1000/s05_eval_Cap200/summary.csv")
OUT = Path(__file__).resolve().parent
EVAL_DATASET = "voc_val"                            # per-class figure
MEAN_DATASETS = ["voc_val", "discovery_samples"]    # merged figure, one column each
STANDARDIZED = "matched_to_full_model_on_full_model_prediction_pixels"
SUBTITLE = ("run Dbottleneck262 · 20 classes · U = backbone · τ = 0.95 · p_del = 0.6 · N = 1000 · target pred · "
            "actgrad\nstandardized = " + STANDARDIZED)

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
       "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Per circuit type: (label, {variant: colour}). Raw = darker shade, standardized = brighter shade of the s06 hue.
CIRCUITS = {"certified": ("certified", {"raw": "#174a8c", STANDARDIZED: "#6aa8f0"}),
            "baseline_circuit": ("baseline circuit", {"raw": "#a8431a", STANDARDIZED: "#f7a074"}),
            "majority": ("majority vote", {"raw": "#0b6b48", STANDARDIZED: "#5fd6a8"})}
VARIANTS = {"raw": ("raw", "-", "s"), STANDARDIZED: ("standardized", "--", "o")}  # (label, line style, marker)
SIZE_LABEL = "Subnetwork size, layer-normalized ratio\n(channels kept per layer of U, mean over layers)"
MEAN_ROWS = [  # as s06 GROUP_CURVES[1]
    ("images_with_class_c", "cIoU", "cIoU (vs GT)", (0, 1.02)),
    ("images_with_class_c", "mIoU_vs_ground_truth", "mIoU, all 21 classes (vs GT)", (0, 1.02)),
    ("images_without_class_c", "mIoU_vs_ground_truth", "mIoU, all 21 classes (vs GT)", (0, 1.02)),
    ("images_without_class_c", "class_c_predicted_pixel_ratio", "share of pixels predicted as c\n"
     "(false positives of c)", None),
    ("images_without_class_c", "images_with_class_c_predicted_ratio", "share of images with\n"
     "≥ 1 pixel predicted as c", (0, 1.02))]
FAITHFULNESS_ROWS = [  # as s06 GROUP_CURVES[2]
    ("images_with_class_c", "faith_IoU", "faithfulness IoU (class c)", (0, 1.02)),
    ("images_with_class_c", "pearson_abs", "|Pearson r| of f_pred", (0, 1.02)),
    ("images_with_class_c", "pixel_agreement_with_full_model", "pixel agreement\nwith the full model", (0, 1.02)),
    ("images_with_class_c", "class_share_total_variation_to_full_model", "class-share total variation\n"
     "to the full model", (0, 1.02)),
    ("images_without_class_c", "pixel_agreement_with_full_model", "pixel agreement\nwith the full model", (0, 1.02)),
    ("images_without_class_c", "class_share_total_variation_to_full_model", "class-share total variation\n"
     "to the full model", (0, 1.02))]
IMAGE_SET_SHORT = {"images_with_class_c": "images with c", "images_without_class_c": "images without c"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})


def look(color, style, marker):
    """Line and solid marker (no hole) in one colour."""
    return dict(color=color, linestyle=style, marker=marker, markersize=4.5 if marker == "s" else 4,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5)


def draw(ax, d, metric):
    """One line per circuit type x variant, plus the grey full-model line (raw)."""
    for circuit, (_, colors) in CIRCUITS.items():
        for variant, (_, style, marker) in VARIANTS.items():
            sub = d[(d.circuit == circuit) & (d.logit_standardization_variant == variant)].sort_values(
                "topk_layer_normalized_ratio")
            ax.plot(sub.subnet_size_layer_normalized_ratio_mean, sub[metric], **look(colors[variant], style, marker))
    full = d[(d.circuit == "full") & (d.logit_standardization_variant == "raw")][metric]
    if len(full):
        ax.axhline(full.iloc[0], color=MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 1.02)


def legend_and_title(fig, title):
    handles, labels = [], []
    for circuit, (name, colors) in CIRCUITS.items():
        for variant, (vname, style, marker) in VARIANTS.items():
            handles.append(plt.Line2D([], [], **look(colors[variant], style, marker)))
            labels.append(f"{name}, {vname}")
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8))
    labels.append("full model")
    legend = fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=3.5)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"{title}\n{SUBTITLE}", y=top + 0.08 / fig.get_figheight(), va="bottom", fontsize=10, color=INK)


def per_class(df):
    d = df[(df.eval_dataset == EVAL_DATASET) & (df.image_set == "images_with_class_c")]
    classes = sorted(d["class"].unique())
    cols = 5
    rows = math.ceil(len(classes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 0.6, 2.2 * rows + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for ax, c in zip(axes.flat, classes):
        draw(ax, d[d["class"] == c], "cIoU")
        ax.set_ylim(0, 1.02)
        ax.set_title(VOC[c], fontsize=9)
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    fig.supxlabel(SIZE_LABEL, fontsize=9, color=INK_2)
    fig.supylabel("cIoU (vs GT)", fontsize=9, color=INK_2)
    fig.tight_layout()
    legend_and_title(fig, f"cIoU per class on {EVAL_DATASET}, images with c: raw vs. standardized")
    path = OUT / f"raw_and_standardized_curves_per_class_{EVAL_DATASET}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def mean_over_classes(df, rows, title, filename):
    """One column per eval dataset in MEAN_DATASETS, one row per entry of rows."""
    keys = ["eval_dataset", "image_set", "logit_standardization_variant", "circuit", "topk_layer_normalized_ratio"]
    metrics = sorted({m for _, m, _, _ in rows} | {"subnet_size_layer_normalized_ratio_mean"})
    means = df.groupby(keys, dropna=False)[metrics].mean().reset_index()  # full model: K is NaN
    fig, axes = plt.subplots(len(rows), len(MEAN_DATASETS), figsize=(3.4 * len(MEAN_DATASETS) + 1.0,
                             2.4 * len(rows)), sharex=True, squeeze=False)
    for col, name in enumerate(MEAN_DATASETS):
        for row, (image_set, metric, label, ylim) in enumerate(rows):
            ax = axes[row, col]
            draw(ax, means[(means.eval_dataset == name) & (means.image_set == image_set)], metric)
            ax.set_ylim(*ylim) if ylim else ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel(f"{label}\n[{IMAGE_SET_SHORT[image_set]}]")
        axes[0, col].set_title(name)
        axes[-1, col].set_xlabel(SIZE_LABEL)
    fig.tight_layout()
    legend_and_title(fig, f"{title} vs. subnetwork size (mean over classes):\nraw vs. standardized")
    path = OUT / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    summary = pd.read_csv(SUMMARY)
    per_class(summary)
    mean_over_classes(summary, MEAN_ROWS, "Accuracy against the ground truth", "raw_and_standardized_curves.png")
    mean_over_classes(summary, FAITHFULNESS_ROWS, "Faithfulness to the full model",
                      "raw_and_standardized_faithfulness_curves.png")
