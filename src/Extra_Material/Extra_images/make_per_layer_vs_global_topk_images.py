"""Extra figures: CERTIFIED circuit, per-layer top-K (TOPK_SELECTION_MODE = "per_layer", K = layer-normalized ratio)
vs. global top-K ("global_raw" over U), each raw and logit-standardized. Not part of the codebase pipeline; same look
as make_extra_images.py.

  per_layer_vs_global_topk_certified_curves_per_class_voc_val.png   cIoU per class on voc_val (images with c)
  per_layer_vs_global_topk_certified_curves.png                     mean over classes; one column each for voc_val
                                                                    and discovery_samples

Hue = top-K mode (per-layer blue as in make_extra_images.py, global purple); raw = darker shade, solid line, squares;
standardized = brighter shade, dashed line, filled circles. The full model is the same in both runs (grey line).

The global run's s05 evaluated only "images_with_class_c", so the mean figure has the two rows of s06's group-1
"curves" for that image set (cIoU, mIoU).

x axis (X_COLUMN), the same measure for both runs so the curves are comparable:
  "pooled"                                   share of all 3,840 channels of U kept, rebuilt from the per-layer shares:
                                             Σ share_l · C_l / Σ C_l (so a summary.csv without
                                             subnet_size_group_pooled_ratio works too).
  "subnet_size_layer_normalized_ratio_mean"  mean over layers of the per-layer share (the x axis of
                                             make_extra_images.py)

Run:  python src/Extra_Material/Extra_images/make_per_layer_vs_global_topk_images.py
"""

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from make_extra_images import INK, INK_2, MUTED, STANDARDIZED, VARIANTS, VOC, look  # also sets rcParams

ROOT = Path(__file__).resolve().parents[3]
RUN = "results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores/"
TAIL = "/s04_backbone_tau0.95_a0.001/s04b_logit_statistics_discovery_samples_minpx1000/s05_eval_CapAll/summary.csv"
# Per top-K mode: (summary.csv, legend label, {variant: colour}).
MODES = {
    "per_layer": (ROOT / (RUN + "s03_pred_actgrad_pdel0.60_n100-1000_sseed0" + TAIL), "per-layer top-K",
                  {"raw": "#174a8c", STANDARDIZED: "#6aa8f0"}),
    "global": (ROOT / (RUN + "s03_pred_actgrad_pdel0.60_n100-1000_sseed0_topk_global_raw_backbone" + TAIL),
               "global top-K", {"raw": "#5b2a86", STANDARDIZED: "#c08ae6"}),
}
LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}  # common/model.py HOOK_POINTS
OUT = Path(__file__).resolve().parent
PREFIX = "per_layer_vs_global_topk_certified_"
EVAL_DATASET = "voc_val"                            # per-class figure
MEAN_DATASETS = ["voc_val", "discovery_samples"]    # mean figure, one column each
CIRCUIT = "certified"
X_COLUMN = "pooled"
SIZE_LABEL = {"pooled": f"Subnetwork size, pooled ratio\n(share of all {sum(LAYER_CHANNELS.values()):,} channels "
                        f"of U kept)",
              "subnet_size_layer_normalized_ratio_mean": "Subnetwork size, layer-normalized ratio\n"
                                                         "(channels kept per layer of U, mean over layers)"}[X_COLUMN]
SUBTITLE = ("run Dbottleneck262 · certified circuit · 20 classes · U = backbone · τ = 0.95 · p_del = 0.6 · N = 1000 · "
            "target pred · actgrad\nstandardized = " + STANDARDIZED)
MEAN_ROWS = [  # s06 GROUP_CURVES[1], images-with-c rows only
    ("images_with_class_c", "cIoU", "cIoU (vs GT)\n[images with c]"),
    ("images_with_class_c", "mIoU_vs_ground_truth", "mIoU, all 21 classes (vs GT)\n[images with c]")]


def load():
    """Both summaries in one table with a "mode" column and the pooled share rebuilt from the per-layer shares."""
    frames = []
    total = sum(LAYER_CHANNELS.values())
    for mode, (path, _, _) in MODES.items():
        d = pd.read_csv(path)
        d = d[d.image_set == "images_with_class_c"].copy()
        d["mode"] = mode
        d["pooled"] = sum(d[f"subnet_size_layer_normalized_ratio_{l}"] * c for l, c in LAYER_CHANNELS.items()) / total
        if "subnet_size_group_pooled_ratio" in d:  # sanity check against s05's own pooled share
            gap = (d.pooled - d.subnet_size_group_pooled_ratio).abs().max()
            print(f"{mode}: rebuilt vs. stored pooled share, max |difference| = {gap:.2e}")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def draw(ax, d, metric):
    """Certified raw and standardized for both modes, plus the grey full-model line (raw, per-layer run)."""
    for mode, (_, _, colors) in MODES.items():
        for variant, (_, style, marker) in VARIANTS.items():
            sub = d[(d["mode"] == mode) & (d.circuit == CIRCUIT) & (d.logit_standardization_variant == variant)
                    ].sort_values("topk_layer_normalized_ratio")
            ax.plot(sub[X_COLUMN], sub[metric], **look(colors[variant], style, marker))
    full = d[(d["mode"] == "per_layer") & (d.circuit == "full") & (d.logit_standardization_variant == "raw")][metric]
    if len(full):
        ax.axhline(full.iloc[0], color=MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)


def legend_and_title(fig, title):
    handles, labels = [], []
    for mode, (_, name, colors) in MODES.items():
        for variant, (vname, style, marker) in VARIANTS.items():
            handles.append(plt.Line2D([], [], **look(colors[variant], style, marker)))
            labels.append(f"{name}, {vname}")
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8))
    labels.append("full model")
    legend = fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), handlelength=3.5)
    fig.canvas.draw()
    top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.suptitle(f"{title}\n{SUBTITLE}", y=top + 0.08 / fig.get_figheight(), va="bottom", fontsize=10, color=INK)


def save(fig, filename):
    path = OUT / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def per_class(df):
    d = df[df.eval_dataset == EVAL_DATASET]
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
    legend_and_title(fig, f"Certified circuit, per-layer vs. global top-K: cIoU per class on {EVAL_DATASET}, "
                          f"images with c (raw and standardized)")
    save(fig, f"{PREFIX}curves_per_class_{EVAL_DATASET}.png")


def mean_over_classes(df):
    keys = ["mode", "eval_dataset", "image_set", "logit_standardization_variant", "circuit",
            "topk_layer_normalized_ratio"]
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
    legend_and_title(fig, "Certified circuit, per-layer vs. global top-K: accuracy against the ground truth vs. "
                          "subnetwork size\n(mean over classes, raw and standardized)")
    save(fig, f"{PREFIX}curves.png")


if __name__ == "__main__":
    summary = load()
    per_class(summary)
    mean_over_classes(summary)
