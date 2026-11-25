"""Paper figure 2: mean cIoU of the certified subnetworks against subnetwork size, in two panels (5.6 x 2.1 in):

  left  = PASCAL VOC only: global top-K (blue, filled markers) vs layer-normalized top-K (deep purple-pink, hollow
          markers), each raw (solid, squares) and standardized (dashed, circles).
  right = global top-K on all four eval datasets in their full bright colours (no muting), PASCAL VOC drawn in front.

  fig1_split_voc_global_vs_layer_normalized_and_all_datasets.png

The bottom axis title is shared by both panels (one centred title); the y title sits on the left panel only. The top K axis
(requested global top-K budget) is on the right panel only; the left panel uses the actual-size axis alone, since its
two modes both request the same share K but end at slightly different actual sizes after certification.

Data: the global run's s05 / s05c summary.csv files; the layer-normalized (per-layer) run's s05
summary.csv, whose pooled share of the 3,840 backbone channels is rebuilt from the per-layer shares weighted by channels
per layer (K below 0.1 is dropped). Certified circuit, images with class c, mean over classes. Bottom axis =
actual certified size; top axis = the global run's requested K, ticks at the actual size that K produced on voc_val.

Run:  python src/Extra_Material/Extra_images/paper_ready_figures/make_fig1_split_voc_global_vs_layer_normalized_and_all_datasets.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SCORES = ROOT / "results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores"
CERT = "s04_backbone_tau0.95_a0.001/s04b_logit_statistics_discovery_samples_minpx1000"
RUN = SCORES / "s03_pred_actgrad_pdel0.60_n100-1000_sseed0_topk_global_raw_backbone" / CERT
LAYER_NORMALIZED_RUN = SCORES / "s03_pred_actgrad_pdel0.60_n100-1000_sseed0" / CERT
LAYER_NORMALIZED_SUMMARY = LAYER_NORMALIZED_RUN / "s05_eval_CapAll/summary.csv"  # PASCAL VOC (voc_val) only
LAYER_NORMALIZED_COLOR = "#b0247a"  # deep purple-pink
LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}  # common/model.py HOOK_POINTS
OUT = Path(__file__).resolve().parent
OUT_STEM = "fig1_split_voc_global_vs_layer_normalized_and_all_datasets"
STANDARDIZED = "matched_to_full_model_on_full_model_prediction_pixels"
CIRCUIT = "certified"
X_COLUMN = "subnet_size_group_pooled_ratio"
X_LABEL = "Mean Subnetwork Size (share of backbone channels kept)"  # mean over classes
TOPK_COLUMN = "topk_layer_normalized_ratio"
TOPK_LABEL = "Global Top-K Budget (share of channels requested)"  # used only if SHOW_TOPK_AXIS
# Shared top title replacing the top K axis: each marker step to the left is one K step down.
SHARED_TOP_TITLE = "Top-K budget K = 1.0 at the rightmost data point, decreasing by 0.1 per data point to the left"
SHOW_TOPK_AXIS = False  # True = a top K tick row on the right panel instead of the shared title
TOPK_TICK_DATASET = "voc_val"  # the dataset whose mean certified size per K places the top-axis ticks
Y_LABEL = "Mean class IoU (cIoU)"
# eval_dataset: (summary.csv, legend name, colour)
DATASETS = {
    "voc_val": (RUN / "s05_eval_CapAll/summary.csv", "PASCAL VOC", "#2a78d6"),
    "coco_val2017_voc": (RUN / "s05c_label_remapped_eval_CapAll/coco_val2017_voc/summary.csv", "COCO", "#eb6834"),
    "ade20k_val_voc": (RUN / "s05c_label_remapped_eval_CapAll/ade20k_val_voc/summary.csv", "ADE20K", "#1baf7a"),
    "cityscapes_val_voc": (RUN / "s05c_label_remapped_eval_CapAll/cityscapes_val_voc/summary.csv", "Cityscapes",
                           "#eda100"),
}
FOCUS_DATASET = "voc_val"  # drawn in front on the right panel
VARIANTS = {"raw": ("raw", "-", "s"), STANDARDIZED: ("standardized", (0, (1.8, 1.4)), "o")}  # (label, line style, marker); short dashes so several fit between two markers

# Text sizes (points).
AXIS_TITLE_SIZE = 8 * 0.8          # bottom, left and top axis titles
TICK_LABEL_SIZE = 8 * 0.9
TOP_TICK_LABEL_SIZE = 7 * 0.9
LEGEND_FONT_SIZE = 5.6 * 0.9

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": AXIS_TITLE_SIZE, "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": TICK_LABEL_SIZE, "ytick.labelsize": TICK_LABEL_SIZE, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})


def look(color, style, marker):
    return dict(color=color, linestyle=style, marker=marker, markersize=4.5 if marker == "s" else 4,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.5)


def look_hollow(color, style, marker):
    return {**look(color, style, marker), "markerfacecolor": SURFACE, "markeredgewidth": 1.0}


def class_mean(path, eval_dataset, rebuild_pooled=False):
    d = pd.read_csv(path)
    d = d[(d.eval_dataset == eval_dataset) & (d.image_set == "images_with_class_c") & (d.circuit == CIRCUIT)].copy()
    if rebuild_pooled:  # per-layer run: pooled share rebuilt from the per-layer shares; K below 0.1 dropped
        d[X_COLUMN] = sum(d[f"subnet_size_layer_normalized_ratio_{l}"] * c for l, c in LAYER_CHANNELS.items()) / sum(
            LAYER_CHANNELS.values())
        d = d[d[TOPK_COLUMN] >= 0.1 - 1e-9]
    n_classes = d["class"].nunique()
    means = (d.groupby(["logit_standardization_variant", TOPK_COLUMN])[[X_COLUMN, "cIoU"]]
             .mean().reset_index())
    return means, n_classes


def plot_variants(ax, means, name, look_function, color, zorder):
    for variant, (vname, style, marker) in VARIANTS.items():
        sub = means[means.logit_standardization_variant == variant].sort_values(TOPK_COLUMN)
        ax.plot(sub[X_COLUMN], sub["cIoU"], zorder=zorder, **look_function(color, style, marker))
        print(f"{name:24s} {vname:12s} " + "  ".join(f"{x:.2f}:{y:.3f}" for x, y in zip(sub[X_COLUMN], sub["cIoU"])))


def style_axes(ax, show_y_title):
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if show_y_title:
        ax.set_ylabel(Y_LABEL)


def add_topk_axis(ax, topk_ticks):
    """Top axis = the global run's requested K, ticks at the actual size each K produced, faint guides down."""
    for x in topk_ticks[X_COLUMN]:
        ax.axvline(x, color=GRID, linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)
    top = ax.secondary_xaxis("top")
    top.set_xticks(topk_ticks[X_COLUMN].to_list(), [f"{k:.1f}" for k in topk_ticks[TOPK_COLUMN]])
    top.tick_params(axis="x", labelsize=TOP_TICK_LABEL_SIZE, length=3)
    top.spines["top"].set_visible(False)
    top.set_xlabel(TOPK_LABEL, fontsize=AXIS_TITLE_SIZE, labelpad=4)


def variant_legend(ax):
    handles = [plt.Line2D([], [], **look(MUTED, style, marker)) for _, style, marker in VARIANTS.values()]
    legend = ax.legend(handles, [vname for vname, _, _ in VARIANTS.values()], loc="upper left",
                       bbox_to_anchor=(0.0, 1.0), handlelength=4.0)
    ax.add_artist(legend)


def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(5.6, 1.75), sharey=True)

    # Right panel: global top-K, all datasets, bright colours, PASCAL VOC in front.
    dataset_handles, dataset_labels = [], []
    voc_global_means = topk_ticks = None
    for eval_dataset, (path, name, color) in DATASETS.items():
        means, n_classes = class_mean(path, eval_dataset)
        if eval_dataset == TOPK_TICK_DATASET:
            topk_ticks = (means[means.logit_standardization_variant == "raw"]
                          .sort_values(TOPK_COLUMN)[[TOPK_COLUMN, X_COLUMN]])
        if eval_dataset == FOCUS_DATASET:
            voc_global_means = means
        plot_variants(ax_right, means, f"{name} ({n_classes})", look, color,
                      zorder=4.0 if eval_dataset == FOCUS_DATASET else 2.0)
        dataset_handles.append(plt.Line2D([], [], color=color, linewidth=2.5))
        dataset_labels.append(f"{name} ({n_classes})")
    print("top-axis ticks (K -> mean actual size on " + TOPK_TICK_DATASET + "): "
          + "  ".join(f"{k:.1f}->{x:.3f}" for k, x in zip(topk_ticks[TOPK_COLUMN], topk_ticks[X_COLUMN])))

    # Left panel: PASCAL VOC, global (blue, filled, in front) vs layer-normalized (purple-pink, hollow).
    voc_color = DATASETS[FOCUS_DATASET][2]
    layer_normalized_means, _ = class_mean(LAYER_NORMALIZED_SUMMARY, "voc_val", rebuild_pooled=True)
    plot_variants(ax_left, layer_normalized_means, "VOC layer-normalized", look_hollow, LAYER_NORMALIZED_COLOR, 3.0)
    plot_variants(ax_left, voc_global_means, "VOC global (left panel)", look, voc_color, 4.0)

    style_axes(ax_left, show_y_title=True)
    style_axes(ax_right, show_y_title=False)
    # Top K axis on the right panel only: the left panel compares two budget modes, which both request the same
    # share K of backbone channels but land at slightly different actual sizes, so it uses the bottom axis alone.
    if SHOW_TOPK_AXIS:
        add_topk_axis(ax_right, topk_ticks)
    ax_right.tick_params(axis="y", labelleft=False)

    # Legends: line style (upper left) on both; left: top-K mode by colour (lower right); right: dataset by colour.
    variant_legend(ax_left)
    variant_legend(ax_right)
    ax_left.legend([plt.Line2D([], [], **look(voc_color, "-", "s")),
                    plt.Line2D([], [], **look_hollow(LAYER_NORMALIZED_COLOR, "-", "o"))],
                   ["global top-K", "per-layer top-K"], loc="lower right", bbox_to_anchor=(1.0, 0.0),
                   handlelength=2.8)
    # Dataset legend: one column under the line-style legend, in the empty upper-left area (small sizes, low cIoU
    # only happen below it).
    ax_right.legend(dataset_handles, dataset_labels, loc="upper left", bbox_to_anchor=(0.0, 0.76), ncol=1,
                    handlelength=1.2, handletextpad=0.5)

    # Shared bottom axis title, centred over both panels.
    # Placed just outside the panels' tick labels (positions measured after layout, in figure coordinates).
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 0.94))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    boxes = [to_figure.transform_bbox(ax.get_tightbbox(renderer)) for ax in (ax_left, ax_right)]
    centre = (ax_left.get_position().x0 + ax_right.get_position().x1) / 2
    fig.text(centre, min(b.y0 for b in boxes) - 0.01, X_LABEL, ha="center", va="top", fontsize=AXIS_TITLE_SIZE,
             color=INK_2)
    if not SHOW_TOPK_AXIS:
        fig.text(centre, max(b.y1 for b in boxes) + 0.01, SHARED_TOP_TITLE, ha="center", va="bottom",
                 fontsize=AXIS_TITLE_SIZE, color=INK_2)
    for suffix in ("png",):  # add "pdf" for a vector version
        path = OUT / f"{OUT_STEM}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
