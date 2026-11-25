"""Paper figure 3 (the _by_median_shift output): logit distributions per class d for global top-K
(TOPK_SELECTION_MODE = "global_raw" over U = backbone): the full model, the CERTIFIED circuit at K = 0.5 raw, and the
same circuit after logit standardization.

  fig3_global_topk_logit_distributions_K0.5.png                    extremes by median
  fig3_global_topk_logit_distributions_K0.5_by_median_shift.png    extremes by |median shift| (the paper's Fig. 3)

Same quantity as s06's box plots (plot_box_plots, Jung Fig. 2 style): the class-d logit on the pixels the RAW full
model predicts as d, voc_val images with class c, each class c's images with its own class-c circuit, pooled over
the 20 classes c. Box = quartiles, line = median, whiskers = 5th and 95th percentile, from s05's fixed-bin
histograms (bin width 0.25; values outside [-20, 20] sit at the edge).

Classes shown (x axis):
  - background, person, car, bicycle, dog (fixed);
  - the two most extreme remaining classes (among the classes not in the fixed list; the ranking is printed):
      "median":       highest and lowest median of the CERTIFIED RAW logit;
      "median_shift": largest and smallest |median shift| of the certified raw logit from the full model's
                      median (the version in the paper);
  - "mean of all classes": all 21 classes d (background included) as one box whose five statistics are the means
    of the per-class statistics (each class weighted equally, not pooled pixels).

Run:  python src/Extra_Material/Extra_images/paper_ready_figures/make_fig3_global_topk_logit_distributions_K0.5.py
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from common.datasets import NUM_CLASSES, VOC_CLASSES  # noqa: E402
from common.logit_standardization import quantiles_from_histogram  # noqa: E402

RUN = ROOT / ("results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores/"
              "s03_pred_actgrad_pdel0.60_n100-1000_sseed0_topk_global_raw_backbone/s04_backbone_tau0.95_a0.001/"
              "s04b_logit_statistics_discovery_samples_minpx1000")
COUNTS = RUN / "s05_eval_CapAll/confusion_matrices_and_logit_histograms/voc_val"
OUT = Path(__file__).resolve().parent
STANDARDIZED = "matched_to_full_model_on_full_model_prediction_pixels"
K = 0.5
FIXED_CLASSES = ["background", "person", "car", "bicycle", "dog"]
PROBABILITIES = [0.05, 0.25, 0.5, 0.75, 0.95]
# (legend label, circuit type, variant, colour)
BOXES = [("Full model", "full", "raw", "#898781"),
         (f"Certified, global top-K, K = {K:g}, raw", "certified", "raw", "#2a78d6"),
         (f"Certified, global top-K, K = {K:g}, standardized", "certified", STANDARDIZED, "#eb6834")]
AXIS_TITLE_SIZE = 8 * 0.8  # y-axis title
LEGEND_FONT_SIZE = 7 * 0.8
Y_TICK_LABEL_SIZE = 8 * 0.8  # y tick numbers
CLASS_LABEL_SIZE = 6.5  # x-axis class names
GROUP_TITLE_SIZE = 5.5  # "Median shift vs full model" title over the two-class group
FIGURE_HEIGHT_INCHES = 1.89 * 0.8

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
})


def model_index(z, t, k=None):
    types, ratios = z["model_circuit_types"], z["model_topk_layer_normalized_ratio"]
    return int(np.flatnonzero((types == t) if t == "full" else (types == t) & np.isclose(ratios, k))[0])


def pooled_histograms():
    """{(circuit type, variant): [21, bins]} summed over the 20 classes c, on the full model's d-pixels."""
    files = sorted(COUNTS.glob("class_*_images_with_class_c.npz"))
    pooled, edges = {}, None
    for f in files:
        z = np.load(f)
        edges = z["histogram_bin_edges"]
        variants = z["logit_standardization_variants"].tolist()
        p = z["distribution_pixel_sets"].tolist().index("full_model_prediction")
        for _, t, v, _ in BOXES:
            h = z["logit_histogram"][model_index(z, t, K), variants.index(v), p]
            pooled[(t, v)] = pooled.get((t, v), 0) + h
    print(f"{len(files)} classes c pooled from {COUNTS}")
    return pooled, edges


# ranking: (file suffix, label under the highest, label under the lowest, title over the two-class group)
# A label of None shows only the class name; a title of None draws no title.
RANKINGS = {"median": ("", "Highest\nmedian", "Lowest\nmedian", None),
            "median_shift": ("_by_median_shift", "Largest", "Smallest", "Median shift vs full model")}


def main():
    pooled, edges = pooled_histograms()
    for ranking in RANKINGS:
        make_figure(pooled, edges, ranking)


def make_figure(pooled, edges, ranking):
    suffix, high_label, low_label, group_title = RANKINGS[ranking]
    print(f"\n==== extremes by {ranking} ====")
    # stats[(t, v)]: [21, 5] quantiles per class d
    stats = {key: np.array([quantiles_from_histogram(h[d], edges, PROBABILITIES) for d in range(NUM_CLASSES)])
             for key, h in pooled.items()}

    fixed = [VOC_CLASSES.index(name) for name in FIXED_CLASSES]
    rest = [d for d in range(NUM_CLASSES) if d not in fixed]
    medians = stats[("certified", "raw")][:, 2]
    shifts = medians - stats[("full", "raw")][:, 2]
    if ranking == "median":
        key = medians
        print("certified raw median, remaining classes (low -> high):")
        print("  " + ", ".join(f"{VOC_CLASSES[d]} {medians[d]:.2f}" for d in sorted(rest, key=lambda d: key[d])))
    else:
        key = np.abs(shifts)
        print("certified raw median - full model median, remaining classes (smallest -> largest |shift|):")
        print("  " + ", ".join(f"{VOC_CLASSES[d]} {shifts[d]:+.2f}" for d in sorted(rest, key=lambda d: key[d])))
    ranked = sorted(rest, key=lambda d: key[d])
    low, high = ranked[0], ranked[-1]

    groups = [(VOC_CLASSES[d], [d]) for d in fixed]
    groups += [(VOC_CLASSES[d] if label is None else f"{VOC_CLASSES[d]}\n({label})", [d])
               for d, label in ((high, high_label), (low, low_label))]
    groups += [("mean of\nall classes", list(range(NUM_CLASSES)))]

    fig, ax = plt.subplots(figsize=(5.6, FIGURE_HEIGHT_INCHES))
    width = 0.8 / len(BOXES)
    for b, (label, t, v, color) in enumerate(BOXES):
        boxes = []
        for g, (_, members) in enumerate(groups):
            q = stats[(t, v)][members].mean(axis=0)
            boxes.append({"whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4]})
        positions = np.arange(len(groups)) + (b - (len(BOXES) - 1) / 2) * width
        ax.bxp(boxes, positions=positions, widths=width * 0.85, showfliers=False, patch_artist=True,
               boxprops={"facecolor": color, "edgecolor": SURFACE, "linewidth": 0.5},
               medianprops={"color": SURFACE, "linewidth": 1.2},
               whiskerprops={"color": color, "linewidth": 0.8}, capprops={"color": color, "linewidth": 0.8})
        print(f"{label:38s} medians: " + "  ".join(f"{name.splitlines()[0]}:{box['med']:.2f}"
                                                   for (name, _), box in zip(groups, boxes)))
    # thin dividers: fixed classes | extremes | mean of the rest
    for x in (len(fixed) - 0.5, len(fixed) + 1.5):
        ax.axvline(x, color=GRID, linewidth=0.8, zorder=0)
    if group_title is not None:
        # title centred over the two-class group, sitting just above the y = 15 grid line
        ax.annotate(group_title, xy=(len(fixed) + 0.5, 15), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=GROUP_TITLE_SIZE, color=INK_2)
    ax.set_xticks(range(len(groups)), [name for name, _ in groups])
    ax.tick_params(axis="x", length=0, labelsize=CLASS_LABEL_SIZE)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Class-wise\nlogit distribution", fontsize=AXIS_TITLE_SIZE)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))  # a tick every 5
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    handles = [Rectangle((0, 0), 1, 1, facecolor=color) for *_, color in BOXES]
    ax.legend(handles, [label for label, *_ in BOXES], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=LEGEND_FONT_SIZE, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    for ext in ("png",):  # add "pdf" for a vector version
        path = OUT / f"fig3_global_topk_logit_distributions_K{K:g}{suffix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
