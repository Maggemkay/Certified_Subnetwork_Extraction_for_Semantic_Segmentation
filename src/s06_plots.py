"""s06 · Plots and tables of the evaluation (from s05's summary.csv and counts, s04's decisions and s05b's merged
circuits).

Output in <plot_dir> (IMAGES_DIR + the s05 folder's path, s05_eval_Cap<MAX_EVAL_IMAGES> included), one folder per
evaluation group, format PLOT_FORMAT.
Means are over CLASSES (each class weighted equally). The x-axis of the curves is the subnetwork size
(subnet_size_layer_normalized_ratio_mean), so at the same top-K layer-normalized ratio K a certified circuit sits left
of the baseline circuit, as in Anani's cACC–K plots. K is named by TOPK_SELECTION_MODE (utils.topk_ratio_label): the
top-K layer-normalized ratio (per_layer) or the top-K global ratio of LAYERS, the share of the whole group kept
(global); every figure's subtitle names the mode. Every figure outside a logit_standardization/ folder shows the RAW
circuits; with LOGIT_STANDARDIZATION_VARIANTS on, a logit_standardization/ folder inside each group holds the same
comparison with one line (or one figure) per variant. "/" in an eval dataset name becomes "_" in folder names.
  curves_mean.csv               the numbers behind every curve: means per eval dataset, image set, variant, circuit, K
  1_accuracy_vs_ground_truth/
    curves                      rows: cIoU and mIoU on the images with c; mIoU and the false positives of c on the
                                images without c. One column per eval dataset
    peak_ciou.csv               per eval dataset, variant and class: the full model's cIoU and each circuit's peak cIoU
                                over the grid, with its K and subnetwork size (also printed)
    logit_standardization/curves_<circuit type>          as curves, one line per variant
    <eval dataset>/curves_per_class                     cIoU per class (images with c)
                   false_positives_per_class            share of pixels predicted as c per class (images without c)
                   misclassification_of_class_c         where class c's GT pixels go (images with c), selected K
                   false_positive_sources               which GT classes the false c-pixels come from (images
                                                        without c), selected K
                   logit_standardization/               curves_per_class_<circuit type>; the two heatmaps per variant
  2_faithfulness_to_full_model/
    curves                      rows: faith, |r|, pixel agreement, class-share total variation (images with c); pixel
                                agreement and total variation (images without c)
    logit_standardization/curves_<circuit type>
    <eval dataset>/class_shares_<image set>              share of pixels per predicted class, full vs circuit per K
                   class_shares_stacked_bars_<image set> the same as stacked bars: background, own class c, the 5
                                                        largest other classes, all other classes
                   logit_standardization/class_shares[_stacked_bars]_<image set>_<variant>
  3_logit_distributions/<eval dataset>/
    shift_and_std_ratio_<image set>                     Δμ and σ ratio of every class's logit vs the full model
    wasserstein_1_<image set>                           Wasserstein-1 distance (needs STORE_LOGIT_HISTOGRAMS)
    box_plots_<image set>_<circuit type>                Jung Fig. 2-style box plots per class, selected K
    logit_standardization/                              the same per variant
  4_size_and_structure/
    layer_breakdown             per layer of U and K: share certified in / abstaining / certified out
    layer_breakdown_per_class   per class: share of each layer's channels in the baseline circuit and the certified
                                circuit against K (which layers lose channels; flat at K for the baseline per_layer)
    size_measures               per-layer channel shares, the pooled share of U, whole-model and reachable weight
                                shares against K
    merged_circuits, all_classes_union                  s05b's merged circuits (only when s05b has run)
"""

import math
import sys
import textwrap
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import config as C  # noqa: E402
from common import circuit_size, logit_standardization, model as M, utils  # noqa: E402
from common.datasets import NUM_CLASSES, VOC_CLASSES  # noqa: E402

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "IMAGES_DIR", "CLASSES",
    "DISCOVERY_DATASET", "EVAL_DATASETS", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "MAX_EVAL_IMAGES",
    "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA",
    "LOGIT_STANDARDIZATION_VARIANTS", "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS",
    "CIRCUIT_TYPES", "PLOT_FORMAT", "PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO", "SMOKE_TEST",
]

# Chart chrome and a fixed look per circuit type (color + line style), the same in every figure.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "certified": ("certified", "#2a78d6", "-"),
    "baseline_circuit": ("baseline circuit", "#eb6834", "--"),
    "majority": ("majority vote", "#1baf7a", ":"),
}
# The variant figures (one circuit type each): a fixed look per logit-standardization variant. raw = ink; the
# variants are green, violet and red, none of them a circuit-type colour; line style and marker are the secondary
# encoding.
VARIANT_STYLE = {
    "raw": ("raw (no logit standardization)", INK, "-", "o"),
    "matched_to_full_model_on_full_model_prediction_pixels": (None, "#008300", "--", "s"),
    "matched_to_full_model_on_all_pixels": (None, "#6250d6", "-.", "^"),
    "jung_standardized_logits_on_own_prediction_pixels": (None, "#e34948", ":", "D"),
}
BREAKDOWN = [  # stacked from the bottom
    ("certified in", "#2a78d6"),
    ("abstain, in baseline circuit", "#eb6834"),
    ("abstain, not in baseline circuit", "#1baf7a"),
    ("certified out", AXIS),
]
# Heatmaps: one-hue sequential blue (magnitude, near zero recedes to the surface) and the blue <-> red diverging pair
# with a neutral grey midpoint (sign). nan cells are hatched.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b5b4", "#e34948", "#8f2223"])
# Class-share stacked bars: background and "all other classes" in greys (never adjacent: background at the bottom,
# the rest at the top), the own class c in blue, the largest other classes in five further categorical colours.
CLASS_SHARE_BAR_TOP_CLASSES = 5
CLASS_SHARE_BAR_BACKGROUND, CLASS_SHARE_BAR_REST = AXIS, GRID
CLASS_SHARE_BAR_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SIZE_LABEL = "Subnetwork size, layer-normalized ratio\n(channels kept per layer of U, mean over layers)"
# layer_breakdown_per_class: a fixed colour per layer (the colour follows the layer). layer1-layer4 are ordered by depth:
# an ordinal blue ramp, shallow = light; aspp and head (only with LAYERS "backbone+head" or "head") are violet and
# red, which no circuit type uses.
LAYER_COLORS = {"layer1": "#86b6ef", "layer2": "#3987e5", "layer3": "#1c5cab", "layer4": "#0d366b",
                "aspp": "#6250d6", "head": "#e34948"}
IMAGE_SETS = ["images_with_class_c", "images_without_class_c"]
IMAGE_SET_SHORT = {"images_with_class_c": "images with c", "images_without_class_c": "images without c"}
GROUP_DIRS = {1: "1_accuracy_vs_ground_truth", 2: "2_faithfulness_to_full_model", 3: "3_logit_distributions",
              4: "4_size_and_structure"}
# Curve rows per group: (image set, summary.csv column, y-axis label, fixed y-range or None = from 0, automatic)
GROUP_CURVES = {
    1: [("images_with_class_c", "cIoU", "cIoU (vs GT)", (0, 1.02)),
        ("images_with_class_c", "mIoU_vs_ground_truth", f"mIoU, all {NUM_CLASSES} classes (vs GT)", (0, 1.02)),
        ("images_without_class_c", "mIoU_vs_ground_truth", f"mIoU, all {NUM_CLASSES} classes (vs GT)", (0, 1.02)),
        ("images_without_class_c", "class_c_predicted_pixel_ratio", "share of pixels predicted as c\n"
         "(false positives of c)", None),
        ("images_without_class_c", "images_with_class_c_predicted_ratio", "share of images with\n"
         "≥ 1 pixel predicted as c", (0, 1.02))],
    2: [("images_with_class_c", "faith_IoU", "faithfulness IoU (class c)", (0, 1.02)),
        ("images_with_class_c", "pearson_abs", "|Pearson r| of f_pred", (0, 1.02)),
        ("images_with_class_c", "pixel_agreement_with_full_model", "pixel agreement\nwith the full model", (0, 1.02)),
        ("images_with_class_c", "class_share_total_variation_to_full_model", "class-share total variation\n"
         "to the full model", (0, 1.02)),
        ("images_without_class_c", "pixel_agreement_with_full_model", "pixel agreement\nwith the full model", (0, 1.02)),
        ("images_without_class_c", "class_share_total_variation_to_full_model", "class-share total variation\n"
         "to the full model", (0, 1.02))],
}
GROUP_TITLES = {1: "Accuracy against the ground truth", 2: "Faithfulness to the full model"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.dash_capstyle": "round",
})


def circuit_types(types=None):
    """The pruned circuit types to draw, in STYLE order: of CIRCUIT_TYPES, or of `types` (s06c passes its own)."""
    return [t for t in STYLE if t in (C.CIRCUIT_TYPES if types is None else types)]


def variants(variant_list=None):
    """"raw" + LOGIT_STANDARDIZATION_VARIANTS, or "raw" + `variant_list` (s06c passes its own)."""
    return ["raw", *(C.LOGIT_STANDARDIZATION_VARIANTS if variant_list is None else variant_list)]


def folder_name(name):
    return name.replace("/", "_")


def variant_line(ax, x, y, v):
    label, color, style, marker = VARIANT_STYLE[v]
    ax.plot(x, y, color=color, linestyle=style, marker=marker, markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1, label=label or v)


def full_variant_lines(ax, d, metric, variant_list=None):
    """Grey full-model lines of a variant figure: raw, plus the variants that change the full model (Jung);
    the matched variants are the identity for the full model, so their line would repeat the raw one.
    variant_list: as variants()."""
    for v in variants(variant_list):
        if v in logit_standardization.MATCHED_VARIANTS:
            continue
        value = d[(d.circuit == "full") & (d.logit_standardization_variant == v)][metric]
        if len(value) and not np.isnan(value.iloc[0]):
            ax.axhline(value.iloc[0], color=MUTED, linewidth=0.8, linestyle=VARIANT_STYLE[v][2], zorder=1)


def full_variant_handles(variant_list=None):
    handles = [plt.Line2D([], [], color=MUTED, linewidth=0.8, linestyle=VARIANT_STYLE[v][2])
               for v in variants(variant_list) if v not in logit_standardization.MATCHED_VARIANTS]
    labels = ["full model" + ("" if v == "raw" else f" after {v}") for v in variants(variant_list)
              if v not in logit_standardization.MATCHED_VARIANTS]
    return handles, labels


def line(ax, x, y, t):
    label, color, style = STYLE[t]
    ax.plot(x, y, color=color, linestyle=style, marker="o", markersize=4,
            markeredgecolor=SURFACE, markeredgewidth=1, label=label)


def full_line(ax, value, label=True):
    if np.isnan(value):
        return
    ax.axhline(value, color=MUTED, linewidth=0.8, zorder=1)
    if label:
        ax.annotate("full model", xy=(1, value), xycoords=("axes fraction", "data"), xytext=(0, 2),
                    textcoords="offset points", ha="right", va="bottom", color=INK_2, fontsize=7)


def save(fig, path, what, base=None):
    """Saves and closes the figure and prints its path below `base` (None = plot_dir(); s06c passes its own)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(utils.plot_dir() if base is None else base)}\n      {what}")


def figure_path(group, *parts):
    """<plot_dir>/<group folder>/<parts...>.<PLOT_FORMAT>"""
    return utils.plot_dir().joinpath(GROUP_DIRS[group], *parts[:-1], f"{parts[-1]}.{C.PLOT_FORMAT}")


def run_subtitle(classes_text=None):
    """The run line under every figure title; classes_text replaces "<len(CLASSES)> classes" (s06c)."""
    selection = (f"top-K selection {C.TOPK_SELECTION_MODE}"
                 + (f" over {C.LAYERS}" if utils.topk_selection_is_global() else ""))
    return (f"{classes_text or f'{len(C.CLASSES)} classes'} · certifiable components U = {C.LAYERS} · {selection} · "
            f"τ = {C.AGREEMENT_THRESHOLD_TAU} · p_del = {C.P_DEL} · N = {C.N} · target {C.TARGET} · "
            f"scoring method {C.CHANNEL_SCORING_METHOD}")


def header(fig, title, handles=(), labels=(), ncol=None, note=None, classes_text=None):
    """Legend directly above the plot area (one row unless ncol is given; none without handles), and title + run
    subtitle above it. Call after tight_layout; bbox_inches="tight" keeps both in the saved file.
    note: an extra line under the title (s06c: the title note of a label-remapped dataset); classes_text: as
    run_subtitle."""
    top = 1.0
    if handles:
        legend = fig.legend(handles, labels, loc="lower center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, 1.0))
        fig.canvas.draw()  # a legend with several rows is taller: keep the title above its top
        top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
    text = f"{title}\n{note}\n{run_subtitle(classes_text)}" if note else f"{title}\n{run_subtitle(classes_text)}"
    fig.suptitle(text, y=max(1.0 + 0.35 / fig.get_figheight(), top + 0.08 / fig.get_figheight()),
                 va="bottom", fontsize=10, color=INK)


# ---- data ------------------------------------------------------------------------------
def load_summary():
    """s05's summary.csv, restricted to CLASSES and to "raw" + LOGIT_STANDARDIZATION_VARIANTS. A file without
    the variant column (an older s05 output) holds raw circuits only, one without the image_set column the images
    with c only."""
    path = utils.eval_dir() / "summary.csv"
    utils.require(path, "s05_evaluate.py")
    df = pd.read_csv(path)
    if "logit_standardization_variant" not in df:
        df.insert(df.columns.get_loc("circuit") + 1, "logit_standardization_variant", "raw")
    if "image_set" not in df:
        df.insert(df.columns.get_loc("eval_dataset") + 1, "image_set", "images_with_class_c")
    missing = [v for v in variants() if v not in set(df.logit_standardization_variant)]
    if missing:
        sys.exit(f"\n[missing input] {path} has no rows for the logit-standardization variant(s) {missing}\n"
                 f"-> re-run s05_evaluate.py with the current config.py.")
    return df[df["class"].isin(C.CLASSES) & df.logit_standardization_variant.isin(variants())]


def mean_curves(df):
    """Mean over classes per eval dataset, image set, variant, circuit and top-K layer-normalized ratio (full model:
    one row per eval dataset, image set and variant), of every numeric column of summary.csv."""
    keys = ["eval_dataset", "image_set", "logit_standardization_variant"]
    columns = [c for c in df.columns if c not in keys + ["class", "class_name", "circuit", "topk_layer_normalized_ratio"]
               and pd.api.types.is_numeric_dtype(df[c])]
    pruned = (df[df.circuit != "full"].groupby(keys + ["circuit", "topk_layer_normalized_ratio"])[columns]
              .mean().reset_index())
    full = df[df.circuit == "full"].groupby(keys)[columns].mean().reset_index()
    full["circuit"] = "full"
    return pd.concat([pruned, full], ignore_index=True)


def peak_table(df):
    rows = []
    with_c = df[df.image_set == "images_with_class_c"]
    for name in with_c.eval_dataset.unique():
        for v in variants():
            for c in C.CLASSES:
                d = with_c[(with_c.eval_dataset == name) & (with_c.logit_standardization_variant == v)
                           & (with_c["class"] == c)]
                rows.append(peak_row(d, name, v, c))
    return pd.DataFrame(rows)


def peak_row(d, name, v, c):
    """One row of peak_table from the summary rows d of one eval dataset, variant and class."""
    full = d[d.circuit == "full"].cIoU
    row = {"eval_dataset": name, "logit_standardization_variant": v, "class": c, "class_name": VOC_CLASSES[c],
           "full_cIoU": full.iloc[0] if len(full) else np.nan}
    for t in circuit_types():
        sub = d[d.circuit == t]
        best = sub.loc[sub.cIoU.idxmax()] if sub.cIoU.notna().any() else None
        row[f"{t}_peak_cIoU"] = np.nan if best is None else best.cIoU
        row[f"{t}_topk_layer_normalized_ratio"] = np.nan if best is None else best["topk_layer_normalized_ratio"]
        row[f"{t}_subnet_size_layer_normalized_ratio_mean"] = \
            np.nan if best is None else best["subnet_size_layer_normalized_ratio_mean"]
    return row


def layer_shares():
    """Mean over classes of the share of each layer's channels in the four BREAKDOWN categories."""
    layers, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    shares = []
    for c in C.CLASSES:
        path = utils.certify_dir() / f"class_{c:02d}.npz"
        utils.require(path, "s04_certify.py")
        certification_results = np.load(path)
        decision, baseline_circuit = certification_results["decision"], certification_results["baseline_circuit"]
        per_layer = []
        for s in certifiable_slices.values():
            decision_layer, baseline_circuit_layer = decision[:, s], baseline_circuit[:, s]
            per_layer.append(np.stack([(decision_layer == 1).mean(axis=1),
                                       ((decision_layer == -1) & baseline_circuit_layer).mean(axis=1),
                                       ((decision_layer == -1) & ~baseline_circuit_layer).mean(axis=1),
                                       (decision_layer == 0).mean(axis=1)], axis=-1))
        shares.append(np.stack(per_layer, axis=1))       # [|grid|, layers, 4]
    return layers, certification_results["topk_layer_normalized_ratio_grid"], np.mean(shares, axis=0)


def load_counts(name, image_set):
    """s05's counts of one eval dataset and image set: {class: npz arrays} for the classes in CLASSES whose file
    exists (none: e.g. s05 ran without EVALUATE_IMAGES_WITHOUT_CLASS)."""
    folder = utils.eval_dir() / "confusion_matrices_and_logit_histograms" / folder_name(name)
    out = {}
    for c in C.CLASSES:
        path = folder / f"class_{c:02d}_{image_set}.npz"
        if path.exists():
            with np.load(path) as stored:
                out[c] = {key: stored[key] for key in stored.files}
    return out


def model_index(counts_c, t, topk_layer_normalized_ratio=None):
    """Position of the full model (t = "full") or of circuit type t at K in an npz file's model axis."""
    types, ratios = counts_c["model_circuit_types"], counts_c["model_topk_layer_normalized_ratio"]
    hit = np.flatnonzero((types == t) if t == "full" else (types == t) & np.isclose(ratios, topk_layer_normalized_ratio))
    return int(hit[0])


def grid_of(counts_c, t):
    types, ratios = counts_c["model_circuit_types"], counts_c["model_topk_layer_normalized_ratio"]
    return ratios[types == t]


def selected_ratios(grid, selected=None, setting="PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO"):
    """The values of PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO (or of `selected`, named `setting` in the note; s06c)
    found in the grid (largest first); a value that is not in the grid is skipped with a note."""
    found = []
    for k in C.PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO if selected is None else selected:
        if np.isclose(grid, k).any():
            found.append(float(grid[np.isclose(grid, k)][0]))
        else:
            print(f"  NOTE: {setting} value {k} is not in the grid {[float(g) for g in grid]}; skipped.")
    return sorted(set(found), reverse=True)


# ---- curve figures (groups 1 and 2) ---------------------------------------------------------
def plot_group_curves(means, group, path, t=None):
    """The rows of GROUP_CURVES[group] against the subnetwork size, one column per eval dataset. t None: the raw
    circuits, one line per circuit type; else circuit type t only, one line per logit-standardization variant."""
    names = [n for n in C.EVAL_DATASETS if n in set(means.eval_dataset)]
    rows = [r for r in GROUP_CURVES[group] if r[0] in set(means.image_set)]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(3.4 * len(names) + 1.0, 2.3 * len(rows)),
                             sharex=True, squeeze=False)
    for col, name in enumerate(names):
        for row, (image_set, metric, label, ylim) in enumerate(rows):
            ax = axes[row, col]
            d = means[(means.eval_dataset == name) & (means.image_set == image_set)]
            if t is None:
                d = d[d.logit_standardization_variant == "raw"]
                for circuit in circuit_types():
                    sub = d[d.circuit == circuit].sort_values("topk_layer_normalized_ratio")
                    line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], circuit)
                full = d[d.circuit == "full"][metric]
                if len(full):
                    full_line(ax, full.iloc[0])
            else:
                for v in variants():
                    sub = d[(d.circuit == t) & (d.logit_standardization_variant == v)].sort_values(
                        "topk_layer_normalized_ratio")
                    variant_line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], v)
                full_variant_lines(ax, d, metric)
            ax.set_xlim(0, 1.02)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"{label}\n[{IMAGE_SET_SHORT[image_set]}]")
            if row == len(rows) - 1:
                ax.set_xlabel(SIZE_LABEL)
    fig.tight_layout()
    if t is None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        header(fig, f"{GROUP_TITLES[group]} vs. subnetwork size (mean over classes)", handles, labels)
        what = (f"{GROUP_TITLES[group].lower()}: the rows (metric [image set]) against the subnetwork size, one column "
                f"per eval dataset, raw circuits;\n      the grey line is the full model.")
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        full_handles, full_labels = full_variant_handles()
        header(fig, f"Logit standardization, circuit type {STYLE[t][0]}: {GROUP_TITLES[group].lower()} vs. subnetwork "
               f"size (mean over classes)", handles + full_handles, labels + full_labels, ncol=2)
        what = (f"as {GROUP_DIRS[group]}/curves, for the {t} circuits only: one line per logit-standardization variant;"
                f"\n      grey lines: the full model (raw; after the Jung variant if it is on).")
    save(fig, path, what)


def plot_per_class(df, name, image_set, metric, ylabel, title, path, what, t=None, ylim=(0, 1.02), classes=None,
                   types=None, variant_list=None, note=None, base=None):
    """One panel per class: `metric` on one eval dataset and image set against the subnetwork size. t None: the raw
    circuits, one line per circuit type; else circuit type t, one line per logit-standardization variant.
    s06c passes its own classes (None = CLASSES), types (circuit_types), variant_list (variants), title note (header)
    and base folder (save)."""
    classes = C.CLASSES if classes is None else classes
    d = df[(df.eval_dataset == name) & (df.image_set == image_set)]
    cols = min(5, len(classes))
    rows = math.ceil(len(classes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols + 0.6, 1.9 * rows + 0.6),
                             sharex=True, sharey=ylim is not None, squeeze=False)
    for ax, c in zip(axes.flat, classes):
        dc = d[d["class"] == c]
        if t is None:
            dc = dc[dc.logit_standardization_variant == "raw"]
            for circuit in circuit_types(types):
                sub = dc[dc.circuit == circuit].sort_values("topk_layer_normalized_ratio")
                line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], circuit)
            full = dc[dc.circuit == "full"][metric]
            if len(full):
                full_line(ax, full.iloc[0], label=False)
        else:
            for v in variants(variant_list):
                sub = dc[(dc.circuit == t) & (dc.logit_standardization_variant == v)].sort_values(
                    "topk_layer_normalized_ratio")
                variant_line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], v)
            full_variant_lines(ax, dc, metric, variant_list)
        ax.set_title(VOC_CLASSES[c], fontsize=9)
        ax.set_xlim(0, 1.02)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if t is None:
        handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.8))
        labels.append("full model")
    else:
        full_handles, full_labels = full_variant_handles(variant_list)
        handles, labels = handles + full_handles, labels + full_labels
    fig.supxlabel(SIZE_LABEL, fontsize=9, color=INK_2)
    fig.supylabel(ylabel, fontsize=9, color=INK_2)
    fig.tight_layout()
    header(fig, title, handles, labels, ncol=None if t is None else (2 if len(classes) > 1 else 1), note=note,
           classes_text=None if classes is C.CLASSES else f"{len(classes)} class{'' if len(classes) == 1 else 'es'}")
    save(fig, path, what, base)


# ---- heatmaps -------------------------------------------------------------------------------
def heatmap(ax, values, row_labels, column_labels, cmap, vmin, vmax):
    """Cells with a surface gap; nan cells hatched. Rows top to bottom."""
    values = np.asarray(values, dtype=float)
    image = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=SURFACE,
                          linewidth=0.6)
    for r, col in zip(*np.nonzero(np.isnan(values))):
        ax.add_patch(Rectangle((col, r), 1, 1, fill=False, hatch="////", edgecolor=AXIS, linewidth=0))
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, column_labels, rotation=90)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, row_labels)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def heatmap_grid(n_rows, n_cols, width, row_height):
    """A constrained-layout figure with one sub-figure per row of heatmap panels (its own title and colorbar):
    the figure, the sub-figures and a list of panel rows."""
    fig = plt.figure(figsize=(width, row_height * n_rows), layout="constrained")
    subfigures = np.atleast_1d(fig.subfigures(n_rows, 1, hspace=0.04))
    return fig, subfigures, [subfigure.subplots(1, n_cols, squeeze=False)[0] for subfigure in subfigures]


def heatmap_title(fig, title, note=None, classes_text=None):
    """Title + run subtitle of a heatmap figure; note and classes_text as in header (s06c)."""
    text = f"{title}\n{note}\n{run_subtitle(classes_text)}" if note else f"{title}\n{run_subtitle(classes_text)}"
    fig.suptitle(text, fontsize=10, color=INK)


def ratio_ticks(limit):
    """Ticks (log2 values) and labels (2, 4, … and 1/2, 1/4, …) of a log2-ratio colorbar, at most 7 of them."""
    top = max(1, math.floor(limit))
    step = max(1, math.ceil(2 * top / 6))
    ticks = np.arange(-(top // step) * step, top + 1, step)
    return ticks, [f"{2.0 ** t:g}" if t >= 0 else f"1/{2.0 ** -t:g}" for t in ticks]


def colorbar(fig, image, axes, label, ticks=None, tick_labels=None):
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label(label, color=INK_2)
    bar.outline.set_visible(False)
    if ticks is not None:
        bar.set_ticks(ticks, labels=tick_labels)
    return bar


def class_d_labels():
    return ["own class c"] + [f"{d} {VOC_CLASSES[d]}" for d in range(NUM_CLASSES)]


def plot_class_shares(counts, name, image_set, variant, path):
    """Share of pixels per predicted class (mean over classes c), full model vs circuit at every K: per circuit type
    the shares (square-root colour scale) and their difference to the full model (percentage points)."""
    types = circuit_types()
    any_c = next(iter(counts.values()))
    v_idx = any_c["logit_standardization_variants"].tolist().index(variant)
    grid = grid_of(any_c, types[0])
    shares = {t: [] for t in types}
    full_shares = []
    for c, z in counts.items():
        cm = z["confusion_matrix_vs_full_model"][:, v_idx].astype(float)        # [M, 21, 21]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = cm.sum(axis=1) / cm.sum(axis=(1, 2))[:, None]                   # [M, 21] column sums = predicted shares
        own = lambda values: np.concatenate([[values[c]], values])            # noqa: E731  "own class c" row first
        full_shares.append(own(s[model_index(z, "full")]))
        for t in types:
            shares[t].append(np.stack([own(s[model_index(z, t, k)]) for k in grid], axis=1))   # [22, K]
    full = np.nanmean(full_shares, axis=0)
    fig, subfigures, axes = heatmap_grid(2, len(types), 2.6 + 0.4 * (len(grid) + 1) * len(types), 7.0)
    columns = ["full"] + [f"{k:g}" for k in grid]
    means = {t: np.nanmean(shares[t], axis=0) for t in types}
    differences = {t: 100 * (means[t] - full[:, None]) for t in types}
    finite = np.concatenate([d[np.isfinite(d)] for d in differences.values()])
    limit = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
    for j, t in enumerate(types):
        top = heatmap(axes[0][j], np.sqrt(np.column_stack([full, means[t]])), class_d_labels() if j == 0 else [],
                      columns, SEQUENTIAL, 0, 1)
        axes[0][j].set_title(STYLE[t][0])
        bottom = heatmap(axes[1][j], differences[t], class_d_labels() if j == 0 else [], columns[1:], DIVERGING,
                         -limit, limit)
        axes[1][j].set_title(STYLE[t][0])
    subfigures[0].suptitle("share of all pixels predicted as each class: full model, then every K", fontsize=9)
    subfigures[1].suptitle("circuit − full model, in percentage points (red = more pixels than the full model)",
                           fontsize=9)
    for subfigure in subfigures:
        subfigure.supxlabel(f"{utils.topk_ratio_label()} K", fontsize=9, color=INK_2)
    colorbar(subfigures[0], top, axes[0], "share of pixels (square-root scale)", np.sqrt([0, 0.01, 0.05, 0.2, 0.5, 1]),
             ["0", "1 %", "5 %", "20 %", "50 %", "100 %"])
    colorbar(subfigures[1], bottom, axes[1], "circuit − full model, percentage points")
    variant_text = "" if variant == "raw" else f", after {variant}"
    heatmap_title(fig, f"Class shares on {name}, {IMAGE_SET_SHORT[image_set]}{variant_text}: share of pixels predicted "
                       f"as each class (mean over classes c)")
    save(fig, path, f"top: share of all pixels predicted as each class (rows; \"own class c\" = the circuit's own "
         f"class), full model and every K;\n      bottom: circuit minus full model in percentage points (red = more "
         f"pixels than the full model). Total variation = half the sum of a column's absolute differences.")


def plot_class_shares_stacked_bars(counts, name, image_set, variant, path):
    """The class shares of plot_class_shares as stacked bars (full model, then every K; one panel per circuit type).
    21 classes are too many hues to tell apart in a stack, so each bar is split into: background, the circuit's own
    class c, the CLASS_SHARE_BAR_TOP_CLASSES other classes with the largest share in any bar of the figure (the same
    classes and colours in every panel), and all remaining classes together. Per class c the parts add up to 1;
    the bars are the mean over classes c."""
    types = circuit_types()
    any_c = next(iter(counts.values()))
    v_idx = any_c["logit_standardization_variants"].tolist().index(variant)
    grid = grid_of(any_c, types[0])
    columns = [("full", None)] + [(t, k) for t in types for k in grid]
    parts = []                                                                  # per c: [bars, 2 + 21]
    for c, z in counts.items():
        cm = z["confusion_matrix_vs_full_model"][:, v_idx].astype(float)        # [M, 21, 21]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = cm.sum(axis=1) / cm.sum(axis=(1, 2))[:, None]                   # [M, 21] predicted shares
        s = s[[model_index(z, t, k) for t, k in columns]]
        others = s.copy()
        others[:, [0, c]] = 0                                                   # classes d other than background, c
        parts.append(np.column_stack([s[:, 0], s[:, c], others]))
    mean = np.nanmean(parts, axis=0)                                            # [bars, 2 + 21]
    others = mean[:, 2:]
    top = [int(d) for d in np.argsort(-np.nanmax(others, axis=0), kind="stable")[:CLASS_SHARE_BAR_TOP_CLASSES]
           if np.nanmax(others[:, d]) > 0]
    rest = np.nansum(others, axis=1) - np.nansum(others[:, top], axis=1)
    segments = ([("background", CLASS_SHARE_BAR_BACKGROUND, mean[:, 0]), ("own class c", STYLE["certified"][1],
                 mean[:, 1])]
                + [(VOC_CLASSES[d], CLASS_SHARE_BAR_COLORS[j], others[:, d]) for j, d in enumerate(top)]
                + [("all other classes", CLASS_SHARE_BAR_REST, rest)])
    fig, axes = plt.subplots(1, len(types), figsize=(1.2 + 0.35 * (len(grid) + 1) * len(types) + 0.6 * len(types), 3.4),
                             sharey=True, squeeze=False)
    for j, (ax, t) in enumerate(zip(axes[0], types)):
        rows = [0] + [columns.index((t, k)) for k in grid]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for label, color, values in segments:
            values = np.nan_to_num(values[rows])
            ax.bar(x, values, bottom=bottom, width=0.7, color=color, edgecolor=SURFACE, linewidth=0.8,
                   label=label if j == 0 else None)
            bottom += values
        ax.set_title(STYLE[t][0])
        ax.set_xticks(x, ["full"] + [f"{k:g}" for k in grid], rotation=90)
        ax.set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("share of all pixels")
    fig.supxlabel(f"{utils.topk_ratio_label()} K", fontsize=9, color=INK_2)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5))   # top of the stack first
    variant_text = "" if variant == "raw" else f", after {variant}"
    header(fig, f"Class shares on {name}, {IMAGE_SET_SHORT[image_set]}{variant_text} (mean over classes c)")
    save(fig, path, f"the class shares as stacked bars (bottom up: background, the circuit's own class c, the "
         f"{len(top)} other classes with the largest share in any bar, all other classes);\n      full model, then every "
         f"K, one panel per circuit type, mean over classes c. Every class: class_shares_{image_set}.")


def plot_confusion_heatmaps(counts, name, kind, variant, path, types=None, selected=None,
                            selected_setting="PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO", note=None, base=None):
    """Per class c (rows) and class d (columns), for the full model and every circuit type at the selected K:
        misclassification_of_class_c  images WITH c: the share of class c's GT pixels predicted as d (row c of the
                                      vs-GT confusion matrix; the diagonal cell = correctly found)
        false_positive_sources        images WITHOUT c: of the pixels predicted as c (all false positives), the share
                                      whose GT is d (column c of the vs-GT confusion matrix).
    s06c passes its own types (circuit_types), selected K and their setting name (selected_ratios), title note and
    base folder (save); the classes are those in `counts`."""
    types = circuit_types(types)
    any_c = next(iter(counts.values()))
    v_idx = any_c["logit_standardization_variants"].tolist().index(variant)
    ratios = selected_ratios(grid_of(any_c, types[0]), selected, selected_setting)
    if not ratios:
        return
    panels = [("full", None)] + [(t, None) for t in types]
    fig, subfigures, axes = heatmap_grid(len(ratios), len(panels), 4.4 * len(panels) + 2.2,
                                         0.24 * len(counts) + 2.4)
    classes = list(counts)
    for r, k in enumerate(ratios):
        subfigures[r].suptitle(f"{utils.topk_ratio_label()} K = {k:g} (the full-model panel is the same in every row)",
                               fontsize=9)
        for j, (t, _) in enumerate(panels):
            table = np.full((len(classes), NUM_CLASSES), np.nan)
            for i, c in enumerate(classes):
                cm = counts[c]["confusion_matrix_vs_ground_truth"][model_index(counts[c], t, k), v_idx].astype(float)
                values = cm[c] if kind == "misclassification_of_class_c" else cm[:, c]
                if values.sum() > 0:
                    table[i] = values / values.sum()
            image = heatmap(axes[r][j], table, [f"{c} {VOC_CLASSES[c]}" for c in classes] if j == 0 else [],
                            [VOC_CLASSES[d] for d in range(NUM_CLASSES)], SEQUENTIAL, 0, 1)
            axes[r][j].set_title("full model" if t == "full" else STYLE[t][0], fontsize=9)
        colorbar(subfigures[r], image, axes[r], "share")
    if kind == "misclassification_of_class_c":
        title = f"Where class c's GT pixels go on {name}, images with c"
        bar_label = "share of class c's GT pixels predicted as class d"
        what = ("per class c (row), the share of its GT pixels predicted as each class d (column), images with c; the "
                "cell in c's own column = found correctly.")
        axes_label = "predicted class d"
    else:
        title = f"Where the false positives of class c come from on {name}, images without c"
        bar_label = "share of the pixels predicted as c whose GT is class d"
        what = ("per class c (row), of the pixels predicted as c on images WITHOUT c (all false positives), the share "
                "whose GT is class d (column); hatched = no pixel predicted as c.")
        axes_label = "GT class d"
    fig.supxlabel(f"{axes_label}; colour: {bar_label}", fontsize=9, color=INK_2)
    heatmap_title(fig, f"{title}{'' if variant == 'raw' else f', after {variant}'}: full model and each circuit type",
                  note, None if note is None else f"{len(counts)} class{'' if len(counts) == 1 else 'es'}")
    save(fig, path, what + f"\n      Rows of panels: K in {ratios} ({selected_setting}).", base)


# ---- logit distributions (group 3) ----------------------------------------------------------
def distribution_tables(counts, variant):
    """Per circuit type: {(measure, pixel set): [22 rows (own class c, then every class d), K]}, the mean over the
    classes c of the circuit's logit against the full model after the same variant: mean_shift (logit units),
    log2_standard_deviation_ratio and, with histograms, wasserstein_1. Also the grid."""
    types = circuit_types()
    any_c = next(iter(counts.values()))
    v_idx = any_c["logit_standardization_variants"].tolist().index(variant)
    pixel_sets = any_c["distribution_pixel_sets"].tolist()
    grid = grid_of(any_c, types[0])
    has_histograms = "logit_histogram" in any_c
    collected = {}
    for c, z in counts.items():
        count = z["logit_pixel_count"][:, v_idx].astype(float)                   # [M, P, 21]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = z["logit_sum"][:, v_idx] / count
            std = np.sqrt(np.maximum(z["logit_squared_sum"][:, v_idx] / count - mean ** 2, 0))
        mean[count == 0], std[count == 0] = np.nan, np.nan
        m_full = model_index(z, "full")
        with np.errstate(divide="ignore", invalid="ignore"):
            measures = {"mean_shift": mean - mean[m_full], "log2_standard_deviation_ratio": np.log2(std / std[m_full])}
        if has_histograms:
            histograms = z["logit_histogram"][:, v_idx]                            # [M, P, 21, B]
            measures["wasserstein_1"] = logit_standardization.wasserstein_1_from_histograms(
                histograms, histograms[m_full][None], float(np.diff(z["histogram_bin_edges"][:2])[0]))
        for t in types:
            idx = [model_index(z, t, k) for k in grid]
            for measure, values in measures.items():
                for p, pixel_set in enumerate(pixel_sets):
                    table = values[idx, p, :].T                                    # [21, K]
                    table = np.vstack([table[c][None], table])                     # own class c first
                    collected.setdefault((t, measure, pixel_set), []).append(table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {key: np.nanmean(tables, axis=0) for key, tables in collected.items()}, grid, pixel_sets


PIXEL_SET_TEXT = {"full_model_prediction": "the pixels the full model predicts as d", "all_pixels": "all pixels"}


def plot_distribution_heatmaps(tables, grid, pixel_sets, name, image_set, variant, kind, path):
    """kind "shift": rows of panels Δμ and log2 σ ratio per pixel set; kind "wasserstein_1": W1 per pixel set.
    One column of panels per circuit type; each panel rows = own class c + 21 classes d, columns = K."""
    types = circuit_types()
    measures = [("mean_shift", "Δμ (logit units)"), ("log2_standard_deviation_ratio", "σ ratio")] \
        if kind == "shift" else [("wasserstein_1", "Wasserstein-1 (logit units)")]
    rows = [(measure, label, p) for p in pixel_sets for measure, label in measures]
    fig, subfigures, axes = heatmap_grid(len(rows), len(types), 0.4 * len(grid) * len(types) + 4.2, 7.0)
    columns = [f"{k:g}" for k in grid]
    for r, (measure, label, p) in enumerate(rows):
        values = [tables[(t, measure, p)] for t in types]
        finite = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.array([])
        if measure == "wasserstein_1":
            cmap, vmin, vmax = SEQUENTIAL, 0, max(float(finite.max()) if finite.size else 1.0, 1e-6)
        else:
            limit = max(float(np.abs(finite).max()) if finite.size else 1.0, 1e-6)
            if measure == "log2_standard_deviation_ratio":
                limit = max(limit, 1.0)
            cmap, vmin, vmax = DIVERGING, -limit, limit
        for j, t in enumerate(types):
            image = heatmap(axes[r][j], values[j], class_d_labels() if j == 0 else [], columns, cmap, vmin, vmax)
            axes[r][j].set_title(STYLE[t][0], fontsize=9)
        subfigures[r].suptitle(f"{label}, on {PIXEL_SET_TEXT[p]}", fontsize=9)
        subfigures[r].supxlabel(f"{utils.topk_ratio_label()} K", fontsize=9, color=INK_2)
        if measure == "log2_standard_deviation_ratio":
            colorbar(subfigures[r], image, axes[r], "σ_circuit / σ_full (log scale)", *ratio_ticks(vmax))
        else:
            colorbar(subfigures[r], image, axes[r], label + (", circuit − full" if measure == "mean_shift" else ""))
    reference = ("the raw full model" if variant == "raw" or variant in logit_standardization.MATCHED_VARIANTS
                 else "the full model after the same variant")
    title = ("Logit shift (mean and spread)" if kind == "shift" else "Wasserstein-1 distance") + \
        f" of every class's logit vs {reference} on {name}, {IMAGE_SET_SHORT[image_set]}" + \
        ("" if variant == "raw" else f", after {variant}")
    heatmap_title(fig, title + " (mean over classes c)")
    save(fig, path, ("Δμ = μ_circuit − μ_full (blue = lower than the full model) and σ ratio (blue = compressed)"
                     if kind == "shift" else "W1 between the circuit's and the full model's distribution of each "
                     "class's logit (0 = the same; binned, resolution = the histogram bin width)")
         + ";\n      rows: \"own class c\" + every class d; columns: K; hatched = no pixels (e.g. the full model never "
           "predicts d there).")


def plot_box_plots(counts, name, image_set, t, variant, path):
    """Jung Fig. 2-style box plots per class d: the distribution of the class-d logit on the pixels the RAW full model
    predicts as d, pooled over the images of every class c's image set (each with its own class-c circuit), for the
    full model and circuit type t (raw; with a variant also after it). Rows: the selected K."""
    any_c = next(iter(counts.values()))
    variants_in_file = any_c["logit_standardization_variants"].tolist()
    edges = any_c["histogram_bin_edges"]
    p = any_c["distribution_pixel_sets"].tolist().index("full_model_prediction")
    ratios = selected_ratios(grid_of(any_c, t))
    if not ratios:
        return
    boxes = [("full model" if variant == "raw" else f"full model after {variant}", "full", variant, MUTED),
             (f"{STYLE[t][0]} (raw)", t, "raw", STYLE[t][1])]
    if variant != "raw":
        boxes.append((f"{STYLE[t][0]} after {variant}", t, variant, VARIANT_STYLE[variant][1]))
    fig, axes = plt.subplots(len(ratios), 1, figsize=(13, 2.9 * len(ratios) + 0.8), sharex=True, squeeze=False)
    width = 0.8 / len(boxes)
    for r, k in enumerate(ratios):
        ax = axes[r, 0]
        for b, (label, model_type, v, color) in enumerate(boxes):
            histogram = sum(z["logit_histogram"][model_index(z, model_type, k), variants_in_file.index(v), p]
                            for z in counts.values())                                   # [21, B]
            stats, positions = [], []
            for d in range(NUM_CLASSES):
                q = logit_standardization.quantiles_from_histogram(histogram[d], edges, [0.05, 0.25, 0.5, 0.75, 0.95])
                if np.isfinite(q).all():
                    stats.append({"whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4]})
                    positions.append(d + (b - (len(boxes) - 1) / 2) * width)
            if stats:
                ax.bxp(stats, positions=positions, widths=width * 0.85, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": color, "edgecolor": SURFACE, "linewidth": 0.5},
                       medianprops={"color": SURFACE, "linewidth": 1.2},
                       whiskerprops={"color": color, "linewidth": 0.8}, capprops={"color": color, "linewidth": 0.8})
        ax.set_title(f"K = {k:g}", fontsize=9)
        ax.set_ylabel("logit")
        ax.set_xlim(-0.6, NUM_CLASSES - 0.4)
    axes[-1, 0].set_xticks(range(NUM_CLASSES), [f"{d} {VOC_CLASSES[d]}" for d in range(NUM_CLASSES)], rotation=90)
    fig.tight_layout()
    handles = [Rectangle((0, 0), 1, 1, facecolor=color) for _, _, _, color in boxes]
    header(fig, f"Logit distribution per class d on the pixels the full model predicts as d, {name}, "
                f"{IMAGE_SET_SHORT[image_set]} (pooled over classes c)", handles, [label for label, *_ in boxes],
           ncol=len(boxes))
    save(fig, path, "per class d: box = quartiles, white line = median, whiskers = 5th and 95th percentile (from the "
         "fixed-bin histograms; values outside the histogram range sit at its edge);\n      each class c's images with "
         "its own class-c circuit, pooled. A class without such pixels has no box.")


# ---- size and structure (group 4) ------------------------------------------------------------
def plot_layer_breakdown(path):
    layers, topk_layer_normalized_ratio_grid, shares = layer_shares()
    x = np.arange(len(topk_layer_normalized_ratio_grid))
    fig, axes = plt.subplots(1, len(layers), figsize=(2.6 * len(layers) + 0.6, 3.0), sharey=True, squeeze=False)
    for l, (ax, name) in enumerate(zip(axes[0], layers)):
        bottom = np.zeros(len(topk_layer_normalized_ratio_grid))
        for j, (label, color) in enumerate(BREAKDOWN):
            ax.bar(x, shares[:, l, j], bottom=bottom, width=0.7, color=color, edgecolor=SURFACE,
                   linewidth=0.8, label=label)
            bottom += shares[:, l, j]
        ax.set_title(name)
        ax.set_xticks(x, [f"{topk_layer_normalized_ratio:g}" for topk_layer_normalized_ratio in topk_layer_normalized_ratio_grid],
                      rotation=90)
        ax.set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("share of the layer's channels")
    fig.supxlabel(utils.topk_ratio_axis_label(), fontsize=9, color=INK_2)  # one label: the global-mode text is long
    fig.tight_layout()
    header(fig, "Certification outcome per layer (mean over classes)", *axes[0, 0].get_legend_handles_labels())
    save(fig, path, f"per layer of U and {utils.topk_ratio_label()}: share of the layer's "
         "channels certified in / abstaining (in or not in the baseline circuit) / certified out;\n      mean over "
         "classes, bars add up to 1.")


def plot_size_measures(df, path):
    """The size measures against K (mean over classes; they do not depend on the eval data)."""
    layers, _, _ = M.certifiable_components(C.LAYERS)
    first = df[(df.eval_dataset == df.eval_dataset.iloc[0]) & (df.image_set == "images_with_class_c")
               & (df.logit_standardization_variant == "raw")]
    if "subnet_size_whole_model_weight_ratio" not in first:
        print("  NOTE: summary.csv has no size measures (an older s05 output); size_measures "
              "skipped.")
        return
    group_channels = sum(M.HOOK_POINTS[layer][1] for layer in layers)
    pooled = ([("subnet_size_group_pooled_ratio", f"POOLED share of U's {group_channels:,} channels kept")]
              if "subnet_size_group_pooled_ratio" in first else [])
    if not pooled:
        print("  NOTE: summary.csv has no subnet_size_group_pooled_ratio (an older s05 output); the "
              "pooled panel of size_measures is skipped.")
    panels = ([(f"subnet_size_layer_normalized_ratio_{layer}", f"{layer}: share of channels kept") for layer in layers]
              + pooled
              + [("subnet_size_whole_model_weight_ratio", "share of the WHOLE model's weights kept"),
                 ("subnet_size_reachable_weight_ratio", "share of the REACHABLE weights kept")])
    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols + 0.6, 2.7 * rows + 0.4), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (column, title) in zip(axes.flat, panels):
        for t in circuit_types():
            sub = first[first.circuit == t].groupby("topk_layer_normalized_ratio")[column].mean()
            line(ax, sub.index, sub.values, t)
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    pooled_note = (f"Pooled share of U = channels of U kept / all {group_channels:,} (big layers dominate; × {group_channels:,} "
                   f"= the channel count). " if pooled else "")
    baseline_note = ("Global top-K: the baseline circuit's POOLED share is K (on the grey line), its per-layer shares "
                     "are not. " if utils.topk_selection_is_global() else
                     "Per-layer top-K: the baseline circuit's per-layer shares are K (on the grey line). ")
    note = (baseline_note + pooled_note
            + f"Binary-mask convention: a weight is 0 if it writes or reads a removed channel of U. Whole model = all "
            f"weights (auxiliary classifier removed); even an empty circuit keeps the weights no channel of U reaches. "
            f"{circuit_size.REACHABLE_WEIGHTS_LABEL[0].upper()}{circuit_size.REACHABLE_WEIGHTS_LABEL[1:]}: the share "
            f"runs from 0 (every channel of U removed) to 1. They are NOT the backbone: most are the ASPP convs reading "
            f"layer4.")
    grey_line = ("share = K, the share of the whole " + C.LAYERS + " group A keeps" if utils.topk_selection_is_global()
                 else "share = K")
    fig.supxlabel(f"{utils.topk_ratio_label()} K (thin grey line: {grey_line})", fontsize=9, color=INK_2)
    free = len(axes.flat) - len(panels)
    if free:  # the note goes into the empty panel slots of the last row
        fig.tight_layout()
        slot = axes.flat[len(panels)]
        slot.set_visible(True)
        slot.axis("off")
        slot.text(0, 1, textwrap.fill(note, 42 * free), transform=slot.transAxes, fontsize=7.5, color=INK_2, va="top")
    else:
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.text(0.01, 0.0, textwrap.fill(note, 150), fontsize=7.5, color=INK_2, va="bottom")
    header(fig, "Circuit size measures (mean over classes)", *axes[0, 0].get_legend_handles_labels())
    save(fig, path, "per-layer channel shares, " + ("the pooled share of U, " if pooled else "") + "and the weight "
         "shares (whole model; reachable weights, see the note in the figure) against K, mean over classes.")


def layer_kept_shares_per_class():
    """Per class in CLASSES, from s04's masks: {class: {"baseline_circuit": [|grid|, layers of U], "certified": [...]}},
    the share of each layer's channels in the circuit; also the layer names and the grid."""
    layers, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    channels = np.array([s.stop - s.start for s in certifiable_slices.values()])
    shares = {}
    for c in C.CLASSES:
        path = utils.certify_dir() / f"class_{c:02d}.npz"
        utils.require(path, "s04_certify.py")
        with np.load(path) as certification_results:
            shares[c] = {
                "baseline_circuit": utils.layer_channel_counts(certification_results["baseline_circuit"],
                                                               certifiable_slices) / channels,
                "certified": utils.layer_channel_counts(certification_results["decision"] == 1,
                                                        certifiable_slices) / channels,
            }
            grid = certification_results["topk_layer_normalized_ratio_grid"]
    return layers, grid, shares


def plot_layer_breakdown_per_class(path):
    """One panel per class: the share of each layer's channels in the baseline circuit (dashed) and in the certified
    circuit (solid) against K, one colour per layer. Shows which layers lose channels for a single class."""
    layers, grid, shares = layer_kept_shares_per_class()
    cols = min(5, len(C.CLASSES))
    rows = math.ceil(len(C.CLASSES) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols + 0.6, 1.9 * rows + 0.6), sharex=True, sharey=True,
                             squeeze=False)
    for ax, c in zip(axes.flat, C.CLASSES):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=0)
        for l, name in enumerate(layers):
            ax.plot(grid, shares[c]["baseline_circuit"][:, l], color=LAYER_COLORS[name], linestyle="--", linewidth=1.4)
            ax.plot(grid, shares[c]["certified"][:, l], color=LAYER_COLORS[name], linestyle="-", marker="o",
                    markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_title(VOC_CLASSES[c], fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(C.CLASSES):]:
        ax.set_visible(False)
    handles = ([plt.Line2D([], [], color=LAYER_COLORS[name], linewidth=2) for name in layers]
               + [plt.Line2D([], [], color=INK_2, linestyle="--", linewidth=1.4),
                  plt.Line2D([], [], color=INK_2, linestyle="-", marker="o", markersize=3.5, markeredgecolor=SURFACE),
                  plt.Line2D([], [], color=GRID, linewidth=0.8)])
    labels = ([f"{name} ({M.HOOK_POINTS[name][1]:,} channels)" for name in layers]
              + ["baseline circuit A(D_c) (dashed)", "certified circuit (solid)", "share = K"])
    fig.supxlabel(utils.topk_ratio_axis_label(), fontsize=9, color=INK_2)
    fig.supylabel("share of the layer's channels in the circuit", fontsize=9, color=INK_2)
    fig.tight_layout()
    header(fig, "Which layers keep channels, per class: share of each layer's channels in the baseline circuit and in "
           "the certified circuit", handles, labels, ncol=math.ceil(len(handles) / 2))
    mode_text = ("global top-K: K is the share of the whole " + C.LAYERS + " group, so the layers spread around the "
                 "grey line (below it = the layer loses more than the group)" if utils.topk_selection_is_global() else
                 "per-layer top-K: the baseline circuit keeps K of every layer, so its dashed lines lie on the grey "
                 "line by construction")
    save(fig, path, f"per class, the share of each layer's channels in the baseline circuit (dashed) and the certified "
         f"circuit (solid) against K;\n      {mode_text}. A line at 0 = an empty layer.")


def plot_merged(path_pairs, path_union):
    """s05b's merged circuits: the mean over class pairs, and the all-classes union, against the merged circuit's
    subnetwork size; one column per eval dataset."""
    folder = utils.merged_circuits_eval_dir()
    if not (folder / "summary_class_pairs.csv").exists():
        print(f"  NOTE: no s05b output in {folder}; merged_circuits and all_classes_union skipped "
              f"(run s05b_evaluate_merged_circuits.py first).")
        return
    pairs = pd.read_csv(folder / "summary_class_pairs.csv")
    pairs = pairs[pairs.class_1.isin(C.CLASSES) & pairs.class_2.isin(C.CLASSES)].copy()
    pairs["cIoU"] = pairs[["cIoU_class_1", "cIoU_class_2"]].mean(axis=1)
    pairs["faith_IoU"] = pairs[["faith_IoU_class_1", "faith_IoU_class_2"]].mean(axis=1)
    union = pd.read_csv(folder / "summary_all_classes_union.csv").rename(
        columns={"cIoU_mean_over_classes": "cIoU", "faith_IoU_mean_over_classes": "faith_IoU"})
    metrics = [("cIoU", "cIoU (vs GT)"), ("faith_IoU", "faithfulness IoU"), ("mIoU_vs_ground_truth", "mIoU (vs GT)"),
               ("pixel_agreement_with_full_model", "pixel agreement\nwith the full model"),
               ("class_share_total_variation_to_full_model", "class-share total\nvariation to the full model")]
    for table, path, title, what in (
            (pairs, path_pairs, "Merged circuits (union of two class circuits), mean over the class pairs",
             "per pair the union of the two class circuits on both classes' images with c; cIoU and faith = mean of "
             "the two classes; mean over pairs."),
            (union, path_union, f"All-classes union (reconstruction check): the union of all {len(C.CLASSES)} class "
             f"circuits", "the union of every class circuit on the union of all eval lists with c; cIoU and faith = "
             "mean over the classes.")):
        names = [n for n in C.EVAL_DATASETS if n in set(table.eval_dataset)]
        if not names:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = (table.groupby(["eval_dataset", "circuit", "topk_layer_normalized_ratio"], dropna=False)
                     [[m for m, _ in metrics] + ["subnet_size_layer_normalized_ratio_mean"]].mean().reset_index())
        fig, axes = plt.subplots(len(metrics), len(names), figsize=(3.4 * len(names) + 1.0, 2.2 * len(metrics)),
                                 sharex=True, sharey=True, squeeze=False)
        for col, name in enumerate(names):
            d = means[means.eval_dataset == name]
            for row, (metric, label) in enumerate(metrics):
                ax = axes[row, col]
                for t in circuit_types():
                    sub = d[d.circuit == t].sort_values("topk_layer_normalized_ratio")
                    line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], t)
                full = d[d.circuit == "full"][metric]
                if len(full):
                    full_line(ax, full.iloc[0])
                ax.set_xlim(0, 1.02)
                ax.set_ylim(0, 1.02)
                if row == 0:
                    ax.set_title(name)
                if col == 0:
                    ax.set_ylabel(label)
                if row == len(metrics) - 1:
                    ax.set_xlabel("Subnetwork size of the MERGED circuit,\nlayer-normalized ratio")
        fig.tight_layout()
        header(fig, title, *axes[0, 0].get_legend_handles_labels())
        save(fig, path, what + " Raw circuits (s05b); the grey line is the full model on the same images.")


# ---- main ------------------------------------------------------------------------------------
def print_peaks(peaks):
    measures = ["peak_cIoU", "topk_layer_normalized_ratio", "subnet_size_layer_normalized_ratio_mean"]
    for (name, variant), table in peaks.groupby(["eval_dataset", "logit_standardization_variant"], sort=False):
        utils.section(
            f"[{name}] Best cIoU over the {utils.topk_ratio_label()} grid per class, with the ratio and the "
            f"subnetwork size where it occurs (peak_ciou.csv), images WITH class c"
            f"{logit_standardization.section_title_suffix(variant)}",
            logit_standardization.how_to_read_variant(variant) + textwrap.dedent(f"""
            One block per circuit type X; in peak_ciou.csv the same columns are named X_<column>
            (e.g. baseline_circuit_peak_cIoU).
            full_cIoU      the unpruned model's class-c IoU vs GT over the class's eval images (s05).
            peak_cIoU      circuit type X's HIGHEST cIoU over the grid.
            topk_layer_normalized_ratio
                           the {utils.topk_ratio_label()} K where the peak occurs ({utils.topk_ratio_meaning()}).
            subnet_size_layer_normalized_ratio_mean
                           the circuit's subnetwork size there (layer-normalized ratio: equal-weight mean over the
                           layers of U, the certifiable components = {C.LAYERS!r}). At K = 1 every circuit is the
                           full model, so a peak at K = 1 only says that no smaller circuit did better than the
                           full model.
            MEAN           mean over the classes of every column (for the last two only a rough summary).
            This table shows the best case only; how cIoU falls as circuits shrink is in curves_mean.csv
            (also printed by s05) and in the "curves" figure.{" discovery_samples is in-sample (D_c itself)." if name == "discovery_samples" else ""}
            """))
        for t in circuit_types():
            block = table[["class_name", "full_cIoU"] + [f"{t}_{m}" for m in measures]].rename(
                columns={f"{t}_{m}": m for m in measures})
            mean_row = block.drop(columns="class_name").mean().to_frame().T.assign(class_name="MEAN")
            print(f"\ncircuit type: {t}")
            print(pd.concat([block, mean_row[block.columns]]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        at_full = ", ".join(f"{t} {int((table[f'{t}_topk_layer_normalized_ratio'] == 1).sum())}/{len(table)}"
                            for t in circuit_types())
        print(f"\nClasses whose peak is at K = 1 (the full model): {at_full}")


def main():
    utils.start_run("s06_plots", USED_PARAMS)
    df = load_summary()
    out_dir = utils.plot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    means = mean_curves(df)
    means.to_csv(out_dir / "curves_mean.csv", index=False)
    peaks = peak_table(df)
    (out_dir / GROUP_DIRS[1]).mkdir(parents=True, exist_ok=True)
    peaks.to_csv(out_dir / GROUP_DIRS[1] / "peak_ciou.csv", index=False)
    print_peaks(peaks)

    names = [n for n in C.EVAL_DATASETS if n in set(df.eval_dataset)]
    image_sets = [s for s in IMAGE_SETS if s in set(df.image_set)]
    utils.section(
        f"Figures in {out_dir}, one folder per evaluation group",
        textwrap.dedent(f"""
        1_accuracy_vs_ground_truth   cIoU, mIoU, false positives of c, where class c's pixels go
        2_faithfulness_to_full_model faith, |r|, pixel agreement, class-share total variation, class shares
        3_logit_distributions        Δμ, σ ratio, Wasserstein-1 and box plots of every class's logit vs the full model
        4_size_and_structure         certification outcome per layer (mean; per class), size measures, merged circuits (s05b)
        Figures outside a logit_standardization/ folder show the RAW circuits (no logit standardization).
        {"Each group's logit_standardization/ folder shows raw + " + ", ".join(C.LOGIT_STANDARDIZATION_VARIANTS) + "." if C.LOGIT_STANDARDIZATION_VARIANTS else ""}
        Per-eval-dataset figures are one folder down: <group>/<eval dataset>/. Image sets: {", ".join(image_sets)}.
        curves_mean.csv (top level) holds the means behind every curve.
        """))

    # ---- groups 1 and 2: curves ---------------------------------------------------------------
    for group in (1, 2):
        plot_group_curves(means, group, figure_path(group, "curves"))
        for t in circuit_types() if C.LOGIT_STANDARDIZATION_VARIANTS else []:
            plot_group_curves(means, group, figure_path(group, "logit_standardization", f"curves_{t}"), t)
    for name in names:
        folder = folder_name(name)
        plot_per_class(df, name, "images_with_class_c", "cIoU", "cIoU (vs GT)", f"cIoU per class on {name}, images "
                       f"with c", figure_path(1, folder, "curves_per_class"),
                       f"cIoU against the subnetwork size on {name}, one panel per class; the grey line is the full "
                       f"model's cIoU of that class.")
        for t in circuit_types() if C.LOGIT_STANDARDIZATION_VARIANTS else []:
            plot_per_class(df, name, "images_with_class_c", "cIoU", "cIoU (vs GT)",
                           f"Logit standardization, circuit type {STYLE[t][0]}: cIoU per class on {name}",
                           figure_path(1, folder, "logit_standardization", f"curves_per_class_{t}"),
                           f"cIoU of the {t} circuits on {name}, one panel per class, one line per logit-"
                           f"standardization variant;\n      grey lines: the class's full-model cIoU (raw; after the "
                           f"Jung variant if it is on).", t=t)
        if "images_without_class_c" in image_sets:
            plot_per_class(df, name, "images_without_class_c", "class_c_predicted_pixel_ratio",
                           "share of pixels predicted as c (false positives)",
                           f"False positives of c per class on {name}, images without c",
                           figure_path(1, folder, "false_positives_per_class"),
                           f"share of the non-void pixels predicted as c on {name}'s images WITHOUT c (all false "
                           f"positives), one panel per class; grey = the full model.", ylim=None)

        # ---- per eval dataset: confusion heatmaps, class shares, logit distributions -----------------
        for image_set in image_sets:
            counts = load_counts(name, image_set)
            if not counts:
                print(f"  NOTE: no s05 counts for {name}, {image_set} (an older s05 output without counts); "
                      f"its heatmaps and box plots are skipped.")
                continue
            for v in variants():
                sub = [] if v == "raw" else ["logit_standardization"]
                suffix = "" if v == "raw" else f"_{v}"
                kind = "misclassification_of_class_c" if image_set == "images_with_class_c" else "false_positive_sources"
                plot_confusion_heatmaps(counts, name, kind, v, figure_path(1, folder, *sub, kind + suffix))
                plot_class_shares(counts, name, image_set, v, figure_path(2, folder, *sub,
                                                                          f"class_shares_{image_set}{suffix}"))
                plot_class_shares_stacked_bars(counts, name, image_set, v, figure_path(
                    2, folder, *sub, f"class_shares_stacked_bars_{image_set}{suffix}"))
                tables, grid, pixel_sets = distribution_tables(counts, v)
                plot_distribution_heatmaps(tables, grid, pixel_sets, name, image_set, v, "shift",
                                           figure_path(3, folder, *sub, f"shift_and_std_ratio_{image_set}{suffix}"))
                if "logit_histogram" in next(iter(counts.values())):
                    plot_distribution_heatmaps(tables, grid, pixel_sets, name, image_set, v, "wasserstein_1",
                                               figure_path(3, folder, *sub, f"wasserstein_1_{image_set}{suffix}"))
                    for t in circuit_types():
                        plot_box_plots(counts, name, image_set, t, v,
                                       figure_path(3, folder, *sub, f"box_plots_{image_set}_{t}{suffix}"))
            del counts

    # ---- group 4 --------------------------------------------------------------------------------
    plot_layer_breakdown(figure_path(4, "layer_breakdown"))
    plot_layer_breakdown_per_class(figure_path(4, "layer_breakdown_per_class"))
    plot_size_measures(df, figure_path(4, "size_measures"))
    plot_merged(figure_path(4, "merged_circuits"), figure_path(4, "all_classes_union"))
    utils.write_run_info(out_dir)


if __name__ == "__main__":
    main()
