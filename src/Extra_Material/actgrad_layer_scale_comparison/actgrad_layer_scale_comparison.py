"""Exploration, not part of the pipeline: are the cached ActGrad scores comparable across layers?

Reads the existing s02 score cache (per image, per channel ActGrad = mean_{h,w} |a · ∂f/∂a|) and the s03
deletion masks of the per-layer run (TOPK_SELECTION_MODE = "per_layer"), and asks what a GLOBAL top-K (one ranking over all channels of a
layer group) would keep per layer, compared with the PER-LAYER top-K.

Run:
    python src/Extra_Material/actgrad_layer_scale_comparison/actgrad_layer_scale_comparison.py
Writes the figures and a text report next to this script. Only reads from results/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCORE_DIRECTORY = REPO_ROOT / "results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores"
SAMPLE_DIRECTORY = SCORE_DIRECTORY / "s03_pred_actgrad_pdel0.60_n100-1000_sseed0"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent
TARGET = "pred"                          # as config.TARGET in the run
CLASSES = range(1, 21)
VOC_CLASS_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                   "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                   "train", "tvmonitor"]
TOPK_LAYER_NORMALIZED_RATIO_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAYER_GROUPS = {"backbone": ["layer1", "layer2", "layer3", "layer4"],
                "backbone+head": ["layer1", "layer2", "layer3", "layer4", "aspp", "head"]}
# Output stride of each hook point in torchvision's DeepLabV3-ResNet50 (layer3/layer4 are dilated, not strided).
LAYER_OUTPUT_STRIDE = {"layer1": 4, "layer2": 8, "layer3": 8, "layer4": 8, "aspp": 8, "head": 8}
# Categorical palette, fixed order.
LAYER_COLORS = {"layer1": "#2a78d6", "layer2": "#eb6834", "layer3": "#1baf7a", "layer4": "#eda100",
                "aspp": "#e87ba4", "head": "#008300"}
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

report_lines = []


def report(line=""):
    print(line)
    report_lines.append(line)


def style_axis(axis):
    axis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY)


def load_class(c):
    cache = torch.load(SCORE_DIRECTORY / f"class_{c:02d}.pt")
    per_image_scores = cache["scores"][TARGET].double().numpy()            # [|D_c|, 4352]
    layer_slices = {name: slice(*bounds) for name, bounds in cache["layer_slices"].items()}
    return per_image_scores, layer_slices


def min_max_within_layer(layer_scores):
    """Hamblin et al. App. 8.5: rescale a layer's scores to [0, 1] before the global top-κ."""
    return (layer_scores - layer_scores.min()) / (layer_scores.max() - layer_scores.min())


def keep_counts_per_layer(scores_of_group, channel_layer_index, number_of_layers, number_kept):
    """Global top-K: the number_kept highest scores of the whole group -> kept channels per layer."""
    order = np.argsort(-scores_of_group, kind="stable")
    kept_layer_index = channel_layer_index[order[:number_kept]]
    return np.bincount(kept_layer_index, minlength=number_of_layers)


def global_topk_share_curves(cumulative_saliency, layer_slices, layers, kept_fractions, normalization):
    """Per kept fraction of the group: share of each layer that a global top-K keeps."""
    pieces, channel_layer_index = [], []
    for layer_number, name in enumerate(layers):
        layer_scores = cumulative_saliency[layer_slices[name]]
        if normalization == "raw":
            normalized = layer_scores
        elif normalization == "times_channel_count":
            normalized = layer_scores * len(layer_scores)
        elif normalization == "divided_by_layer_mean":
            normalized = layer_scores / layer_scores.mean()
        elif normalization == "min_max_within_layer":
            normalized = min_max_within_layer(layer_scores)
        pieces.append(normalized)
        channel_layer_index.append(np.full(len(layer_scores), layer_number))
    group_scores, channel_layer_index = np.concatenate(pieces), np.concatenate(channel_layer_index)
    channels_per_layer = np.bincount(channel_layer_index)
    order = np.argsort(-group_scores, kind="stable")
    ranked_layer_index = channel_layer_index[order]
    shares = np.zeros((len(kept_fractions), len(layers)))
    for fraction_index, fraction in enumerate(kept_fractions):
        number_kept = int(round(fraction * len(group_scores)))
        shares[fraction_index] = np.bincount(ranked_layer_index[:number_kept], minlength=len(layers)) / channels_per_layer
    return shares


def main():
    all_classes = {c: load_class(c) for c in CLASSES}
    layer_slices = all_classes[1][1]
    layer_names = list(layer_slices)
    channels_per_layer = {name: s.stop - s.start for name, s in layer_slices.items()}

    # ---------------------------------------------------------------- 1. scale per layer
    report("=" * 100)
    report("[1] Scale of the cumulative saliency (Hamblin Eq. 3: sum over the |D_c| images of the per-image ActGrad")
    report("    score) per channel, by layer. Target f = mean class-c logit over the 'pred' pixels (the run's TARGET).")
    report("    Each cell: median over the layer's channels, divided by the median of layer4 in the same class,")
    report("    then the median of that ratio over the 20 classes [min - max over classes in brackets].")
    report("    1.0 = same typical channel score as layer4. If layers were comparable, all cells would be near 1.")
    report("=" * 100)
    median_ratio_to_layer4 = {name: [] for name in layer_names}
    layer_mass_share = {name: [] for name in layer_names}
    pooled_normalized_scores = {name: [] for name in layer_names}
    for c, (per_image_scores, _) in all_classes.items():
        cumulative_saliency = per_image_scores.sum(axis=0)
        model_wide_median = np.median(cumulative_saliency)
        layer4_median = np.median(cumulative_saliency[layer_slices["layer4"]])
        for name in layer_names:
            layer_scores = cumulative_saliency[layer_slices[name]]
            median_ratio_to_layer4[name].append(np.median(layer_scores) / layer4_median)
            layer_mass_share[name].append(layer_scores.sum() / cumulative_saliency.sum())
            pooled_normalized_scores[name].append(layer_scores / model_wide_median)
    report(f"{'layer':8s} {'channels':>8s} {'stride':>6s} {'median / layer4 median':>30s} {'share of model score mass':>27s} {'channel share':>14s}")
    for name in layer_names:
        ratios = np.array(median_ratio_to_layer4[name])
        mass = np.array(layer_mass_share[name])
        report(f"{name:8s} {channels_per_layer[name]:8d} {LAYER_OUTPUT_STRIDE[name]:6d} "
               f"{np.median(ratios):10.2f} [{ratios.min():6.2f} - {ratios.max():6.2f}]   "
               f"{np.median(mass):12.3f} [{mass.min():.3f} - {mass.max():.3f}] {channels_per_layer[name] / 4352:14.3f}")

    figure, axis = plt.subplots(figsize=(8, 4.6))
    positions = np.arange(len(layer_names))
    boxes = axis.boxplot([np.concatenate(pooled_normalized_scores[name]) for name in layer_names], positions=positions,
                         widths=0.55, patch_artist=True, showfliers=True, whis=(1, 99),
                         flierprops=dict(marker=".", markersize=2, alpha=0.25, markeredgecolor=TEXT_SECONDARY),
                         medianprops=dict(color="#0b0b0b", linewidth=1.5))
    for patch, name in zip(boxes["boxes"], layer_names):
        patch.set_facecolor(LAYER_COLORS[name])
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    axis.set_yscale("log")
    axis.set_xticks(positions, [f"{name}\n{channels_per_layer[name]} ch, stride {LAYER_OUTPUT_STRIDE[name]}"
                                for name in layer_names], fontsize=8)
    axis.set_ylabel("cumulative saliency per channel\n(÷ the class's model-wide median channel)")
    axis.set_title("ActGrad channel scores by layer, all 20 classes pooled (target 'pred')\n"
                   "box = 25–75 %, whiskers = 1–99 %, line = median; label at top = median ÷ layer4 median",
                   fontsize=10)
    for position, name in zip(positions, layer_names):
        axis.annotate(f"×{np.median(median_ratio_to_layer4[name]):.1f}", (position, axis.get_ylim()[1]),
                      xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "fig1_score_scale_per_layer.png", dpi=150)
    plt.close(figure)

    # ---------------------------------------------------------------- 1b. raw values, no rescaling
    report("")
    report("=" * 100)
    report("[1b] RAW values, no rescaling at all (fig. 1b). Left: one image × one channel (the cached ActGrad score).")
    report("     Right: cumulative saliency (sum over the class's images). All 20 classes pooled. Images whose 'pred'")
    report("     region is empty (all-zero scores) are left out of both. 'zeros' = share of exactly-zero values,")
    report("     which a log axis cannot show.")
    report("=" * 100)
    raw_per_image, raw_cumulative = {name: [] for name in layer_names}, {name: [] for name in layer_names}
    for per_image_scores, _ in all_classes.values():
        scores_of_nonempty_images = per_image_scores[per_image_scores.sum(axis=1) > 0]
        for name in layer_names:
            raw_per_image[name].append(scores_of_nonempty_images[:, layer_slices[name]].ravel())
            raw_cumulative[name].append(scores_of_nonempty_images[:, layer_slices[name]].sum(axis=0))
    raw_per_image = {name: np.concatenate(values) for name, values in raw_per_image.items()}
    raw_cumulative = {name: np.concatenate(values) for name, values in raw_cumulative.items()}
    for title, values in (("one image × one channel", raw_per_image), ("cumulative saliency", raw_cumulative)):
        report(f"\n-- {title}")
        report(f"{'layer':8s} {'p1':>9s} {'p25':>9s} {'median':>9s} {'p75':>9s} {'p99':>9s} {'max':>9s} {'zeros':>7s}")
        for name in layer_names:
            v = values[name]
            report(f"{name:8s}" + "".join(f" {np.percentile(v, q):9.2e}" for q in (1, 25, 50, 75, 99))
                   + f" {v.max():9.2e} {np.mean(v == 0):7.1%}")

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw=dict(height_ratios=[1, 1.15]))
    panel_kinds = (("One image × one channel (cached ActGrad score)\nwhiskers = 1–99 %, values beyond them not drawn",
                    raw_per_image, False),
                   ("Cumulative saliency per channel (sum over the class's images)\nwhiskers = 1–99 %, dots = beyond",
                    raw_cumulative, True))
    rows = (("backbone only (the certified LAYERS), zoomed", LAYER_GROUPS["backbone"]),
            ("all six hook points", layer_names))
    for row, (row_label, row_layers) in enumerate(rows):
        row_positions = np.arange(len(row_layers))
        for column, (title, values, show_beyond_whiskers) in enumerate(panel_kinds):
            axis = axes[row, column]
            boxes = axis.boxplot([values[name][values[name] > 0] for name in row_layers], positions=row_positions,
                                 widths=0.55, patch_artist=True, showfliers=show_beyond_whiskers, whis=(1, 99),
                                 flierprops=dict(marker=".", markersize=2, alpha=0.25, markeredgecolor=TEXT_SECONDARY),
                                 medianprops=dict(color="#0b0b0b", linewidth=1.5))
            for patch, name in zip(boxes["boxes"], row_layers):
                patch.set_facecolor(LAYER_COLORS[name])
                patch.set_alpha(0.85 if name in LAYER_GROUPS["backbone"] else 0.35)
                patch.set_edgecolor("white")
            if len(row_layers) > 4:
                axis.axvspan(3.5, 5.5, color=GRID_COLOR, alpha=0.5, zorder=0)
                axis.text(4.5, 0.02, "not certified\n(LAYERS = backbone)", transform=axis.get_xaxis_transform(),
                          ha="center", fontsize=7, color=TEXT_SECONDARY)
            axis.set_yscale("log")
            axis.set_xticks(row_positions, [f"{name}\nzeros {np.mean(values[name] == 0):.1%}" for name in row_layers],
                            fontsize=8)
            # median written above the highest drawn value (upper whisker, or the highest dot), so it covers nothing
            for position, name in zip(row_positions, row_layers):
                highest_drawn = values[name].max() if show_beyond_whiskers else np.percentile(values[name], 99)
                axis.annotate(f"median {np.median(values[name]):.1e}", (position, highest_drawn),
                              xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7,
                              color="#0b0b0b")
            axis.set_ylim(top=axis.get_ylim()[1] * 4)
            axis.set_title(f"{title}\n[{row_label}]", fontsize=9)
            style_axis(axis)
        axes[row, 0].set_ylabel("raw ActGrad = mean over pixels of |a · ∂f/∂a|  (log scale)")
    figure.suptitle("ActGrad channel scores by layer, RAW values (no rescaling), 20 classes pooled, target 'pred'.\n"
                    "Exact zeros cannot be shown on a log axis; their share is under each layer name.", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(OUTPUT_DIRECTORY / "fig1b_raw_score_scale_per_layer.png", dpi=150)
    plt.close(figure)

    # ---------------------------------------------------------------- 2. per image consistency
    report("")
    report("=" * 100)
    report("[2] Is the layer ordering a property of single images, or only of the sum over D_c?")
    report("    Per image: median channel score of the layer ÷ median channel score of layer4 (same image).")
    report("    Cells: percentiles over all images of all 20 classes (5,240 images); '% images > 1' = the layer's")
    report("    typical channel scores higher than layer4's typical channel in that image.")
    report("=" * 100)
    per_image_ratio = {name: [] for name in layer_names}
    for c, (per_image_scores, _) in all_classes.items():
        layer4_median = np.median(per_image_scores[:, layer_slices["layer4"]], axis=1)
        valid = layer4_median > 0                                  # images with an empty 'pred' region are all zero
        for name in layer_names:
            per_image_ratio[name].append(np.median(per_image_scores[valid][:, layer_slices[name]], axis=1) / layer4_median[valid])
    report(f"{'layer':8s} {'p5':>7s} {'p50':>7s} {'p95':>7s} {'% images > 1':>13s}")
    for name in layer_names:
        ratios = np.concatenate(per_image_ratio[name])
        report(f"{name:8s} {np.percentile(ratios, 5):7.2f} {np.percentile(ratios, 50):7.2f} {np.percentile(ratios, 95):7.2f} "
               f"{np.mean(ratios > 1):12.1%}")
    report(f"(images excluded because their 'pred' region is empty, all scores 0: "
           f"{sum((np.median(s[:, layer_slices['layer4']], axis=1) == 0).sum() for s, _ in all_classes.values())})")

    # ---------------------------------------------------------------- 3. global top-K share per layer
    kept_fractions = np.linspace(0.01, 1.0, 100)
    normalizations = {"raw": "raw ActGrad (as cached)",
                      "times_channel_count": "score × channels in layer",
                      "divided_by_layer_mean": "score ÷ layer mean",
                      "min_max_within_layer": "min-max per layer (Hamblin App. 8.5)"}
    curves = {}  # (group, normalization) -> [classes, fractions, layers]
    for group, layers in LAYER_GROUPS.items():
        for normalization in normalizations:
            curves[group, normalization] = np.stack([
                global_topk_share_curves(per_image_scores.sum(axis=0), layer_slices, layers, kept_fractions, normalization)
                for per_image_scores, _ in all_classes.values()])

    report("")
    report("=" * 100)
    report("[3] What a GLOBAL top-K would keep per layer, at the same total circuit size as the per-layer")
    report("    top-K (\"per_layer\"). Rows: the per-layer ratio K -> global top-K keeping the same number of channels")
    report("    over the whole group. Cells: share of the layer's channels kept, mean over the 20 classes")
    report("    [min - max]. Under per-layer top-K every cell would equal K (column 'K').")
    report("=" * 100)
    for group, layers in LAYER_GROUPS.items():
        for normalization, label in normalizations.items():
            report(f"\n-- group '{group}' ({sum(channels_per_layer[n] for n in layers)} channels), scores: {label}")
            report(f"{'K':>5s}" + "".join(f" {name:>17s}" for name in layers))
            for K in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
                shares = np.stack([global_topk_share_curves(s.sum(axis=0), layer_slices, layers, [K], normalization)[0]
                                   for s, _ in all_classes.values()])
                report(f"{K:5.2f}" + "".join(f" {shares[:, l].mean():5.2f} [{shares[:, l].min():.2f}-{shares[:, l].max():.2f}]"
                                             for l in range(len(layers))))

    report("")
    report("Smallest group-wide kept fraction at which a raw global top-K keeps >= 1 channel in EVERY layer")
    report("(below it some layer is pruned completely; every hook point is a full cut of DeepLabV3, so the model")
    report("output would no longer depend on the input):")
    for group, layers in LAYER_GROUPS.items():
        thresholds = []
        for per_image_scores, _ in all_classes.values():
            cumulative_saliency = per_image_scores.sum(axis=0)
            group_scores = np.concatenate([cumulative_saliency[layer_slices[n]] for n in layers])
            layer_index = np.concatenate([np.full(channels_per_layer[n], i) for i, n in enumerate(layers)])
            ranked = layer_index[np.argsort(-group_scores, kind="stable")]
            first_appearance = [np.argmax(ranked == i) + 1 for i in range(len(layers))]
            thresholds.append(max(first_appearance) / len(group_scores))
        thresholds = np.array(thresholds)
        report(f"  {group:14s}: median {np.median(thresholds):.3%} of the group  [min {thresholds.min():.3%}, max {thresholds.max():.3%}]")

    figure, axes = plt.subplots(len(LAYER_GROUPS), len(normalizations), figsize=(17, 7.6), sharex=True, sharey=True)
    for row, (group, layers) in enumerate(LAYER_GROUPS.items()):
        for column, (normalization, label) in enumerate(normalizations.items()):
            axis = axes[row, column]
            axis.plot([0, 1], [0, 1], linestyle="--", color=TEXT_SECONDARY, linewidth=1, label="per-layer top-K (\"per_layer\")")
            for l, name in enumerate(layers):
                share = curves[group, normalization][:, :, l]
                axis.fill_between(kept_fractions, share.min(axis=0), share.max(axis=0), color=LAYER_COLORS[name],
                                  alpha=0.15, linewidth=0)
                axis.plot(kept_fractions, share.mean(axis=0), color=LAYER_COLORS[name], linewidth=2, label=name)
            axis.set_title(f"{group}  ·  {label}", fontsize=9)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            style_axis(axis)
            if row == len(LAYER_GROUPS) - 1:
                axis.set_xlabel("channels kept in the whole group (global top-K)")
            if column == 0:
                axis.set_ylabel("share of the layer's channels kept")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=9)
    figure.suptitle("Global top-K: share kept per layer vs total kept (line = mean over 20 classes, band = min–max)\n"
                    "Dashed diagonal = what the per-layer top-K (\"per_layer\") keeps in every layer", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(OUTPUT_DIRECTORY / "fig2_global_topk_share_per_layer.png", dpi=150)
    plt.close(figure)

    # ---------------------------------------------------------------- 4. overlap with per-layer top-K
    report("")
    report("=" * 100)
    report("[4] Overlap (Jaccard) between the global top-K circuit and the per-layer top-K (\"per_layer\") circuit of the")
    report("    same total size, group 'backbone' (the certified LAYERS). Mean over classes. 1 = identical circuits.")
    report("=" * 100)
    layers = LAYER_GROUPS["backbone"]
    report(f"{'K':>5s}" + "".join(f" {label.split(' (')[0]:>24s}" for label in normalizations.values()))
    for K in TOPK_LAYER_NORMALIZED_RATIO_GRID[:-1]:
        cells = []
        for normalization in normalizations:
            jaccards = []
            for per_image_scores, _ in all_classes.values():
                cumulative_saliency = per_image_scores.sum(axis=0)
                per_layer_circuit, normalized = [], []
                for name in layers:
                    layer_scores = cumulative_saliency[layer_slices[name]]
                    kept = max(1, int(np.floor(K * len(layer_scores) + 0.5)))
                    mask = np.zeros(len(layer_scores), dtype=bool)
                    mask[np.argsort(-layer_scores, kind="stable")[:kept]] = True
                    per_layer_circuit.append(mask)
                    normalized.append({"raw": layer_scores, "times_channel_count": layer_scores * len(layer_scores),
                                       "divided_by_layer_mean": layer_scores / layer_scores.mean(),
                                       "min_max_within_layer": min_max_within_layer(layer_scores)}[normalization])
                per_layer_circuit, normalized = np.concatenate(per_layer_circuit), np.concatenate(normalized)
                global_circuit = np.zeros(len(normalized), dtype=bool)
                global_circuit[np.argsort(-normalized, kind="stable")[:per_layer_circuit.sum()]] = True
                jaccards.append((global_circuit & per_layer_circuit).sum() / (global_circuit | per_layer_circuit).sum())
            cells.append(np.mean(jaccards))
        report(f"{K:5.2f}" + "".join(f" {cell:24.3f}" for cell in cells))

    # ---------------------------------------------------------------- 5. RS-Del: layer sizes become random
    report("")
    report("=" * 100)
    report("[5] Under RS-Del (the s03 deletion masks, P_DEL = 0.6, the N = 1000 certification samples), a raw global")
    report("    top-K no longer has a fixed size per layer: each sub-dataset gives its own split between layers.")
    report("    Group 'backbone', total size matched to per-layer K. Cells: share of the layer kept, mean over")
    report("    samples, and the standard deviation over samples (mean over classes of both). Under per-layer top-K")
    report("    the standard deviation is 0 by construction.")
    report("=" * 100)
    layer_index = np.concatenate([np.full(channels_per_layer[n], i) for i, n in enumerate(layers)])
    group_index = np.concatenate([np.arange(layer_slices[n].start, layer_slices[n].stop) for n in layers])
    report(f"{'K':>5s}" + "".join(f" {name + ' mean ± sd':>20s}" for name in layers))
    variability_for_plot = {}
    for K in [0.05, 0.2, 0.4]:
        number_kept = sum(max(1, int(np.floor(K * channels_per_layer[n] + 0.5))) for n in layers)
        per_class_mean, per_class_sd = [], []
        for c, (per_image_scores, _) in all_classes.items():
            deletion_masks = np.load(SAMPLE_DIRECTORY / f"class_{c:02d}.npz")["eps"]
            certification_masks = deletion_masks[100:]                       # N0 = 100 selection samples first
            totals = certification_masks.astype(np.float64) @ per_image_scores[:, group_index]
            totals = totals[certification_masks.sum(axis=1) > 0]
            top = np.argpartition(-totals, number_kept - 1, axis=1)[:, :number_kept]
            counts = np.stack([np.bincount(row, minlength=len(layers)) for row in layer_index[top]])
            shares = counts / np.array([channels_per_layer[n] for n in layers])
            per_class_mean.append(shares.mean(axis=0))
            per_class_sd.append(shares.std(axis=0))
            if K == 0.2:
                variability_for_plot[c] = shares
        per_class_mean, per_class_sd = np.array(per_class_mean), np.array(per_class_sd)
        report(f"{K:5.2f}" + "".join(f" {per_class_mean[:, l].mean():9.3f} ± {per_class_sd[:, l].mean():.3f}   "
                                     for l in range(len(layers))))

    figure, axis = plt.subplots(figsize=(9, 4.4))
    for l, name in enumerate(layers):
        values = np.concatenate([shares[:, l] for shares in variability_for_plot.values()])
        violin = axis.violinplot([values], positions=[l], widths=0.7, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(LAYER_COLORS[name])
            body.set_alpha(0.8)
            body.set_edgecolor("white")
        axis.plot([l - 0.3, l + 0.3], [0.2, 0.2], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    axis.set_xticks(range(len(layers)), layers)
    axis.set_ylabel("share of the layer's channels kept")
    axis.set_title("Raw global top-K over the backbone at total size of K = 0.2, across RS-Del samples\n"
                   "(20 classes × 1000 certification samples pooled; dashed = 0.2, what per-layer top-K keeps in every sample)",
                   fontsize=9)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "fig3_global_topk_layer_share_across_rsdel_samples.png", dpi=150)
    plt.close(figure)

    (OUTPUT_DIRECTORY / "actgrad_layer_scale_comparison_report.txt").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
