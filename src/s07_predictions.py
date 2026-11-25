"""s07 · Prediction examples: the circuits of s04 run on single eval images and drawn as figures.

A qualitative spot check, not a metric. For every class c in CLASSES and every position p in
PREDICT_POSITIONS, the image at position p of class c's s01 eval list in PREDICT_DATASET is taken (so its
GT contains class c; the same position gives a different image per class). The full model and the
circuits of class c named in PREDICT_CIRCUITS are run on it at every top-K layer-normalized ratio in
PREDICT_TOPK_LAYER_NORMALIZED_RATIO, at native resolution (the same forward pass as s05: channels of U, the
certifiable components, outside the circuit are zeroed, layers outside U untouched).
No ratio is skipped, even when the circuit is empty and its prediction degenerate.

Two figures per (class, image), with the same grid:
    <...>__classes.png   GT and predictions in the VOC palette (all 21 classes)
    <...>__binary.png    class c vs everything else (void in grey in the GT)
Panel titles give the class-c IoU on this image (vs GT, vs the raw full model) and the pixel agreement with the
raw full model: the share of all pixels with the full model's 21-class label.
Top row: PREDICT_TOP_PANELS (any of image, gt, full; independent of the ratio). Below it one row per value in
PREDICT_TOPK_LAYER_NORMALIZED_RATIO, from the largest at the top to the smallest at the bottom, with one column
per circuit type in PREDICT_CIRCUITS (any of baseline_circuit, majority, certified). Both lists also give the
column order.
Logit standardization (PREDICT_LOGIT_STANDARDIZATION_VARIANTS, statistics from s04b): next to every
circuit panel, one panel per variant with the prediction after that variant (the same forward pass). Next to the
"full" panel, one panel for the Jung variant, which changes the full model too (the matched variants are the
identity for the full model).
Confidence maps (PREDICT_CONFIDENCE_MAPS; statistics from s04b): Jung et
al.'s standardized max logit as written, S = (L − μ_Ŷ) / σ_Ŷ per pixel (the winning logit L, standardized with the
statistics of the winning class Ŷ on s04b's pixel set own_prediction, i.e. on D_c). The label is not changed; low S =
unsure. A third figure per (class, image):
    <...>__confidence.png  top row: image, the full model's prediction, the full model's own S (full model's logits
                           and statistics); one row per ratio with two panels per circuit type: the circuit's own S
                           (its logits, its statistics) and S with the full model's statistics (its logits, the full
                           model's statistics). Fixed colour range ±CONFIDENCE_MAP_RANGE; pixels whose winning class has
                           no usable statistics are dark grey.

Output: <prediction_dir>/class_XX_<name>__<image id>__{classes,binary[,confidence]}.png and run_info.json. The folder
name holds the cap of the eval lists (s07_<dataset>_Cap<MAX_EVAL_IMAGES>_pos..., CapAll for "all"), because the
positions index s01's eval lists and the cap changes them.
"""

import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config as C  # noqa: E402
from common import datasets, logit_standardization, model as M, utils  # noqa: E402
from common.datasets import NUM_CLASSES, VOC_CLASSES, VOID_LABEL  # noqa: E402

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "IMAGES_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "MAX_EVAL_IMAGES", "REQUIRE_PRED_IN_DISCOVERY",
    "DISCOVERY_AND_EVAL_SEED", "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU",
    "FAILURE_PROB_ALPHA", "PREDICT_DATASET", "PREDICT_POSITIONS", "PREDICT_TOPK_LAYER_NORMALIZED_RATIO",
    "PREDICT_TOP_PANELS", "PREDICT_CIRCUITS", "PREDICT_LOGIT_STANDARDIZATION_VARIANTS", "PREDICT_CONFIDENCE_MAPS",
    "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS", "SMOKE_TEST",
]
STATIC_PANELS = ["image", "gt", "full"]  # allowed in PREDICT_TOP_PANELS (top row, independent of the ratio)
CIRCUIT_PANELS = ["baseline_circuit", "majority", "certified"]  # allowed in PREDICT_CIRCUITS (one column per ratio row)
# Binary figure colours: class c, every other class, void.
NOT_C, VOID_GREY = np.array([255, 255, 255], dtype=np.uint8), np.array([190, 190, 190], dtype=np.uint8)
# Confidence maps: S on a fixed range (our display choice; values beyond it saturate), a diverging map with a neutral
# grey at S = 0 (the class's typical winning logit on D_c): red = low S (unsure), blue = high S. No statistics: dark grey.
CONFIDENCE_MAP_RANGE = 4.0
CONFIDENCE_COLORMAP = LinearSegmentedColormap.from_list(
    "confidence_red_blue", ["#8f2223", "#e34948", "#f3b5b4", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
CONFIDENCE_COLORMAP.set_bad("#52514e")


def iou(pred_c, ref_c, valid=None):
    """Class-c IoU of two boolean masks on one image (nan if neither contains c)."""
    if valid is not None:
        pred_c, ref_c = pred_c & valid, ref_c & valid
    union = (pred_c | ref_c).sum()
    return (pred_c & ref_c).sum() / union if union > 0 else np.nan


def binary_rgb(is_c, c, palette, void=None):
    rgb = np.where(is_c[..., None], palette[c], NOT_C)
    if void is not None:
        rgb[void] = VOID_GREY
    return rgb


def check_config(discovery_and_eval_lists):
    for name, given, allowed in (("PREDICT_TOP_PANELS", C.PREDICT_TOP_PANELS, STATIC_PANELS),
                                 ("PREDICT_CIRCUITS", C.PREDICT_CIRCUITS, CIRCUIT_PANELS)):
        unknown = [p for p in given if p not in allowed]
        if unknown:
            raise SystemExit(f"\n[config] unknown {name} {unknown}; allowed: {allowed}")
    if C.PREDICT_DATASET not in discovery_and_eval_lists["eval"]:
        raise SystemExit(f"\n[missing input] no eval lists for PREDICT_DATASET = {C.PREDICT_DATASET!r} in "
                         f"discovery_and_eval_lists.json (it has {list(discovery_and_eval_lists['eval'])})\n"
                         f"-> add it to EVAL_DATASETS and re-run s01_discovery_samples_dataset.py.")
    if any(p < 0 for p in C.PREDICT_POSITIONS):
        raise SystemExit(f"\n[config] PREDICT_POSITIONS must be >= 0 (0-based), got {C.PREDICT_POSITIONS}")
    logit_standardization.check_variants(C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS,
                                         "PREDICT_LOGIT_STANDARDIZATION_VARIANTS")
    if utils.needs_logit_statistics_in_predictions():
        utils.require(utils.logit_statistics_dir() / "run_info.json", "s04b_logit_statistics.py")


def agreement(pred, full):
    """Share of all pixels whose 21-class label equals the raw full model's (pixel_agreement_with_full_model)."""
    return float((pred == full).mean())


def confidence_summary(score):
    """Mean S over the pixels with statistics, and the share of pixels without (nan)."""
    finite = np.isfinite(score)
    return (float(score[finite].mean()) if finite.any() else np.nan), float(1 - finite.mean())


def grid_indices(topk_layer_normalized_ratio_grid):
    """Grid index of every PREDICT_TOPK_LAYER_NORMALIZED_RATIO value in s04's top-K layer-normalized ratio grid,
    ordered from the largest ratio to the smallest (the order of the figure rows and the log lines); stops if
    one is missing."""
    found = []
    for topk_layer_normalized_ratio in C.PREDICT_TOPK_LAYER_NORMALIZED_RATIO:
        hit = np.flatnonzero(np.isclose(topk_layer_normalized_ratio_grid, topk_layer_normalized_ratio))
        if len(hit) == 0:
            raise SystemExit(f"\n[config] PREDICT_TOPK_LAYER_NORMALIZED_RATIO value {topk_layer_normalized_ratio} is "
                             f"not in the top-K layer-normalized ratio grid of s04 "
                             f"({[float(value) for value in topk_layer_normalized_ratio_grid]}).")
        found.append(int(hit[0]))
    return sorted(set(found), key=lambda grid_idx: -topk_layer_normalized_ratio_grid[grid_idx])


def predictions(logits, statistics, full_statistics):
    """{variant: [H, W] prediction}: "raw" and every PREDICT_LOGIT_STANDARDIZATION_VARIANTS entry."""
    out = {"raw": logits.argmax(dim=0).cpu().numpy()}
    for v in C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS:
        out[v] = logit_standardization.adjust_logits(logits, v, statistics, full_statistics).argmax(dim=0).cpu().numpy()
    return out


def predict_image(model, image, mask, c, circuits, subnet_size_layer_normalized_ratio_mean, grid_idxs, device,
                  logit_statistics):
    """Full-model prediction, and per (circuit type, grid index) the prediction and its numbers. With
    PREDICT_LOGIT_STANDARDIZATION_VARIANTS also ("full", variant) and (circuit type, grid index, variant)."""
    full_statistics, circuit_statistics = logit_statistics
    x = M.preprocess(image, device)
    full_logits = M.forward(model, x)
    full_predictions = predictions(full_logits, full_statistics, full_statistics)
    full = full_predictions["raw"]
    valid, gt_c, full_c = mask != VOID_LABEL, mask == c, full == c
    out = {"full": {"pred": full, "iou_gt": iou(full_c, gt_c, valid)}}
    if C.PREDICT_CONFIDENCE_MAPS:
        out["full"]["confidence"] = logit_standardization.standardized_max_logit(full_logits, full_statistics)
    for v in C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS:
        pred = full_predictions[v]
        out[("full", v)] = {"pred": pred, "iou_gt": iou(pred == c, gt_c, valid), "iou_full": iou(pred == c, full_c),
                            "agreement": agreement(pred, full)}
    cache = {}
    for t, circuits_per_grid_idx in circuits.items():
        for grid_idx in grid_idxs:
            masks, key = circuits_per_grid_idx[grid_idx]
            if key not in cache:
                statistics = circuit_statistics[t][grid_idx] if circuit_statistics else None
                logits = M.forward(model, x, masks)
                cache[key] = (predictions(logits, statistics, full_statistics),
                              {"confidence_own": logit_standardization.standardized_max_logit(logits, statistics),
                               "confidence_with_full_model_statistics":
                                   logit_standardization.standardized_max_logit(logits, full_statistics)}
                              if C.PREDICT_CONFIDENCE_MAPS else {})
            circuit_predictions, confidence = cache[key]
            for v, pred in circuit_predictions.items():
                out[(t, grid_idx) if v == "raw" else (t, grid_idx, v)] = {
                    "pred": pred,
                    "iou_gt": iou(pred == c, gt_c, valid),
                    "iou_full": iou(pred == c, full_c),
                    "agreement": agreement(pred, full),
                    "kept": int(sum(m.sum() for m in masks.values())),
                    "empty": [name for name, m in masks.items() if not m.any()],
                    "subnet_size_layer_normalized_ratio_mean":
                        float(subnet_size_layer_normalized_ratio_mean[t][grid_idx]),
                    **(confidence if v == "raw" else {}),
                }
    return out


def draw(kind, image, mask, c, results, grid_idxs, topk_layer_normalized_ratio_grid, n_certifiable, title, palette):
    """One figure. Top row: the panels that do not depend on the top-K layer-normalized ratio (image, gt, full);
    below it one row per ratio with the circuit panels. PREDICT_TOP_PANELS and PREDICT_CIRCUITS give the column
    order; columns = the longer of the two. With PREDICT_LOGIT_STANDARDIZATION_VARIANTS, every circuit panel is
    followed by one panel per variant, and the "full" panel by one for the Jung variant."""
    variants = list(C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS)
    static, circuit = [], []  # (panel, variant): variant None = raw
    for p in C.PREDICT_TOP_PANELS:
        static += [(p, None)] + [(p, v) for v in variants
                                 if p == "full" and v not in logit_standardization.MATCHED_VARIANTS]
    for p in C.PREDICT_CIRCUITS:
        circuit += [(p, None)] + [(p, v) for v in variants]
    cells = [(0, j, p, None, v) for j, (p, v) in enumerate(static)]  # (row, column, panel, grid index, variant)
    first = 1 if static else 0
    cells += [(first + r, j, p, grid_idx, v) for r, grid_idx in enumerate(grid_idxs if circuit else [])
              for j, (p, v) in enumerate(circuit)]
    rows, cols = first + (len(grid_idxs) if circuit else 0), max(len(static), len(circuit))
    width, height = image.size
    panel_w = 4.0
    panel_h = panel_w * height / width + 1.05 + (0.15 if variants else 0)  # room for the panel titles
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_h * rows + 0.6), squeeze=False)
    for ax in axes.flat:  # cells without a panel stay blank
        ax.axis("off")
    void = mask == VOID_LABEL
    shown = set(np.unique(mask[~void]).tolist())

    def colour(pred, gt=False):
        if kind == "classes":
            return palette[pred]
        return binary_rgb(pred == c, c, palette, void if gt else None)

    for r, j, panel, grid_idx, variant in cells:
        ax = axes[r, j]
        ax.axis("on")
        if panel == "image":
            ax.imshow(image)
            ax.set_title("image")
        elif panel == "gt":
            ax.imshow(colour(mask, gt=True))
            ax.set_title("ground truth" if kind == "classes" else f"ground truth: {VOC_CLASSES[c]} vs rest")
        elif panel == "full" and variant is None:
            res = results["full"]
            shown |= set(np.unique(res["pred"]).tolist())
            ax.imshow(colour(res["pred"]))
            ax.set_title(f"full model\nIoU_c vs GT {res['iou_gt']:.3f}")
        elif panel == "full":
            res = results[("full", variant)]
            shown |= set(np.unique(res["pred"]).tolist())
            ax.imshow(colour(res["pred"]))
            ax.set_title(f"full model\nafter {variant}\nIoU_c vs GT {res['iou_gt']:.3f} · vs full {res['iou_full']:.3f}"
                         f"\nagreement with full {res['agreement']:.3f}")
        else:
            res = results[(panel, grid_idx) if variant is None else (panel, grid_idx, variant)]
            shown |= set(np.unique(res["pred"]).tolist())
            ax.imshow(colour(res["pred"]))
            text = (f"{panel}\n" + ("" if variant is None else f"after {variant}\n") +
                    f"{utils.topk_ratio_label()} {topk_layer_normalized_ratio_grid[grid_idx]:g} · "
                    f"subnetwork size {res['subnet_size_layer_normalized_ratio_mean']:.3f}\n"
                    f"{res['kept']:,}/{n_certifiable:,} channels of U · IoU_c vs GT {res['iou_gt']:.3f}"
                    f" · vs full {res['iou_full']:.3f}\nagreement with full {res['agreement']:.3f}")
            if res["empty"]:
                text += f"\nEMPTY layers (all channels zeroed): {', '.join(res['empty'])}"
            ax.set_title(text, color="#b00020" if res["empty"] else "black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_fontsize(8)

    if kind == "classes":
        handles = [Patch(color=palette[v] / 255, label=f"{v} {VOC_CLASSES[v]}")
                   for v in sorted(shown) if v < NUM_CLASSES]
        if void.any():
            handles.append(Patch(color=palette[VOID_LABEL] / 255, label="void (GT only)"))
    else:
        handles = [Patch(color=palette[c] / 255, label=f"{c} {VOC_CLASSES[c]}"),
                   Patch(facecolor=NOT_C / 255, edgecolor="black", label="every other class"),
                   Patch(color=VOID_GREY / 255, label="void (GT only)")]
    fig_h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig_h))  # keep the top 0.6 inch for the two-line suptitle
    fig.suptitle(title, fontsize=10, y=1 - 0.1 / fig_h, va="top")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 7),
               frameon=False, fontsize=8)
    return fig


def draw_confidence(image, results, grid_idxs, topk_layer_normalized_ratio_grid, title, palette):
    """The confidence-map figure (PREDICT_CONFIDENCE_MAPS). Top row: the image, the full model's prediction and its own
    S; below it one row per ratio with, per circuit type in PREDICT_CIRCUITS, the circuit's own S and S with the full
    model's statistics."""
    types = list(C.PREDICT_CIRCUITS)
    rows, cols = 1 + (len(grid_idxs) if types else 0), max(3, 2 * len(types))
    width, height = image.size
    panel_w = 4.0
    panel_h = panel_w * height / width + 0.9
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_h * rows + 1.2), squeeze=False)
    for ax in axes.flat:  # cells without a panel stay blank
        ax.axis("off")

    def panel(ax, text):
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(text, fontsize=8)

    def score_panel(ax, score, text):
        mapped = ax.imshow(np.ma.masked_invalid(score), cmap=CONFIDENCE_COLORMAP, vmin=-CONFIDENCE_MAP_RANGE,
                           vmax=CONFIDENCE_MAP_RANGE)
        mean, without = confidence_summary(score)
        panel(ax, f"{text}\nmean S {mean:.2f} · no statistics {without:.1%} of pixels")
        return mapped

    axes[0, 0].imshow(image)
    panel(axes[0, 0], "image")
    axes[0, 1].imshow(palette[results["full"]["pred"]])
    panel(axes[0, 1], "full model: prediction")
    mapped = score_panel(axes[0, 2], results["full"]["confidence"], "full model: own S\n(its logits, its statistics)")
    for r, grid_idx in enumerate(grid_idxs if types else [], start=1):
        ratio = f"{utils.topk_ratio_label()} {topk_layer_normalized_ratio_grid[grid_idx]:g}"
        for j, t in enumerate(types):
            res = results[(t, grid_idx)]
            score_panel(axes[r, 2 * j], res["confidence_own"], f"{t} · {ratio}\nown S (its logits, its statistics)")
            score_panel(axes[r, 2 * j + 1], res["confidence_with_full_model_statistics"],
                        f"{t} · {ratio}\nS with the FULL MODEL's statistics (its logits)")
    fig_h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.7 / fig_h, 1, 1 - 0.6 / fig_h))
    fig.suptitle(title, fontsize=10, y=1 - 0.1 / fig_h, va="top")
    bar = fig.colorbar(mapped, cax=fig.add_axes([0.3, 0.25 / fig_h, 0.4, 0.12 / fig_h]), orientation="horizontal")
    bar.set_label("S = (winning logit − μ_Ŷ) / σ_Ŷ, statistics on D_c (s04b, own_prediction): red = low = unsure, "
                  "blue = high; dark grey = the winning class has no usable statistics", fontsize=8)
    return fig


def main():
    utils.start_run("s07_predictions", USED_PARAMS)
    utils.require(utils.certify_dir() / "run_info.json", "s04_certify.py")
    discovery_and_eval_lists = utils.load_discovery_and_eval_lists()
    utils.check_eval_list_cap(discovery_and_eval_lists)
    check_config(discovery_and_eval_lists)
    out_dir = utils.prediction_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)
    dataset = utils.eval_dataset(C.PREDICT_DATASET)
    eval_lists = discovery_and_eval_lists["eval"][C.PREDICT_DATASET]
    palette = datasets.voc_palette()
    _, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    n_certifiable = sum(s.stop - s.start for s in certifiable_slices.values())
    types = list(C.PREDICT_CIRCUITS)
    ratio_column = utils.topk_ratio_label()

    utils.section(
        f"Prediction examples on {C.PREDICT_DATASET}: circuit types {types or 'none'}, {ratio_column} values "
        f"{list(C.PREDICT_TOPK_LAYER_NORMALIZED_RATIO)}",
        textwrap.dedent(f"""
        Images  for every class c: position(s) {C.PREDICT_POSITIONS} (0-based) of c's s01 eval list in
                {C.PREDICT_DATASET}, so the GT of every image contains c. The same position is a DIFFERENT
                image for each class. {"These are the D_c images the circuits were discovered on (in-sample)." if C.PREDICT_DATASET == "discovery_samples" else ""}
        Circuit the full model with the channels of U (the certifiable components, LAYERS = {C.LAYERS!r},
                {n_certifiable:,} channels) outside the circuit of class c set to zero; layers outside U are never
                masked (as in s05).
        {ratio_column}
{textwrap.fill(utils.topk_ratio_description() + '.', 104, initial_indent=" " * 16, subsequent_indent=" " * 16)}
        size    the subnetwork size, layer-normalized ratio (subnet_size_layer_normalized_ratio_mean): share of
                each layer's channels kept, equal-weight mean over the layers of U (s04).
        kept    channels of U in the circuit, out of {n_certifiable:,}.
        IoU_c   class-c IoU on THIS ONE IMAGE: "vs GT" over the non-void pixels, "vs full" against the full
                model's class-c prediction over all pixels. nan = neither mask contains c. This is NOT s05's
                cIoU, which pools the pixels of all eval images of c. Both look at class c only.
        agreement
                share of ALL pixels of this image whose 21-class label equals the RAW full model's label
                (pixel_agreement_with_full_model, as in s05 but for one image): what the circuit does to every
                other class too. 1 = the full model's segmentation.
        EMPTY   a layer of U without any kept channel: its whole output is zeroed, so the prediction is
                close to constant (often all background). A correct result, not a bug; it is drawn anyway.
        Every ratio is run and drawn, also when the prediction is degenerate.
        """) + ("" if not C.PREDICT_CONFIDENCE_MAPS else textwrap.dedent(f"""
        Confidence maps (PREDICT_CONFIDENCE_MAPS; Jung et al.'s standardized max logit as written):
        per pixel S = (L − μ_Ŷ) / σ_Ŷ, L = the winning logit, Ŷ = the winning class, μ and σ = class Ŷ's statistics
        on s04b's pixel set own_prediction (fitted on D_c, not on this image). The label is NOT changed; low S =
        less confident than the class is on D_c. Third figure (__confidence): the full model's own S; per circuit
        its own S (its statistics) and S with the full model's statistics. Log: mean S over the pixels with
        statistics, and the share of pixels whose winning class has no usable statistics (S undefined).
        """)) + ("" if not C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS else textwrap.dedent(f"""
        Logit standardization (PREDICT_LOGIT_STANDARDIZATION_VARIANTS, statistics from s04b on
        LOGIT_STATISTICS_IMAGES = {C.LOGIT_STATISTICS_IMAGES!r}, i.e. D_c, not this image): after each circuit's raw
        lines, one block per variant, with the prediction after that variant from the same forward pass. In the
        figures, every circuit panel is followed by one panel per variant ("after <variant>" in the title).
        "vs full" is always against the RAW full model. The full model after the Jung variant gets its own line
        and a panel next to "full"; the matched variants are the identity for the full model.
        """) + "\n".join(f"  {v}\n" + textwrap.indent(logit_standardization.VARIANT_DESCRIPTIONS[v], "      ")
                         for v in C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS)))

    n_figures = 0
    for c in C.CLASSES:
        circuits, subnet_size_layer_normalized_ratio_mean, topk_layer_normalized_ratio_grid = \
            utils.load_circuits(c, certifiable_slices, types)
        logit_statistics = (logit_standardization.load_class_logit_statistics(c, circuits)
                            if utils.needs_logit_statistics_in_predictions() else (None, None))
        grid_idxs = grid_indices(topk_layer_normalized_ratio_grid)
        entry = eval_lists[str(c)]
        n_list = len(entry["indices"])
        for pos in C.PREDICT_POSITIONS:
            if pos >= n_list:
                print(f"\nclass {c:2d} {VOC_CLASSES[c]}: NOTE position {pos} skipped, the eval list holds only "
                      f"{n_list} images (positions 0-{n_list - 1}).")
                continue
            i, image_id = entry["indices"][pos], entry["ids"][pos]
            image, mask = dataset[i]
            results = predict_image(model, image, mask, c, circuits, subnet_size_layer_normalized_ratio_mean,
                                    grid_idxs, device, logit_statistics)

            print(f"\nclass {c:2d} {VOC_CLASSES[c]} | {C.PREDICT_DATASET} image {image_id} (dataset index {i}, "
                  f"eval-list position {pos} of {n_list}) | {image.size[0]}x{image.size[1]} px, "
                  f"{int((mask == c).sum()):,} GT pixels of class c")
            print(f"  full model: IoU_c vs GT {results['full']['iou_gt']:.3f}, "
                  f"{int((results['full']['pred'] == c).sum()):,} predicted class-c pixels")
            if types:
                print(f"  {'circuit':16s} {ratio_column} {'size':>6s} {'kept':>12s} {'IoU_c GT':>9s} {'vs full':>8s} "
                      f"{'agreement':>9s}  empty layers")
            for variant in [None, *C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS]:
                if variant is not None:
                    print(f"  logit standardization variant {variant}:")
                    if variant not in logit_standardization.MATCHED_VARIANTS:
                        res = results[("full", variant)]
                        print(f"  full model after this variant: IoU_c vs GT {res['iou_gt']:.3f}, vs full "
                              f"{res['iou_full']:.3f}, agreement {res['agreement']:.3f}, "
                              f"{int((res['pred'] == c).sum()):,} predicted class-c pixels")
                for t in types:
                    for grid_idx in grid_idxs:
                        res = results[(t, grid_idx) if variant is None else (t, grid_idx, variant)]
                        print(f"  {t:16s} {topk_layer_normalized_ratio_grid[grid_idx]:{len(ratio_column)}g} "
                              f"{res['subnet_size_layer_normalized_ratio_mean']:6.3f} {res['kept']:>5,}/{n_certifiable:<6,} "
                              f"{res['iou_gt']:9.3f} {res['iou_full']:8.3f} {res['agreement']:9.3f}  "
                              f"{', '.join(res['empty']) or '-'}")
            if C.PREDICT_CONFIDENCE_MAPS:
                mean, without = confidence_summary(results["full"]["confidence"])
                print(f"  confidence maps (S): full model own S: mean {mean:.2f}, no statistics {without:.1%} of pixels")
                for t in types:
                    for grid_idx in grid_idxs:
                        res = results[(t, grid_idx)]
                        own, own_without = confidence_summary(res["confidence_own"])
                        full_s, full_without = confidence_summary(res["confidence_with_full_model_statistics"])
                        print(f"  {t:16s} {topk_layer_normalized_ratio_grid[grid_idx]:{len(ratio_column)}g} "
                              f"own S: mean {own:6.2f}, no statistics {own_without:6.1%} | with the full model's "
                              f"statistics: mean {full_s:6.2f}, no statistics {full_without:6.1%}")

            title = (f"class {c} {VOC_CLASSES[c]} · {C.PREDICT_DATASET} image {image_id} (eval-list position {pos})\n"
                     f"certifiable components U = {C.LAYERS} · top-K selection {C.TOPK_SELECTION_MODE}"
                     f"{f' over {C.LAYERS}' if utils.topk_selection_is_global() else ''} · "
                     f"τ = {C.AGREEMENT_THRESHOLD_TAU} · p_del = {C.P_DEL} · "
                     f"N = {C.N} · target {C.TARGET} · scoring method {C.CHANNEL_SCORING_METHOD}")
            for kind in ("classes", "binary") + (("confidence",) if C.PREDICT_CONFIDENCE_MAPS else ()):
                fig = (draw_confidence(image, results, grid_idxs, topk_layer_normalized_ratio_grid, title, palette)
                       if kind == "confidence" else
                       draw(kind, image, mask, c, results, grid_idxs, topk_layer_normalized_ratio_grid, n_certifiable,
                            title, palette))
                path = out_dir / f"class_{c:02d}_{VOC_CLASSES[c]}__{image_id}__{kind}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                n_figures += 1
                print(f"  saved {path.name}")

    utils.write_run_info(out_dir)
    print(f"\nSaved {n_figures} figures and run_info.json to {out_dir}")


if __name__ == "__main__":
    main()
