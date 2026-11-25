"""s05 · Evaluation of the circuits: accuracy (sufficiency), faithfulness, logit distributions and size (groups 1-4).

For every eval dataset in EVAL_DATASETS, class c, image set, top-K layer-normalized ratio K and circuit type in
CIRCUIT_TYPES, the model runs with only the circuit's channels (ChannelMasker: channels of U = the certifiable
components outside the circuit are zeroed, layers outside U are untouched):
    full              the unpruned model (independent of K)
    baseline_circuit  A(D_c) on the whole discovery set
    majority          channels selected in > 50 % of the N samples
    certified         certified-in channels (abstaining channels are left out)

Image sets (s01's lists), reported apart and never pooled:
    images_with_class_c     the s01 eval list of c: images whose GT contains c, every one (MAX_EVAL_IMAGES = "all") or
                            at most MAX_EVAL_IMAGES; on "discovery_samples" D_c itself
    images_without_class_c  only with EVALUATE_IMAGES_WITHOUT_CLASS: every image whose GT has no c (no cap); on
                            "discovery_samples" the other classes' D_c' without the images that contain c

Metrics per eval dataset, class, image set, circuit and K, pixels pooled over the images of the set:
  group 1, accuracy against the ground truth (void excluded):
    cIoU     class-c IoU against the GT (the segmentation analogue of Anani's cACC). On the images without c it is 0,
             or nan if c is never predicted: every predicted c-pixel is a false positive there.
    mIoU_vs_ground_truth                 mean IoU of all 21 classes (background included) against the GT
    class_c_predicted_pixel_ratio        share of the non-void pixels predicted as c; on the images without c these
                                         are the false positives of c
    images_with_class_c_predicted_ratio  share of the images with at least one non-void pixel predicted as c
  group 2, faithfulness to the full model (all pixels; the reference is always the RAW full model):
    faith    class-c IoU against the full model's own prediction
    |r|      Hamblin-style |Pearson r| between f_pred(x) of the full model and of the circuit, where
             f_pred = mean class-c logit over the pixels the full model predicts as c. Images on which
             the full model predicts no class-c pixel are left out; their number is num_no_pred.
             With fewer than 3 remaining images |r| is nan.
    pixel_agreement_with_full_model            share of pixels with the full model's 21-class label
    class_share_total_variation_to_full_model  ½ Σ_d |s_d^circuit − s_d^full|, s_d = share of pixels predicted as d
  group 3, logit distributions: per class d (all 21) and pixel set (the pixels the RAW full model predicts as
    d; all pixels), the circuit's class-d logit against the full model's after the same variant (raw and the matched
    variants: the raw full model; the Jung variant: the standardized full model): mean shift μ_d^circuit − μ_d^full,
    standard deviation ratio σ_d^circuit / σ_d^full and, with STORE_LOGIT_HISTOGRAMS, the Wasserstein-1 distance
    from fixed-bin histograms. summary.csv holds class c and background; every class d is in the npz files.
  group 4, size (common/circuit_size.py; independent of the eval data):
    subnet_size_layer_normalized_ratio_mean (from s04) and _<layer> per layer of U, the pooled sizes over all channels
    of U subnet_size_group_channel_count (absolute) and subnet_size_group_pooled_ratio (relative),
    subnet_size_kept_weight_count, subnet_size_whole_model_weight_ratio, subnet_size_reachable_weight_ratio
    (binary-mask convention)
K is the top-K layer-normalized ratio with TOPK_SELECTION_MODE = "per_layer" and the share of the whole LAYERS group
kept with a global mode (s03); the logs name it accordingly (utils.topk_ratio_label). The evaluation itself is the
same for every mode.
Reference: the full model's per-class IoU and mIoU on every eval dataset (all of its images; for
"discovery_samples", and for every eval set under SMOKE_TEST, the union of the class eval lists).

Logit standardization (LOGIT_STANDARDIZATION_VARIANTS, statistics from s04b): from the same
forward pass of every circuit, the prediction is also taken after each variant (common/logit_standardization.py),
and every metric above is computed for "raw" (no adjustment) and for each variant. The full model gets the same
variants (the identity for the matched variants; standardized for the Jung variant). faith, |r|, num_no_pred, the
pixel agreement and the class-share total variation always use the RAW full model as the reference, and the mIoU
reference is the raw full model only.

Circuits with identical masks (e.g. all channels at K = 1.0) are run only once per image.

Output in <eval_dir> (s05_eval_Cap<MAX_EVAL_IMAGES>, s05_eval_CapAll for "all"):
    eval_lists_used.json    the image lists this run evaluated (s01's lists with c; the lists without c as used), so the
                            folder records its own images after s01 is re-run with another cap
    metrics_class_XX.json   per eval dataset: the images with c at the top level ("circuits": raw,
                            "circuits_per_logit_standardization_variant") and "images_without_class_c" (same layout)
    summary.csv             one row per class, eval dataset, image set, circuit, variant and K; columns by group
    reference.json, run_info.json
    confusion_matrices_and_logit_histograms/<eval dataset>/class_XX_<image set>.npz   ("/" in the name -> "_"):
                            the counts every metric above follows from (layout: save_counts), so later metrics need
                            no new s05 run
"""

import csv
import textwrap
import time
import warnings

import numpy as np
import torch

import config as C
from common import circuit_size, logit_standardization, metrics, model as M, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES, VOID_LABEL

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "EVAL_DATASETS", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "MAX_EVAL_IMAGES",
    "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA",
    "LOGIT_STANDARDIZATION_VARIANTS", "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS",
    "CIRCUIT_TYPES", "EVALUATE_IMAGES_WITHOUT_CLASS", "STORE_LOGIT_HISTOGRAMS", "LOGIT_HISTOGRAM_LOWER_EDGE",
    "LOGIT_HISTOGRAM_UPPER_EDGE", "LOGIT_HISTOGRAM_BIN_WIDTH", "SMOKE_TEST",
]
PRUNED_TYPES = ["baseline_circuit", "majority", "certified"]
IMAGE_SETS = ["images_with_class_c", "images_without_class_c"]
IMAGE_SET_TITLES = {"images_with_class_c": "images WITH class c", "images_without_class_c": "images WITHOUT class c"}
DISTRIBUTION_PIXEL_SETS = logit_standardization.DISTRIBUTION_PIXEL_SETS
PIXEL_SET_IN_COLUMN_NAME = {"full_model_prediction": "full_model_prediction_pixels", "all_pixels": "all_pixels"}
RATIO_COLUMN = utils.topk_ratio_label()  # the name of K: "top-K layer-normalized ratio" (per_layer) or "top-K global ratio of <LAYERS>"


def distribution_columns():
    """summary.csv columns of group 3 (class c and background, both pixel sets)."""
    measures = ["mean_shift", "standard_deviation_ratio"] + (["wasserstein_1"] if C.STORE_LOGIT_HISTOGRAMS else [])
    return [f"{label}_logit_{measure}_on_{PIXEL_SET_IN_COLUMN_NAME[p]}"
            for label in ("class_c", "background") for p in DISTRIBUTION_PIXEL_SETS for measure in measures]


def summary_columns(layers):
    """summary.csv columns: keys, then the metrics ordered by evaluation group (1 accuracy vs GT, 2 faithfulness,
    3 logit distributions, 4 size)."""
    return (["class", "class_name", "eval_dataset", "image_set", "circuit", "logit_standardization_variant",
             "topk_layer_normalized_ratio", "num_images", "num_no_pred"]
            + ["cIoU", "mIoU_vs_ground_truth", "class_c_predicted_pixel_ratio", "images_with_class_c_predicted_ratio"]
            + ["faith_IoU", "pearson_abs", "pixel_agreement_with_full_model", "class_share_total_variation_to_full_model"]
            + distribution_columns()
            + ["subnet_size_layer_normalized_ratio_mean"] + [f"subnet_size_layer_normalized_ratio_{layer}" for layer in layers]
            + ["subnet_size_group_channel_count", "subnet_size_group_pooled_ratio"]
            + ["subnet_size_kept_weight_count", "subnet_size_whole_model_weight_ratio", "subnet_size_reachable_weight_ratio"])


def reference_indices(name, eval_lists, dataset):
    """Images for the full-model reference: the whole eval dataset; for "discovery_samples" (and for every
    eval set under SMOKE_TEST) the union of the class eval lists."""
    if name == "discovery_samples" or C.SMOKE_TEST:
        return sorted({i for entry in eval_lists.values() for i in entry["indices"]})
    return range(len(dataset))


def without_class_entry(entry):
    """s01's list of the images without c; under SMOKE_TEST only its first MAX_EVAL_IMAGES images (a fast
    functionality check, like the reference over the eval-list images; with "all", every image). Real runs use every
    image."""
    if not C.SMOKE_TEST:
        return entry
    cap = utils.eval_image_cap(C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES")  # None ("all"): the slice keeps everything
    return {**entry, "indices": entry["indices"][:cap], "ids": entry["ids"][:cap]}


def histogram_bin_count():
    """Bins per logit histogram (inner bins + underflow + overflow), 0 without STORE_LOGIT_HISTOGRAMS."""
    if not C.STORE_LOGIT_HISTOGRAMS:
        return 0
    return logit_standardization.histogram_inner_bin_count(
        C.LOGIT_HISTOGRAM_LOWER_EDGE, C.LOGIT_HISTOGRAM_UPPER_EDGE, C.LOGIT_HISTOGRAM_BIN_WIDTH) + 2


def new_accumulator(n_images, device, accuracy_only=False):
    """Zeroed sums of one model (one mask) after one variant over the images of one image set:
        counts        [tp, fp, fn] of class c against the GT (valid pixels) + [tp, fp, fn] against the full model
        f_pred        per image (nan where the full model predicts no c-pixel)
        confusion_matrix_vs_ground_truth / _vs_full_model   [21, 21], rows = GT / raw full model, columns = model
        images_with_class_c_predicted                       number of images with ≥ 1 non-void pixel predicted as c
        logit_sums        {pixel set: logit_standardization.new_sums}       (the two DISTRIBUTION_PIXEL_SETS)
        logit_histograms  {pixel set: [21, bins] int64}, only with STORE_LOGIT_HISTOGRAMS
    accuracy_only (s05c): only the sums of group 1 (accuracy against the GT): counts [tp, fp, fn] against the GT, the
    vs-GT confusion matrix, images_with_class_c_predicted, and num_images (no f_pred to count them from)."""
    if accuracy_only:
        return {
            "counts": np.zeros(3, dtype=np.int64),
            "num_images": n_images,
            "confusion_matrix_vs_ground_truth": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64,
                                                            device=device),
            "images_with_class_c_predicted": 0,
        }
    n_bins = histogram_bin_count()
    return {
        "counts": np.zeros(6, dtype=np.int64),
        "f_pred": np.full(n_images, np.nan),
        "confusion_matrix_vs_ground_truth": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device),
        "confusion_matrix_vs_full_model": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device),
        "images_with_class_c_predicted": 0,
        "logit_sums": logit_standardization.new_sums(NUM_CLASSES, device, DISTRIBUTION_PIXEL_SETS),
        "logit_histograms": {p: torch.zeros((NUM_CLASSES, n_bins), dtype=torch.int64, device=device)
                             for p in DISTRIBUTION_PIXEL_SETS} if n_bins else None,
    }


@torch.inference_mode()
def add_image(accumulator, j, logits, c, gt, valid, gt_c, full_labels, full_c, accuracy_only=False):
    """Adds image j of the image set to `accumulator` (new_accumulator): one model's logits after one variant.
    gt: [H, W] GT labels (255 = void); full_labels: the RAW full model's prediction on the same image.
    accuracy_only: the accumulator of new_accumulator(accuracy_only=True); only its group-1 sums are added."""
    pred = logits.argmax(dim=0)
    pred_c = pred == c
    if accuracy_only:
        accumulator["counts"] += torch.stack([
            (pred_c & gt_c & valid).sum(), (pred_c & ~gt_c & valid).sum(), (~pred_c & gt_c & valid).sum(),
        ]).cpu().numpy()
        accumulator["confusion_matrix_vs_ground_truth"] += metrics.confusion_matrix_torch(pred, gt)
        accumulator["images_with_class_c_predicted"] += int((pred_c & valid).any())
        return
    accumulator["counts"] += torch.stack([
        (pred_c & gt_c & valid).sum(), (pred_c & ~gt_c & valid).sum(), (~pred_c & gt_c & valid).sum(),
        (pred_c & full_c).sum(), (pred_c & ~full_c).sum(), (~pred_c & full_c).sum(),
    ]).cpu().numpy()
    f_pred = logits[c][full_c].mean().item() if full_c.any() else np.nan
    if not np.isfinite(f_pred):  # −∞: the Jung variant left class c out of the argmax
        f_pred = np.nan
    accumulator["f_pred"][j] = f_pred
    accumulator["confusion_matrix_vs_ground_truth"] += metrics.confusion_matrix_torch(pred, gt)
    accumulator["confusion_matrix_vs_full_model"] += metrics.confusion_matrix_torch(pred, full_labels)
    accumulator["images_with_class_c_predicted"] += int((pred_c & valid).any())
    for p, labels in (("full_model_prediction", full_labels), ("all_pixels", None)):
        logit_standardization.accumulate_class_logit_statistics(logits, labels, accumulator["logit_sums"][p])
        if accumulator["logit_histograms"] is not None:
            logit_standardization.accumulate_class_logit_histograms(
                logits, labels, accumulator["logit_histograms"][p], C.LOGIT_HISTOGRAM_LOWER_EDGE,
                C.LOGIT_HISTOGRAM_BIN_WIDTH)


def accumulator_to_numpy(accumulator):
    def convert(value):
        if torch.is_tensor(value):
            return value.cpu().numpy()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value
    return {key: convert(value) for key, value in accumulator.items()}


@torch.inference_mode()
def evaluate_class(model, dataset, indices, c, circuits, full_key, device, logit_statistics,
                   logit_standardization_variants=None, accuracy_only=False):
    """Runs the full model and every distinct circuit of class c on the images `indices` and returns their
    accumulators {(mask key, variant): new_accumulator (numpy)}; variant "raw" = no logit standardization.
    logit_statistics: s04b's (full model, circuits) statistics of class c, or (None, None) when no variant is used.
    A circuit that keeps every channel has the full model's key.
    logit_standardization_variants: the variants besides "raw" (None = LOGIT_STANDARDIZATION_VARIANTS; s05c passes
    its own). accuracy_only: group-1 sums only (new_accumulator)."""
    full_statistics, circuit_statistics = logit_statistics
    variants = ["raw", *(C.LOGIT_STANDARDIZATION_VARIANTS if logit_standardization_variants is None
                         else logit_standardization_variants)]
    distinct = {full_key: (None, full_statistics)}  # mask key -> (masks, statistics); each runs once per image
    for t, circuits_per_grid_idx in circuits.items():
        for grid_idx, (masks, key) in enumerate(circuits_per_grid_idx):
            if key not in distinct:
                distinct[key] = (masks, circuit_statistics[t][grid_idx] if circuit_statistics else None)
    accumulators = {(key, v): new_accumulator(len(indices), device, accuracy_only) for key in distinct
                    for v in variants}

    for j, i in enumerate(indices):
        image, mask = dataset[i]
        x = M.preprocess(image, device)
        gt = torch.as_tensor(mask, device=device)
        valid, gt_c = gt != VOID_LABEL, gt == c
        full_logits = M.forward(model, x)
        full_labels = full_logits.argmax(dim=0)
        full_c = full_labels == c
        for key, (masks, statistics) in distinct.items():
            logits = full_logits if masks is None else M.forward(model, x, masks)
            for v in variants:
                adjusted = logits if v == "raw" else logit_standardization.adjust_logits(logits, v, statistics,
                                                                                         full_statistics)
                add_image(accumulators[(key, v)], j, adjusted, c, gt, valid, gt_c, full_labels, full_c, accuracy_only)
    return {key: accumulator_to_numpy(accumulator) for key, accumulator in accumulators.items()}


def iou(tp, fp, fn):
    return tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan


def pearson_abs(f_full, f_circuit):
    """|Pearson r| over the images where the full model predicts class c (f_full not nan; f_circuit is nan only
    there, or everywhere when the Jung variant left class c out of the argmax).
    Needs at least 3 such images; with 2 points r is always ±1."""
    keep = ~np.isnan(f_full) & ~np.isnan(f_circuit)
    if keep.sum() < 3 or np.std(f_full[keep]) == 0 or np.std(f_circuit[keep]) == 0:
        return np.nan
    return abs(np.corrcoef(f_full[keep], f_circuit[keep])[0, 1])


def accumulator_metrics(accumulator, reference, f_full, c, accuracy_only=False, mean_iou_classes=None):
    """Every metric of one accumulator (one model after one variant); reference = the full model's accumulator after
    the same variant (group 3), f_full = the RAW full model's f_pred per image (|r|).
    accuracy_only: group 1 only (reference and f_full are not used). mean_iou_classes: the classes of
    mIoU_vs_ground_truth (None = all 21; s05c passes background + the dataset's evaluable classes)."""
    cm_gt = accumulator["confusion_matrix_vs_ground_truth"]
    n_images = accumulator["num_images"] if accuracy_only else len(accumulator["f_pred"])
    out = {
        "ciou": iou(*accumulator["counts"][:3]),
        "mIoU_vs_ground_truth": metrics.mean_iou(cm_gt, mean_iou_classes),
        "class_c_predicted_pixel_ratio": metrics.predicted_class_share(cm_gt, c),
        "images_with_class_c_predicted_ratio":
            accumulator["images_with_class_c_predicted"] / n_images if n_images else np.nan,
    }
    if accuracy_only:
        return out
    cm_full = accumulator["confusion_matrix_vs_full_model"]
    out.update({
        "faith": iou(*accumulator["counts"][3:]),
        "pearson_abs": pearson_abs(f_full, accumulator["f_pred"]),
        "pixel_agreement_with_full_model": metrics.pixel_agreement(cm_full),
        "class_share_total_variation_to_full_model": metrics.class_share_total_variation(cm_full),
    })
    for p in DISTRIBUTION_PIXEL_SETS:
        model = logit_standardization.statistics_from_sums(accumulator["logit_sums"][p], 0)
        full = logit_standardization.statistics_from_sums(reference["logit_sums"][p], 0)
        for label, d in (("class_c", c), ("background", 0)):
            column = f"{label}_logit_{{}}_on_{PIXEL_SET_IN_COLUMN_NAME[p]}"
            with np.errstate(divide="ignore", invalid="ignore"):
                out[column.format("mean_shift")] = float(model["logit_mean"][d] - full["logit_mean"][d])
                out[column.format("standard_deviation_ratio")] = float(
                    model["logit_standard_deviation"][d] / full["logit_standard_deviation"][d])
            if C.STORE_LOGIT_HISTOGRAMS:
                out[column.format("wasserstein_1")] = float(logit_standardization.wasserstein_1_from_histograms(
                    accumulator["logit_histograms"][p][d], reference["logit_histograms"][p][d],
                    C.LOGIT_HISTOGRAM_BIN_WIDTH))
    return out


def summarise(accumulators, circuits, full_key, c, sizes, subnet_size_layer_normalized_ratio_mean,
              topk_layer_normalized_ratio_grid, logit_standardization_variants=None, circuit_types=None,
              accuracy_only=False, mean_iou_classes=None):
    """{variant: {type: {topk_layer_normalized_ratio, subnet_size_*, ciou, faith, pearson_abs, ...: one value per grid
    index}}} of one class and image set ("full": one value), and the raw full model's f_pred per image (None with
    accuracy_only). sizes: {type: [circuit_size measures per grid index]} with "full" included.
    logit_standardization_variants, circuit_types: None = LOGIT_STANDARDIZATION_VARIANTS, CIRCUIT_TYPES (s05c passes
    its own); accuracy_only, mean_iou_classes: as accumulator_metrics."""
    variants = C.LOGIT_STANDARDIZATION_VARIANTS if logit_standardization_variants is None \
        else logit_standardization_variants
    circuit_types = C.CIRCUIT_TYPES if circuit_types is None else circuit_types
    f_full = None if accuracy_only else accumulators[(full_key, "raw")]["f_pred"]
    out = {}
    for v in ["raw", *variants]:
        reference = accumulators[(full_key, v)]
        out[v] = {}
        for t in ["full", *circuits]:
            if t == "full" and "full" not in circuit_types:
                continue
            entries = [reference] if t == "full" else [accumulators[(key, v)] for _, key in circuits[t]]
            result = {
                "topk_layer_normalized_ratio": [None] if t == "full" else topk_layer_normalized_ratio_grid.tolist(),
                "subnet_size_layer_normalized_ratio_mean":
                    [1.0] if t == "full" else subnet_size_layer_normalized_ratio_mean[t].tolist(),
            }
            per_grid_idx = [accumulator_metrics(entry, reference, f_full, c, accuracy_only, mean_iou_classes)
                            for entry in entries]
            for metric in per_grid_idx[0]:
                result[metric] = [values[metric] for values in per_grid_idx]
            for measure in sizes[t][0]:
                result[measure] = [size[measure] for size in sizes[t]]
            out[v][t] = result
    return out, f_full


def save_counts(path, accumulators, circuits, full_key, topk_layer_normalized_ratio_grid, entry,
                logit_standardization_variants=None, accuracy_only=False):
    """The counts behind every metric of one class, eval dataset and image set, as a compressed npz (M = models: the
    full model, then every circuit type × grid value; V = "raw" + LOGIT_STANDARDIZATION_VARIANTS (or the
    logit_standardization_variants given); P = DISTRIBUTION_PIXEL_SETS; B = histogram bins, see
    logit_standardization.histogram_bin_edges). With accuracy_only (s05c) only the vs-GT confusion matrix,
    images_with_class_c_predicted and the labels of the axes:
        confusion_matrix_vs_ground_truth  [M, V, 21, 21] int64  rows = GT (void excluded), columns = model
        confusion_matrix_vs_full_model    [M, V, 21, 21] int64  rows = RAW full model (all pixels), columns = model
        images_with_class_c_predicted     [M, V]
        logit_pixel_count, logit_sum, logit_squared_sum   [M, V, P, 21]   (sums of F_d over the pixel set)
        logit_histogram                   [M, V, P, 21, B] int64  (only with STORE_LOGIT_HISTOGRAMS)
        model_circuit_types, model_topk_layer_normalized_ratio (nan for the full model) [M], logit_standardization_
        variants [V], distribution_pixel_sets [P], histogram_bin_edges, image_indices, image_ids"""
    models = [("full", np.nan, full_key)] + [(t, topk_layer_normalized_ratio_grid[grid_idx], key)
                                             for t, circuits_per_grid_idx in circuits.items()
                                             for grid_idx, (_, key) in enumerate(circuits_per_grid_idx)]
    variants = ["raw", *(C.LOGIT_STANDARDIZATION_VARIANTS if logit_standardization_variants is None
                         else logit_standardization_variants)]

    def stacked(get):
        return np.stack([np.stack([get(accumulators[(key, v)]) for v in variants]) for _, _, key in models])

    arrays = {"confusion_matrix_vs_ground_truth": stacked(lambda a: a["confusion_matrix_vs_ground_truth"])}
    if not accuracy_only:
        arrays["confusion_matrix_vs_full_model"] = stacked(lambda a: a["confusion_matrix_vs_full_model"])
    arrays["images_with_class_c_predicted"] = stacked(lambda a: np.int64(a["images_with_class_c_predicted"]))
    if not accuracy_only:
        for quantity, name in (("pixel_count", "logit_pixel_count"), ("logit_sum", "logit_sum"),
                               ("logit_squared_sum", "logit_squared_sum")):
            arrays[name] = stacked(lambda a: np.stack([a["logit_sums"][p][quantity] for p in DISTRIBUTION_PIXEL_SETS]))
        if C.STORE_LOGIT_HISTOGRAMS:
            arrays["logit_histogram"] = stacked(lambda a: np.stack([a["logit_histograms"][p]
                                                                    for p in DISTRIBUTION_PIXEL_SETS]))
            arrays["histogram_bin_edges"] = logit_standardization.histogram_bin_edges(
                C.LOGIT_HISTOGRAM_LOWER_EDGE, C.LOGIT_HISTOGRAM_UPPER_EDGE, C.LOGIT_HISTOGRAM_BIN_WIDTH)
        arrays["distribution_pixel_sets"] = np.array(DISTRIBUTION_PIXEL_SETS)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays,
                        model_circuit_types=np.array([t for t, _, _ in models]),
                        model_topk_layer_normalized_ratio=np.array([k for _, k, _ in models], dtype=np.float64),
                        logit_standardization_variants=np.array(variants),
                        image_indices=np.array(entry["indices"], dtype=np.int64), image_ids=np.array(entry["ids"]))


def histogram_overflow(accumulators):
    """{variant: (counts outside the histogram range, all counts)} over every model, pixel set and class d."""
    out = {}
    for (_, v), accumulator in accumulators.items():
        outside, total = out.get(v, (0, 0))
        for histograms in accumulator["logit_histograms"].values():
            outside += int(histograms[:, 0].sum() + histograms[:, -1].sum())
            total += int(histograms.sum())
        out[v] = (outside, total)
    return out


def reference(model, dataset, indices, device):
    """Confusion matrix of the full model over the given images."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for n, i in enumerate(indices, start=1):
        image, mask = dataset[i]
        pred = M.forward(model, M.preprocess(image, device)).argmax(dim=0).cpu().numpy()
        cm += metrics.confusion_matrix(pred, mask)
        if n % 250 == 0:
            print(f"  reference {n}/{len(indices)}", flush=True)
    return cm


def unmasked_layers():
    layers, _, _ = M.certifiable_components(C.LAYERS)
    return [name for name in M.HOOK_POINTS if name not in layers]


def print_means(name, results, variant):
    """Mean over classes per circuit type and top-K layer-normalized ratio, for one variant ("raw" or a
    LOGIT_STANDARDIZATION_VARIANTS entry), on the images with class c."""
    outside = unmasked_layers()
    utils.section(
        f"[{name}] Circuit metrics per {RATIO_COLUMN}, mean over the {len(C.CLASSES)} classes "
        f"(each class weighted equally), images WITH class c{logit_standardization.section_title_suffix(variant)}",
        logit_standardization.how_to_read_variant(variant) + textwrap.dedent(f"""
        A circuit = the full model with the channels of U (the certifiable components, LAYERS = {C.LAYERS!r})
        outside the circuit set to zero.
        {"Layers outside U (" + ", ".join(outside) + ") are NEVER masked, so they stay complete in every circuit." if outside else "U holds every hook point."}
        {RATIO_COLUMN}
{textwrap.fill(utils.topk_ratio_description() + '. "-" = the full model.', 104, initial_indent=" " * 16, subsequent_indent=" " * 16)}
        size    the subnetwork size, layer-normalized ratio (subnet_size_layer_normalized_ratio_mean, s04): the
                ratio of each layer's channels kept, equal-weight mean over the layers of U. At the same K the
                certified circuit is smaller than the baseline circuit, so compare circuit types at equal size
                (as in s06's curves), not at equal K.
        cIoU    sufficiency: class-c IoU against the GROUND TRUTH, pixels pooled over the class's eval
                images (void excluded). Compare with the full-model row; 0 = the circuit never finds class c.
        faith   faithfulness: class-c IoU against the FULL MODEL's own class-c prediction (all pixels).
                1.000 = the circuit predicts exactly the full model's class-c mask (the full model is 1 by
                definition). nan for a class = neither model predicts any class-c pixel.
        |r|     |Pearson r| over the eval images between the full model's and the circuit's f_pred
                (mean class-c logit over the pixels the full model predicts as c). 1 = the circuit's logit
                follows the full model's linearly across images; it can still be shifted or scaled, so a
                high |r| does NOT mean the circuit predicts class c (check cIoU / faith). A class gets nan
                with < 3 images that have a full-model class-c prediction, or with constant f_pred.
        n|r|    number of classes in the |r| mean (classes with nan are skipped).
        At K = 1 every circuit keeps every channel, i.e. it is the full model.
        """))
    with warnings.catch_warnings():  # all-nan columns print as nan
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in C.CIRCUIT_TYPES:
            per_class = [results[c]["images_with_class_c"][variant][t] for c in C.CLASSES]
            print(f"  circuit type: {t}")
            print(f"  {RATIO_COLUMN} {'size':>6s} {'cIoU':>6s} {'faith':>6s} {'|r|':>6s} {'n|r|':>5s}")
            for grid_idx, topk_layer_normalized_ratio in enumerate(per_class[0]["topk_layer_normalized_ratio"]):
                values = [np.nanmean([r[m][grid_idx] for r in per_class])
                          for m in ("subnet_size_layer_normalized_ratio_mean", "ciou", "faith", "pearson_abs")]
                n_r = sum(not np.isnan(r["pearson_abs"][grid_idx]) for r in per_class)
                label = "-" if topk_layer_normalized_ratio is None else f"{topk_layer_normalized_ratio:g}"
                print(f"  {label:>{len(RATIO_COLUMN)}s}" + "".join(f" {v:6.3f}" for v in values) + f" {n_r:5d}")
            print()


def print_mean_table(results, variant, image_set, columns):
    """Mean over classes per circuit type and top-K layer-normalized ratio of the metrics in `columns` ([(metric,
    column header, format)]), for one variant and image set. The caller prints the section and its "How to read"."""
    with warnings.catch_warnings():  # all-nan cells print as nan
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in C.CIRCUIT_TYPES:
            per_class = [results[c][image_set][variant][t] for c in C.CLASSES]
            widths = [max(len(header), len(fmt.format(0.0))) for _, header, fmt in columns]
            print(f"  circuit type: {t}")
            print(f"  {RATIO_COLUMN} {'size':>6s}" + "".join(f" {header:>{w}s}"
                                                           for (_, header, _), w in zip(columns, widths)))
            for grid_idx, topk_layer_normalized_ratio in enumerate(per_class[0]["topk_layer_normalized_ratio"]):
                size = np.nanmean([r["subnet_size_layer_normalized_ratio_mean"][grid_idx] for r in per_class])
                values = [np.nanmean([r[metric][grid_idx] for r in per_class]) for metric, _, _ in columns]
                label = "-" if topk_layer_normalized_ratio is None else f"{topk_layer_normalized_ratio:g}"
                print(f"  {label:>{len(RATIO_COLUMN)}s} {size:6.3f}"
                      + "".join(f" {'nan' if np.isnan(v) else fmt.format(v):>{w}s}"
                                for v, (_, _, fmt), w in zip(values, columns, widths)))
            print()


def print_multi_class_means(name, results, variant, image_set):
    """Groups 1 and 2 beyond class c: all 21 classes, and on the images without c the false positives of c."""
    without = image_set == "images_without_class_c"
    columns = ([("class_c_predicted_pixel_ratio", "c-pixels", "{:.4f}"),
                ("images_with_class_c_predicted_ratio", "c-images", "{:.3f}")] if without else []) + [
        ("mIoU_vs_ground_truth", "mIoU", "{:.3f}"),
        ("pixel_agreement_with_full_model", "agreement", "{:.3f}"),
        ("class_share_total_variation_to_full_model", "class-share TV", "{:.3f}"),
    ] + ([("faith", "faith", "{:.3f}")] if without else [])
    utils.section(
        f"[{name}] Multi-class metrics (all {NUM_CLASSES} classes, not only c), mean over the {len(C.CLASSES)} classes c, "
        f"{IMAGE_SET_TITLES[image_set]}{logit_standardization.section_title_suffix(variant)}",
        logit_standardization.how_to_read_variant(variant) + textwrap.dedent(f"""
        Pixels are pooled over the images of class c's image set ({IMAGE_SET_TITLES[image_set]}), then each class c
        is weighted equally in the mean. size and K as in the cIoU table above. "-" = the full model.
        """ + ("""
        c-pixels   class_c_predicted_pixel_ratio: share of the non-void pixels predicted as class c. These images
                   have NO class c in their GT, so every such pixel is a FALSE POSITIVE of c. Compare
                   with the full-model row (the full model's own false positives). 0 = c is never predicted.
        c-images   images_with_class_c_predicted_ratio: share of the images with at least one non-void pixel
                   predicted as c. Which GT classes the false c-pixels come from is column c of the vs-GT confusion
                   matrix in the npz files.
        """ if without else "") + f"""
        mIoU       mIoU_vs_ground_truth: IoU against the GT of every one of the {NUM_CLASSES} classes (background included),
                   pixels pooled over the image set, void excluded; mean over the classes whose IoU is defined.
                   It shows whether the circuit breaks the OTHER classes, not only c.{" On these images class c has no GT pixel, so its IoU is 0 (or undefined)." if without else ""}
        agreement  pixel_agreement_with_full_model: share of ALL pixels whose 21-class label equals the RAW full model's
                   label. The full model against itself: 1.
        class-share TV
                   class_share_total_variation_to_full_model: ½ Σ_d |s_d^circuit − s_d^full|, s_d = share of the
                   pixels predicted as class d. 0 = the same class MIX as the raw full model (the full model: 0),
                   1 = no class in common. It ignores WHERE the pixels are: a class moved to the wrong places but
                   kept in number gives TV 0 and still a low agreement.
        """ + ("""faith      class-c IoU against the full model's class-c pixels (here: both models' false positives).
                   nan for a class = neither model predicts any c-pixel on these images.
        """ if without else "") + """At K = 1 every circuit is the full model.
        """))
    print_mean_table(results, variant, image_set, columns)


def print_distribution_means(name, results, variant, image_set):
    """Group 3: the logit shift of class c and background, circuit vs full model."""
    histograms = C.STORE_LOGIT_HISTOGRAMS
    columns = []
    for label, short in (("class_c", "c"), ("background", "bg")):
        for p, short_p in (("full_model_prediction", "fp"), ("all_pixels", "all")):
            pixel_set = PIXEL_SET_IN_COLUMN_NAME[p]
            columns += [(f"{label}_logit_mean_shift_on_{pixel_set}", f"Δμ {short} {short_p}", "{:+.2f}"),
                        (f"{label}_logit_standard_deviation_ratio_on_{pixel_set}", f"σ× {short} {short_p}", "{:.3f}")]
            if histograms:
                columns.append((f"{label}_logit_wasserstein_1_on_{pixel_set}", f"W1 {short} {short_p}", "{:.2f}"))
    full_reference = ("the RAW full model" if variant == "raw" or variant in logit_standardization.MATCHED_VARIANTS
                      else "the full model after the same variant (both sides are z-scores)")
    utils.section(
        f"[{name}] Logit distributions: class c and background, circuit vs full model, mean over the "
        f"{len(C.CLASSES)} classes c, {IMAGE_SET_TITLES[image_set]}{logit_standardization.section_title_suffix(variant)}",
        logit_standardization.how_to_read_variant(variant) + textwrap.dedent(f"""
        These compare the 21 LOGITS before the argmax (the shift the logit standardization corrects), not the
        labels. Reference: {full_reference}, on the same images and pixels.
        Columns: <measure> <class: c = class c, bg = background> <pixel set>, pixels pooled over the image set:
          fp   the pixels the RAW full model predicts as that class (s04b's pixel set full_model_prediction)
          all  every pixel
        Δμ     μ_circuit − μ_full of that class's logit, in logit units (< 0 = lower than the full model's)
        σ×     σ_circuit / σ_full of that class's logit (1 = the same spread, < 1 = compressed)
        """) + (textwrap.dedent(f"""
        W1     Wasserstein-1 distance between the two distributions of that class's logit, in logit units (0 = the
               same distribution). From fixed-bin histograms, [{C.LOGIT_HISTOGRAM_LOWER_EDGE:g}, {C.LOGIT_HISTOGRAM_UPPER_EDGE:g}] in bins of {C.LOGIT_HISTOGRAM_BIN_WIDTH:g} plus one
               overflow bin at each end: an approximation with resolution {C.LOGIT_HISTOGRAM_BIN_WIDTH:g}.
        """) if histograms else "") + textwrap.dedent(f"""
        nan = the full model predicts no pixel as that class on these images (fp), or a class the Jung variant leaves
        out of the argmax (its logit is −∞ and is not counted).
        The full-model row is 0 / 1 / 0 by definition. At K = 1 every circuit is the full model. The matched variants
        are fitted on D_c, so on discovery_samples WITH class c they give Δμ = 0 and σ× = 1 on their own pixel set
        (by construction, where the class has usable statistics); elsewhere the rest shows how the correction
        carries over. Every class d is in the npz files (s06 group 3).
        """))
    print_mean_table(results, variant, image_set, columns)


def print_size_table(sizes, subnet_size_layer_normalized_ratio_mean, topk_layer_normalized_ratio_grid, counter):
    """Group 4: the size measures per circuit type and K, mean over the classes."""
    layers = counter.layers
    counter_channels = sum(M.HOOK_POINTS[layer][1] for layer in layers)
    breakdown = "\n".join(f"{'':27s}{count:>12,}  {description}" for description, count in counter.reachable_weight_breakdown())
    whole, reachable = counter.whole_model_weight_count, counter.reachable_weight_count
    utils.section(
        f"Circuit size measures, mean over the {len(C.CLASSES)} classes (the same for every eval "
        f"dataset)",
        textwrap.dedent(f"""
        Binary-mask convention: a circuit is a 0/1 mask over the model's weights, 1 = used, 0 = removed (a zero weight
        is effectively removed; nothing is deleted). A zeroed channel of U makes the weights that WRITE it (filter j
        of the last conv before the hook point and its BatchNorm entry) and the weights that READ it (input slice j of
        the next convs, incl. the ASPP branch convs after layer4) removable: they are 0. Everything else stays 1.
          whole model        {whole:,} weights (auxiliary classifier removed)
          reachable weights  weights tied to a certifiable channel (LAYERS = {C.LAYERS!r}): its writers in the model
                             and its readers, incl. the ASPP input slices. {reachable:,} of {whole:,}
                             ({reachable / whole:.1%} of the whole model):
        """) + breakdown + textwrap.dedent(f"""
                             They are NOT the backbone: most of them are the ASPP convs reading layer4.
        {RATIO_COLUMN}
{textwrap.fill(utils.topk_ratio_description() + '.', 104, initial_indent=" " * 21, subsequent_indent=" " * 21)}
        size         subnet_size_layer_normalized_ratio_mean (s04): the per-layer channel shares below, equal-weight mean
        {layers[0]}…{layers[-1]:6s} subnet_size_layer_normalized_ratio_<layer>, one column per layer of U: the share of that
                     layer's channels kept
        pooled       subnet_size_group_pooled_ratio: channels of U kept / all |U| = {counter_channels:,} channels of U, all layers
                     POOLED (big layers dominate; with a global top-K the baseline circuit's value is K)
        channels     subnet_size_group_channel_count: channels of U kept (absolute; the same count as pooled × {counter_channels:,})
        whole        subnet_size_whole_model_weight_ratio: weights kept / all {whole:,} weights. Even an empty
                     circuit keeps {1 - reachable / whole:.1%} (the weights no channel of U reaches: stem, block interiors,
                     ASPP projection, head).
        reachable    subnet_size_reachable_weight_ratio: reachable weights kept / all {reachable:,} reachable
                     weights: 0 = every channel of U removed, 1 = the full model.
        kept (M)     subnet_size_kept_weight_count, in millions of weights.
        "-" = the full model (1.000 everywhere). At K = 1 every circuit is the full model.
        """))
    header = (f"  {RATIO_COLUMN} {'size':>6s}" + "".join(f" {layer:>7s}" for layer in layers)
              + f" {'pooled':>6s} {'channels':>8s} {'whole':>6s} {'reachable':>9s} {'kept (M)':>8s}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in C.CIRCUIT_TYPES:
            print(f"  circuit type: {t}")
            print(header)
            grid = [None] if t == "full" else topk_layer_normalized_ratio_grid
            for grid_idx, topk_layer_normalized_ratio in enumerate(grid):
                def mean(measure):
                    return np.mean([sizes[c][t][grid_idx][measure] for c in C.CLASSES])
                size = 1.0 if t == "full" else np.mean([subnet_size_layer_normalized_ratio_mean[c][t][grid_idx]
                                                        for c in C.CLASSES])
                label = "-" if topk_layer_normalized_ratio is None else f"{topk_layer_normalized_ratio:g}"
                print(f"  {label:>{len(RATIO_COLUMN)}s} {size:6.3f}"
                      + "".join(f" {mean(f'subnet_size_layer_normalized_ratio_{layer}'):7.3f}" for layer in layers)
                      + f" {mean('subnet_size_group_pooled_ratio'):6.3f} {mean('subnet_size_group_channel_count'):8.0f}"
                      + f" {mean('subnet_size_whole_model_weight_ratio'):6.3f}"
                      + f" {mean('subnet_size_reachable_weight_ratio'):9.3f}"
                      + f" {mean('subnet_size_kept_weight_count') / 1e6:8.2f}")
            print()


def best_cells(results_c, circuit_types, topk_layer_normalized_ratio_grid, best_width):
    """The "best X" cells of a per-class line: each circuit type's highest cIoU over the grid and where."""
    peaks = ""
    for t in circuit_types:
        ciou = np.array(results_c[t]["ciou"], dtype=float)
        if np.isnan(ciou).all():
            peaks += f" {'nan':>{best_width}s}"
        else:
            grid_idx = int(np.nanargmax(ciou))
            peaks += f" {f'{ciou[grid_idx]:.3f} @ {topk_layer_normalized_ratio_grid[grid_idx]:g}':>{best_width}s}"
    return peaks


def summary_rows(c, name, image_set, results_c, num_images, num_no_pred, columns):
    """summary.csv rows of one class, eval dataset and image set ({column: value})."""
    renamed = {"ciou": "cIoU", "faith": "faith_IoU"}
    rows = []
    for v, per_type in results_c.items():
        for t, r in per_type.items():
            for grid_idx, topk_layer_normalized_ratio in enumerate(r["topk_layer_normalized_ratio"]):
                row = {"class": c, "class_name": VOC_CLASSES[c], "eval_dataset": name, "image_set": image_set,
                       "circuit": t, "logit_standardization_variant": v,
                       "topk_layer_normalized_ratio": "" if topk_layer_normalized_ratio is None else topk_layer_normalized_ratio,
                       "num_images": num_images, "num_no_pred": num_no_pred}
                for metric, values in r.items():
                    if metric != "topk_layer_normalized_ratio":
                        row[renamed.get(metric, metric)] = values[grid_idx]
                rows.append({column: row[column] for column in columns})
    return rows


def main():
    utils.start_run("s05_evaluate", USED_PARAMS)
    utils.require(utils.certify_dir() / "run_info.json", "s04_certify.py")
    logit_standardization.check_variants(C.LOGIT_STANDARDIZATION_VARIANTS, "LOGIT_STANDARDIZATION_VARIANTS")
    if C.LOGIT_STANDARDIZATION_VARIANTS:
        utils.require(utils.logit_statistics_dir() / "run_info.json", "s04b_logit_statistics.py")
    discovery_and_eval_lists = utils.load_discovery_and_eval_lists()
    utils.check_eval_list_cap(discovery_and_eval_lists)
    missing = [name for name in C.EVAL_DATASETS if name not in discovery_and_eval_lists["eval"]]
    if missing:
        raise SystemExit(f"\n[missing input] no eval lists for {missing} in discovery_and_eval_lists.json\n"
                         f"-> re-run s01_discovery_samples_dataset.py with them in EVAL_DATASETS.")
    if C.EVALUATE_IMAGES_WITHOUT_CLASS:
        missing = [name for name in C.EVAL_DATASETS
                   if name not in discovery_and_eval_lists.get("eval_without_class", {})]
        if missing:
            raise SystemExit(f"\n[missing input] no eval lists WITHOUT class c for {missing} in "
                             f"discovery_and_eval_lists.json (an older s01 output)\n"
                             f"-> re-run s01_discovery_samples_dataset.py (cheap; D_c and the lists with c stay the "
                             f"same, so s02-s04b stay valid), or set EVALUATE_IMAGES_WITHOUT_CLASS = False.")
    histogram_bin_count()  # stops early on an invalid histogram range
    out_dir = utils.eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)

    layers, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    full_masks = {name: np.ones(s.stop - s.start, dtype=bool) for name, s in certifiable_slices.items()}
    full_key = utils.mask_key(full_masks)
    types = [t for t in PRUNED_TYPES if t in C.CIRCUIT_TYPES]
    circuits = {c: utils.load_circuits(c, certifiable_slices, types) for c in C.CLASSES}
    variants = ["raw", *C.LOGIT_STANDARDIZATION_VARIANTS]
    image_sets = IMAGE_SETS if C.EVALUATE_IMAGES_WITHOUT_CLASS else IMAGE_SETS[:1]
    logit_statistics = {c: logit_standardization.load_class_logit_statistics(c, circuits[c][0])
                        if C.LOGIT_STANDARDIZATION_VARIANTS else (None, None) for c in C.CLASSES}
    counter = circuit_size.WeightMaskCounter(model, layers)
    sizes = {c: {"full": [counter.size_measures(full_masks)],
                 **{t: [counter.size_measures(masks) for masks, _ in circuits[c][0][t]] for t in types}}
             for c in C.CLASSES}
    columns = summary_columns(layers)
    if C.LOGIT_STANDARDIZATION_VARIANTS:
        utils.section(
            "Logit standardization (LOGIT_STANDARDIZATION_VARIANTS)",
            textwrap.dedent(f"""
            Every circuit and the full model are evaluated RAW and after each variant below, from the same forward
            pass. The statistics come from s04b ({utils.logit_statistics_dir().name}): the full model and every
            circuit of class c, measured on LOGIT_STATISTICS_IMAGES = {C.LOGIT_STATISTICS_IMAGES!r} (D_c of class c),
            usable with ≥ LOGIT_STATISTICS_MIN_PIXELS = {C.LOGIT_STATISTICS_MIN_PIXELS:,} pixels. Each table below says which variant it shows;
            in summary.csv the column logit_standardization_variant is "raw" or the variant's name.
            The full-model reference (mIoU over the eval set) is the raw full model only.
            """) + "\n".join(f"  {v}\n" + textwrap.indent(logit_standardization.VARIANT_DESCRIPTIONS[v], "      ")
                             for v in C.LOGIT_STANDARDIZATION_VARIANTS))
    utils.section(
        "Image sets (reported apart, never pooled)",
        textwrap.dedent(f"""
        images WITH class c     the s01 eval list of c: images whose GT contains c; on discovery_samples D_c itself.
                                {utils.cap_text(C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES")}. Every table so far.
        images WITHOUT class c  {"every image whose GT has no pixel of c, no cap; on discovery_samples the other classes' D_c' without the images that contain c (count printed)." if C.EVALUATE_IMAGES_WITHOUT_CLASS else "OFF (EVALUATE_IMAGES_WITHOUT_CLASS = False)."}{" SMOKE_TEST: only the first MAX_EVAL_IMAGES of them." if C.SMOKE_TEST and C.EVALUATE_IMAGES_WITHOUT_CLASS else ""}
        Each table states its image set. In summary.csv the column image_set is images_with_class_c or
        images_without_class_c.
        """))
    print_size_table(sizes, {c: circuits[c][1] for c in C.CLASSES}, circuits[C.CLASSES[0]][2], counter)

    references, per_class, rows = {}, {c: {} for c in C.CLASSES}, []
    eval_lists_used = {"discovery_dataset": discovery_and_eval_lists["discovery_dataset"],
                       "max_eval_images": discovery_and_eval_lists["max_eval_images"], "eval": {},
                       "eval_without_class": {}}
    for name in C.EVAL_DATASETS:
        dataset = utils.eval_dataset(name)
        eval_lists = discovery_and_eval_lists["eval"][name]
        counts_dir = out_dir / "confusion_matrices_and_logit_histograms" / name.replace("/", "_")

        # ---- full-model reference ----------------------------------------------------
        indices = reference_indices(name, eval_lists, dataset)
        print(f"\n{'=' * 100}\nEVAL DATASET: {name}\n{'=' * 100}", flush=True)
        if name == "discovery_samples":
            print("NOTE: discovery_samples = the D_c images the circuits were DISCOVERED on (in-sample, Anani's ID\n"
                  "setting). Expect these numbers to be optimistic compared with held-out eval sets.")
        whole = "the union of the per-class eval lists" if (name == "discovery_samples" or C.SMOKE_TEST) \
            else "ALL images of the eval set"
        utils.section(
            f"[{name}] Full-model reference: the unpruned model's IoU per class over {len(indices)} images",
            f"""
            IoU per class over {whole},
            including images that do not contain the class (false positives there count). This is the
            model's ordinary benchmark score; mIoU averages all {NUM_CLASSES} classes, background included.
            nan = the class neither occurs nor is predicted in these images. It is NOT the same number as the per-class "full cIoU" further down, which
            only uses the eval images that contain class c.
            """)
        cm = reference(model, dataset, indices, device)
        class_iou = metrics.class_scores(cm)["iou"]
        references[name] = {"num_images": len(indices), "iou": class_iou.tolist(),
                            "miou": float(np.nanmean(class_iou)), "confusion_matrix": cm.tolist()}
        for row in range(0, NUM_CLASSES, 3):
            print("  " + "".join(f"{c:3d} {VOC_CLASSES[c]:13s} {class_iou[c]:5.3f}     "
                                 for c in range(row, min(row + 3, NUM_CLASSES))))
        print(f"  mIoU ({NUM_CLASSES} classes incl. background) = {np.nanmean(class_iou):.4f}")

        # ---- circuits, images WITH class c ----------------------------------------------
        class_circuit_types = [t for t in PRUNED_TYPES if t in C.CIRCUIT_TYPES]
        best_width = max([17] + [len("best " + t) for t in class_circuit_types])
        utils.section(
            f"[{name}] Per class: eval images, the full model's cIoU, and each circuit type's best cIoU over the "
            f"{RATIO_COLUMN} grid, images WITH class c{logit_standardization.section_title_suffix('raw')}",
            f"""
            images     eval images of class c: images whose GT contains c (any number of pixels).
                       {"These are the D_c images (in-sample), never capped." if name == "discovery_samples" else utils.cap_text(C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES") + "."}
            no pred    of those, images where the full model predicts no class-c pixel; they are left out of |r|
                       only (cIoU and faith use every image).
            full cIoU  the unpruned model's class-c IoU against the GT, pixels pooled over these images.
            best X     circuit type X's HIGHEST cIoU over the grid; "@ value" = the {RATIO_COLUMN}
                       K where it occurs. A best at K = 1 is the full model itself (every channel kept),
                       i.e. no smaller circuit did better. The full curves over the grid are in the mean tables
                       below and in s06.
            """)
        print(f"{'':3s} {'class':13s} {'images':>6s} {'no pred':>7s} {'full cIoU':>9s}"
              + "".join(f" {'best ' + t:>{best_width}s}" for t in class_circuit_types) + f" {'time':>6s}")
        results, lines, overflow = {c: {} for c in C.CLASSES}, {v: {} for v in variants}, {}
        for c in C.CLASSES:
            start = time.time()
            class_circuits, subnet_size_layer_normalized_ratio_mean, topk_layer_normalized_ratio_grid = circuits[c]
            entry = eval_lists[str(c)]
            eval_lists_used["eval"].setdefault(name, {})[str(c)] = entry
            indices = entry["indices"]
            accumulators = evaluate_class(model, dataset, indices, c, class_circuits, full_key, device,
                                          logit_statistics[c])
            results[c]["images_with_class_c"], f_full = summarise(
                accumulators, class_circuits, full_key, c, sizes[c], subnet_size_layer_normalized_ratio_mean,
                topk_layer_normalized_ratio_grid)
            save_counts(counts_dir / f"class_{c:02d}_images_with_class_c.npz", accumulators, class_circuits, full_key,
                        topk_layer_normalized_ratio_grid, entry)
            if C.STORE_LOGIT_HISTOGRAMS:
                overflow[("images_with_class_c", c)] = histogram_overflow(accumulators)
            num_no_pred = int(np.isnan(f_full).sum())
            results_c = results[c]["images_with_class_c"]
            per_class[c][name] = {"num_images": len(indices), "num_no_pred": num_no_pred, "circuits": results_c["raw"]}
            if C.LOGIT_STANDARDIZATION_VARIANTS:
                per_class[c][name]["circuits_per_logit_standardization_variant"] = {
                    v: results_c[v] for v in C.LOGIT_STANDARDIZATION_VARIANTS}
            rows += summary_rows(c, name, "images_with_class_c", results_c, len(indices), num_no_pred, columns)
            for v in variants:
                full_counts = accumulators[(full_key, v)]["counts"]
                lines[v][c] = (f"{c:3d} {VOC_CLASSES[c]:13s} {len(indices):6d} {num_no_pred:7d} "
                               f"{iou(*full_counts[:3]):9.3f}"
                               + best_cells(results_c[v], class_circuits, topk_layer_normalized_ratio_grid, best_width))
            print(f"{lines['raw'][c]} {time.time() - start:5.0f}s", flush=True)
        print_means(name, results, "raw")
        print_multi_class_means(name, results, "raw", "images_with_class_c")
        print_distribution_means(name, results, "raw", "images_with_class_c")

        # ---- circuits, images WITHOUT class c -------------------------------------------
        if C.EVALUATE_IMAGES_WITHOUT_CLASS:
            in_sample = name == "discovery_samples"
            utils.section(
                f"[{name}] Per class: images WITHOUT class c (false positives of c), the full model's "
                f"numbers{logit_standardization.section_title_suffix('raw')}",
                f"""
                images     {"the other classes' D_c' (c' in CLASSES), each image once, minus the images whose GT contains c" if in_sample else "every image of the eval set whose GT has no pixel of class c (all of them, no cap)"}.
                {"dropped    D_c' images that contain c: neither in D_c nor free of c, so left out of both image sets." if in_sample else ""}
                no pred    of those, images where the full model predicts NO class-c pixel (left out of |r|). On images
                           without c that is the normal case, so it is usually close to "images" and |r| mostly nan.
                full c-pixels  the full model's share of non-void pixels predicted as c: its own false positives.
                full c-images  the full model's share of images with at least one non-void pixel predicted as c.
                The circuits' values are in the mean tables below; per class and K in summary.csv.
                """)
            print(f"{'':3s} {'class':13s} {'images':>7s}" + (f" {'dropped':>7s}" if in_sample else "")
                  + f" {'no pred':>7s} {'full c-pixels':>13s} {'full c-images':>13s} {'time':>6s}")
            for c in C.CLASSES:
                start = time.time()
                class_circuits, subnet_size_layer_normalized_ratio_mean, topk_layer_normalized_ratio_grid = circuits[c]
                entry = without_class_entry(discovery_and_eval_lists["eval_without_class"][name][str(c)])
                eval_lists_used["eval_without_class"].setdefault(name, {})[str(c)] = entry
                indices = entry["indices"]
                accumulators = evaluate_class(model, dataset, indices, c, class_circuits, full_key, device,
                                              logit_statistics[c])
                results[c]["images_without_class_c"], f_full = summarise(
                    accumulators, class_circuits, full_key, c, sizes[c], subnet_size_layer_normalized_ratio_mean,
                    topk_layer_normalized_ratio_grid)
                save_counts(counts_dir / f"class_{c:02d}_images_without_class_c.npz", accumulators, class_circuits,
                            full_key, topk_layer_normalized_ratio_grid, entry)
                if C.STORE_LOGIT_HISTOGRAMS:
                    overflow[("images_without_class_c", c)] = histogram_overflow(accumulators)
                num_no_pred = int(np.isnan(f_full).sum())
                results_c = results[c]["images_without_class_c"]
                per_class[c][name]["images_without_class_c"] = {
                    "num_images": len(indices), "num_no_pred": num_no_pred, "circuits": results_c["raw"]}
                if in_sample:
                    per_class[c][name]["images_without_class_c"]["num_dropped_containing_class"] = \
                        entry["num_dropped_containing_class"]
                if C.LOGIT_STANDARDIZATION_VARIANTS:
                    per_class[c][name]["images_without_class_c"]["circuits_per_logit_standardization_variant"] = {
                        v: results_c[v] for v in C.LOGIT_STANDARDIZATION_VARIANTS}
                rows += summary_rows(c, name, "images_without_class_c", results_c, len(indices), num_no_pred, columns)
                full_result = accumulator_metrics(accumulators[(full_key, "raw")], accumulators[(full_key, "raw")],
                                                  f_full, c)
                print(f"{c:3d} {VOC_CLASSES[c]:13s} {len(indices):7,d}"
                      + (f" {entry['num_dropped_containing_class']:7,d}" if in_sample else "")
                      + f" {num_no_pred:7,d} {full_result['class_c_predicted_pixel_ratio']:13.4f}"
                      f" {full_result['images_with_class_c_predicted_ratio']:13.3f} {time.time() - start:5.0f}s",
                      flush=True)
            print_multi_class_means(name, results, "raw", "images_without_class_c")
            print_distribution_means(name, results, "raw", "images_without_class_c")

        # ---- every logit-standardization variant ----------------------------------------
        for v in C.LOGIT_STANDARDIZATION_VARIANTS:
            utils.section(
                f"[{name}] Per class: eval images, the full model's cIoU, and each circuit type's best cIoU over the "
                f"{RATIO_COLUMN} grid, images WITH class c{logit_standardization.section_title_suffix(v)}",
                logit_standardization.how_to_read_variant(v) + "Columns as in the raw per-class table above (without the time).")
            print(f"{'':3s} {'class':13s} {'images':>6s} {'no pred':>7s} {'full cIoU':>9s}"
                  + "".join(f" {'best ' + t:>{best_width}s}" for t in class_circuit_types))
            for c in C.CLASSES:
                print(lines[v][c])
            print_means(name, results, v)
            for image_set in image_sets:
                print_multi_class_means(name, results, v, image_set)
                print_distribution_means(name, results, v, image_set)

        # ---- histogram range check --------------------------------------------------------
        if C.STORE_LOGIT_HISTOGRAMS:
            utils.section(
                f"[{name}] Logit histograms: share of the counted logits OUTSIDE [{C.LOGIT_HISTOGRAM_LOWER_EDGE:g}, "
                f"{C.LOGIT_HISTOGRAM_UPPER_EDGE:g}] (the two overflow bins)",
                f"""
                Every logit F_d of every model, class d, pixel set and image is binned (−∞ from the Jung variant is
                not counted). Values outside the range land in the overflow bins: they are counted, but their
                position is lost, so W1 and the box plots treat them as lying at the range edge. A share well above 0
                means the range is too narrow (LOGIT_HISTOGRAM_LOWER_EDGE / _UPPER_EDGE; changing them needs a new s05
                run). Per variant, pooled over all classes c, models, pixel sets and classes d; "max per class c" =
                the largest share of any single class c.
                """)
            for image_set in image_sets:
                for v in variants:
                    per_c = [overflow[(image_set, c)][v] for c in C.CLASSES]
                    outside, total = sum(o for o, _ in per_c), sum(t for _, t in per_c)
                    largest = max((o / t for o, t in per_c if t), default=np.nan)
                    print(f"  {IMAGE_SET_TITLES[image_set]:24s} {v:55s} "
                          f"{outside / total if total else np.nan:8.4%} (max per class c {largest:.4%})")

    # ---- save ------------------------------------------------------------------------
    for c in C.CLASSES:
        utils.save_json(out_dir / f"metrics_class_{c:02d}.json",
                        {"class": c, "name": VOC_CLASSES[c], "eval": per_class[c]})
    utils.save_json(out_dir / "reference.json", references)
    utils.save_json(out_dir / "eval_lists_used.json", eval_lists_used)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    utils.write_run_info(out_dir)
    print(f"\nSaved metrics for {len(C.CLASSES)} classes, summary.csv, reference.json, eval_lists_used.json and the "
          f"counts (confusion_matrices_and_logit_histograms/) to {out_dir}")


if __name__ == "__main__":
    main()
