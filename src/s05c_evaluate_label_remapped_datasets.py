"""s05c · Evaluation on label-remapped datasets: the accuracy of s04's circuits on COCO val2017, ADE20K and Cityscapes,
with the ground truth remapped to the VOC labels.

Runs after s04 (and s04b when LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS is non-empty); nothing upstream
is re-run. The circuits are s04's, discovered on VOC+SBD, unchanged. Accuracy only (s05's group 1): no faithfulness,
no logit distributions, no images without c.

Per dataset in LABEL_REMAPPED_EVAL_DATASETS (mapping tables: common/datasets.py; the setup check's label check shows
them with counts):
    images     only the images whose remapped GT holds at least one object class (1-20). Images with only background
               and / or 255 are left out completely: from the class lists and from the full-model reference.
    reference  the full model over those images: per-class IoU, the mIoU over background + the dataset's evaluable
               classes, pixel accuracy. Next to it VOC val (all 1,449 images, the same full model) with the mIoU over
               the same classes (like-for-like).
    classes    the classes of CLASSES the dataset can score (its evaluable classes); the others never occur in its GT.
    circuits   per class c: the images whose remapped GT holds at least one pixel of c (LABEL_REMAPPED_MAX_EVAL_IMAGES:
               "all", or a seeded random subset; on the discovery dataset without D_c, as s01). The full model and every
               circuit type of LABEL_REMAPPED_CIRCUIT_TYPES at every K (top-K ratio of s03), raw and after each
               LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS entry (s04b's statistics, measured on VOC, applied as
               they are). Circuits with identical masks run once per image (s05.evaluate_class).
Metrics per class, circuit, variant and K, pixels pooled over the images of class c, GT 255 excluded:
    cIoU                                    class-c IoU against the GT
    evaluable_classes_mIoU_vs_ground_truth  mean IoU over background + the dataset's evaluable classes
    class_c_predicted_pixel_ratio           share of the non-void pixels predicted as c
    images_with_class_c_predicted_ratio     share of the images with at least one non-void pixel predicted as c
Predictions of non-evaluable classes are kept, not masked: a "dog" on a Cityscapes road pixel is a background error.
The non-evaluable classes' own IoU is left out of the mean (it would be 0 from false positives alone).

Under SMOKE_TEST the full-model references (each dataset and VOC val) run on the first LABEL_REMAPPED_MAX_EVAL_IMAGES
images only.

Output in <label_remapped_eval_dir> (s05c_label_remapped_eval_CapAll, ...; below the s04b folder when variants are
on), one subfolder per dataset. A run replaces the subfolders of its datasets and leaves the others as they are; each
dataset is saved as soon as it is done:
    <dataset>/summary.csv      one row per class, circuit, variant and K
    <dataset>/reference.json   the full-model reference of the dataset and the VOC val like-for-like row
    <dataset>/lists.json       the images: left out (no object class), the reference images, the eval list per class
    <dataset>/run_info.json
    <dataset>/confusion_matrices/class_XX_images_with_class_c.npz   the vs-GT confusion matrices of every model and
                               variant (s05.save_counts, accuracy only), so later metrics need no new run
"""

import csv
import sys
import textwrap
import time
import warnings

import numpy as np

import config as C
import s05_evaluate as s05
from common import circuit_size, datasets, logit_standardization, metrics, model as M, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA",
    "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS",
    "LABEL_REMAPPED_EVAL_DATASETS", "LABEL_REMAPPED_MAX_EVAL_IMAGES", "LABEL_REMAPPED_EVAL_SEED",
    "LABEL_REMAPPED_CIRCUIT_TYPES", "LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS",
    "LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE", "SMOKE_TEST",
]
IMAGE_SET = "images_with_class_c"           # the only image set here (s05's name, for the npz files and summary.csv)
MEAN_IOU_COLUMN = "evaluable_classes_mIoU_vs_ground_truth"
VOC_REFERENCE_DATASET = "voc_val"
CAP_SETTING = "LABEL_REMAPPED_MAX_EVAL_IMAGES"
RATIO_COLUMN = s05.RATIO_COLUMN


def check_config():
    """Stops with a message on an unknown dataset, circuit type, variant or cap."""
    names = C.LABEL_REMAPPED_EVAL_DATASETS
    unknown = [name for name in names if name not in datasets.LABEL_REMAPPED_DATASETS]
    if not names or unknown or len(set(names)) != len(names):
        sys.exit(f"\n[config] LABEL_REMAPPED_EVAL_DATASETS = {names!r}: "
                 f"{'unknown ' + str(unknown) if unknown else 'empty or a dataset listed twice'}; "
                 f"allowed: {list(datasets.LABEL_REMAPPED_DATASETS)}")
    types = C.LABEL_REMAPPED_CIRCUIT_TYPES
    allowed = ["full", *s05.PRUNED_TYPES]
    wrong = [t for t in types if t not in allowed]
    if not types or wrong or len(set(types)) != len(types):
        sys.exit(f"\n[config] LABEL_REMAPPED_CIRCUIT_TYPES = {types!r}: "
                 f"{'unknown ' + str(wrong) if wrong else 'empty or a type listed twice'}; allowed: {allowed}")
    utils.eval_image_cap(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING)
    logit_standardization.check_variants(C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS,
                                         "LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS")


def variant_title(variant):
    """Title suffix of a log section: nothing without variants, else which variant the section shows."""
    if not C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS:
        return ""
    return (", RAW circuits (no logit standardization)" if variant == "raw"
            else f", logit standardization variant {variant}")


def how_to_read_variant(variant):
    """Opening lines of the "How to read" block of a section of one variant ("" for raw)."""
    if variant == "raw":
        return ""
    full = ("identical to raw (the matched variants are the identity for the full model)"
            if variant in logit_standardization.MATCHED_VARIANTS
            else "the full model standardized as well, so it differs from raw")
    return (f"LOGIT STANDARDIZATION VARIANT {variant}\n"
            f"Every number is computed from the prediction AFTER this variant:\n"
            + textwrap.indent(logit_standardization.VARIANT_DESCRIPTIONS[variant], "  ") + "\n"
            f"The statistics are s04b's, measured on VOC (LOGIT_STATISTICS_IMAGES = {C.LOGIT_STATISTICS_IMAGES!r}) and\n"
            f"applied to this dataset as they are. Same forward passes as the raw numbers. The full model's numbers:\n"
            f"{full}.\n")


def class_label(c):
    return f"{c} {VOC_CLASSES[c]}"


# ------------------------------------------------------------------
# Image lists
# ------------------------------------------------------------------
def build_lists(name, dataset, classes, discovery_samples):
    """The images of one dataset, from the remapped GT of every image (s05c's lists.json):
        num_images, images_without_object_class {indices, ids}  (left out: only background and / or 255)
        reference_indices   the images with at least one object class (SMOKE_TEST: the first CAP of them)
        eval {str(c): {indices, ids, num_available}}  per evaluated class c: the images with at least one pixel of c,
                            minus D_c on the discovery dataset (discovery_samples given), capped as s01 does
        excluded_discovery_samples  whether D_c was left out (the dataset is DISCOVERY_DATASET)"""
    print(f"\n[{name}] Counting the remapped GT pixels per VOC class in {len(dataset):,} images ...", flush=True)
    counts = datasets.class_pixel_counts(dataset)  # [images, 21], 255 excluded
    has_object = counts[:, 1:].sum(axis=1) > 0
    without = np.flatnonzero(~has_object)
    cap = utils.eval_image_cap(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING)
    reference_indices = np.flatnonzero(has_object)
    if C.SMOKE_TEST:
        reference_indices = reference_indices[:cap]   # cap None ("all"): every image
    eval_lists = {}
    for c in classes:
        exclude = discovery_samples[str(c)]["indices"] if discovery_samples is not None else []
        idx, n_available = utils.eval_list(c, counts, C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING,
                                           C.LABEL_REMAPPED_EVAL_SEED, exclude)
        eval_lists[str(c)] = {"indices": idx.tolist(), "ids": [dataset.ids[i] for i in idx],
                              "num_available": int(n_available)}
    return {
        "dataset": name,
        "max_eval_images": C.LABEL_REMAPPED_MAX_EVAL_IMAGES,
        "eval_seed": C.LABEL_REMAPPED_EVAL_SEED,
        "num_images": len(dataset),
        "images_without_object_class": {"indices": without.tolist(), "ids": [dataset.ids[i] for i in without]},
        "reference_indices": reference_indices.tolist(),
        "reference_smoke_test_slice": bool(C.SMOKE_TEST),
        "excluded_discovery_samples": discovery_samples is not None,
        "eval": eval_lists,
    }


# ------------------------------------------------------------------
# Full-model reference
# ------------------------------------------------------------------
def reference_record(cm, mean_iou_classes, all_classes_miou=False):
    """The full model's numbers from one vs-GT confusion matrix (reference.json)."""
    record = {
        "num_pixels": int(cm.sum()),
        "iou": metrics.class_scores(cm)["iou"].tolist(),
        "evaluable_classes_miou": metrics.mean_iou(cm, mean_iou_classes),
        "pixel_accuracy": metrics.pixel_agreement(cm),
        "confusion_matrix": cm.tolist(),
    }
    if all_classes_miou:
        record["miou_all_21_classes"] = metrics.mean_iou(cm)
    return record


def print_reference(name, info, dataset_cm, num_images, voc_cm, num_voc_images):
    evaluable = info["evaluable_classes"]
    mean_classes = [0, *evaluable]
    not_evaluable = [c for c in range(1, NUM_CLASSES) if c not in evaluable]
    smoke = (f" SMOKE_TEST: only the first {CAP_SETTING} = {C.LABEL_REMAPPED_MAX_EVAL_IMAGES!r} images of each."
             if C.SMOKE_TEST else "")
    utils.section(
        f"[{name}] Full-model reference: the unpruned model over {num_images:,} images, next to VOC val "
        f"({num_voc_images:,} images) on the same classes",
        f"""
        {name}: every image whose remapped GT holds at least one object class (images with only background and / or
        255 are left out, see the image counts above). {VOC_REFERENCE_DATASET}: all of its images.{smoke}
        IoU      per class, pixels pooled over all these images (also those without the class: false positives there
                 count), GT 255 excluded. nan = the class neither occurs nor is predicted.
        mIoU     evaluable-class mIoU: the mean IoU over background + the {len(evaluable)} evaluable classes of {name}
                 (the rows below), on both datasets: like-for-like. Not the usual 21-class mIoU; the VOC val 21-class
                 mIoU is printed below for comparison with s05's reference.
        pixel accuracy  share of the non-void pixels predicted correctly, all classes.
        Predictions of non-evaluable classes are kept (not masked): on {name} every such pixel is an error
        (usually a background pixel predicted as an object class).
        """)
    dataset_iou = metrics.class_scores(dataset_cm)["iou"]
    voc_iou = metrics.class_scores(voc_cm)["iou"]
    print(f"  {'class':16s} {name:>20s} {VOC_REFERENCE_DATASET:>12s}")
    for c in mean_classes:
        print(f"  {class_label(c):16s} {dataset_iou[c]:20.4f} {voc_iou[c]:12.4f}")
    print(f"  {'mIoU':16s} {metrics.mean_iou(dataset_cm, mean_classes):20.4f} "
          f"{metrics.mean_iou(voc_cm, mean_classes):12.4f}   (background + {len(evaluable)} evaluable classes)")
    print(f"  {'pixel accuracy':16s} {metrics.pixel_agreement(dataset_cm):20.4f} {metrics.pixel_agreement(voc_cm):12.4f}")
    print(f"\n  {VOC_REFERENCE_DATASET} mIoU over all {NUM_CLASSES} classes: {metrics.mean_iou(voc_cm):.4f}")
    if not_evaluable:
        total = dataset_cm.sum()
        shares = {c: dataset_cm[:, c].sum() / total if total else np.nan for c in not_evaluable}
        largest = sorted((c for c in not_evaluable if shares[c] > 0), key=lambda c: -shares[c])[:5]
        print(f"  {name}: pixels predicted as a non-evaluable class (all errors): "
              f"{sum(shares.values()):.3%} of the non-void pixels"
              + (f"; largest: {', '.join(f'{VOC_CLASSES[c]} {shares[c]:.3%}' for c in largest)}" if largest else ""))


# ------------------------------------------------------------------
# Circuit tables and summary.csv
# ------------------------------------------------------------------
def summary_columns(layers):
    return (["class", "class_name", "eval_dataset", "image_set", "circuit", "logit_standardization_variant",
             "topk_layer_normalized_ratio", "num_images"]
            + ["cIoU", MEAN_IOU_COLUMN, "class_c_predicted_pixel_ratio", "images_with_class_c_predicted_ratio"]
            + ["subnet_size_layer_normalized_ratio_mean"] + [f"subnet_size_layer_normalized_ratio_{layer}" for layer in layers]
            + ["subnet_size_group_channel_count", "subnet_size_group_pooled_ratio"]
            + ["subnet_size_kept_weight_count", "subnet_size_whole_model_weight_ratio", "subnet_size_reachable_weight_ratio"])


def summary_rows(c, name, results_c, num_images, columns):
    """summary.csv rows of one class (results_c: s05.summarise's {variant: {type: {metric: [per grid index]}}})."""
    renamed = {"ciou": "cIoU", "mIoU_vs_ground_truth": MEAN_IOU_COLUMN}
    rows = []
    for v, per_type in results_c.items():
        for t, r in per_type.items():
            for grid_idx, ratio in enumerate(r["topk_layer_normalized_ratio"]):
                row = {"class": c, "class_name": VOC_CLASSES[c], "eval_dataset": name, "image_set": IMAGE_SET,
                       "circuit": t, "logit_standardization_variant": v,
                       "topk_layer_normalized_ratio": "" if ratio is None else ratio, "num_images": num_images}
                for metric, values in r.items():
                    if metric != "topk_layer_normalized_ratio":
                        row[renamed.get(metric, metric)] = values[grid_idx]
                rows.append({column: row[column] for column in columns})
    return rows


def print_per_class_header(name, variant, pruned_types, best_width, with_time):
    utils.section(
        f"[{name}] Per class: images, the full model's cIoU, and each circuit type's best cIoU over the "
        f"{RATIO_COLUMN} grid{variant_title(variant)}",
        how_to_read_variant(variant) + textwrap.dedent(f"""
        images     images of class c used: remapped GT with at least one pixel of c. {utils.cap_text(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING)}.
        available  images with c in the dataset (on the discovery dataset without D_c); "(capped)" = fewer used.
        full cIoU  the unpruned model's class-c IoU against the remapped GT, pixels pooled over these images.
        full c-img the unpruned model's share of these images with at least one pixel predicted as c.
        best X     circuit type X's HIGHEST cIoU over the grid; "@ value" = the K where it occurs. A best at K = 1 is the
                   full model itself (every channel kept). The full curves are in the mean tables below and in s06c.
        """) if variant == "raw" else how_to_read_variant(variant) + "Columns as in the raw per-class table above "
                                                                       "(without the time).")
    print(f"{'':3s} {'class':13s} {'images':>7s} {'available':>9s} {'full cIoU':>9s} {'full c-img':>10s}"
          + "".join(f" {'best ' + t:>{best_width}s}" for t in pruned_types) + (f" {'time':>6s}" if with_time else ""))


def print_mean_table(name, results, classes, variant, num_evaluable):
    """Mean over the evaluated classes per circuit type and K, for one variant."""
    utils.section(
        f"[{name}] Circuit metrics per {RATIO_COLUMN}, mean over the {len(classes)} evaluated classes "
        f"(each class weighted equally){variant_title(variant)}",
        how_to_read_variant(variant) + textwrap.dedent(f"""
        Classes: {', '.join(VOC_CLASSES[c] for c in classes)} (the classes of CLASSES that {name} can score).
        Per class, pixels are pooled over the images of class c (remapped GT with at least one pixel of c); then
        each class is weighted equally. "-" = the full model. At K = 1 every circuit is the full model.
        {RATIO_COLUMN}
{textwrap.fill(utils.topk_ratio_description() + '.', 104, initial_indent=" " * 18, subsequent_indent=" " * 18)}
        size      subnet_size_layer_normalized_ratio_mean (s04): the ratio of each layer's channels kept, equal-weight
                  mean over the layers of U. Compare circuit types at equal size, not at equal K.
        cIoU      class-c IoU against the remapped GT. Compare with the full-model row.
        mIoU      evaluable_classes_mIoU_vs_ground_truth: mean IoU over background + the {num_evaluable} evaluable classes
                  of {name}, on the images of class c. It shows whether the circuit breaks the other classes.
        c-pixels  class_c_predicted_pixel_ratio: share of the non-void pixels predicted as c.
        c-images  images_with_class_c_predicted_ratio: share of the images with at least one pixel predicted as c.
        """))
    columns = [("ciou", "cIoU", "{:.3f}"), ("mIoU_vs_ground_truth", "mIoU", "{:.3f}"),
               ("class_c_predicted_pixel_ratio", "c-pixels", "{:.4f}"),
               ("images_with_class_c_predicted_ratio", "c-images", "{:.3f}")]
    widths = [max(len(header), len(fmt.format(0.0))) for _, header, fmt in columns]
    with warnings.catch_warnings():  # all-nan cells print as nan
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in C.LABEL_REMAPPED_CIRCUIT_TYPES:
            per_class = [results[c][variant][t] for c in classes]
            print(f"  circuit type: {t}")
            print(f"  {RATIO_COLUMN} {'size':>6s}" + "".join(f" {header:>{w}s}" for (_, header, _), w in zip(columns, widths)))
            for grid_idx, ratio in enumerate(per_class[0]["topk_layer_normalized_ratio"]):
                size = np.nanmean([r["subnet_size_layer_normalized_ratio_mean"][grid_idx] for r in per_class])
                values = [np.nanmean([r[metric][grid_idx] for r in per_class]) for metric, _, _ in columns]
                label = "-" if ratio is None else f"{ratio:g}"
                print(f"  {label:>{len(RATIO_COLUMN)}s} {size:6.3f}"
                      + "".join(f" {'nan' if np.isnan(v) else fmt.format(v):>{w}s}"
                                for v, (_, _, fmt), w in zip(values, columns, widths)))
            print()


# ------------------------------------------------------------------
def main():
    utils.start_run("s05c_evaluate_label_remapped_datasets", USED_PARAMS)
    check_config()
    utils.require(utils.certify_dir() / "run_info.json", "s04_certify.py")
    variants = list(C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS)
    if variants:
        utils.require(utils.logit_statistics_dir() / "run_info.json", "s04b_logit_statistics.py")
    utils.require_datasets([*C.LABEL_REMAPPED_EVAL_DATASETS, VOC_REFERENCE_DATASET])
    discovery_samples = None
    if C.DISCOVERY_DATASET in C.LABEL_REMAPPED_EVAL_DATASETS:  # e.g. coco_val2017_voc as both discovery and eval dataset
        discovery_samples = utils.load_discovery_and_eval_lists()["discovery_samples"]
    out_dir = utils.label_remapped_eval_dir()
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)

    layers, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    full_masks = {layer: np.ones(s.stop - s.start, dtype=bool) for layer, s in certifiable_slices.items()}
    full_key = utils.mask_key(full_masks)
    pruned_types = [t for t in s05.PRUNED_TYPES if t in C.LABEL_REMAPPED_CIRCUIT_TYPES]
    infos = {name: datasets.LABEL_REMAPPED_DATASETS[name] for name in C.LABEL_REMAPPED_EVAL_DATASETS}
    classes_of = {name: [c for c in C.CLASSES if c in info["evaluable_classes"]] for name, info in infos.items()}
    needed = sorted({c for classes in classes_of.values() for c in classes})
    circuits = {c: utils.load_circuits(c, certifiable_slices, pruned_types) for c in needed}
    logit_statistics = {c: logit_standardization.load_class_logit_statistics(c, circuits[c][0]) if variants
                        else (None, None) for c in needed}
    counter = circuit_size.WeightMaskCounter(model, layers)
    sizes = {c: {"full": [counter.size_measures(full_masks)],
                 **{t: [counter.size_measures(masks) for masks, _ in circuits[c][0][t]] for t in pruned_types}}
             for c in needed}
    distinct_models = {c: len({full_key} | {key for t in pruned_types for _, key in circuits[c][0][t]}) for c in needed}
    columns = summary_columns(layers)

    utils.section(
        "Label-remapped datasets: accuracy of the circuits (s04, discovered on VOC+SBD) with the GT remapped to VOC",
        textwrap.dedent(f"""
        Output: {out_dir}
                one subfolder per dataset; this run replaces {', '.join(C.LABEL_REMAPPED_EVAL_DATASETS)} and leaves any
                other dataset subfolder as it is. Each dataset is saved as soon as it is done.
        Circuit types: {', '.join(C.LABEL_REMAPPED_CIRCUIT_TYPES)} (LABEL_REMAPPED_CIRCUIT_TYPES), every K of s04's grid.
        Logit standardization: {"raw, and after each of " + ", ".join(variants) + " (LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS; s04b's statistics from " + utils.logit_statistics_dir().name + ")" if variants else "off (raw circuits only)"}.
        {utils.cap_text(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING)}.
        Cityscapes resolution: {"native 2048x1024 (no resize)" if C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE is None else f"resized to {C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE} (width, height): image bilinear, labels nearest"} (LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE).
        Accuracy only (s05's group 1). Metrics: see the "How to read" block of each table.
        """) + (f"\nD_c is left out of the class lists of {C.DISCOVERY_DATASET} (it is also DISCOVERY_DATASET), as s01 "
                f"does; the full-model reference uses all of its images with an object class, as s05 does.\n"
                if discovery_samples is not None else ""))

    # ---- lists of every dataset first, so the cost is known before the long part ----------------
    loaded, lists = {}, {}
    for name in C.LABEL_REMAPPED_EVAL_DATASETS:
        loaded[name] = datasets.get_dataset(name)
        lists[name] = build_lists(name, loaded[name], classes_of[name],
                                  discovery_samples if name == C.DISCOVERY_DATASET else None)
    voc_dataset = datasets.get_dataset(VOC_REFERENCE_DATASET)
    voc_cap = utils.eval_image_cap(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING) if C.SMOKE_TEST else None
    voc_indices = list(range(len(voc_dataset)))[:voc_cap]

    utils.section(
        "Cost of this run: images and forward passes per dataset",
        textwrap.dedent(f"""
        images         images in the dataset
        no object      images whose remapped GT holds only background and / or 255: left out completely
        reference      the full-model reference images (one forward pass each){" (SMOKE_TEST: the first images only)" if C.SMOKE_TEST else ""}
        classes        classes evaluated: those of CLASSES the dataset can score
        image-class    the sum over these classes of the images used per class
        passes         forward passes of the circuit part: per class, images × distinct models (the full model and
                       every circuit with a distinct mask; identical masks run once). The variants add none.
        Time scales with the passes and the image size (Cityscapes native: ≈ 11× a VOC image's pixels).
        {VOC_REFERENCE_DATASET} like-for-like reference: {len(voc_indices):,} forward passes, once for all datasets.
        """))
    print(f"  {'dataset':20s} {'images':>7s} {'no object':>9s} {'reference':>9s} {'classes':>7s} {'image-class':>11s} "
          f"{'passes':>9s}")
    for name in C.LABEL_REMAPPED_EVAL_DATASETS:
        entries = lists[name]["eval"]
        pairs = sum(len(entries[str(c)]["indices"]) for c in classes_of[name])
        passes = sum(len(entries[str(c)]["indices"]) * distinct_models[c] for c in classes_of[name])
        print(f"  {name:20s} {lists[name]['num_images']:7,d} {len(lists[name]['images_without_object_class']['ids']):9,d} "
              f"{len(lists[name]['reference_indices']):9,d} {len(classes_of[name]):7d} {pairs:11,d} {passes:9,d}")

    print(f"\n[{VOC_REFERENCE_DATASET}] Full-model reference over {len(voc_indices):,} images (like-for-like rows) ...",
          flush=True)
    voc_cm = s05.reference(model, voc_dataset, voc_indices, device)

    for name in C.LABEL_REMAPPED_EVAL_DATASETS:
        dataset, info, classes, dataset_lists = loaded[name], infos[name], classes_of[name], lists[name]
        evaluable = info["evaluable_classes"]
        mean_iou_classes = [0, *evaluable]
        folder = out_dir / name
        print(f"\n{'=' * 100}\nLABEL-REMAPPED DATASET: {name}\n{'=' * 100}", flush=True)

        # ---- mapping and images ------------------------------------------------------------
        utils.section(
            f"[{name}] Label mapping: {info['source']}",
            textwrap.dedent(f"""
            Each source class goes to a VOC class, to 0 background (clearly none of the 20 VOC classes) or to 255
            ignore (may contain VOC-class pixels but cannot be mapped one-to-one; excluded from every metric). The
            source dataset's own void / unlabelled pixels go to 255. Counts per source class: the setup check's
            label check (CHECK_LABEL_REMAPPING).
            {info['title_note']}: {', '.join(VOC_CLASSES[c] for c in evaluable)}.
            Evaluated here (evaluable and in CLASSES): {', '.join(VOC_CLASSES[c] for c in classes) or 'none'}.
            """) + (f"In CLASSES but not evaluable on {name} (never in its GT): "
                    f"{', '.join(VOC_CLASSES[c] for c in C.CLASSES if c not in evaluable)}.\n"
                    if any(c not in evaluable for c in C.CLASSES) else ""))
        datasets.print_mapping_table(dataset)

        utils.section(
            f"[{name}] Images",
            textwrap.dedent(f"""
            no object class   images whose remapped GT holds only background and / or 255: left out completely, from
                              the class lists and from the full-model reference.
            reference         the images with at least one object class: the full-model reference.{" SMOKE_TEST: only the first of them." if C.SMOKE_TEST else ""}
            per class         images whose remapped GT holds at least one pixel of c: available -> used
                              ({utils.cap_text(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, CAP_SETTING)}){"; D_c left out" if dataset_lists["excluded_discovery_samples"] else ""}.
            """))
        n_without = len(dataset_lists["images_without_object_class"]["ids"])
        print(f"  images in the dataset:      {dataset_lists['num_images']:7,d}")
        print(f"  without any object class:   {n_without:7,d}  "
              f"({n_without / dataset_lists['num_images']:.1%}; left out)")
        print(f"  reference images:           {len(dataset_lists['reference_indices']):7,d}")
        for c in classes:
            entry = dataset_lists["eval"][str(c)]
            used, available = len(entry["indices"]), entry["num_available"]
            print(f"    {class_label(c):16s} {available:6,d} -> {used:6,d}" + (" (capped)" if used < available else "")
                  + ("   WARNING: no image of this class" if available == 0 else ""))

        # ---- full-model reference ----------------------------------------------------------
        print(f"\n[{name}] Full-model reference over {len(dataset_lists['reference_indices']):,} images ...", flush=True)
        dataset_cm = s05.reference(model, dataset, dataset_lists["reference_indices"], device)
        print_reference(name, info, dataset_cm, len(dataset_lists["reference_indices"]), voc_cm, len(voc_indices))
        reference_json = {
            "dataset": name, "source": info["source"], "title_note": info["title_note"],
            "evaluable_classes": evaluable, "mean_iou_classes": mean_iou_classes,
            "num_images": len(dataset_lists["reference_indices"]),
            "smoke_test_slice": bool(C.SMOKE_TEST),
            **reference_record(dataset_cm, mean_iou_classes),
            "voc_val_like_for_like": {"dataset": VOC_REFERENCE_DATASET, "num_images": len(voc_indices),
                                      **reference_record(voc_cm, mean_iou_classes, all_classes_miou=True)},
        }

        # ---- circuits, per class -------------------------------------------------------------
        rows, results, lines = [], {}, {v: {} for v in ["raw", *variants]}
        best_width = max([17] + [len("best " + t) for t in pruned_types])
        if classes:
            print_per_class_header(name, "raw", pruned_types, best_width, with_time=True)
        for c in classes:
            start = time.time()
            class_circuits, subnet_size_layer_normalized_ratio_mean, grid = circuits[c]
            entry = dataset_lists["eval"][str(c)]
            accumulators = s05.evaluate_class(model, dataset, entry["indices"], c, class_circuits, full_key, device,
                                              logit_statistics[c], variants, accuracy_only=True)
            results[c], _ = s05.summarise(accumulators, class_circuits, full_key, c, sizes[c],
                                          subnet_size_layer_normalized_ratio_mean, grid, variants,
                                          C.LABEL_REMAPPED_CIRCUIT_TYPES, accuracy_only=True,
                                          mean_iou_classes=mean_iou_classes)
            s05.save_counts(folder / "confusion_matrices" / f"class_{c:02d}_{IMAGE_SET}.npz", accumulators,
                            class_circuits, full_key, grid, entry, variants, accuracy_only=True)
            rows += summary_rows(c, name, results[c], len(entry["indices"]), columns)
            for v in ["raw", *variants]:
                full = accumulators[(full_key, v)]
                n = len(entry["indices"])
                lines[v][c] = (f"{c:3d} {VOC_CLASSES[c]:13s} {n:7,d} {entry['num_available']:9,d} "
                               f"{s05.iou(*full['counts'][:3]):9.3f} "
                               f"{full['images_with_class_c_predicted'] / n if n else np.nan:10.3f}"
                               + s05.best_cells(results[c][v], class_circuits, grid, best_width))
            print(f"{lines['raw'][c]} {time.time() - start:5.0f}s", flush=True)
        if classes:
            print_mean_table(name, results, classes, "raw", len(evaluable))
            for v in variants:
                print_per_class_header(name, v, pruned_types, best_width, with_time=False)
                for c in classes:
                    print(lines[v][c])
                print_mean_table(name, results, classes, v, len(evaluable))
        else:
            print(f"\n[{name}] No class of CLASSES is evaluable here: the reference only, no circuits.")

        # ---- save this dataset -------------------------------------------------------------
        utils.save_json(folder / "reference.json", reference_json)
        utils.save_json(folder / "lists.json", dataset_lists)
        with open(folder / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        utils.write_run_info(folder)
        print(f"\n[{name}] Saved summary.csv ({len(rows)} rows), reference.json, lists.json, run_info.json and "
              f"{len(classes)} confusion-matrix files (confusion_matrices/) to {folder}", flush=True)

    print(f"\nDone: {len(C.LABEL_REMAPPED_EVAL_DATASETS)} dataset(s) in {out_dir}")


if __name__ == "__main__":
    main()
