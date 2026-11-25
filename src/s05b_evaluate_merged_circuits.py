"""s05b · Evaluation of merged circuits.

A circuit evaluated only on its own class can look better than it is (Gullstrand et al. 2026, §5.1.1:
a subnetwork scores close to perfect on its own class but poorly on the others). Here every class circuit is merged
with every other class circuit, and the merged circuit is tested on both classes:
    class pairs     every pair (c, c') of CLASSES with c < c' (190 pairs for the 20 VOC classes)
    merged circuit  per circuit type in CIRCUIT_TYPES (baseline_circuit, majority, certified) and K (the top-K
                    layer-normalized ratio, or with a global TOPK_SELECTION_MODE the top-K global ratio): the UNION of the two class circuits' channel masks from s04 (a channel
                    of U is kept if either circuit keeps it). It is NOT certified again: each class circuit is
                    certified on its own (s04), then merged. Nothing changes in s03/s04.
    images          per eval dataset in EVAL_DATASETS: the union of the two classes' s01 eval lists WITH their class
                    (e.g. dog ∪ cat circuit on all dog and all cat eval images; an image in both lists counts once).
                    On "discovery_samples": D_c ∪ D_c'.
    reconstruction  the all-classes union: the union of the circuits of every class in CLASSES, on the union of all
                    their eval lists. It shows how much of the full model the class circuits rebuild together.

Metrics, pixels pooled over the images of the pair (the full model on the same images is the reference row):
    cIoU_class_1, cIoU_class_2            class-c and class-c' IoU against the GT, void excluded. The pooled images
                                          include the other class's images, so a c-pixel predicted on a c' image is a
                                          false positive of c here
    faith_IoU_class_1, faith_IoU_class_2  the same against the RAW full model's prediction (all pixels)
    mIoU_vs_ground_truth, pixel_agreement_with_full_model, class_share_total_variation_to_full_model   as in s05
    size of the merged circuit: subnet_size_layer_normalized_ratio_mean (channel share per layer of U, equal-weight
    mean), the per-layer shares, the pooled sizes over all channels of U and the weight measures of
    common/circuit_size.py (binary-mask convention)
All-classes union: mIoU, agreement and total variation as above, plus the mean over CLASSES of each class's IoU
against the GT and against the full model (per class in the npz file).

Raw circuits only: no logit-standardization variants, because s04b fits statistics per class circuit and a merged
circuit has none.

Cost: per pair and eval dataset, one full-model pass and one pass per distinct merged circuit on every image of the
two lists; up to ≈ 19× s05's single-class passes per eval set. Circuits with
identical masks run once per image (at K = 1 every merged circuit is the full model).

Output in <certify_dir>/s05b_merged_circuits_eval_Cap<MAX_EVAL_IMAGES>/ (_CapAll for "all"; the eval lists with c
depend on the cap):
    class_pair_XX_YY.npz          per pair, written as soon as the pair is done (layout: save_pair)
    all_classes_union.npz
    summary_class_pairs.csv, summary_all_classes_union.csv, run_info.json
"""

import csv
import itertools
import sys
import textwrap
import time
import warnings

import numpy as np
import torch

import config as C
from common import circuit_size, metrics, model as M, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "EVAL_DATASETS", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "MAX_EVAL_IMAGES",
    "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA", "CIRCUIT_TYPES", "SMOKE_TEST",
]
PRUNED_TYPES = ["baseline_circuit", "majority", "certified"]
RATIO_COLUMN = utils.topk_ratio_label()  # the name of K: "top-K layer-normalized ratio" (per_layer) or "top-K global ratio of <LAYERS>"


def merge(class_circuits, types, n_grid_values):
    """The merged circuits of the given classes: {type: [({layer: 0/1 mask}, key) per grid index]}, each mask the
    union (logical or) of the classes' masks. class_circuits: a list of utils.load_circuits(...)[0]."""
    out = {}
    for t in types:
        out[t] = []
        for grid_idx in range(n_grid_values):
            layers = class_circuits[0][t][grid_idx][0]
            masks = {layer: np.logical_or.reduce([circuits[t][grid_idx][0][layer] for circuits in class_circuits])
                     for layer in layers}
            out[t].append((masks, utils.mask_key(masks)))
    return out


def subnet_size_layer_normalized_ratio_mean(masks):
    """As in s04: the share of each layer's channels in the circuit, equal-weight mean over the layers of U."""
    return float(np.mean([np.count_nonzero(m) / len(m) for m in masks.values()]))


@torch.inference_mode()
def evaluate(model, dataset, indices, merged, full_key, device):
    """{mask key: (confusion matrix vs GT, vs the raw full model)} of the full model and every distinct merged circuit,
    pixels pooled over the images `indices` (rows = GT / full model, columns = the model's prediction)."""
    distinct = {full_key: None}
    for circuits_per_grid_idx in merged.values():
        for masks, key in circuits_per_grid_idx:
            distinct.setdefault(key, masks)
    matrices = {key: [torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device) for _ in range(2)]
                for key in distinct}
    for i in indices:
        image, mask = dataset[i]
        x = M.preprocess(image, device)
        gt = torch.as_tensor(mask, device=device)
        full_labels = M.forward(model, x).argmax(dim=0)
        for key, masks in distinct.items():
            pred = full_labels if masks is None else M.forward(model, x, masks).argmax(dim=0)
            matrices[key][0] += metrics.confusion_matrix_torch(pred, gt)
            matrices[key][1] += metrics.confusion_matrix_torch(pred, full_labels)
    return {key: (vs_gt.cpu().numpy(), vs_full.cpu().numpy()) for key, (vs_gt, vs_full) in matrices.items()}, \
        len(distinct) - 1


def models_of(merged, full_key, topk_layer_normalized_ratio_grid):
    """[(circuit type, K or None, masks or None, mask key)]: the full model first, then every merged circuit."""
    return [("full", None, None, full_key)] + [
        (t, topk_layer_normalized_ratio_grid[grid_idx], masks, key)
        for t, circuits_per_grid_idx in merged.items() for grid_idx, (masks, key) in enumerate(circuits_per_grid_idx)]


def multi_class_metrics(vs_gt, vs_full):
    return {"mIoU_vs_ground_truth": metrics.mean_iou(vs_gt),
            "pixel_agreement_with_full_model": metrics.pixel_agreement(vs_full),
            "class_share_total_variation_to_full_model": metrics.class_share_total_variation(vs_full)}


def size_columns(masks, counter, full_masks):
    masks = full_masks if masks is None else masks
    return {"subnet_size_layer_normalized_ratio_mean": subnet_size_layer_normalized_ratio_mean(masks),
            **counter.size_measures(masks)}


def save(path, models, matrices_per_dataset, num_images):
    """npz of one pair (or of the all-classes union): E = EVAL_DATASETS, M = models (the full model, then every
    circuit type × K)
        confusion_matrix_vs_ground_truth, confusion_matrix_vs_full_model   [E, M, 21, 21] int64
        num_images [E], eval_datasets [E], model_circuit_types [M], model_topk_layer_normalized_ratio [M] (nan: full)"""
    np.savez_compressed(
        path,
        confusion_matrix_vs_ground_truth=np.stack([np.stack([matrices[key][0] for *_, key in models])
                                                   for matrices in matrices_per_dataset]),
        confusion_matrix_vs_full_model=np.stack([np.stack([matrices[key][1] for *_, key in models])
                                                 for matrices in matrices_per_dataset]),
        num_images=np.array(num_images), eval_datasets=np.array(C.EVAL_DATASETS),
        model_circuit_types=np.array([t for t, *_ in models]),
        model_topk_layer_normalized_ratio=np.array([np.nan if k is None else k for _, k, *_ in models], dtype=np.float64))


def print_pair_means(name, rows, layers):
    """Mean over the class pairs per circuit type and K, for one eval dataset."""
    columns = [("cIoU", "cIoU", "{:.3f}"), ("faith_IoU", "faith", "{:.3f}"), ("mIoU_vs_ground_truth", "mIoU", "{:.3f}"),
               ("pixel_agreement_with_full_model", "agreement", "{:.3f}"),
               ("class_share_total_variation_to_full_model", "class-share TV", "{:.3f}"),
               ("subnet_size_whole_model_weight_ratio", "whole", "{:.3f}"),
               ("subnet_size_reachable_weight_ratio", "reachable", "{:.3f}")]
    rows = [r for r in rows if r["eval_dataset"] == name]
    n_pairs = len({(r["class_1"], r["class_2"]) for r in rows})
    utils.section(
        f"[{name}] Merged circuits (union of two class circuits), mean over the {n_pairs} class pairs",
        textwrap.dedent(f"""
        Each pair (c, c') is tested on the union of the eval images WITH c and WITH c' ({name}), pixels pooled.
        size       subnet_size_layer_normalized_ratio_mean of the MERGED circuit (channel share per layer of U, equal-
                   weight mean). A union is larger than either class circuit, so compare at equal size.
        cIoU       the mean of cIoU_class_1 and cIoU_class_2: each class's IoU against the GT over ALL images of the
                   pair, so c-pixels predicted on c' images count as false positives of c. Compare with the full row.
        faith      the same against the RAW full model's prediction (faith_IoU_class_1 / _class_2).
        mIoU, agreement, class-share TV   as in s05 (all {NUM_CLASSES} classes; the full model: agreement 1, TV 0).
        whole      subnet_size_whole_model_weight_ratio of the merged circuit (weights kept / all weights).
        reachable  subnet_size_reachable_weight_ratio ({circuit_size.REACHABLE_WEIGHTS_LABEL}).
        "-" = the full model. At K = 1 every merged circuit is the full model. Per pair: summary_class_pairs.csv.
        """))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in [t for t in C.CIRCUIT_TYPES if t == "full" or t in PRUNED_TYPES]:
            selected = [r for r in rows if r["circuit"] == t]
            if not selected:
                continue
            ratios = sorted({r["topk_layer_normalized_ratio"] for r in selected}, key=lambda k: -1 if k == "" else k)
            widths = [max(len(header), 6) for _, header, _ in columns]
            print(f"  circuit type: {t}")
            print(f"  {RATIO_COLUMN} {'size':>6s}" + "".join(f" {h:>{w}s}" for (_, h, _), w in zip(columns, widths)))
            for k in ratios:
                group = [r for r in selected if r["topk_layer_normalized_ratio"] == k]
                values = {
                    "cIoU": np.nanmean([[r["cIoU_class_1"], r["cIoU_class_2"]] for r in group]),
                    "faith_IoU": np.nanmean([[r["faith_IoU_class_1"], r["faith_IoU_class_2"]] for r in group]),
                    **{key: np.nanmean([r[key] for r in group]) for key, _, _ in columns[2:]},
                }
                size = np.mean([r["subnet_size_layer_normalized_ratio_mean"] for r in group])
                label = "-" if k == "" else f"{k:g}"
                print(f"  {label:>{len(RATIO_COLUMN)}s} {size:6.3f}"
                      + "".join(f" {'nan' if np.isnan(values[key]) else fmt.format(values[key]):>{w}s}"
                                for (key, _, fmt), w in zip(columns, widths)))
            print()


def main():
    utils.start_run("s05b_evaluate_merged_circuits", USED_PARAMS)
    utils.require(utils.certify_dir() / "run_info.json", "s04_certify.py")
    discovery_and_eval_lists = utils.load_discovery_and_eval_lists()
    utils.check_eval_list_cap(discovery_and_eval_lists)
    missing = [name for name in C.EVAL_DATASETS if name not in discovery_and_eval_lists["eval"]]
    if missing:
        sys.exit(f"\n[missing input] no eval lists for {missing} in discovery_and_eval_lists.json\n"
                 f"-> re-run s01_discovery_samples_dataset.py with them in EVAL_DATASETS.")
    if len(C.CLASSES) < 2:
        sys.exit("\n[config] s05b merges pairs of class circuits, so CLASSES needs at least 2 classes.")
    out_dir = utils.merged_circuits_eval_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)
    datasets = {name: utils.eval_dataset(name) for name in C.EVAL_DATASETS}

    layers, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    full_masks = {name: np.ones(s.stop - s.start, dtype=bool) for name, s in certifiable_slices.items()}
    full_key = utils.mask_key(full_masks)
    types = [t for t in PRUNED_TYPES if t in C.CIRCUIT_TYPES]
    loaded = {c: utils.load_circuits(c, certifiable_slices, types) for c in C.CLASSES}
    grid = loaded[C.CLASSES[0]][2]
    if not all(np.array_equal(loaded[c][2], grid) for c in C.CLASSES):
        sys.exit("\n[input] the classes have different top-K layer-normalized ratio grids in s04, so their circuits "
                 "cannot be merged K by K.\n-> re-run s03 and s04 with the same TOPK_LAYER_NORMALIZED_RATIO_GRID for "
                 "every class.")
    counter = circuit_size.WeightMaskCounter(model, layers)
    pairs = list(itertools.combinations(C.CLASSES, 2))

    utils.section(
        f"Merged circuits: {len(pairs)} class pairs + the all-{len(C.CLASSES)}-classes union, circuit types {types}, "
        f"{len(grid)} values of K, eval datasets {C.EVAL_DATASETS}",
        textwrap.dedent(f"""
        Merged circuit of (c, c') = the union of the two class circuits' channel masks from s04 (a channel of U is kept
        if either circuit keeps it), per circuit type and K. Not certified again. Raw circuits only (no logit-
        standardization variants: a merged circuit has no s04b statistics).
        Images per pair and eval dataset: the s01 eval lists WITH c and WITH c', joined (an image in both counts once);
        on discovery_samples D_c ∪ D_c' (in-sample). The full model runs on the same images as the reference.
        Per pair below: the images per eval dataset, the distinct merged circuits (identical masks run once; at K = 1
        every merged circuit is the full model and is not counted) and the time. Per-pair numbers are in
        summary_class_pairs.csv; the tables after the loop are means over the pairs.
        """))
    header = f"  {'pair':28s}" + "".join(f" {name[:22]:>22s}" for name in C.EVAL_DATASETS) + f" {'distinct':>8s} {'time':>6s}"
    print(header)
    columns = (["class_1", "class_1_name", "class_2", "class_2_name", "eval_dataset", "circuit",
                "topk_layer_normalized_ratio", "num_images", "cIoU_class_1", "cIoU_class_2", "faith_IoU_class_1",
                "faith_IoU_class_2", "mIoU_vs_ground_truth", "pixel_agreement_with_full_model",
                "class_share_total_variation_to_full_model", "subnet_size_layer_normalized_ratio_mean"]
               + [f"subnet_size_layer_normalized_ratio_{layer}" for layer in layers]
               + ["subnet_size_group_channel_count", "subnet_size_group_pooled_ratio"]
               + ["subnet_size_kept_weight_count", "subnet_size_whole_model_weight_ratio",
                  "subnet_size_reachable_weight_ratio"])
    rows = []
    for c1, c2 in pairs:
        start = time.time()
        merged = merge([loaded[c1][0], loaded[c2][0]], types, len(grid))
        models = models_of(merged, full_key, grid)
        matrices_per_dataset, num_images = [], []
        for name in C.EVAL_DATASETS:
            lists = discovery_and_eval_lists["eval"][name]
            indices = sorted(set(lists[str(c1)]["indices"]) | set(lists[str(c2)]["indices"]))
            matrices, n_distinct = evaluate(model, datasets[name], indices, merged, full_key, device)
            matrices_per_dataset.append(matrices)
            num_images.append(len(indices))
            for t, k, masks, key in models:
                vs_gt, vs_full = matrices[key]
                rows.append({"class_1": c1, "class_1_name": VOC_CLASSES[c1], "class_2": c2,
                             "class_2_name": VOC_CLASSES[c2], "eval_dataset": name, "circuit": t,
                             "topk_layer_normalized_ratio": "" if k is None else float(k), "num_images": len(indices),
                             "cIoU_class_1": metrics.class_scores(vs_gt)["iou"][c1],
                             "cIoU_class_2": metrics.class_scores(vs_gt)["iou"][c2],
                             "faith_IoU_class_1": metrics.class_scores(vs_full)["iou"][c1],
                             "faith_IoU_class_2": metrics.class_scores(vs_full)["iou"][c2],
                             **multi_class_metrics(vs_gt, vs_full), **size_columns(masks, counter, full_masks)})
        save(out_dir / f"class_pair_{c1:02d}_{c2:02d}.npz", models, matrices_per_dataset, num_images)
        label = f"{c1:2d} {VOC_CLASSES[c1]} + {c2:2d} {VOC_CLASSES[c2]}"
        print(f"  {label:28s}" + "".join(f" {n:22,d}" for n in num_images) + f" {n_distinct:8d}"
              f" {time.time() - start:5.0f}s", flush=True)

    for name in C.EVAL_DATASETS:
        print_pair_means(name, rows, layers)

    # ---- the all-classes union (reconstruction check) --------------------------------------------
    merged = merge([loaded[c][0] for c in C.CLASSES], types, len(grid))
    models = models_of(merged, full_key, grid)
    union_columns = (["eval_dataset", "circuit", "topk_layer_normalized_ratio", "num_images", "mIoU_vs_ground_truth",
                      "cIoU_mean_over_classes", "faith_IoU_mean_over_classes", "pixel_agreement_with_full_model",
                      "class_share_total_variation_to_full_model"] + columns[15:])
    union_rows, matrices_per_dataset, num_images = [], [], []
    for name in C.EVAL_DATASETS:
        start = time.time()
        lists = discovery_and_eval_lists["eval"][name]
        indices = sorted({i for c in C.CLASSES for i in lists[str(c)]["indices"]})
        matrices, n_distinct = evaluate(model, datasets[name], indices, merged, full_key, device)
        matrices_per_dataset.append(matrices)
        num_images.append(len(indices))
        for t, k, masks, key in models:
            vs_gt, vs_full = matrices[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                union_rows.append({
                    "eval_dataset": name, "circuit": t, "topk_layer_normalized_ratio": "" if k is None else float(k),
                    "num_images": len(indices),
                    "cIoU_mean_over_classes": float(np.nanmean(metrics.class_scores(vs_gt)["iou"][C.CLASSES])),
                    "faith_IoU_mean_over_classes": float(np.nanmean(metrics.class_scores(vs_full)["iou"][C.CLASSES])),
                    **multi_class_metrics(vs_gt, vs_full), **size_columns(masks, counter, full_masks)})
        print(f"  all-classes union on {name}: {len(indices):,} images, {n_distinct} distinct merged circuits, "
              f"{time.time() - start:.0f}s", flush=True)
    save(out_dir / "all_classes_union.npz", models, matrices_per_dataset, num_images)

    utils.section(
        f"All-{len(C.CLASSES)}-classes union (reconstruction check): the union of every class circuit in CLASSES",
        textwrap.dedent(f"""
        One merged circuit per circuit type and K: the union of the circuits of all {len(C.CLASSES)} classes. Images: the
        union of every class's eval list WITH its class (per eval dataset). If the class circuits together rebuild the
        full model, every number here approaches the full row.
        cIoU   cIoU_mean_over_classes: each class in CLASSES' IoU against the GT over these images, mean over the
               classes (per class in all_classes_union.npz).
        faith  faith_IoU_mean_over_classes: the same against the RAW full model.
        mIoU, agreement, class-share TV   as in s05 (all {NUM_CLASSES} classes).
        size / whole / reachable          the union's size measures ({circuit_size.REACHABLE_WEIGHTS_LABEL}).
        "-" = the full model.
        """))
    for name in C.EVAL_DATASETS:
        print(f"\n  [{name}]")
        print(f"  {'circuit type':16s} {RATIO_COLUMN} {'size':>6s} {'cIoU':>6s} {'faith':>6s} {'mIoU':>6s} "
              f"{'agreement':>9s} {'class-share TV':>14s} {'whole':>6s} {'reachable':>9s}")
        for r in [r for r in union_rows if r["eval_dataset"] == name]:
            k = r["topk_layer_normalized_ratio"]
            print(f"  {r['circuit']:16s} {'-' if k == '' else f'{k:g}':>{len(RATIO_COLUMN)}s} "
                  f"{r['subnet_size_layer_normalized_ratio_mean']:6.3f} {r['cIoU_mean_over_classes']:6.3f} "
                  f"{r['faith_IoU_mean_over_classes']:6.3f} {r['mIoU_vs_ground_truth']:6.3f} "
                  f"{r['pixel_agreement_with_full_model']:9.3f} {r['class_share_total_variation_to_full_model']:14.3f} "
                  f"{r['subnet_size_whole_model_weight_ratio']:6.3f} {r['subnet_size_reachable_weight_ratio']:9.3f}")

    for path, header_columns, table in ((out_dir / "summary_class_pairs.csv", columns, rows),
                                        (out_dir / "summary_all_classes_union.csv", union_columns, union_rows)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_columns)
            writer.writeheader()
            writer.writerows(table)
    utils.write_run_info(out_dir)
    print(f"\nSaved {len(pairs)} class-pair files, all_classes_union.npz and the two summary tables to {out_dir}")


if __name__ == "__main__":
    main()
