"""Setup check & baseline (run before s01).

1. Datasets: with CHECK_DOWNLOAD, VOC 2012 and SBD are first downloaded if missing. Then,
   for every dataset in CHECK_DATASETS, its size and the number of images and GT pixels
   per VOC class. Datasets that are not on disk are skipped.
1b. Label check (only with CHECK_LABEL_REMAPPING), for every label-remapped dataset in CHECK_DATASETS
   (COCO val2017, ADE20K, Cityscapes; their GT is remapped into the VOC labels): the mapping table
   with the images and pixels of every source class, the remapped GT's pixels and images per VOC
   class, the images without any object class (s05c leaves them out), values outside the class
   list, and CHECK_LABEL_REMAPPING_NUM_EXAMPLES figures (image | original labels | remapped labels).
   Check this output before running s05c.
2. Model: loads MODEL_NAME, prints the hook points and, with PRINT_MODEL, the
   architecture and parameters per layer.
3. Example: runs image CHECK_EXAMPLE_INDEX of every loaded dataset through the model
   at native resolution and saves image / GT / prediction side by side.
4. Baseline (only with RUN_BASELINE): confusion matrix and IoU / Dice / precision /
   recall of the unpruned model over BASELINE_DATASETS (capped at BASELINE_MAX_IMAGES).
   For a label-remapped dataset also the mIoU over its evaluable classes.

Output: images/setup_check/<dataset>_example<index>.png, with CHECK_LABEL_REMAPPING
images/setup_check/label_remapping/<dataset>/example_<index>_<image id>.png and, with RUN_BASELINE,
results/baseline/<MODEL_NAME>[_max<n>]/baseline_metrics.json and run_info.json.
"""

import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, Dataset, Subset

import config as C
from common import datasets, metrics, model as M, utils
from common.datasets import LABEL_REMAPPED_DATASETS, NUM_CLASSES, VOC_CLASSES, VOID_LABEL

USED_PARAMS = [
    "DATA_DIR", "RESULTS_DIR", "IMAGES_DIR", "MODEL_NAME", "DEVICE",
    "CHECK_DOWNLOAD", "CHECK_DATASETS", "CHECK_EXAMPLE_INDEX", "PRINT_MODEL",
    "CHECK_LABEL_REMAPPING", "CHECK_LABEL_REMAPPING_NUM_EXAMPLES", "LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE",
    "DISCOVERY_DATASET",  # only to say in the label check whether s05c leaves D_c out of a dataset
    "RUN_BASELINE", "BASELINE_DATASETS", "BASELINE_MAX_IMAGES",
]


def load_dataset(name):
    """The dataset, or None (with a message) if it is not on disk."""
    try:
        return datasets.get_dataset(name)
    except (FileNotFoundError, RuntimeError, ImportError) as err:
        print(f"  [skipped] {name}: {err}")
        return None


@torch.no_grad()
def predict(model, image, device):
    """[H, W] predicted VOC classes at native resolution, and the raw output dict."""
    output = model(M.preprocess(image, device))
    return output["out"].argmax(dim=1).squeeze(0).cpu().numpy(), output


# ------------------------------------------------------------------
# 1. Datasets
# ------------------------------------------------------------------
def dataset_stats(ds):
    print(f"\n{ds.name}: {len(ds):,} images. Counting GT pixels per class ...", flush=True)
    counts = datasets.class_pixel_counts(ds)  # [images, 21], void excluded
    images_per_class = (counts > 0).sum(axis=0)
    pixels_per_class = counts.sum(axis=0)
    print(f"  {'idx':>3s}  {'class':15s} {'# images':>10s} {'# pixels':>15s}")
    for c, name in enumerate(VOC_CLASSES):
        print(f"  {c:3d}  {name:15s} {images_per_class[c]:>10,} {pixels_per_class[c]:>15,}")


# ------------------------------------------------------------------
# 1b. Label check of the label-remapped datasets
# ------------------------------------------------------------------
class _SourceAndRemappedCounts(Dataset):
    """Per image: [256] pixels per source label value and [256] pixels per remapped label value (VOC 0-20, 255)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return (torch.from_numpy(np.bincount(self.dataset.source_mask(i).ravel(), minlength=256)),
                torch.from_numpy(np.bincount(self.dataset.mask(i).ravel(), minlength=256)))


def source_and_remapped_counts(ds, num_workers=8):
    """[images, 256] pixels per source value and [images, 256] pixels per remapped value (one pass, in parallel)."""
    loader = DataLoader(_SourceAndRemappedCounts(ds), batch_size=16, num_workers=num_workers)
    source, remapped = zip(*loader)
    return torch.cat(source).numpy(), torch.cat(remapped).numpy()


def source_palette():
    """[256, 3] uint8 colours for source label values: well-separated hues (golden-ratio steps), dark grey for 0
    (unlabelled / no annotation in every source dataset here), VOC's light beige for 255."""
    palette = np.array([[round(255 * v) for v in colorsys.hsv_to_rgb((value * 0.618034) % 1.0, 0.65,
                                                                      0.95 if value % 2 else 0.75)]
                        for value in range(256)], dtype=np.uint8)
    palette[0] = (64, 64, 64)
    palette[VOID_LABEL] = datasets.voc_palette()[VOID_LABEL]
    return palette


def remapping_example_figure(ds, i, source_palette_colors, voc_palette_colors, max_legend_entries=18):
    """image | original labels (each source class present, with where it goes) | remapped labels (VOC palette)."""
    image, source, remapped = ds.image(i), ds.source_mask(i), ds.mask(i)
    target_of = {value: voc_class for value, _, voc_class in ds.source_classes}
    name_of = {value: name for value, name, _ in ds.source_classes}

    def target_name(v):
        return "255 ignore" if v == VOID_LABEL else "0 background" if v == 0 else f"{v} {VOC_CLASSES[v]}"

    values, pixels = np.unique(source, return_counts=True)
    order = np.argsort(-pixels, kind="stable")
    source_entries = [(Patch(facecolor=source_palette_colors[values[k]] / 255),
                       f"{values[k]} {name_of.get(int(values[k]), 'NOT IN THE CLASS LIST')[:32]}  →  "
                       f"{target_name(target_of.get(int(values[k]), VOID_LABEL))}  ({pixels[k] / source.size:.1%})")
                      for k in order]
    if len(source_entries) > max_legend_entries:
        rest = len(source_entries) - max_legend_entries + 1
        source_entries = source_entries[:max_legend_entries - 1] + [(Patch(facecolor="none", edgecolor="none"),
                                                                     f"... {rest} more source classes")]
    voc_values, voc_pixels = np.unique(remapped, return_counts=True)
    voc_entries = [(Patch(facecolor=voc_palette_colors[v] / 255), f"{target_name(int(v))}  ({n / remapped.size:.1%})")
                   for v, n in zip(voc_values, voc_pixels)]

    aspect = image.size[0] / image.size[1]
    fig, axes = plt.subplots(1, 3, figsize=(3 * 4.6 * max(aspect, 1.0) ** 0.5 + 0.5, 4.6 / max(aspect, 1.0) ** 0.5 + 0.6))
    panels = ((image, "image"),
              (source_palette_colors[source], f"original labels ({len(values)} source values in this image)"),
              (voc_palette_colors[remapped], "remapped to VOC labels (light beige = 255 ignore)"))
    for ax, (picture, title) in zip(axes, panels):
        ax.imshow(picture)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax, entries, title in ((axes[1], source_entries, "source value, class  →  VOC label  (share of pixels)"),
                               (axes[2], voc_entries, "VOC label  (share of pixels)")):
        handles, labels = zip(*entries)
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=7,
                  frameon=False, title=title, title_fontsize=7)
    fig.suptitle(f"{ds.name} [{i}] {ds.ids[i]}: {LABEL_REMAPPED_DATASETS[ds.name]['source']}\n"
                 f"{LABEL_REMAPPED_DATASETS[ds.name]['title_note']}", fontsize=9)
    path = C.IMAGES_DIR / "setup_check" / "label_remapping" / ds.name / f"example_{i:05d}_{ds.ids[i]}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def label_remapping_check(ds):
    """The label check of one label-remapped dataset (module docstring, 1b)."""
    info = LABEL_REMAPPED_DATASETS[ds.name]
    evaluable = info["evaluable_classes"]
    not_evaluable = [c for c in range(1, NUM_CLASSES) if c not in evaluable]
    utils.section(
        f"Label check: {ds.name} ({len(ds):,} images)",
        f"""
        Source labels: {info['source']}.
        Rules: a source class that clearly is a VOC class -> that class (several may merge); clearly none of the 20 VOC
        classes -> 0 background; may contain VOC-class pixels but cannot be mapped one-to-one -> 255 ignore (excluded
        from every metric); the source dataset's own void / unlabelled pixels -> 255.
        {info['title_note']}: {', '.join(VOC_CLASSES[c] for c in evaluable)}.
        """ + (f"""Resolution: LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE = {C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE!r} """
               f"""({'native 2048x1024, no resize' if C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE is None else 'image bilinear, labels nearest'}).
        """ if ds.name == "cityscapes_val_voc" else ""))
    print(f"Counting source and remapped label values in {len(ds):,} images ...", flush=True)
    source_counts, remapped_counts = source_and_remapped_counts(ds)
    total_pixels = int(remapped_counts.sum())

    # ---- A. mapping table ----
    utils.section(
        f"[{ds.name}] A. Mapping table: every source class, where it goes, and how often it occurs",
        f"""
        -> VOC class  the label the source class becomes ("255 ignore" = excluded from every metric).
        value         the source label value (COCO: category id; ADE20K: Idx of objectInfo150.txt; Cityscapes: labelId).
        images        images whose ORIGINAL labels contain at least one pixel of the source class.
        pixels/share  its pixels in all {len(ds):,} images, and their share of all {total_pixels:,} pixels.
        Check: no source class with many pixels in a group where it does not belong.
        """ + ("""COCO: value 255 here = a pixel covered by more than one instance of ANY category; the VOC GT voids only
        overlaps of VOC instances, so its 255 share (table B) is smaller.
        """ if ds.name.startswith("coco") else ""))
    datasets.print_mapping_table(ds, (source_counts > 0).sum(axis=0), source_counts.sum(axis=0))
    listed = {value for value, _, _ in ds.source_classes}
    unexpected = [v for v in np.flatnonzero(source_counts.sum(axis=0)) if int(v) not in listed]
    if unexpected:
        print(f"\n  WARNING: source values NOT IN THE CLASS LIST occur (they map to 255): "
              + ", ".join(f"{v}: {int(source_counts[:, v].sum()):,} px in {int((source_counts[:, v] > 0).sum())} images"
                          for v in unexpected))
    else:
        print("\n  Source values outside the class list: none (expected).")

    # ---- B. remapped GT per VOC label ----
    pixels, images = remapped_counts.sum(axis=0), (remapped_counts > 0).sum(axis=0)
    utils.section(
        f"[{ds.name}] B. Remapped GT: pixels and images per VOC label",
        f"""
        images   images whose REMAPPED GT holds at least one pixel of the label; for an evaluable class this is the
                 number of images s05c evaluates the class's circuits on (LABEL_REMAPPED_MAX_EVAL_IMAGES = "all").
        share    of all {total_pixels:,} pixels. Background usually dominates (ADE20K and Cityscapes are mostly
                 "stuff": walls, road, sky).
        status   evaluable = some source class maps to it; NOT evaluable classes must have 0 pixels (else: a
                 mapping bug). An evaluable class with 0 images is flagged as well.
        """)
    print(f"  {'label':16s} {'status':15s} {'images':>8s} {'pixels':>15s} {'share':>8s}")
    problems = []
    for v in list(range(NUM_CLASSES)) + [VOID_LABEL]:
        status = ("background" if v == 0 else "ignore" if v == VOID_LABEL
                  else "evaluable" if v in evaluable else "NOT evaluable")
        label = "255 ignore" if v == VOID_LABEL else f"{v} {VOC_CLASSES[v]}"
        print(f"  {label:16s} {status:15s} {int(images[v]):8,d} {int(pixels[v]):15,d} "
              f"{pixels[v] / total_pixels if total_pixels else np.nan:8.3%}")
        if v in not_evaluable and pixels[v] > 0:
            problems.append(f"{label} is not evaluable but has {int(pixels[v]):,} pixels")
        if v in evaluable and images[v] == 0:
            problems.append(f"{label} is evaluable but occurs in no image")
    others = [v for v in np.flatnonzero(pixels) if NUM_CLASSES <= v < VOID_LABEL]
    if others:
        problems.append(f"values outside 0-20 and 255 in the remapped GT: {others}")
    object_pixels = int(pixels[1:NUM_CLASSES].sum())
    print(f"\n  background {pixels[0] / total_pixels:.2%} · object classes {object_pixels / total_pixels:.2%} · "
          f"ignore {pixels[VOID_LABEL] / total_pixels:.2%} of all pixels")
    if object_pixels == 0:
        problems.append("no object pixel at all after the remapping (all background / ignore)")
    print("\n  " + ("PROBLEMS: " + "; ".join(problems) if problems else "No problem found (no pixels in non-evaluable "
                                                                         "classes, every evaluable class occurs)."))

    # ---- C. images ----
    has_object = remapped_counts[:, 1:NUM_CLASSES].sum(axis=1) > 0
    utils.section(
        f"[{ds.name}] C. Images used by s05c",
        f"""
        without any object class  images whose remapped GT holds only background and / or 255 pixels. s05c leaves them
                                  out completely: from the class lists AND from the full-model baseline.
        image-class pairs         the sum over the evaluable classes of the images with that class: how many
                                  (image, class) evaluations s05c runs, each with every distinct circuit (cost).
        """)
    print(f"  images:                      {len(ds):8,d}")
    print(f"  without any object class:    {int((~has_object).sum()):8,d}  ({(~has_object).mean():.1%}; left out by s05c)")
    print(f"  with at least one object:    {int(has_object.sum()):8,d}  (s05c's full-model baseline)")
    print(f"  image-class pairs:           {int(sum(images[c] for c in evaluable)):8,d}  (over the {len(evaluable)} "
          f"evaluable classes)")
    if ds.name == C.DISCOVERY_DATASET:
        print(f"  NOTE: {ds.name} is also DISCOVERY_DATASET, so s05c leaves each class's D_c images out of its list "
              f"(the counts above include them).")

    # ---- D. example figures ----
    n = min(C.CHECK_LABEL_REMAPPING_NUM_EXAMPLES, len(ds))
    indices = sorted({int(round(k)) for k in np.linspace(0, len(ds) - 1, n)}) if n > 0 else []
    utils.section(
        f"[{ds.name}] D. Example figures ({len(indices)} images, evenly spaced over the dataset)",
        """
        Each figure: the image, its ORIGINAL labels (one colour per source class; legend: value, source class ->
        VOC label, share of the image's pixels) and the REMAPPED labels in the VOC palette (light beige = 255 ignore).
        """)
    source_colors, voc_colors = source_palette(), datasets.voc_palette()
    for i in indices:
        print(f"  saved {remapping_example_figure(ds, i, source_colors, voc_colors)}", flush=True)


# ------------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------------
def model_summary(model):
    print("\nHook points (circuit components):")
    for name, (module, channels) in M.HOOK_POINTS.items():
        print(f"  {name:7s} {module:18s} {channels:>5,} channels")
    print(f"  total   {'':18s} {M.NUM_COMPONENTS:>5,} components")

    if not C.PRINT_MODEL:
        return
    print("\nModel architecture (aux_classifier removed):")
    print(model)
    print("\nTrainable parameters per layer:")
    print(f"{'Layer':70s} {'Shape':25s} {'# Params':>12s}")
    print("-" * 110)
    total_trainable, total_all = 0, 0
    for name, param in model.named_parameters():
        total_all += param.numel()
        if param.requires_grad:
            total_trainable += param.numel()
            print(f"{name:70s} {str(list(param.shape)):25s} {param.numel():>12,}")
    print("-" * 110)
    aux = getattr(model, "removed_aux_params", 0)
    print(f"Total trainable parameters: {total_trainable:,}")
    print(f"Total parameters:           {total_all:,}  (inference model, aux_classifier removed)")
    print(f"  + removed aux_classifier: {aux:,}  (training-only auxiliary loss head on layer3)")
    print(f"  = full torchvision model: {total_all + aux:,}  ({C.MODEL_NAME} as shipped)")


# ------------------------------------------------------------------
# 3. Example prediction
# ------------------------------------------------------------------
def example_prediction(model, ds, device, palette):
    i = C.CHECK_EXAMPLE_INDEX
    image, gt = ds[i]
    pred, output = predict(model, image, device)
    print(f"\n{ds.name}[{i}] ({ds.ids[i]}): image {image.size[1]}x{image.size[0]}, "
          f"output keys {list(output.keys())}, 'out' shape {tuple(output['out'].shape)}")
    for label, mask in (("GT", gt), ("predicted", pred)):
        classes = [f"{c} {VOC_CLASSES[c]}" for c in np.unique(mask) if c < NUM_CLASSES]
        print(f"  {label:9s} classes: {', '.join(classes)}")
    valid = gt != datasets.VOID_LABEL
    print(f"  pixel accuracy (void excluded): {(pred[valid] == gt[valid]).mean():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axes, (image, palette[gt], palette[pred]),
                              ("Image", f"GT ({ds.name})", f"{C.MODEL_NAME} prediction")):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    path = C.IMAGES_DIR / "setup_check" / f"{ds.name.replace('/', '_')}_example{i}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


# ------------------------------------------------------------------
# 4. Baseline evaluation
# ------------------------------------------------------------------
def _first(batch):
    # batch_size=1; items are (PIL image, mask array) of variable size.
    return batch[0]


@torch.no_grad()
def baseline_confusion_matrix(model, ds, device, log_every=100):
    subset = ds if C.BASELINE_MAX_IMAGES is None else Subset(ds, range(min(C.BASELINE_MAX_IMAGES, len(ds))))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=8, collate_fn=_first)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total = len(subset)
    print(f"\nEvaluating on {ds.name} ({total:,} images) ...", flush=True)
    for i, (image, gt) in enumerate(loader, start=1):
        pred, _ = predict(model, image, device)
        assert pred.shape == gt.shape, f"prediction/mask shape mismatch: {pred.shape} vs {gt.shape}"
        cm += metrics.confusion_matrix(pred, gt)
        if i % log_every == 0 or i == total:
            print(f"  [{ds.name}] {i}/{total}", flush=True)
    return cm, total


def baseline_metrics(cm, name=None):
    """Per-class and mean metrics of a confusion matrix; for a label-remapped dataset (name) also the mean IoU over
    its evaluable classes and background."""
    scores = metrics.class_scores(cm)
    tp = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(gt_sum + pred_sum > 0, 2 * tp / (gt_sum + pred_sum), np.nan)
    freq = gt_sum / gt_sum.sum() if gt_sum.sum() > 0 else np.zeros_like(gt_sum)
    evaluable = {}
    if name in LABEL_REMAPPED_DATASETS:
        classes = [0] + LABEL_REMAPPED_DATASETS[name]["evaluable_classes"]
        evaluable = {"evaluable_classes": classes,
                     "evaluable_classes_mean_iou": float(np.nanmean(scores["iou"][classes]))}
    return {
        "per_class": {
            "iou": scores["iou"].tolist(),
            "precision": scores["precision"].tolist(),
            "recall": scores["recall"].tolist(),
            "dice": dice.tolist(),
            "support_pixels": gt_sum.astype(np.int64).tolist(),
        },
        "overall": {
            "pixel_accuracy": float(tp.sum() / cm.sum()) if cm.sum() > 0 else float("nan"),
            "mean_iou": float(np.nanmean(scores["iou"])),
            "mean_dice": float(np.nanmean(dice)),
            "mean_precision": float(np.nanmean(scores["precision"])),
            "mean_recall": float(np.nanmean(scores["recall"])),
            "frequency_weighted_iou": float(np.nansum(freq * np.nan_to_num(scores["iou"]))),
            **evaluable,
        },
    }


def print_report(name, result):
    def fmt(x):
        return "     nan" if np.isnan(x) else f"{x:8.4f}"

    print(f"\n==== {name}: per-class metrics ====")
    header = f"{'class':15s} {'IoU':>8s} {'Dice':>8s} {'Prec':>8s} {'Recall':>8s} {'Pixels':>14s}"
    print(header)
    print("-" * len(header))
    pc = result["per_class"]
    for c, cname in enumerate(VOC_CLASSES):
        print(f"{cname:15s} {fmt(pc['iou'][c])} {fmt(pc['dice'][c])} {fmt(pc['precision'][c])} "
              f"{fmt(pc['recall'][c])} {pc['support_pixels'][c]:>14,}")
    o = result["overall"]
    print(f"\n  Overall pixel accuracy  : {o['pixel_accuracy']:.4f}")
    print(f"  Mean IoU                : {o['mean_iou']:.4f}")
    print(f"  Mean Dice / F1          : {o['mean_dice']:.4f}")
    print(f"  Mean precision          : {o['mean_precision']:.4f}")
    print(f"  Mean recall             : {o['mean_recall']:.4f}")
    print(f"  Frequency-weighted IoU  : {o['frequency_weighted_iou']:.4f}")
    if "evaluable_classes_mean_iou" in o:
        print(f"  Mean IoU, evaluable     : {o['evaluable_classes_mean_iou']:.4f}  (background + the "
              f"{len(o['evaluable_classes']) - 1} classes this label-remapped dataset can score; the 'Mean IoU' above "
              f"also averages the classes that never occur in its GT, whose IoU is 0 from false positives alone or nan)")
        print(f"  Images: every image of the dataset; s05c's full-model reference leaves out the images whose remapped "
              f"GT holds no object class.")


def run_baseline(model, loaded, device):
    splits = {}
    for name in C.BASELINE_DATASETS:
        ds = loaded[name] if name in loaded else load_dataset(name)
        if ds is None:
            continue
        cm, num_images = baseline_confusion_matrix(model, ds, device)
        result = baseline_metrics(cm, name)
        print_report(name, result)
        splits[name] = {"num_images": num_images, "confusion_matrix": cm.tolist(), "metrics": result}

    folder = C.RESULTS_DIR / "baseline" / C.MODEL_NAME
    if C.BASELINE_MAX_IMAGES is not None:
        folder = folder.with_name(f"{folder.name}_max{C.BASELINE_MAX_IMAGES}")
    utils.save_json(folder / "baseline_metrics.json", {
        "model": C.MODEL_NAME,
        "weights": "torchvision DEFAULT (COCO_WITH_VOC_LABELS_V1)",
        "classes": VOC_CLASSES,
        "void_label": datasets.VOID_LABEL,
        "eval_resolution": "native (no resize)" + (
            f"; cityscapes_val_voc resized to {C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE} (width, height)"
            if C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE and "cityscapes_val_voc" in splits else ""),
        "splits": splits,
    })
    utils.write_run_info(folder)
    print(f"\nSaved baseline metrics to: {folder / 'baseline_metrics.json'}")


def main():
    utils.start_run("setup_check_and_baseline", USED_PARAMS)
    device = utils.get_device()

    print("\n---- 1. Datasets " + "-" * 55)
    if C.CHECK_DOWNLOAD:
        try:
            datasets.download_missing_voc_and_sbd()
        except Exception as err:  # offline node, host down, ... -> the per-dataset check below reports what is missing
            print(f"  [download failed] {type(err).__name__}: {err}")
    loaded = {}
    for name in C.CHECK_DATASETS:
        ds = load_dataset(name)
        if ds is not None:
            loaded[name] = ds
            dataset_stats(ds)

    print("\n---- 1b. Label check of the label-remapped datasets " + "-" * 20)
    remapped = [ds for name, ds in loaded.items() if name in LABEL_REMAPPED_DATASETS]
    if not C.CHECK_LABEL_REMAPPING:
        print("CHECK_LABEL_REMAPPING = False: skipped.")
    elif not remapped:
        print(f"No label-remapped dataset ({', '.join(LABEL_REMAPPED_DATASETS)}) among the loaded CHECK_DATASETS: "
              f"skipped.")
    for ds in remapped if C.CHECK_LABEL_REMAPPING else []:
        label_remapping_check(ds)

    print("\n---- 2. Model " + "-" * 58)
    model = M.load_model(C.MODEL_NAME, device)
    print(f"Loaded {C.MODEL_NAME} on {device}.")
    model_summary(model)

    print("\n---- 3. Example predictions " + "-" * 44)
    palette = datasets.voc_palette()
    for ds in loaded.values():
        example_prediction(model, ds, device, palette)

    print("\n---- 4. Baseline " + "-" * 55)
    if C.RUN_BASELINE:
        run_baseline(model, loaded, device)
    else:
        print("RUN_BASELINE = False: skipped.")


if __name__ == "__main__":
    main()
