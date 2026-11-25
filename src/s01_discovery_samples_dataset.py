"""s01 · Discovery samples dataset.

For every class c in CLASSES:
  - D_c, the discovery samples: images from DISCOVERY_DATASET whose GT holds at least MIN_PIXELS pixels
    of c (the candidates), drawn with DISCOVERY_AND_EVAL_SEED. DISCOVERY_SAMPLE_SIZE sets how many:
    a fixed number, "all" candidates, a fraction of them, or "bottleneck" (the fewest candidates of any
    class in CLASSES, the same |D_c| for every class). With REQUIRE_PRED_IN_DISCOVERY
    an image is only taken if the full model also predicts at least one pixel of class c in it; with
    "bottleneck" the model then runs on every candidate first, and the bottleneck counts only those.
  - eval lists: for every dataset in EVAL_DATASETS, the images whose GT contains c (any pixel). MAX_EVAL_IMAGES = "all"
    keeps every such image; an int caps them (a random subset, seeded per class with DISCOVERY_AND_EVAL_SEED, the same
    for every eval set). If the eval dataset is DISCOVERY_DATASET itself, the D_c images are left out, so the list stays
    held out. The eval set "discovery_samples" is D_c itself (Anani's "ID" setting, i.e. the discovery images).
  - eval lists WITHOUT class c (s05 evaluates them only with
    EVALUATE_IMAGES_WITHOUT_CLASS): for every dataset in EVAL_DATASETS, every image whose GT has no pixel of c, with no
    cap and no random draw. For "discovery_samples" (a train-set derivative) they are the other classes' discovery
    samples D_c' (c' in CLASSES), each image once, minus the images whose GT contains c: such an image is neither "with
    c" in the in-sample sense (it is not in D_c) nor "without c", so it is dropped, and the number dropped is stored.
    D_c and the "with c" lists above do not change: these lists use no random stream.

D_c must not change on a re-run (s02-s04b were built on it): if discovery_and_eval_lists.json already exists, the new
D_c of every class in both is compared with the stored one first, and s01 stops without writing if one differs. D_c is
deterministic (same config, data and seed), so this is a safety net, e.g. for a re-run with another MAX_EVAL_IMAGES
(the cap is drawn from its own random streams and never changes D_c).

Output: <discovery_dir>/discovery_and_eval_lists.json and run_info.json. The JSON holds "discovery_samples" (D_c),
"eval" (the lists with c), "eval_without_class" (the lists without c) and "max_eval_images" (the cap the lists with c
were drawn with; s05, s05b and s07 stop if it differs from config.py's).
"""

import sys

import numpy as np
import torch

import config as C
from common import datasets, model as M, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES",
    "DISCOVERY_DATASET", "EVAL_DATASETS", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "MAX_EVAL_IMAGES",
    "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "MODEL_NAME", "DEVICE",  # only used with REQUIRE_PRED_IN_DISCOVERY
    "SMOKE_TEST",
]


@torch.no_grad()
def predicted_classes(model, dataset, i, device):
    """[21] bool: the classes the full model predicts at least one pixel of in image i."""
    pred = model(M.preprocess(dataset.image(i), device))["out"].argmax(dim=1)
    return torch.bincount(pred.ravel(), minlength=NUM_CLASSES).cpu().numpy() > 0


def discovery_sample_size(n_candidates, bottleneck):
    """Target |D_c| for a class with n_candidates candidates (DISCOVERY_SAMPLE_SIZE: int, "all", "bottleneck"
    or a fraction). bottleneck is the fewest (usable) candidates of any class in CLASSES."""
    if C.DISCOVERY_SAMPLE_SIZE == "all":
        return n_candidates
    if C.DISCOVERY_SAMPLE_SIZE == "bottleneck":
        return bottleneck
    if isinstance(C.DISCOVERY_SAMPLE_SIZE, float):
        if not 0 < C.DISCOVERY_SAMPLE_SIZE <= 1:
            raise ValueError(f"A fractional DISCOVERY_SAMPLE_SIZE must be in (0, 1], got {C.DISCOVERY_SAMPLE_SIZE}.")
        return max(1, round(C.DISCOVERY_SAMPLE_SIZE * n_candidates))
    return C.DISCOVERY_SAMPLE_SIZE


def draw_discovery_samples(c, candidates, size, predicted, dataset, model, device):
    """Indices of D_c (sorted) and the number of candidates skipped for no prediction. predicted holds
    predicted_classes() per candidate if the model already ran on all of them ("bottleneck"), else None."""
    order = utils.rng(C.DISCOVERY_AND_EVAL_SEED, c).permutation(candidates)
    if not C.REQUIRE_PRED_IN_DISCOVERY:
        return np.sort(order[:size]), 0

    chosen, skipped = [], 0
    for i in order:
        if len(chosen) == size:
            break
        classes = predicted[i] if predicted is not None else predicted_classes(model, dataset, i, device)
        if classes[c]:
            chosen.append(i)
        else:
            skipped += 1
    return np.sort(chosen), skipped


def eval_list_without_class(c, counts):
    """Indices (sorted) of the eval images whose GT has no pixel of class c: all of them (no cap, no random draw)."""
    return np.flatnonzero(counts[:, c] == 0)


def discovery_samples_without_class(c, discovery_samples, counts):
    """The images without class c on the eval set "discovery_samples": the other classes' D_c' (c' in CLASSES), each
    image once, minus the images whose GT contains c (any pixel; counts = GT pixels per class of the discovery
    dataset). Returns the sorted indices and the number of D_c' images dropped because they contain c."""
    others = np.unique(np.concatenate([np.asarray(discovery_samples[str(other)]["indices"], dtype=np.int64)
                                       for other in C.CLASSES if other != c] or [np.zeros(0, dtype=np.int64)]))
    keep = counts[others, c] == 0
    return others[keep], int((~keep).sum())


def check_discovery_samples_unchanged(path, discovery_samples):
    """Stops (before anything is written) if the stored discovery_and_eval_lists.json holds another D_c for a class
    that is in both: s02-s04b were built on the stored D_c."""
    if not path.exists():
        return
    stored = utils.load_json(path).get("discovery_samples", {})
    common = [c for c in C.CLASSES if str(c) in stored]
    changed = [c for c in common if stored[str(c)]["indices"] != discovery_samples[str(c)]["indices"]]
    if changed:
        sys.exit(f"\n[D_c changed] the new D_c differs from the one stored in {path}\nfor the classes "
                 f"{[f'{c} {VOC_CLASSES[c]}' for c in changed]}. s02-s04b were built on the stored D_c, so nothing was "
                 f"written.\n-> check what changed (data on disk, DISCOVERY_DATASET, MIN_PIXELS, DISCOVERY_SAMPLE_SIZE, "
                 f"REQUIRE_PRED_IN_DISCOVERY, DISCOVERY_AND_EVAL_SEED). To start over on purpose, move the s01 folder "
                 f"away first (s02-s04b inside it would have to be re-run).")
    print(f"D_c check: {path.name} exists; the new D_c equals the stored one for all {len(common)} classes in both "
          f"(s02-s04b, built on it, stay valid).")


def main():
    utils.start_run("s01_discovery_samples_dataset", USED_PARAMS)
    utils.eval_image_cap(C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES")  # stops early on an invalid value
    utils.require_datasets([C.DISCOVERY_DATASET] + [name for name in C.EVAL_DATASETS if name != "discovery_samples"])
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device) if C.REQUIRE_PRED_IN_DISCOVERY else None

    # ---- discovery samples D_c -------------------------------------------------
    discovery = datasets.get_dataset(C.DISCOVERY_DATASET)
    print(f"\nCounting GT pixels per class in {discovery.name} ({len(discovery)} images) ...", flush=True)
    counts = datasets.class_pixel_counts(discovery)
    candidates = {c: np.flatnonzero(counts[:, c] >= C.MIN_PIXELS) for c in C.CLASSES}

    # "bottleneck" with REQUIRE_PRED_IN_DISCOVERY: run the model on every candidate, so that the bottleneck
    # counts only the usable candidates (those the model predicts the class in)
    predicted = None
    if C.REQUIRE_PRED_IN_DISCOVERY and C.DISCOVERY_SAMPLE_SIZE == "bottleneck":
        images = np.unique(np.concatenate(list(candidates.values())))
        print(f"Predicting all {len(images):,} candidate images (\"bottleneck\" with REQUIRE_PRED_IN_DISCOVERY) ...",
              flush=True)
        predicted = {}
        for k, i in enumerate(images, 1):
            predicted[i] = predicted_classes(model, discovery, i, device)
            if k % 1000 == 0 or k == len(images):
                print(f"  {k:,} / {len(images):,}", flush=True)
    num_usable = {c: len(idx) if predicted is None else int(sum(predicted[i][c] for i in idx))
                  for c, idx in candidates.items()}
    bottleneck_class = min(num_usable, key=num_usable.get)
    bottleneck = num_usable[bottleneck_class]
    if C.DISCOVERY_SAMPLE_SIZE == "bottleneck":
        if predicted is not None:
            print("Candidates the model predicts the class in (usable): "
                  f"{ {VOC_CLASSES[c]: n for c, n in num_usable.items()} }")
        print(f"DISCOVERY_SAMPLE_SIZE = \"bottleneck\": |D_c| = {bottleneck} for every class "
              f"(the fewest {'usable ' if predicted is not None else ''}candidates, "
              f"class {bottleneck_class} {VOC_CLASSES[bottleneck_class]}).")
    out_dir = utils.discovery_dir(bottleneck)

    discovery_samples, summary = {}, {}
    for c in C.CLASSES:
        n_candidates = len(candidates[c])
        idx, n_skipped = draw_discovery_samples(c, candidates[c], discovery_sample_size(n_candidates, bottleneck),
                                                predicted, discovery, model, device)
        discovery_samples[str(c)] = {
            "name": VOC_CLASSES[c],
            "indices": idx.tolist(),
            "ids": [discovery.ids[i] for i in idx],
            "gt_pixels": counts[idx, c].tolist(),
            "num_images_with_class": int((counts[:, c] > 0).sum()),
            "num_candidates": n_candidates,
            "num_skipped_no_pred": n_skipped,
        }
        summary[c] = [n_candidates, len(idx), int(np.median(counts[idx, c])) if len(idx) else 0]
        if len(idx) < discovery_sample_size(n_candidates, bottleneck):
            print(f"WARNING: {VOC_CLASSES[c]} has only {len(idx)} usable images "
                  f"(< {discovery_sample_size(n_candidates, bottleneck)}, "
                  f"DISCOVERY_SAMPLE_SIZE = {C.DISCOVERY_SAMPLE_SIZE}).")
    check_discovery_samples_unchanged(out_dir / "discovery_and_eval_lists.json", discovery_samples)

    # ---- eval lists ------------------------------------------------------------
    eval_lists, eval_lists_without_class = {}, {}
    for name in C.EVAL_DATASETS:
        if name == "discovery_samples":  # D_c itself (Anani's "ID" setting): the discovery images
            eval_lists[name] = {str(c): {"indices": discovery_samples[str(c)]["indices"],
                                         "ids": discovery_samples[str(c)]["ids"],
                                         "num_available": len(discovery_samples[str(c)]["ids"])} for c in C.CLASSES}
            for c in C.CLASSES:
                summary[c] += [len(discovery_samples[str(c)]["ids"])] * 2
            eval_lists_without_class[name] = {}
            for c in C.CLASSES:  # the other classes' D_c', without the images that contain c
                idx, n_dropped = discovery_samples_without_class(c, discovery_samples, counts)
                eval_lists_without_class[name][str(c)] = {"indices": idx.tolist(), "ids": [discovery.ids[i] for i in idx],
                                                          "num_dropped_containing_class": n_dropped}
            continue
        dataset = datasets.get_dataset(name)
        print(f"Counting GT pixels per class in {name} ({len(dataset)} images) ...", flush=True)
        eval_counts = datasets.class_pixel_counts(dataset)
        is_discovery = name == C.DISCOVERY_DATASET
        if is_discovery:
            print(f"NOTE: {name} is also the discovery dataset; the D_c images are left out of its eval lists.")
        eval_lists[name] = {}
        for c in C.CLASSES:
            exclude = discovery_samples[str(c)]["indices"] if is_discovery else []
            idx, n_available = utils.eval_list(c, eval_counts, C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES",
                                               C.DISCOVERY_AND_EVAL_SEED, exclude)
            eval_lists[name][str(c)] = {
                "indices": idx.tolist(),
                "ids": [dataset.ids[i] for i in idx],
                "num_available": n_available,
            }
            summary[c] += [n_available, len(idx)]
            if len(idx) == 0:
                print(f"WARNING: no {VOC_CLASSES[c]} images in {name}.")
        eval_lists_without_class[name] = {}
        for c in C.CLASSES:
            idx = eval_list_without_class(c, eval_counts)
            eval_lists_without_class[name][str(c)] = {"indices": idx.tolist(), "ids": [dataset.ids[i] for i in idx]}

    # ---- report ----------------------------------------------------------------
    print(f"\nPer class: images of {discovery.name} with >= {C.MIN_PIXELS} px of the class (candidates), "
          f"|D_c|, median class pixels in D_c;\nfor each eval set: images containing the class (any pixel; on the "
          f"discovery dataset without D_c) -> used.\n{utils.cap_text(C.MAX_EVAL_IMAGES, 'MAX_EVAL_IMAGES')}. "
          f"\"(capped)\" = the cap applied: fewer used than available. discovery_samples = D_c itself (never capped).")
    header = f"{'':3s} {'class':12s} {'candidates':>10s} {'|D_c|':>6s} {'median px':>10s}"
    header += "".join(f" {name:>22s}" for name in C.EVAL_DATASETS)
    print(header)
    print("-" * len(header))
    for c in C.CLASSES:
        n_candidates, size, median_px, *ev = summary[c]
        row = f"{c:3d} {VOC_CLASSES[c]:12s} {n_candidates:10,d} {size:6d} {median_px:10,d}"
        row += "".join(f" {f'{ev[2 * k]} -> {ev[2 * k + 1]}' + (' (capped)' if ev[2 * k + 1] < ev[2 * k] else ''):>22s}"
                       for k in range(len(C.EVAL_DATASETS)))
        print(row)
    if C.REQUIRE_PRED_IN_DISCOVERY:
        skipped = {VOC_CLASSES[c]: discovery_samples[str(c)]["num_skipped_no_pred"] for c in C.CLASSES}
        print(f"\nCandidates skipped because the model predicts no class-c pixel: {skipped}")

    utils.section(
        "Per class: eval images WITHOUT class c (s05 uses them with EVALUATE_IMAGES_WITHOUT_CLASS)",
        f"""
        Each cell: the number of images in the eval set whose GT has NO pixel of class c. All of them are used (no cap,
        no random draw), so these numbers are the image counts s05 evaluates "without c".
        discovery_samples: the other classes' D_c' (c' in CLASSES = {len(C.CLASSES)} classes), each image once, minus the
                           images whose GT contains c ("dropped": they are neither in D_c nor free of c). With a single
                           class in CLASSES this set is empty.
        These lists use no random stream, so D_c and the lists with c above are unchanged.
        """)
    header = f"{'':3s} {'class':12s}" + "".join(f" {name:>32s}" for name in C.EVAL_DATASETS)
    print(header)
    print("-" * len(header))
    for c in C.CLASSES:
        cells = []
        for name in C.EVAL_DATASETS:
            entry = eval_lists_without_class[name][str(c)]
            cells.append(f"{len(entry['indices']):,}" + (f" ({entry['num_dropped_containing_class']:,} dropped)"
                                                          if "num_dropped_containing_class" in entry else ""))
        print(f"{c:3d} {VOC_CLASSES[c]:12s}" + "".join(f" {cell:>32s}" for cell in cells))

    utils.save_json(out_dir / "discovery_and_eval_lists.json", {
        "discovery_dataset": C.DISCOVERY_DATASET,
        "discovery_samples": discovery_samples,
        "eval": eval_lists,
        "eval_without_class": eval_lists_without_class,
        "max_eval_images": C.MAX_EVAL_IMAGES,
    })
    utils.write_run_info(out_dir)
    print(f"\nSaved {out_dir / 'discovery_and_eval_lists.json'}")


if __name__ == "__main__":
    main()
