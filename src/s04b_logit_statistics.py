"""s04b · Class-wise logit statistics (Jung et al. 2021, adapted), for the logit standardization after certification.

Certification (s01-s04) is not changed. This stage only measures the logits that the
LOGIT_STANDARDIZATION_VARIANTS of s05-s07 adjust (formulas in common/logit_standardization.py).

For every class c, the full model and every distinct circuit of class c from s04 (circuit types baseline_circuit,
majority and certified, at every top-K layer-normalized ratio K) run once on each statistics image, without
gradients. Statistics images: LOGIT_STATISTICS_IMAGES = "discovery_samples", i.e. D_c itself. Each pass gives the
21 logit maps F_d. For every class d (background included) and every pixel set
    full_model_prediction   the pixels the full model predicts as d (the same pixels for every circuit)
    own_prediction          the pixels the model itself predicts as d (Jung Eq. 3-4)
    all_pixels              every pixel
the pixel count, the mean μ_d and the standard deviation σ_d of F_d over those pixels of all statistics images are
stored. A class's statistics are usable with ≥ LOGIT_STATISTICS_MIN_PIXELS pixels and σ > 1e-6.

Circuits with identical masks run once per image. A circuit that keeps every channel of U is the full model and
gets the full model's statistics, so the matched variants are exactly the identity for it.

Output: <logit_statistics_dir>/class_XX.npz and run_info.json, with (3 pixel sets in pixel_sets order, 21 classes)
    full_model_{pixel_count, logit_mean, logit_standard_deviation, statistics_usable}   [3, 21]
    circuit_{pixel_count, logit_mean, logit_standard_deviation, statistics_usable}      [circuit types, |grid|, 3, 21]
    circuit_mask_hash   [circuit types, |grid|]  id of each circuit's mask; s05 and s07 stop if s04 changed since
    circuit_types, pixel_sets, topk_layer_normalized_ratio_grid, image_ids, image_indices (the statistics images),
    logit_statistics_min_pixels, minimum_usable_standard_deviation
"""

import sys
import textwrap
import time
import warnings

import numpy as np
import torch

import config as C
from common import datasets, logit_standardization, model as M, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "DATA_DIR", "CLASSES", "MODEL_NAME", "DEVICE",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA",
    "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS", "SMOKE_TEST",
]
PRUNED_TYPES = ["baseline_circuit", "majority", "certified"]
PIXEL_SETS = logit_standardization.PIXEL_SETS


@torch.inference_mode()
def class_sums(model, dataset, entry, circuits, full_key, device):
    """Sums of the full model and of every distinct circuit over the statistics images:
    {mask key: logit_standardization.new_sums(...)}, and the number of pixels per image."""
    distinct = {key: masks for circuits_per_grid_idx in circuits.values()
                for masks, key in circuits_per_grid_idx if key != full_key}
    sums = {key: logit_standardization.new_sums(NUM_CLASSES, device) for key in [full_key, *distinct]}
    pixels_per_image = []
    for i, image_id in zip(entry["indices"], entry["ids"]):
        assert dataset.ids[i] == image_id, f"dataset order changed: {dataset.ids[i]} != {image_id}"
        x = M.preprocess(dataset.image(i), device)
        full_logits = M.forward(model, x)
        full_labels = full_logits.argmax(dim=0)
        pixels_per_image.append(full_labels.numel())
        logit_standardization.accumulate_pixel_set_statistics(full_logits, full_labels, sums[full_key])
        for key, masks in distinct.items():
            logit_standardization.accumulate_pixel_set_statistics(M.forward(model, x, masks), full_labels, sums[key])
    return sums, pixels_per_image


def stacked(statistics):
    """{pixel set: {quantity: [21]}} -> {quantity: [len(PIXEL_SETS), 21]}"""
    return {q: np.stack([statistics[p][q] for p in PIXEL_SETS]) for q in logit_standardization.QUANTITIES}


def per_pixel_set(arrays):
    """{quantity: [len(PIXEL_SETS), 21]} -> {pixel set: {quantity: [21]}}, the form class_adjustment reads."""
    return {p: {q: arrays[q][j] for q in arrays} for j, p in enumerate(PIXEL_SETS)}


def names(classes):
    return ", ".join(VOC_CLASSES[d] for d in classes) or "-"


def main():
    utils.start_run("s04b_logit_statistics", USED_PARAMS)
    utils.require(utils.certify_dir() / "run_info.json", "s04_certify.py")
    if C.LOGIT_STATISTICS_IMAGES != "discovery_samples":
        sys.exit(f"\n[config] LOGIT_STATISTICS_IMAGES = {C.LOGIT_STATISTICS_IMAGES!r} is not implemented; "
                 f"implemented: \"discovery_samples\" (D_c of each class).")
    discovery_and_eval_lists = utils.load_discovery_and_eval_lists()
    out_dir = utils.logit_statistics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = utils.get_device()
    model = M.load_model(C.MODEL_NAME, device)
    dataset = datasets.get_dataset(C.DISCOVERY_DATASET)

    _, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    n_certifiable = sum(s.stop - s.start for s in certifiable_slices.values())
    full_key = utils.mask_key({name: np.ones(s.stop - s.start, dtype=bool) for name, s in certifiable_slices.items()})
    min_pixels = C.LOGIT_STATISTICS_MIN_PIXELS
    variants = "\n".join(f"  {variant}\n" + textwrap.indent(description, "      ")
                         for variant, description in logit_standardization.VARIANT_DESCRIPTIONS.items())

    utils.section(
        "[1/4] Setup: what is measured",
        textwrap.dedent(f"""
        Certification (s01-s04) is NOT changed. This stage only measures logits, for the
        LOGIT_STANDARDIZATION_VARIANTS of s05-s07 (currently {C.LOGIT_STANDARDIZATION_VARIANTS or "[] = off"}).
        Statistics images: LOGIT_STATISTICS_IMAGES = {C.LOGIT_STATISTICS_IMAGES!r} = D_c of each class c, the images
        its circuits were discovered on (from {C.DISCOVERY_DATASET}).
        Models: per class c, the full model and every circuit of class c from s04 ({", ".join(PRUNED_TYPES)},
        at every {utils.topk_ratio_label()} K). A circuit = the full model with the channels of U (LAYERS =
        {C.LAYERS!r}, {n_certifiable:,} channels) outside the circuit set to zero. Circuits with identical masks run
        once per image; a circuit that keeps every channel of U IS the full model and gets its statistics.
        Per model, class d (all {NUM_CLASSES}, background included) and pixel set: the pixel count, the mean μ_d and the
        standard deviation σ_d of the class-d logit F_d, over those pixels of all statistics images:
          full_model_prediction  the pixels the FULL MODEL predicts as d (the same pixels for every circuit)
          own_prediction         the pixels the model ITSELF predicts as d (Jung Eq. 3-4); for the full model the
                                 same pixels as full_model_prediction
          all_pixels             every pixel, for every class d
        Usable statistics: ≥ LOGIT_STATISTICS_MIN_PIXELS = {min_pixels:,} pixels (our choice) and
        σ > {logit_standardization.MINIMUM_USABLE_STANDARD_DEVIATION:g} (smaller = a constant logit).
        The variants s05-s07 build from these statistics (the argmax over the adjusted logits gives the prediction):
        """) + variants)

    utils.section(
        "Progress per class",
        """
        images            statistics images of class c (D_c)
        pixels            all pixels of those images (every pixel counts once per model and class in all_pixels)
        distinct circuits circuits of class c with different masks that are not the full model (at most 3 circuit
                          types × |grid|); each runs once per image
        forward passes    images × (1 + distinct circuits), without gradients
        """)
    print(f"{'':3s} {'class':13s} {'images':>6s} {'pixels':>12s} {'distinct circuits':>17s} {'forward passes':>14s} "
          f"{'time':>6s}")
    results = {}
    for c in C.CLASSES:
        start = time.time()
        circuits, _, topk_layer_normalized_ratio_grid = utils.load_circuits(c, certifiable_slices, PRUNED_TYPES)
        entry = discovery_and_eval_lists["discovery_samples"][str(c)]
        sums, pixels_per_image = class_sums(model, dataset, entry, circuits, full_key, device)

        statistics = {key: stacked({p: logit_standardization.statistics_from_sums(model_sums[p], min_pixels)
                                    for p in PIXEL_SETS})
                      for key, model_sums in sums.items()}
        full_arrays = statistics[full_key]
        circuit_arrays = {q: np.stack([np.stack([statistics[key][q] for _, key in circuits[t]]) for t in PRUNED_TYPES])
                          for q in logit_standardization.QUANTITIES}             # [types, |grid|, 3, 21]
        np.savez_compressed(
            out_dir / f"class_{c:02d}.npz",
            **{f"full_model_{q}": full_arrays[q] for q in logit_standardization.QUANTITIES},
            **{f"circuit_{q}": circuit_arrays[q] for q in logit_standardization.QUANTITIES},
            circuit_mask_hash=np.array([[logit_standardization.mask_hash(key) for _, key in circuits[t]]
                                        for t in PRUNED_TYPES]),
            circuit_types=np.array(PRUNED_TYPES), pixel_sets=np.array(PIXEL_SETS),
            topk_layer_normalized_ratio_grid=topk_layer_normalized_ratio_grid,
            image_ids=np.array(entry["ids"]), image_indices=np.array(entry["indices"]),
            logit_statistics_min_pixels=min_pixels,
            minimum_usable_standard_deviation=logit_standardization.MINIMUM_USABLE_STANDARD_DEVIATION)
        results[c] = {"full": full_arrays, "circuit": circuit_arrays, "grid": topk_layer_normalized_ratio_grid,
                      "images": len(entry["ids"]), "pixels": sum(pixels_per_image)}
        n_distinct = len(sums) - 1
        print(f"{c:3d} {VOC_CLASSES[c]:13s} {len(entry['ids']):6d} {sum(pixels_per_image):12,d} {n_distinct:17d} "
              f"{len(entry['ids']) * (1 + n_distinct):14d} {time.time() - start:5.0f}s", flush=True)

    # ---- [2/4] the full model's pixel sets ------------------------------------------------------
    p_full_prediction, p_all, p_own = (PIXEL_SETS.index(p) for p in ("full_model_prediction", "all_pixels",
                                                                     "own_prediction"))
    utils.section(
        "[2/4] Per class c: the pixels the FULL MODEL predicts as each class d on D_c, and which classes have "
        "statistics",
        f"""
        One block per class c; the statistics images are c's D_c. Rows: every class d the full model predicts on at
        least one pixel of D_c, most pixels first.
        pixels     pixels the full model predicts as d, summed over the images of D_c = the size of d's pixel set
                   full_model_prediction (for the full model also own_prediction).
        share      of all pixels of D_c.
        statistics "usable" = ≥ {min_pixels:,} pixels and σ > {logit_standardization.MINIMUM_USABLE_STANDARD_DEVIATION:g}. A class without usable statistics on
                   full_model_prediction keeps its raw logit in matched_to_full_model_on_full_model_prediction_pixels
                   (for every circuit of c), and the Jung variant leaves it out of the full model's argmax.
        never predicted: classes with 0 pixels, so no statistics on full_model_prediction either. In all_pixels
                   EVERY class has statistics (every pixel counts for every class).
        """)
    for c, result in results.items():
        counts = result["full"]["pixel_count"][p_full_prediction]
        usable = result["full"]["statistics_usable"][p_full_prediction]
        print(f"\nclass {c} {VOC_CLASSES[c]} | D_c: {result['images']} images, {result['pixels']:,} pixels")
        print(f"  {'':3s} {'class d':13s} {'pixels':>12s} {'share':>8s}  statistics")
        for d in np.argsort(-counts, kind="stable"):
            if counts[d] == 0:
                break
            reason = "usable" if usable[d] else (f"NONE: fewer than {min_pixels:,} pixels" if counts[d] < min_pixels
                                                 else "NONE: σ ≈ 0")
            print(f"  {d:3d} {VOC_CLASSES[d]:13s} {counts[d]:12,d} {counts[d] / result['pixels']:8.2%}  {reason}")
        never = np.flatnonzero(counts == 0)
        print(f"  never predicted ({len(never)} classes, no statistics): {names(never)}")

    # ---- [3/4] per class c, circuit type and K ----------------------------------------------------
    ratio_column = utils.topk_ratio_label()
    utils.section(
        "[3/4] Per class c, circuit type and K: classes with usable statistics per variant, the shift of class c and "
        "background, and the classes the Jung variant loses",
        f"""
        One table per class c (statistics images: c's D_c). First row: the full model; then every circuit type and K.
        usable (of {NUM_CLASSES}), per pixel set = the classes the variant on that pixel set CORRECTS (the rest fall back):
          full_model_prediction  matched_to_full_model_on_full_model_prediction_pixels: usable in the circuit AND the
                                 full model. Both are measured on the same pixels, so the counts are the same; they
                                 differ only where a circuit's σ ≈ 0.
          all_pixels             matched_to_full_model_on_all_pixels: every class has pixels, so {NUM_CLASSES} unless σ ≈ 0.
          own_prediction         jung_standardized_logits_on_own_prediction_pixels: usable in the model itself, on
                                 the pixels it predicts as d. The others are left out of its argmax.
        class c / background: the circuit's class-c logit and background logit compared with the full model's, on the
        pixels the FULL MODEL predicts as c / as background (pixel set full_model_prediction), in logit units:
          Δμ        μ_circuit − μ_full: < 0 = the circuit's logit is lower there than the full model's
          σ ratio   σ_circuit / σ_full: < 1 = compressed, > 1 = spread out
          matched_to_full_model_on_full_model_prediction_pixels undoes exactly these shifts (when usable).
          nan = the full model predicts no pixel as that class on D_c.
        Jung variant vs full model (own_prediction):
          lost      classes the full model has usable statistics for and the circuit not: the circuit can never
                    predict them after the Jung variant
          new       classes the circuit has usable statistics for and the full model not
        other fallbacks: classes a matched variant leaves raw only because the CIRCUIT's σ ≈ 0 (beyond the full
                    model's own fallbacks). Normally "-".
        At K = 1 every circuit is the full model: Δμ = 0, σ ratio = 1, nothing lost.
        """)
    for c, result in results.items():
        full = result["full"]
        full_statistics = per_pixel_set(full)
        name_c = VOC_CLASSES[c]
        print(f"\nclass {c} {name_c} | full model on its full_model_prediction pixels: "
              + "; ".join(f"{VOC_CLASSES[d]} μ = {full['logit_mean'][p_full_prediction, d]:.2f}, "
                          f"σ = {full['logit_standard_deviation'][p_full_prediction, d]:.2f} "
                          f"({full['pixel_count'][p_full_prediction, d]:,} pixels)" for d in (c, 0)))
        group = f"usable (of {NUM_CLASSES}) per pixel set"
        print(f"  {'':16s} {'':{len(ratio_column)}s} | {group:^49s} | {'class ' + name_c:^17s} | {'background':^17s} |")
        print(f"  {'circuit type':16s} {ratio_column} | {'full_model_prediction':>21s} {'all_pixels':>10s} "
              f"{'own_prediction':>14s} | {'Δμ':>7s} {'σ ratio':>8s} | {'Δμ':>7s} {'σ ratio':>8s} | "
              f"Jung variant vs full model; other fallbacks")
        full_without = {v: logit_standardization.class_adjustment(v, full_statistics, full_statistics)[2]
                        for v in logit_standardization.VARIANTS}
        rows = [("full model", "-", full)] + [
            (t, f"{grid_value:g}", {q: result["circuit"][q][t_idx, grid_idx] for q in result["circuit"]})
            for t_idx, t in enumerate(PRUNED_TYPES) for grid_idx, grid_value in enumerate(result["grid"])]
        for label, grid_label, arrays in rows:
            statistics = per_pixel_set(arrays)
            without = {v: logit_standardization.class_adjustment(v, statistics, full_statistics)[2]
                       for v in logit_standardization.VARIANTS}
            usable_counts = [NUM_CLASSES - int(without[v].sum()) for v in logit_standardization.VARIANTS]
            shifts = ""
            for d in (c, 0):
                with np.errstate(divide="ignore", invalid="ignore"):
                    shift = arrays["logit_mean"][p_full_prediction, d] - full["logit_mean"][p_full_prediction, d]
                    ratio = (arrays["logit_standard_deviation"][p_full_prediction, d]
                             / full["logit_standard_deviation"][p_full_prediction, d])
                shifts += f" {shift:+7.2f} {ratio:8.3f} |"
            own_full = full["statistics_usable"][p_own]
            own_model = arrays["statistics_usable"][p_own]
            jung = f"lost: {names(np.flatnonzero(own_full & ~own_model))}; new: {names(np.flatnonzero(own_model & ~own_full))}"
            other = "; ".join(f"{logit_standardization.VARIANT_PIXEL_SET[v]}: {names(np.flatnonzero(without[v] & ~full_without[v]))}"
                              for v in logit_standardization.MATCHED_VARIANTS if (without[v] & ~full_without[v]).any())
            print(f"  {label:16s} {grid_label:>{len(ratio_column)}s} | {usable_counts[0]:21d} {usable_counts[1]:10d} "
                  f"{usable_counts[2]:14d} |{shifts} {jung}; other fallbacks: {other or '-'}")

    # ---- [4/4] mean over classes: every class d over all pixels ------------------------------------
    grid = next(iter(results.values()))["grid"]
    same_grid = all(np.allclose(r["grid"], grid) for r in results.values())
    utils.section(
        f"[4/4] Shift of EVERY class's logit, circuit vs full model, over ALL pixels of D_c, mean over the "
        f"{len(results)} classes c",
        f"""
        One pair of tables per circuit type. Rows: the {NUM_CLASSES} classes d. Columns: the {utils.topk_ratio_label()} K.
        μ_full   the full model's mean class-d logit over all pixels of D_c, for reference (logit units).
        Δμ       μ_d^circuit − μ_d^full over ALL pixels of D_c (pixel set all_pixels), in logit units. This is the
                 shift matched_to_full_model_on_all_pixels undoes. > 0 = the circuit says "not d" less strongly
                 (or "d" more strongly) than the full model.
        σ ratio  σ_d^circuit / σ_d^full on the same pixels: 1 = unchanged spread, < 1 = compressed.
        Each cell is the mean over the classes c, each c's circuits measured on its own D_c (each class c weighted
        equally). Per class c, all values are in the class_XX.npz files.
        At K = 1 every circuit is the full model: Δμ = 0, σ ratio = 1.
        """)
    if not same_grid:
        print(f"NOTE: the classes have different {utils.topk_ratio_label()} grids (s03 re-run with another grid for "
              "some class); the tables below are skipped.")
    else:
        with warnings.catch_warnings():  # all-nan cells print as nan
            warnings.simplefilter("ignore", RuntimeWarning)
            full_mean = np.nanmean([r["full"]["logit_mean"][p_all] for r in results.values()], axis=0)
            for t_idx, t in enumerate(PRUNED_TYPES):
                for title, measure, fmt in [
                    ("Δμ", lambda r: r["circuit"]["logit_mean"][t_idx, :, p_all] - r["full"]["logit_mean"][p_all], "{:+7.2f}"),
                    ("σ ratio", lambda r: r["circuit"]["logit_standard_deviation"][t_idx, :, p_all]
                     / r["full"]["logit_standard_deviation"][p_all], "{:7.3f}")]:
                    values = np.nanmean([measure(r) for r in results.values()], axis=0)     # [|grid|, 21]
                    print(f"\ncircuit type: {t} | {title} over all pixels, mean over classes c | columns: {ratio_column}")
                    print(f"  {'':3s} {'class d':13s} {'μ_full':>7s}" + "".join(f" {g:>7g}" for g in grid))
                    for d in range(NUM_CLASSES):
                        print(f"  {d:3d} {VOC_CLASSES[d]:13s} {full_mean[d]:+7.2f}"
                              + "".join(" " + fmt.format(values[g, d]) for g in range(len(grid))))

    utils.write_run_info(out_dir)
    print(f"\nSaved {len(results)} class files to {out_dir}")


if __name__ == "__main__":
    main()
