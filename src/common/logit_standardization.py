"""Class-wise logit statistics and the logit-standardization variants (Jung et al. 2021, adapted).

A model m (the full model or a circuit) gives NUM_CLASSES = 21 logit maps F_d per image. For every class d and
every pixel set, s04b sums the pixel count, Σ F_d and Σ F_d² over the statistics images; from the sums follow the
mean μ_d^m and the standard deviation σ_d^m of m's class-d logit. Pixel sets (PIXEL_SETS):
    full_model_prediction   the pixels the FULL MODEL predicts as d (the same pixels for every circuit)
    own_prediction          the pixels m ITSELF predicts as d (Jung Eq. 3-4; for the full model = full_model_prediction)
    all_pixels              every pixel, for every class

Variants (VARIANTS); each adjusts all 21 logit maps, and the prediction is the argmax over the adjusted logits:
    matched_to_full_model_on_full_model_prediction_pixels
        F'_d = μ_d^full + σ_d^full · (F_d − μ_d^m) / σ_d^m      statistics on full_model_prediction
    matched_to_full_model_on_all_pixels
        the same map                                              statistics on all_pixels
    jung_standardized_logits_on_own_prediction_pixels
        z_d = (F_d − μ_d^m) / σ_d^m                               statistics on own_prediction (Jung Eq. 3-4)
A class's statistics are usable with at least LOGIT_STATISTICS_MIN_PIXELS pixels in its pixel set and
σ > MINIMUM_USABLE_STANDARD_DEVIATION (for a matched variant: in m and in the full model). A class without usable
statistics keeps its raw logit in the matched variants, and is left out of the argmax (z = −∞) in the Jung variant.
The matched map is applied as F'_d = scale_d · F_d + shift_d, so it is exactly the identity when m's statistics
equal the full model's (the full model itself, and every circuit that keeps all channels): scale 1, shift 0.
"""

import hashlib
import sys
import textwrap

import numpy as np
import torch

import config as C
from common import utils

PIXEL_SETS = ["full_model_prediction", "own_prediction", "all_pixels"]
VARIANT_PIXEL_SET = {
    "matched_to_full_model_on_full_model_prediction_pixels": "full_model_prediction",
    "matched_to_full_model_on_all_pixels": "all_pixels",
    "jung_standardized_logits_on_own_prediction_pixels": "own_prediction",
}
VARIANTS = list(VARIANT_PIXEL_SET)
MATCHED_VARIANTS = VARIANTS[:2]   # the identity for the full model
JUNG_VARIANT = VARIANTS[2]        # changes the full model too
VARIANT_DESCRIPTIONS = {
    "matched_to_full_model_on_full_model_prediction_pixels":
        "F'_d = μ_d^full + σ_d^full · (F_d − μ_d^circuit) / σ_d^circuit for every class d, with both models'\n"
        "statistics on the pixels the FULL MODEL predicts as d. A class without usable statistics\n"
        "keeps its raw logit. The identity for the full model.",
    "matched_to_full_model_on_all_pixels":
        "the same map as the matched variant above, with both models' statistics over ALL pixels of the\n"
        "statistics images, so every class has statistics. The identity for the full model.",
    "jung_standardized_logits_on_own_prediction_pixels":
        "z_d = (F_d − μ_d^m) / σ_d^m for every class d, with the statistics of the model m ITSELF (full model or\n"
        "circuit) on the pixels m itself predicts as d (Jung Eq. 3-4, applied to all 21 logits). A class without\n"
        "usable statistics is left out of the argmax (z = −∞). Changes the full model too.",
}
QUANTITIES = ["pixel_count", "logit_mean", "logit_standard_deviation", "statistics_usable"]
# Numerical guard (our choice): a smaller σ counts as σ ≈ 0, i.e. a constant logit, and gives no usable statistics.
MINIMUM_USABLE_STANDARD_DEVIATION = 1e-6


def section_title_suffix(variant):
    """Title suffix of a log section in s05 / s06: nothing without logit standardization, else which variant
    ("raw" or a LOGIT_STANDARDIZATION_VARIANTS entry) the section shows."""
    if not C.LOGIT_STANDARDIZATION_VARIANTS:
        return ""
    return (", RAW circuits (no logit standardization)" if variant == "raw"
            else f", logit standardization variant {variant}")


def how_to_read_variant(variant):
    """Lines that open the "How to read" block of an s05 / s06 section of one variant ("" for raw)."""
    if variant == "raw":
        return ""
    full = ("identical to raw: the matched variants are the identity for the full model" if variant in
            MATCHED_VARIANTS else "the full model standardized as well; differs from raw")
    return (f"LOGIT STANDARDIZATION VARIANT {variant}\n"
            f"Every number is computed from the prediction AFTER this variant:\n"
            + textwrap.indent(VARIANT_DESCRIPTIONS[variant], "  ") + "\n"
            f"Same forward passes as the raw numbers. The full model's numbers are the full model after this\n"
            f"variant ({full}).\n"
            f"faith, |r| and num_no_pred keep the RAW full model as their reference. |r| equals the raw value (the\n"
            f"adjustment is affine per class, with a positive slope), except nan where the Jung variant leaves\n"
            f"class c out of the argmax.\n")


def check_variants(variants, parameter_name):
    """Stop with the allowed names if `variants` holds an unknown or repeated variant."""
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or len(set(variants)) != len(variants):
        sys.exit(f"\n[config] {parameter_name} = {list(variants)}: "
                 f"{'unknown ' + str(unknown) if unknown else 'a variant is listed twice'}; allowed: {VARIANTS}")


def mask_hash(key):
    """Short text id of a circuit's mask key (utils.mask_key), stored by s04b to detect stale statistics."""
    return hashlib.sha1(key).hexdigest()


# ------------------------------------------------------------------
# Statistics (s04b)
# ------------------------------------------------------------------
def new_sums(num_classes, device, pixel_sets=PIXEL_SETS):
    """Zeroed sums of one model: {pixel set: {"pixel_count", "logit_sum", "logit_squared_sum": [num_classes]}}."""
    return {pixel_set: {"pixel_count": torch.zeros(num_classes, dtype=torch.int64, device=device),
                        "logit_sum": torch.zeros(num_classes, dtype=torch.float64, device=device),
                        "logit_squared_sum": torch.zeros(num_classes, dtype=torch.float64, device=device)}
            for pixel_set in pixel_sets}


def accumulate_class_logit_statistics(logits, pixel_labels, sums):
    """Adds to `sums` (one pixel set of new_sums), for every class d, the pixel count, Σ F_d and Σ F_d² (float64)
    over the pixels with pixel_labels == d. logits: [num_classes, H, W]; pixel_labels: [H, W] class indices, or
    None for "all pixels" (every pixel counts for every class).
    Non-finite logits are not counted: they occur only in s05, where the Jung variant sets a class without usable
    statistics to −∞ (raw logits, as in s04b, are always finite, so this changes nothing there)."""
    values = logits.double()
    if pixel_labels is None:
        finite = torch.isfinite(values)
        values = torch.where(finite, values, 0.0)
        sums["pixel_count"] += finite.sum(dim=(1, 2))
        sums["logit_sum"] += values.sum(dim=(1, 2))
        sums["logit_squared_sum"] += (values ** 2).sum(dim=(1, 2))
        return
    num_classes = logits.shape[0]
    labels = pixel_labels.reshape(-1)
    values = values.reshape(num_classes, -1).gather(0, labels[None])[0]  # F_d at each pixel labelled d
    finite = torch.isfinite(values)
    labels, values = labels[finite], values[finite]
    sums["pixel_count"] += torch.bincount(labels, minlength=num_classes)
    sums["logit_sum"] += torch.bincount(labels, weights=values, minlength=num_classes)
    sums["logit_squared_sum"] += torch.bincount(labels, weights=values ** 2, minlength=num_classes)


def accumulate_pixel_set_statistics(logits, full_model_labels, model_sums):
    """One forward pass of a model (full model or circuit) into its sums of every pixel set in PIXEL_SETS.
    full_model_labels: the full model's argmax on the same image."""
    accumulate_class_logit_statistics(logits, full_model_labels, model_sums["full_model_prediction"])
    accumulate_class_logit_statistics(logits, logits.argmax(dim=0), model_sums["own_prediction"])
    accumulate_class_logit_statistics(logits, None, model_sums["all_pixels"])


def statistics_from_sums(sums, min_pixels):
    """μ, σ and a usable flag per class from one pixel set's sums:
    {"pixel_count", "logit_mean", "logit_standard_deviation", "statistics_usable": [num_classes] numpy}.
    μ and σ are nan for a class with no pixel. Usable = at least min_pixels pixels and σ > MINIMUM_USABLE_STANDARD_DEVIATION."""
    def to_numpy(value):
        return value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    pixel_count = to_numpy(sums["pixel_count"]).astype(np.int64)
    logit_sum = to_numpy(sums["logit_sum"]).astype(np.float64)
    logit_squared_sum = to_numpy(sums["logit_squared_sum"]).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logit_mean = np.where(pixel_count > 0, logit_sum / pixel_count, np.nan)
        variance = np.where(pixel_count > 0, logit_squared_sum / pixel_count - logit_mean ** 2, np.nan)
    logit_standard_deviation = np.sqrt(np.maximum(variance, 0.0))  # rounding can make a zero variance negative
    statistics_usable = (pixel_count >= max(min_pixels, 1)) & \
        (logit_standard_deviation > MINIMUM_USABLE_STANDARD_DEVIATION)  # False where σ is nan
    return {"pixel_count": pixel_count, "logit_mean": logit_mean,
            "logit_standard_deviation": logit_standard_deviation, "statistics_usable": statistics_usable}


# ------------------------------------------------------------------
# Logit distributions vs the full model (s05, s06)
# ------------------------------------------------------------------
# The two pixel sets, both defined by the RAW full model's prediction, so a circuit and its reference are
# always measured on the same pixels: the pixels the full model predicts as d, and every pixel.
DISTRIBUTION_PIXEL_SETS = ["full_model_prediction", "all_pixels"]


def histogram_inner_bin_count(lower_edge, upper_edge, bin_width):
    """Number of inner bins between the two edges; stops if the range is not a whole number of bins."""
    count = (upper_edge - lower_edge) / bin_width
    if count < 1 or abs(count - round(count)) > 1e-9:
        sys.exit(f"\n[config] the logit histogram range [{lower_edge}, {upper_edge}] is not a whole number of bins of "
                 f"width {bin_width}.")
    return int(round(count))


def histogram_bin_edges(lower_edge, upper_edge, bin_width):
    """The inner bin edges in logit units, [inner bins + 1]. Every histogram has one more bin at each end: index 0
    counts the values below lower_edge (underflow), the last index the values ≥ upper_edge (overflow); inner bin k
    (1 ≤ k ≤ inner bins) is [lower_edge + (k − 1)·bin_width, lower_edge + k·bin_width)."""
    return lower_edge + bin_width * np.arange(histogram_inner_bin_count(lower_edge, upper_edge, bin_width) + 1)


def accumulate_class_logit_histograms(logits, pixel_labels, histograms, lower_edge, bin_width):
    """Adds to `histograms` ([num_classes, inner bins + 2] int64 tensor, bins as in histogram_bin_edges), for every
    class d, the binned F_d over the pixels with pixel_labels == d (pixel_labels None: every pixel, for every class).
    Non-finite logits (−∞ from the Jung variant) are not counted, as in accumulate_class_logit_statistics."""
    num_classes, n_bins = histograms.shape
    position = (logits.float() - lower_edge) / bin_width
    finite = torch.isfinite(position)
    bins = (torch.floor(position.clamp(-1, n_bins - 2)).long() + 1).reshape(num_classes, -1)
    finite = finite.reshape(num_classes, -1)
    if pixel_labels is None:
        classes = torch.arange(num_classes, device=logits.device)[:, None].expand_as(bins)
        index = (classes * n_bins + bins)[finite]
    else:
        labels = pixel_labels.reshape(-1)
        keep = finite.gather(0, labels[None])[0]
        index = (labels * n_bins + bins.gather(0, labels[None])[0])[keep]
    histograms += torch.bincount(index, minlength=num_classes * n_bins).reshape(num_classes, n_bins)


def wasserstein_1_from_histograms(first, second, bin_width):
    """Approximate Wasserstein-1 distance (logit units) between two distributions given as histograms with the same
    bins (last axis; underflow and overflow bins included): every count is placed at its bin's centre, the underflow
    and overflow bins half a bin outside the range. Then W1 = bin_width · Σ |CDF_first − CDF_second| over the bins, a
    binned approximation with resolution bin_width. nan where either histogram is empty."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf_first = np.cumsum(first, axis=-1) / first.sum(axis=-1, keepdims=True)
        cdf_second = np.cumsum(second, axis=-1) / second.sum(axis=-1, keepdims=True)
    return bin_width * np.abs(cdf_first - cdf_second)[..., :-1].sum(axis=-1)


def quantiles_from_histogram(histogram, bin_edges, probabilities):
    """Approximate quantiles of a histogram (bins as in histogram_bin_edges), linearly interpolated inside the bin
    that holds each quantile. A quantile in the underflow / overflow bin is reported as the range edge. [len
    (probabilities)] numpy, nan for an empty histogram."""
    histogram = np.asarray(histogram, dtype=np.float64)
    total = histogram.sum()
    if total == 0:
        return np.full(len(probabilities), np.nan)
    cumulative = np.cumsum(histogram) / total
    out = []
    for p in probabilities:
        k = int(np.searchsorted(cumulative, p))  # first bin whose cumulative share reaches p
        if k == 0:
            out.append(bin_edges[0])
        elif k >= len(histogram) - 1:
            out.append(bin_edges[-1])
        else:
            below = cumulative[k - 1]
            fraction = (p - below) / (cumulative[k] - below) if cumulative[k] > below else 0.5
            out.append(bin_edges[k - 1] + fraction * (bin_edges[k] - bin_edges[k - 1]))
    return np.array(out)


# ------------------------------------------------------------------
# Adjustment (s05, s07)
# ------------------------------------------------------------------
def class_adjustment(variant, statistics_model, statistics_full_model):
    """The per-class map F'_d = scale_d · F_d + shift_d of a variant, and the classes without usable statistics:
    scale, shift [num_classes] float64 and without_statistics [num_classes] bool. statistics_*: {pixel set:
    statistics_from_sums(...)} of the model m being adjusted and of the full model (only read by the matched
    variants). Classes without statistics get scale 1, shift 0 here; adjust_logits keeps them raw (matched) or
    sets them to −∞ (Jung)."""
    pixel_set = VARIANT_PIXEL_SET[variant]
    model = statistics_model[pixel_set]
    without_statistics = ~model["statistics_usable"]
    if variant in MATCHED_VARIANTS:
        full = statistics_full_model[pixel_set]
        without_statistics |= ~full["statistics_usable"]
        target_mean, target_standard_deviation = full["logit_mean"], full["logit_standard_deviation"]
    else:  # Jung: standardize to mean 0, standard deviation 1
        target_mean = np.zeros_like(model["logit_mean"])
        target_standard_deviation = np.ones_like(model["logit_standard_deviation"])
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = target_standard_deviation / model["logit_standard_deviation"]
        shift = target_mean - scale * model["logit_mean"]
    scale = np.where(without_statistics, 1.0, scale)
    shift = np.where(without_statistics, 0.0, shift)
    return scale, shift, without_statistics


@torch.inference_mode()
def adjust_logits(logits, variant, statistics_model, statistics_full_model):
    """[num_classes, H, W] logits after the variant (see the module docstring); the argmax gives the prediction."""
    scale, shift, without_statistics = class_adjustment(variant, statistics_model, statistics_full_model)

    def per_class(values):
        return torch.as_tensor(values, dtype=logits.dtype, device=logits.device).view(-1, 1, 1)

    adjusted = logits * per_class(scale) + per_class(shift)
    if variant == JUNG_VARIANT and without_statistics.any():
        adjusted[torch.as_tensor(without_statistics, device=logits.device)] = float("-inf")
    return adjusted


@torch.inference_mode()
def standardized_max_logit(logits, statistics):
    """Jung et al.'s standardized max logit as written (Eq. 5): per pixel,
    S = (L − μ_Ŷ) / σ_Ŷ, with L the winning logit, Ŷ the winning class and μ, σ the statistics of class Ŷ on the
    pixel set own_prediction from `statistics` ({pixel set: statistics_from_sums(...)}, e.g. a circuit's own or the
    full model's). The label is not changed; S is a per-pixel confidence score (low S = unsure). [H, W] float32 numpy
    array, nan where class Ŷ has no usable statistics."""
    own = statistics["own_prediction"]
    winning_logit, winning_class = logits.max(dim=0)
    winning_class = winning_class.cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (winning_logit.cpu().numpy().astype(np.float64) - own["logit_mean"][winning_class]) \
            / own["logit_standard_deviation"][winning_class]
    return np.where(own["statistics_usable"][winning_class], score, np.nan).astype(np.float32)


def load_class_logit_statistics(c, circuits):
    """s04b's statistics of class c for the circuit types in `circuits` (from utils.load_circuits):
    (the full model's statistics, {circuit type: a list over the grid indices of statistics}), each statistics
    {pixel set: {"pixel_count", "logit_mean", "logit_standard_deviation", "statistics_usable": [num_classes]}}.
    Stops if s04b is missing, or if it ran on other circuits (e.g. s04 was re-run after s04b)."""
    path = utils.logit_statistics_dir() / f"class_{c:02d}.npz"
    utils.require(path, "s04b_logit_statistics.py")
    stored = np.load(path)
    pixel_sets = stored["pixel_sets"].tolist()
    stored_types = stored["circuit_types"].tolist()

    def statistics(prefix, index):
        return {pixel_set: {q: stored[f"{prefix}_{q}"][index + (p,)] for q in QUANTITIES}
                for p, pixel_set in enumerate(pixel_sets)}

    circuit_statistics = {}
    for t, circuits_per_grid_idx in circuits.items():
        type_idx = stored_types.index(t)
        if [mask_hash(key) for _, key in circuits_per_grid_idx] != stored["circuit_mask_hash"][type_idx].tolist():
            sys.exit(f"\n[stale input] {path}\nholds the statistics of other {t} circuits than s04's current "
                     f"class_{c:02d}.npz, e.g. because s04 was re-run after s04b.\n"
                     f"-> re-run s04b_logit_statistics.py with the current config.py.")
        circuit_statistics[t] = [statistics("circuit", (type_idx, grid_idx))
                                 for grid_idx in range(len(circuits_per_grid_idx))]
    return statistics("full_model", ()), circuit_statistics
