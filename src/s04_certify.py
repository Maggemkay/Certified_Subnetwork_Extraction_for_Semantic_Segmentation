"""s04 · Certification (Anani Alg. 1: SegCertify decision rule on RS-Del samples).

U = the certifiable components: the channels of the LAYERS group. For every class, every top-K
layer-normalized ratio K and every u ∈ U (τ = AGREEMENT_THRESHOLD_TAU, α = FAILURE_PROB_ALPHA):
    ĉ_u  = majority decision over the N0 selection samples (1 = in, 0 = out; a tie counts as out)
    n_u  = number of the N certification samples that agree with ĉ_u
    pv_u = P(Binom(N, τ) ≥ n_u)          one-sided binomial test of H0: P(A agrees with ĉ_u) ≤ τ
    decision_u = ĉ_u if pv_u ≤ α / |U| (Bonferroni), else abstain (⊘, stored as -1)
Every certified decision holds for all discovery sets within edit distance
    r = ⌊log(1.5 − τ) / log P_DEL⌋    (Anani Eq. 5)
with confidence 1 − α.

Also: the majority-vote circuit (u in if it is selected in > 50 % of the N samples, no τ; Anani
App. F.5) and the subnetwork sizes of the three circuit types (baseline_circuit, majority, certified):
subnet_size_layer_normalized_ratio_mean = mean over the layers in U of the ratio of the layer's channels in the
circuit (equal weight per layer), and the pooled sizes over all channels of U (absolute and relative).

The certification is the same for every TOPK_SELECTION_MODE of s03 (it only counts how often each channel is
selected); s04 checks that s03 ran with config.py's TOPK_SELECTION_MODE (and, in a global mode, with LAYERS as the
group of the global top-K).

Output: <certify_dir>/class_XX.npz and run_info.json, with (restricted to U; |grid| = number of grid values;
<type> = baseline_circuit, majority, certified)
    decision           [|grid|, |U|] int8     1 = certified in, 0 = certified out, -1 = abstain
    pvalues            [|grid|, |U|]
    majority           [|grid|, |U|] bool
    baseline_circuit   [|grid|, |U|] bool
    subnet_size_layer_normalized_ratio_mean_<type>   [|grid|]            equal-weight mean of the per-layer shares
    subnet_size_group_channel_count_<type>           [|grid|] int        channels of U kept (absolute, pooled)
    subnet_size_group_pooled_ratio_<type>            [|grid|]            that count / |U| (relative, pooled)
    subnet_size_layer_channel_count_<type>           [|grid|, layers of U] int   channels kept per layer of U
    topk_layer_normalized_ratio_grid, layers (names in U), certifiable_index (positions of U in the 4,352
    components), radius, topk_selection_mode and topk_selection_group (as s03 stored them)
"""

import math
import sys
import textwrap
import warnings

import numpy as np
from scipy.stats import binom

import config as C
from common import model as M, utils
from common.datasets import VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "CLASSES",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA", "SMOKE_TEST",
]
CIRCUIT_SIZE_TYPES = ["baseline_circuit", "majority", "certified"]


def subnet_size_layer_normalized_ratio_mean(mask, certifiable_slices):
    """Ratio of each layer's channels in the circuit, averaged over the layers of U with equal weight:
    [|grid|, |U|] -> [|grid|]."""
    return np.mean([mask[:, s].mean(axis=1) for s in certifiable_slices.values()], axis=0)


def check_topk_selection(samples, path):
    """Stops if s03 ran with another TOPK_SELECTION_MODE, or (global mode) another group, than config.py's.
    s03 files without these keys (older s03 outputs) are per_layer. Returns (mode, group) as stored."""
    stored_mode = str(samples["topk_selection_mode"]) if "topk_selection_mode" in samples.files else "per_layer"
    stored_group = str(samples["topk_selection_group"]) if "topk_selection_group" in samples.files else "none"
    expected = (utils.topk_selection_mode(), utils.topk_selection_group())
    if (stored_mode, stored_group) != expected:
        sys.exit(f"\n[config mismatch] {path}\nwas written by s03 with TOPK_SELECTION_MODE = {stored_mode!r} and "
                 f"group {stored_group!r}; config.py has TOPK_SELECTION_MODE = {expected[0]!r} and group "
                 f"{expected[1]!r} (the group is LAYERS in a global mode, \"none\" for per_layer).\n"
                 f"-> re-run s03_sample_circuits.py with the current config.py, or set config.py back.")
    return stored_mode, stored_group


def circuit_sizes(name, mask, certifiable_slices):
    """The stored size measures of one circuit type `name`, mask [|grid|, |U|] -> {key: array}."""
    layer_channel_count = utils.layer_channel_counts(mask, certifiable_slices)       # [|grid|, layers of U]
    group_channel_count = layer_channel_count.sum(axis=1)                            # [|grid|]
    return {
        f"subnet_size_layer_normalized_ratio_mean_{name}": subnet_size_layer_normalized_ratio_mean(mask, certifiable_slices),
        f"subnet_size_group_channel_count_{name}": group_channel_count,
        f"subnet_size_group_pooled_ratio_{name}": group_channel_count / mask.shape[1],
        f"subnet_size_layer_channel_count_{name}": layer_channel_count,
    }


def certify_class(c, certifiable_index, certifiable_slices):
    path = utils.sample_dir() / f"class_{c:02d}.npz"
    utils.require(path, "s03_sample_circuits.py")
    samples = np.load(path)
    topk_selection_mode, topk_selection_group = check_topk_selection(samples, path)
    counts_N0 = samples["counts_N0"][:, certifiable_index]
    counts_N = samples["counts_N"][:, certifiable_index]

    c_hat = counts_N0 > C.N0 / 2
    n_agree = np.where(c_hat, counts_N, C.N - counts_N)
    pvalues = binom.sf(n_agree - 1, C.N, C.AGREEMENT_THRESHOLD_TAU)        # P(Binom(N, τ) ≥ n_agree)
    certified = pvalues <= C.FAILURE_PROB_ALPHA / len(certifiable_index)
    decision = np.where(certified, c_hat, -1).astype(np.int8)

    majority = counts_N / C.N > 0.5
    baseline_circuit = samples["baseline_circuit"][:, certifiable_index]
    result = {
        "decision": decision,
        "pvalues": pvalues,
        "majority": majority,
        "baseline_circuit": baseline_circuit,
        "topk_layer_normalized_ratio_grid": samples["topk_layer_normalized_ratio_grid"],
        "topk_selection_mode": np.array(topk_selection_mode),
        "topk_selection_group": np.array(topk_selection_group),
    }
    for name, mask in zip(CIRCUIT_SIZE_TYPES, (baseline_circuit, majority, decision == 1)):
        result.update(circuit_sizes(name, mask, certifiable_slices))
    return result


def share_text(value):
    """A kept share for the per-layer tables: EMPTY for exactly 0, <0.1% for a share that rounds to 0.0 % but keeps a
    channel, nan for nan."""
    return "nan" if np.isnan(value) else "EMPTY" if value == 0 else "<0.1%" if value < 0.0005 else f"{value:.1%}"


def print_layer_breakdown(n_sections, ratio_column, results, layers, sizes, topk_layer_normalized_ratio_grid):
    """Section 8: per class, K and layer of U, the share of the layer's channels in the baseline circuit and in the
    certified circuit; then the class means and the circuits with an empty layer."""
    shares = {c: {t: r[f"subnet_size_layer_channel_count_{t}"] / np.array(list(sizes.values()))
                  for t in CIRCUIT_SIZE_TYPES} for c, r in results.items()}
    layer_text = ", ".join(f"{name} {ch:,}" for name, ch in sizes.items())
    mode_text = (
        "In this mode (per_layer) the baseline column is round(K · C) / C ≈ K in every layer by construction; the "
        "certified column shows where certification drops channels."
        if not utils.topk_selection_is_global() else
        f"In this mode ({C.TOPK_SELECTION_MODE}) the baseline split comes from the global top-K (s03 table 6); the "
        f"certified column shows the split after certification.")
    utils.section(
        f"[{n_sections}/{n_sections}] WHICH LAYERS keep channels: share of each layer's channels in the baseline "
        f"circuit and in the certified circuit, per class",
        f"""
        One block per class. Rows: K (the {ratio_column}).
        Per layer of U ({layer_text} channels):
          baseline    channels of the baseline circuit A(D_c) in the layer / the layer's channels
          certified   certified-in channels of the layer / the layer's channels (abstaining channels are not in it)
        {textwrap.fill(mode_text, 104, subsequent_indent=" " * 8)}
        EMPTY = no channel of the layer is in the circuit. An empty layer cuts the network: no skip connection goes
        around a stage, so the circuit's output no longer depends on the image. <0.1% = at least one channel, but
        less than 0.05 % of the layer (e.g. 1 of layer4's 2,048 channels is 0.049 %).
        After the blocks: the mean over the classes, then per K the classes whose circuit has an empty layer.
        """)
    cell_width = 23
    header_names = "".join(f" | {f'{name} ({ch:,} channels)':^{cell_width}s}" for name, ch in sizes.items())
    header_columns = "".join(f" | {'baseline':>9s} {'certified':>13s}" for _ in sizes)

    def block(title, baseline, certified):
        print(f"\n{title}; rows: {ratio_column} K")
        print((f"{'K':>6s}" + header_names).rstrip())
        print(f"{'':6s}" + header_columns)
        for grid_idx, topk_layer_normalized_ratio in enumerate(topk_layer_normalized_ratio_grid):
            print(f"{topk_layer_normalized_ratio:6g}" + "".join(
                f" | {share_text(baseline[grid_idx, l]):>9s} {share_text(certified[grid_idx, l]):>13s}"
                for l in range(len(layers))))

    for c in results:
        block(f"class {c} {VOC_CLASSES[c]}", shares[c]["baseline_circuit"], shares[c]["certified"])
    block(f"mean over the {len(results)} classes",
          np.mean([shares[c]["baseline_circuit"] for c in results], axis=0),
          np.mean([shares[c]["certified"] for c in results], axis=0))

    print(f"\nCircuits with at least one EMPTY layer of U, per K: classes out of {len(results)} (which class and "
          f"layers: listed below the table)")
    print(f"{'K':>6s}" + "".join(f" | {t:>16s}" for t in CIRCUIT_SIZE_TYPES))
    listed = []
    for grid_idx, topk_layer_normalized_ratio in enumerate(topk_layer_normalized_ratio_grid):
        cells = []
        for t in CIRCUIT_SIZE_TYPES:
            empty = {c: [name for l, name in enumerate(layers) if shares[c][t][grid_idx, l] == 0] for c in results}
            empty = {c: names for c, names in empty.items() if names}
            cells.append(f"{len(empty)}/{len(results)}")
            if empty:
                listed.append(f"  K = {topk_layer_normalized_ratio:g}, {t}: "
                              + "; ".join(f"{c} {VOC_CLASSES[c]} ({', '.join(names)})" for c, names in empty.items()))
        print(f"{topk_layer_normalized_ratio:6g}" + "".join(f" | {cell:>16s}" for cell in cells))
    print("Circuits with an empty layer: " + ("none" if not listed else "\n" + "\n".join(listed)))


def main():
    utils.start_run("s04_certify", USED_PARAMS)
    utils.require(utils.sample_dir() / "run_info.json", "s03_sample_circuits.py")
    out_dir = utils.certify_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    layers, certifiable_index, certifiable_slices = M.certifiable_components(C.LAYERS)
    n_certifiable = len(certifiable_index)
    threshold = C.FAILURE_PROB_ALPHA / n_certifiable
    radius = math.floor(math.log(1.5 - C.AGREEMENT_THRESHOLD_TAU) / math.log(C.P_DEL))
    n_needed = next((n for n in range(C.N + 1) if binom.sf(n - 1, C.N, C.AGREEMENT_THRESHOLD_TAU) <= threshold),
                    None)
    sizes = {name: s.stop - s.start for name, s in certifiable_slices.items()}
    outside = [name for name in M.HOOK_POINTS if name not in layers]
    rule = (f"n_u ≥ {n_needed} of the N = {C.N} samples agree with ĉ_u ({n_needed / C.N:.1%})" if n_needed is not None
            else f"never: N = {C.N} is too small for τ = {C.AGREEMENT_THRESHOLD_TAU}, so EVERY channel abstains")
    is_global = utils.topk_selection_is_global()
    ratio_column = utils.topk_ratio_label()
    n_sections = 8

    utils.section(
        f"[1/{n_sections}] Certification setup",
        f"""
        U = the certifiable components (channels) = LAYERS = {C.LAYERS!r}: {", ".join(f"{name} ({ch})" for name, ch in sizes.items())};
        |U| = {n_certifiable} channels. Every channel of U gets a decision (in, out or abstain); the certified circuit is
        the part of U certified in. {"Layers outside U (" + ", ".join(outside) + ") are not certified and s05 never masks them." if outside else "U holds every hook point."}
        The circuits come from s03 with TOPK_SELECTION_MODE = {C.TOPK_SELECTION_MODE!r}: K = the {ratio_column}
        ({utils.topk_ratio_meaning()}). The certification below is the same for every mode.
        τ = AGREEMENT_THRESHOLD_TAU = {C.AGREEMENT_THRESHOLD_TAU}, α = FAILURE_PROB_ALPHA = {C.FAILURE_PROB_ALPHA}.
        For every class, every {ratio_column} K and every channel u in U:
          ĉ_u = "in" if u is in > 50 % of the N0 = {C.N0} selection samples, else "out"
          n_u = number of the N = {C.N} certification samples that agree with ĉ_u
          certified ĉ_u  if  P(Binom(N, τ) ≥ n_u) ≤ α / |U| = {C.FAILURE_PROB_ALPHA} / {n_certifiable} = {threshold:.3g}  (Bonferroni)
                         ⟺  {rule}
          otherwise abstain (⊘): u is neither certified in nor certified out.
        So "certified in" = u is in nearly every sub-sample, "certified out" = u is out in nearly every one.
        The CERTIFIED CIRCUIT = the certified-in channels only; abstaining channels are left out of it.
        Certified radius r = floor(log(1.5 − τ) / log p_del) = floor(log({1.5 - C.AGREEMENT_THRESHOLD_TAU:g}) / log({C.P_DEL})) = {radius}:
        with confidence 1 − α = {1 - C.FAILURE_PROB_ALPHA}, every certified decision holds for every discovery set within
        edit distance r of D_c (Anani Thm. 3.1).
        """)

    shares = ", ".join(f"{name} {ch / n_certifiable:.0%}" for name, ch in sizes.items())
    if not is_global:
        ratio_text = f"""
        top-K layer-normalized ratio K
                 from s03: A keeps round(K · C) channels in every layer, so the baseline circuit A(D_c) and
                 every sample have exactly that size per layer."""
        layer_normalized_baseline_text = """ Baseline circuit: equal to K up to rounding.
                 Certified circuit: at most ≈ K, since each sample keeps K per layer and a certified-in channel
                 must be in almost all of them."""
        pooled_baseline_text = "Baseline circuit: ≈ K as well (Σ_l round(K · C_l) / |U|, the same for every class)."
    else:
        ratio_text = f"""
        {ratio_column} K
                 from s03 (TOPK_SELECTION_MODE = {C.TOPK_SELECTION_MODE!r}): A keeps round(K · |U|) channels of U IN
                 TOTAL (one global top-K over all |U| = {n_certifiable} channels, raw scores), so the baseline circuit
                 A(D_c) and every sample have a POOLED share of U of K; how many of them come from each layer
                 varies per class and sample (table [{n_sections}/{n_sections}])."""
        layer_normalized_baseline_text = """
                 Baseline circuit: NOT equal to K in this mode: the per-layer shares differ (e.g. a nearly full
                 layer1 and a nearly empty layer4 average to far more than K), and they differ per class.
                 Certified circuit: per layer at most ≈ the baseline circuit's share."""
        pooled_baseline_text = ("Baseline circuit: exactly round(K · |U|) / |U| ≈ K (fixed by the global top-K, the "
                                "same for every class).\n                 Certified circuit: at most ≈ K.")
    utils.section(
        f"[2/{n_sections}] Size measures used in this log (two DIFFERENT weightings, read the units)",
        ratio_text + f"""
        subnetwork size, layer-normalized ratio (subnet_size_layer_normalized_ratio_mean)
                 the ratio of a layer's channels that the circuit contains, computed per layer of U and then
                 AVERAGED OVER THE LAYERS WITH EQUAL WEIGHT ({len(layers)} layers; layer1 with {sizes.get("layer1", "-")} channels
                 counts as much as layer4 with {sizes.get("layer4", "-")}).{layer_normalized_baseline_text}
        channels / shares of U, pooled (subnet_size_group_channel_count, subnet_size_group_pooled_ratio)
                 counted over all |U| = {n_certifiable} channels POOLED, so big layers dominate
                 (share of U per layer: {shares}).
                 subnet_size_group_channel_count = the circuit's channels of U (absolute),
                 subnet_size_group_pooled_ratio = that count / |U| (relative).
                 {pooled_baseline_text}
                 This weighting differs from the subnetwork size above; table [{n_sections}/{n_sections}] and s06's
                 layer_breakdown figures show the channels per layer.
        """)

    results = {}
    for c in C.CLASSES:
        result = certify_class(c, certifiable_index, certifiable_slices)
        if not np.allclose(result["topk_layer_normalized_ratio_grid"], C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
            print(f"NOTE: s03 used TOPK_LAYER_NORMALIZED_RATIO_GRID = "
                  f"{result['topk_layer_normalized_ratio_grid'].tolist()}, which differs from config.py; "
                  f"that grid is used here.")
        np.savez_compressed(out_dir / f"class_{c:02d}.npz", **result, layers=np.array(layers),
                            certifiable_index=certifiable_index, radius=radius)
        results[c] = result
    topk_layer_normalized_ratio_grid = next(iter(results.values()))["topk_layer_normalized_ratio_grid"]
    ratio_header = "".join(f" {topk_layer_normalized_ratio:>7g}"
                           for topk_layer_normalized_ratio in topk_layer_normalized_ratio_grid)

    # ---- per class: certified subnetwork size (two weightings) and abstain share per K ------
    baseline_pooled = np.array([r["subnet_size_group_pooled_ratio_baseline_circuit"] for r in results.values()])
    baseline_pooled_label = ("baseline circuit" + (" (every class)" if np.allclose(baseline_pooled, baseline_pooled[0])
                                                   else " (class mean)"))
    if not is_global:
        layer_normalized_how_to_read = """
         Each cell: subnet_size_layer_normalized_ratio_mean of the certified-in channels (see [2/8]). Compare
         with the "top-K layer-normalized ratio" row = the baseline circuit's size. Lower than K = channels that
         were not selected stably abstain and are left out.
         0.000 = nothing certified in at that K (the certified circuit is empty and s05 zeroes all of U).
         """
    else:
        layer_normalized_how_to_read = f"""
         Each cell: subnet_size_layer_normalized_ratio_mean of the certified-in channels (see [2/{n_sections}]). No reference
         row in this mode: the baseline circuit's value is NOT K here and differs per class (its class mean is
         the column "size baseline circuit" of table [6/{n_sections}]). To compare the certified circuit with K, use
         the pooled share in table [4/{n_sections}].
         0.000 = nothing certified in at that K (the certified circuit is empty and s05 zeroes all of U).
         """
    for title, how_to_read, value, fmt, reference_row in [
        (f"[3/{n_sections}] Size of the CERTIFIED circuit per class: subnetwork size, layer-normalized ratio "
         "(equal-weight mean over the layers of U)",
         layer_normalized_how_to_read,
         lambda r: r["subnet_size_layer_normalized_ratio_mean_certified"], lambda v: f"{v:.3f}",
         None if is_global else (ratio_column, topk_layer_normalized_ratio_grid)),
        (f"[4/{n_sections}] Size of the CERTIFIED circuit per class: pooled share of U (subnet_size_group_pooled_ratio "
         f"= certified-in channels / |U| = {n_certifiable}, all layers pooled)",
         f"""
         Each cell: the certified-in channels of U / |U|; in channels: × {n_certifiable}. Compare with the row
         "{baseline_pooled_label}" = the baseline circuit's pooled share ({"exactly round(K · |U|) / |U| in this mode" if is_global else "Σ_l round(K · C_l) / |U| ≈ K in this mode"}).
         Lower = channels that were not selected stably abstain and are left out.
         0.000 = nothing certified in at that K.
         """,
         lambda r: r["subnet_size_group_pooled_ratio_certified"], lambda v: f"{v:.3f}",
         (baseline_pooled_label, baseline_pooled.mean(axis=0))),
        (f"[5/{n_sections}] Share of U that ABSTAINS per class (all |U| = {n_certifiable} certifiable channels pooled)",
         """
         Each cell: abstaining channels / |U|. 0 % = every channel got a decision (in or out).
         Abstaining channels are neither in the certified circuit nor certified out.
         """,
         lambda r: (r["decision"] == -1).mean(axis=1), lambda v: f"{v:.1%}", None),
    ]:
        utils.section(title, how_to_read)
        label_width = max(len(ratio_column), len(baseline_pooled_label))
        print(f"{'':3s} {'':{label_width}s} columns: {ratio_column}")
        print(f"{'':3s} {'class':{label_width}s}" + ratio_header)
        if reference_row is not None:
            print(f"{'':3s} {reference_row[0]:{label_width}s}" + "".join(f" {v:7.3f}" for v in reference_row[1]))
        for c, result in results.items():
            print(f"{c:3d} {VOC_CLASSES[c]:{label_width}s}" + "".join(f" {fmt(v):>7s}" for v in value(result)))

    # ---- mean over classes per K -------------------------------------------------------------
    def class_mean(fn):
        """Mean over classes of fn(result, grid_idx) per grid index; classes where fn is nan are skipped."""
        with warnings.catch_warnings():  # all-nan (e.g. nothing certified in for any class) prints as nan
            warnings.simplefilter("ignore", RuntimeWarning)
            return [np.nanmean([fn(r, grid_idx) for r in results.values()])
                    for grid_idx in range(len(topk_layer_normalized_ratio_grid))]

    def in_baseline_circuit(r, grid_idx, d):
        baseline_circuit = r["baseline_circuit"][grid_idx]
        return (baseline_circuit & (r["decision"][grid_idx] == d)).sum() / baseline_circuit.sum()

    def print_columns(columns, cell):
        """One row per K; every column as wide as its name, columns separated by |."""
        print(ratio_column + "".join(f" | {name:>{max(len(name), 7)}s}" for name in columns))
        for grid_idx, topk_layer_normalized_ratio in enumerate(topk_layer_normalized_ratio_grid):
            print(f"{topk_layer_normalized_ratio:{len(ratio_column)}g}"
                  + "".join(f" | {cell(values, grid_idx):>{max(len(name), 7)}s}" for name, values in columns.items()))

    size_columns = {
        "size baseline circuit": (class_mean(lambda r, grid_idx: r["subnet_size_layer_normalized_ratio_mean_baseline_circuit"][grid_idx]), "{:.3f}"),
        "size majority": (class_mean(lambda r, grid_idx: r["subnet_size_layer_normalized_ratio_mean_majority"][grid_idx]), "{:.3f}"),
        "size certified": (class_mean(lambda r, grid_idx: r["subnet_size_layer_normalized_ratio_mean_certified"][grid_idx]), "{:.3f}"),
        "pooled baseline circuit": (class_mean(lambda r, grid_idx: r["subnet_size_group_pooled_ratio_baseline_circuit"][grid_idx]), "{:.3f}"),
        "pooled majority": (class_mean(lambda r, grid_idx: r["subnet_size_group_pooled_ratio_majority"][grid_idx]), "{:.3f}"),
        "pooled certified": (class_mean(lambda r, grid_idx: r["subnet_size_group_pooled_ratio_certified"][grid_idx]), "{:.3f}"),
    }
    count_columns = {
        "#baseline circuit": (class_mean(lambda r, grid_idx: r["baseline_circuit"][grid_idx].sum()), "{:.0f}"),
        "#majority": (class_mean(lambda r, grid_idx: r["majority"][grid_idx].sum()), "{:.0f}"),
        "#certified in": (class_mean(lambda r, grid_idx: (r["decision"][grid_idx] == 1).sum()), "{:.0f}"),
        "#certified out": (class_mean(lambda r, grid_idx: (r["decision"][grid_idx] == 0).sum()), "{:.0f}"),
        "#abstain": (class_mean(lambda r, grid_idx: (r["decision"][grid_idx] == -1).sum()), "{:.0f}"),
    }
    utils.section(
        f"[6/{n_sections}] Subnetwork sizes, mean over the {len(results)} classes",
        f"""
        size baseline circuit / size majority / size certified
                 subnetwork size, layer-normalized ratio (subnet_size_layer_normalized_ratio_mean, the equal-weight
                 mean over layers) of the baseline circuit A(D_c), the majority-vote circuit (in > 50 % of the
                 N samples, no τ) and the certified circuit.
        pooled baseline circuit / pooled majority / pooled certified
                 the same circuits' pooled share of U (subnet_size_group_pooled_ratio = channels of U kept / |U|,
                 all layers pooled; see [2/{n_sections}]).
        #baseline circuit / #majority
                 channels of U in the baseline circuit / the majority-vote circuit (out of |U| = {n_certifiable};
                 subnet_size_group_channel_count).
        #certified in / #certified out / #abstain
                 channels of U per certification outcome; the three add up to |U| = {n_certifiable}.
        Two tables: the ratios first, then the channel counts.
        """)
    print_columns(size_columns, lambda column, grid_idx: column[1].format(column[0][grid_idx]))
    print()
    print_columns(count_columns, lambda column, grid_idx: column[1].format(column[0][grid_idx]))

    columns = {
        "baseline circuit→in": class_mean(lambda r, grid_idx: in_baseline_circuit(r, grid_idx, 1)),
        "baseline circuit→⊘": class_mean(lambda r, grid_idx: in_baseline_circuit(r, grid_idx, -1)),
        "baseline circuit→out": class_mean(lambda r, grid_idx: in_baseline_circuit(r, grid_idx, 0)),
        "in⊆baseline circuit": class_mean(
            lambda r, grid_idx: ((r["decision"][grid_idx] == 1) & r["baseline_circuit"][grid_idx]).sum()
            / (r["decision"][grid_idx] == 1).sum() if (r["decision"][grid_idx] == 1).any() else np.nan),
    }
    utils.section(
        f"[7/{n_sections}] Certified circuit vs. baseline circuit A(D_c), mean over the {len(results)} classes",
        f"""
        baseline circuit→in / baseline circuit→⊘ / baseline circuit→out
                 of the baseline circuit's channels: the share certified in / abstaining / certified out (the
                 three add up to 100 %). baseline circuit→out = the baseline circuit keeps a channel that almost
                 every sub-sample drops (expected to be rare).
        in⊆baseline circuit
                 of the certified-in channels: the share the baseline circuit also keeps.
                 100 % = the certified circuit is a subset of the baseline circuit.
                 nan = no class has any certified-in channel at that {ratio_column}.
        """)
    print_columns(columns, lambda values, grid_idx: "nan" if np.isnan(values[grid_idx]) else f"{values[grid_idx]:.1%}")

    # ---- per class and layer: which layers keep channels --------------------------------------
    print_layer_breakdown(n_sections, ratio_column, results, layers, sizes, topk_layer_normalized_ratio_grid)

    utils.write_run_info(out_dir)
    print(f"\nSaved {len(results)} class files to {out_dir}")


if __name__ == "__main__":
    main()
