"""s03 · Monte Carlo RS-Del sampling (Anani Alg. 2, with the score cache of App. D).

Base algorithm A (Hamblin cumulative saliency, Eq. 3, + a top-K selection chosen by TOPK_SELECTION_MODE):
    A(D') = top-K selection of  Σ_{i ∈ D'} S_i
            S_i = cached score of image i (TARGET × CHANNEL_SCORING_METHOD)
    K = a value of TOPK_LAYER_NORMALIZED_RATIO_GRID.
    "per_layer" (Anani's per-layer top-K selection): K = the top-K layer-normalized ratio. A layer with
        C channels keeps its topk_layer_count = round(K·C) highest-scoring channels (at least 1).
    "global_raw" (the default; Hamblin Eq. 3 as written: one top-κ, without the min-max normalization of App.
        8.5): K = the top-K global ratio, the share of the group LAYERS (the channels s04 certifies, e.g. the
        backbone's 3,840) that is kept. The group keeps its topk_group_count = round(K·|group|) highest RAW summed scores, ranked over all its
        channels at once (at least 1). Per layer the share varies, and a layer can lose every channel (not prevented,
        counted in the log). The layers outside the group (e.g. aspp, head) keep the per-layer selection; s05 never
        masks them, so they only keep the arrays complete.
    Exactly tied scores are ordered randomly per sample (how often this decides the circuit is
    logged and stored as tie_rate, in global_raw also as global_boundary_tie_rate).

For every class c:
  - draw N0 + N deletion masks ε ~ Bernoulli(1 − P_DEL)^|D| (ε_i = 1: image i is kept);
  - run A on every sub-dataset D ⊙ ε for every K in TOPK_LAYER_NORMALIZED_RATIO_GRID (a matrix op, no
    forward passes); an empty sub-dataset gives the all-zero circuit;
  - count inclusions: counts_N0 over the first N0 samples (selection), counts_N over the last N (certification);
  - baseline circuit A(D) on the full D_c.

Output: <sample_dir>/class_XX.npz and run_info.json, with (|grid| = number of grid values)
    eps                               [N0+N, |D|]               bool
    circuits                          [N0+N, |grid|, 4352/8]    packed bits (np.unpackbits(..., axis=-1) -> [N0+N, |grid|, 4352])
    counts_N0                         [|grid|, 4352]            int
    counts_N                          [|grid|, 4352]            int
    baseline_circuit                  [|grid|, 4352]            bool
    topk_layer_normalized_ratio_grid  [|grid|]
    tie_rate                          [|grid|, 6 layers]        share of non-empty samples whose circuit a random tie-break decided
                                                                (per layer; global_raw: nan in the group's columns)
    baseline_circuit_ties             [|grid|, 6 layers]        bool, the same for the baseline circuit (global_raw: float,
                                                                1.0 / 0.0, nan in the group's columns)
    topk_selection_mode               str                       TOPK_SELECTION_MODE
    topk_selection_group              str                       the group of the global top-K (LAYERS); "none" in per_layer
  global_raw only:
    global_boundary_tie_rate              [|grid|]              as tie_rate, for the group's single boundary
    baseline_circuit_global_boundary_tie  [|grid|]              bool, the same for the baseline circuit
"""

import numpy as np
import torch

import config as C
from common import model as M, utils
from common.datasets import VOC_CLASSES

USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "CLASSES",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "TOPK_LAYER_NORMALIZED_RATIO_GRID", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "SMOKE_TEST",
]


def load_scores(c):
    """Cached per-image scores [|D|, 4352] of TARGET × CHANNEL_SCORING_METHOD."""
    path = utils.score_dir() / f"class_{c:02d}.pt"
    utils.require(path, "s02_score_cache.py")
    cache = torch.load(path)
    if C.CHANNEL_SCORING_METHOD == "activation":
        return cache["act_mean"].double().numpy()
    if C.TARGET not in cache["scores"]:
        raise SystemExit(f"TARGET {C.TARGET!r} is not cached in {path} (cached: {list(cache['scores'])}).")
    return cache["scores"][C.TARGET].double().numpy()


def topk_layer_count(topk_layer_normalized_ratio, channels):
    """Channels A keeps in a layer: round(topk_layer_normalized_ratio · channels) (half rounds up), at least 1."""
    return max(1, int(np.floor(topk_layer_normalized_ratio * channels + 0.5)))


def topk_group_count(topk_global_ratio, group_channels):
    """Channels A keeps in the whole group (global_raw): round(topk_global_ratio · group_channels) (half rounds up), at
    least 1 — the same rounding as topk_layer_count."""
    return topk_layer_count(topk_global_ratio, group_channels)


def select_topk_per_layer(totals, tie_keys):
    """Per-layer top-K selection for every top-K layer-normalized ratio in TOPK_LAYER_NORMALIZED_RATIO_GRID.
    totals, tie_keys: [n, 4352] -> circuits [n, |grid|, 4352] bool, and
    boundary_ties [n, |grid|, n_layers] bool: the topk_layer_count-th and (topk_layer_count + 1)-th highest
    scores of the layer are equal, i.e. the random tie-break decided which channels are in."""
    circuits = np.zeros((totals.shape[0], len(C.TOPK_LAYER_NORMALIZED_RATIO_GRID), M.NUM_COMPONENTS), dtype=bool)
    boundary_ties = np.zeros((totals.shape[0], len(C.TOPK_LAYER_NORMALIZED_RATIO_GRID), len(M.LAYER_SLICES)),
                             dtype=bool)
    for l, s in enumerate(M.LAYER_SLICES.values()):
        channels = s.stop - s.start
        # rank 0 = highest summed score; exact ties are ordered by the random tie_keys
        order = np.lexsort((tie_keys[:, s], -totals[:, s]), axis=-1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(channels), order.shape), axis=-1)
        ordered = np.take_along_axis(totals[:, s], order, axis=-1)
        for grid_idx, topk_layer_normalized_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
            channels_kept = topk_layer_count(topk_layer_normalized_ratio, channels)
            circuits[:, grid_idx, s] = ranks < channels_kept
            if channels_kept < channels:
                boundary_ties[:, grid_idx, l] = ordered[:, channels_kept - 1] == ordered[:, channels_kept]
    return circuits, boundary_ties


def select_topk_global(totals, tie_keys, group_index):
    """Global top-K selection (global_raw) over the group's channels for every top-K global ratio in
    TOPK_LAYER_NORMALIZED_RATIO_GRID: the RAW summed scores of all group channels are ranked together.
    totals, tie_keys: [n, 4352]; group_index: positions of the group's channels in the 4352 vector ->
    group_circuits [n, |grid|, |group|] bool, and global_boundary_ties [n, |grid|] bool: the topk_group_count-th and
    (topk_group_count + 1)-th highest scores of the whole group are equal (the random tie-break decided)."""
    group_totals, group_tie_keys = totals[:, group_index], tie_keys[:, group_index]
    group_channels = len(group_index)
    group_circuits = np.zeros((totals.shape[0], len(C.TOPK_LAYER_NORMALIZED_RATIO_GRID), group_channels), dtype=bool)
    global_boundary_ties = np.zeros((totals.shape[0], len(C.TOPK_LAYER_NORMALIZED_RATIO_GRID)), dtype=bool)
    # rank 0 = highest summed score of the group; exact ties are ordered by the random tie_keys
    order = np.lexsort((group_tie_keys, -group_totals), axis=-1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(group_channels), order.shape), axis=-1)
    ordered = np.take_along_axis(group_totals, order, axis=-1)
    for grid_idx, topk_global_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
        channels_kept = topk_group_count(topk_global_ratio, group_channels)
        group_circuits[:, grid_idx] = ranks < channels_kept
        if channels_kept < group_channels:
            global_boundary_ties[:, grid_idx] = ordered[:, channels_kept - 1] == ordered[:, channels_kept]
    return group_circuits, global_boundary_ties


def select_topk(totals, tie_keys, group_index, group_layers):
    """A's top-K selection by TOPK_SELECTION_MODE, totals, tie_keys: [n, 4352] ->
        circuits                [n, |grid|, 4352] bool
        boundary_ties           [n, |grid|, 6 layers] per layer (per_layer: bool; global_raw: float with nan in the
                                group's columns, whose channels have one global boundary instead)
        global_boundary_ties    [n, |grid|] bool (global_raw), None (per_layer)
    global_raw: the group's channels (group_index, layers group_layers) by select_topk_global, every other layer by
    select_topk_per_layer."""
    circuits, boundary_ties = select_topk_per_layer(totals, tie_keys)
    if not utils.topk_selection_is_global():
        return circuits, boundary_ties, None
    circuits[:, :, group_index], global_boundary_ties = select_topk_global(totals, tie_keys, group_index)
    boundary_ties = boundary_ties.astype(np.float64)
    boundary_ties[:, :, [list(M.LAYER_SLICES).index(name) for name in group_layers]] = np.nan
    return circuits, boundary_ties, global_boundary_ties


def group_layer_statistics(circuits, baseline_circuit, empty, group_layers):
    """global_raw: which layers of the group lose channels. Per K and layer of the group (C_l = its channels):
        baseline_layer_kept_share       [|grid|, layers]  baseline circuit's kept channels / C_l
        sample_layer_kept_share_mean    [|grid|, layers]  the same share, mean over the non-empty samples (nan if none)
        sample_layer_kept_share_std     [|grid|, layers]  ... and its standard deviation
        sample_empty_layer_count        [|grid|, layers]  non-empty samples in which the layer keeps no channel
        samples_with_an_empty_layer     [|grid|]          non-empty samples with at least one such layer
        nonempty_sample_count           int               samples whose sub-dataset is not empty (the denominator)
    circuits [n, |grid|, 4352], baseline_circuit [|grid|, 4352], empty [n]: the empty sub-datasets, left out (their
    circuit is empty by definition, not because of the global top-K)."""
    nonempty = circuits[~empty]
    sample_share = np.stack([nonempty[:, :, M.LAYER_SLICES[name]].mean(axis=-1) for name in group_layers], axis=-1)
    has_samples = len(nonempty) > 0
    return {
        "baseline_layer_kept_share": np.stack([baseline_circuit[:, M.LAYER_SLICES[name]].mean(axis=-1)
                                               for name in group_layers], axis=-1),
        "sample_layer_kept_share_mean": sample_share.mean(axis=0) if has_samples else np.full(sample_share.shape[1:], np.nan),
        "sample_layer_kept_share_std": sample_share.std(axis=0) if has_samples else np.full(sample_share.shape[1:], np.nan),
        "sample_empty_layer_count": (sample_share == 0).sum(axis=0),
        "samples_with_an_empty_layer": (sample_share == 0).any(axis=-1).sum(axis=0),
        "nonempty_sample_count": len(nonempty),
    }


def sample_class(c, group_index, group_layers):
    scores = load_scores(c)                                   # [|D|, 4352]
    rng = utils.rng(C.SAMPLING_SEED, c)
    eps = rng.random((C.N0 + C.N, len(scores))) >= C.P_DEL    # True = kept (probability 1 − P_DEL)

    totals = eps.astype(np.float64) @ scores                  # Σ_{i ∈ D ⊙ ε} S_i, [N0+N, 4352]
    circuits, ties, global_ties = select_topk(totals, rng.random(totals.shape), group_index, group_layers)
    empty = eps.sum(axis=1) == 0
    circuits[empty] = False                                   # empty sub-dataset -> empty circuit

    baseline_circuit, baseline_circuit_ties, baseline_circuit_global_tie = select_topk(
        scores.sum(axis=0, keepdims=True), rng.random((1, M.NUM_COMPONENTS)), group_index, group_layers)
    if (~empty).any():
        tie_rate = ties[~empty].mean(axis=0)                  # [|grid|, layers]
    else:
        tie_rate = np.where(np.isnan(ties[0]), np.nan, 0.0)   # no non-empty sample: 0 (nan stays nan)
    result = {
        "eps": eps,
        "circuits": np.packbits(circuits, axis=-1),
        "counts_N0": circuits[:C.N0].sum(axis=0),
        "counts_N": circuits[C.N0:].sum(axis=0),
        "baseline_circuit": baseline_circuit[0],
        "topk_layer_normalized_ratio_grid": np.array(C.TOPK_LAYER_NORMALIZED_RATIO_GRID),
        # share of the non-empty samples in which the random tie-break decided the layer's circuit
        "tie_rate": tie_rate,                                                                   # [|grid|, layers]
        "baseline_circuit_ties": baseline_circuit_ties[0],                                      # [|grid|, layers]
        "topk_selection_mode": np.array(utils.topk_selection_mode()),
        "topk_selection_group": np.array(utils.topk_selection_group()),
    }
    statistics = None
    if global_ties is not None:
        result["global_boundary_tie_rate"] = (global_ties[~empty].mean(axis=0) if (~empty).any()
                                              else np.zeros(global_ties.shape[1:]))            # [|grid|]
        result["baseline_circuit_global_boundary_tie"] = baseline_circuit_global_tie[0]         # [|grid|]
        statistics = group_layer_statistics(circuits, baseline_circuit[0], empty, group_layers)
    return result, circuits, statistics


def shared_share(circuits, reference):
    """Mean over the circuits [n, 4352] of |circuit ∩ reference| / |reference|."""
    return np.mean((circuits & reference).sum(axis=1) / max(reference.sum(), 1))


def jaccard(circuits, reference):
    """Mean Jaccard index between each circuit [n, 4352] and a reference circuit [4352]."""
    return np.mean((circuits & reference).sum(axis=1) / np.maximum((circuits | reference).sum(axis=1), 1))


def chance_overlap(topk_layer_normalized_ratio, group_channels):
    """Expected overlap with the baseline circuit for a circuit of the same size whose channels are drawn at
    random: (share of the baseline circuit kept, ≈ Jaccard index).
    per_layer: random channels per layer, with each layer's topk_layer_count (all 4,352 channels).
    global_raw: random channels over the whole group (group_channels channels) with the same total topk_group_count,
    the null hypothesis of a global top-K: the share is topk_group_count / group_channels ≈ K, the Jaccard index
    s / (2 − s)."""
    if utils.topk_selection_is_global():
        expected_shared_share = topk_group_count(topk_layer_normalized_ratio, group_channels) / group_channels
        return expected_shared_share, expected_shared_share / (2 - expected_shared_share)
    channels = [s.stop - s.start for s in M.LAYER_SLICES.values()]
    size = sum(topk_layer_count(topk_layer_normalized_ratio, ch) for ch in channels)
    expected_shared = sum(topk_layer_count(topk_layer_normalized_ratio, ch) ** 2 / ch for ch in channels)
    return expected_shared / size, expected_shared / (2 * size - expected_shared)


def print_class_table(title, how_to_read, per_class, fmt, chance=None):
    """Classes × top-K layer-normalized ratio table with a mean row and an optional chance row."""
    utils.section(title, how_to_read)
    grid = C.TOPK_LAYER_NORMALIZED_RATIO_GRID
    print(f"{'':3s} {'':17s} columns: {utils.topk_ratio_label()}")
    print(f"{'':3s} {'class':17s}" + "".join(f" {topk_layer_normalized_ratio:>7g}" for topk_layer_normalized_ratio in grid))
    for c, values in per_class.items():
        print(f"{c:3d} {VOC_CLASSES[c]:17s}" + "".join(f" {fmt(v):>7s}" for v in values))
    print("    " + "-" * (17 + 8 * len(grid)))
    print(f"{'':3s} {'mean over classes':17s}" + "".join(f" {fmt(v):>7s}" for v in np.mean(list(per_class.values()), axis=0)))
    if chance is not None:
        print(f"{'':3s} {'chance level':17s}" + "".join(f" {fmt(v):>7s}" for v in chance))


def print_size_section(n_sections, ratio_column, channels, group_layers, group_channels, outside):
    """Section 1: the circuit size per K, by TOPK_SELECTION_MODE."""
    if not utils.topk_selection_is_global():
        utils.section(
            f"[1/{n_sections}] Circuit size per top-K layer-normalized ratio: FIXED by construction, identical for every class, "
            "every sample and the baseline circuit",
            f"""
            K = the top-K layer-normalized ratio (a value of TOPK_LAYER_NORMALIZED_RATIO_GRID). A keeps the
            topk_layer_count = round(K · C) highest-scoring channels in EVERY layer (C = the layer's channel count,
            at least 1 channel) and drops all others. So every sampled circuit A(D ⊙ ε) and the baseline circuit
            A(D) has exactly the size below; the samples only differ in WHICH channels they keep (tables 3 and 4).
            Example: K = 0.05 keeps 5 % of each layer and drops 95 %. K = 1 keeps the whole model.
            Each cell: kept channels (topk_layer_count) / channels in the layer. Circuits here span all
            {len(channels)} hook points ({M.NUM_COMPONENTS:,} channels); s04 certifies only the layers of LAYERS = {C.LAYERS!r}.
            """)
        print(ratio_column + "".join(f" {name:>10s}" for name in channels) + f" {'total':>11s} {'% of all':>9s}")
        for topk_layer_normalized_ratio in C.TOPK_LAYER_NORMALIZED_RATIO_GRID:
            channels_kept = [topk_layer_count(topk_layer_normalized_ratio, ch) for ch in channels.values()]
            print(f"{topk_layer_normalized_ratio:{len(ratio_column)}g}"
                  + "".join(f" {f'{kept}/{ch}':>10s}" for kept, ch in zip(channels_kept, channels.values()))
                  + f" {f'{sum(channels_kept)}/{M.NUM_COMPONENTS}':>11s} {sum(channels_kept) / M.NUM_COMPONENTS:9.1%}")
        return
    group_text = ", ".join(f"{name} {channels[name]:,}" for name in group_layers)
    outside_text = (f"The layers outside the group ({', '.join(outside)}) keep the per-layer top-K, round(K · C) each. "
                    f"They are not in U, so s05 never masks them; their columns only keep the arrays complete."
                    if outside else "The group holds every hook point.")
    utils.section(
        f"[1/{n_sections}] Circuit size per {ratio_column}: the TOTAL over the group LAYERS = {C.LAYERS!r} is FIXED "
        f"by construction (identical for every class, every sample and the baseline circuit); its split over the "
        f"layers is NOT",
        f"""
        TOPK_SELECTION_MODE = {C.TOPK_SELECTION_MODE!r}. K = the {ratio_column} (a value of
        TOPK_LAYER_NORMALIZED_RATIO_GRID) = the share of the group's channels that A keeps.
        A ranks the RAW summed scores of all {group_channels:,} channels of the group LAYERS = {C.LAYERS!r}
        ({group_text}) TOGETHER and keeps the topk_group_count = round(K · {group_channels:,}) highest (at least 1).
        So every sampled circuit A(D ⊙ ε) and the baseline circuit A(D) keep exactly that many channels of the
        group, but HOW MANY COME FROM EACH LAYER depends on the class and the sample (table 6). A layer can lose
        every channel: this is not prevented, it is counted in table 7. Raw scores are not on a common scale
        across layers, so the split also follows each layer's score scale.
        Example: K = 0.1 keeps {topk_group_count(0.1, group_channels):,} of the {group_channels:,} channels of the group. K = 1 keeps the whole model.
        {outside_text}
        Each cell: kept channels / channels. Circuits here span all {len(channels)} hook points ({M.NUM_COMPONENTS:,} channels).
        """)
    group_column = f"{C.LAYERS} (global)"
    width = max(len(group_column), 11)
    print(ratio_column + f" {group_column:>{width}s}" + "".join(f" {name:>10s}" for name in outside)
          + f" {'total':>11s} {'% of all':>9s}")
    for topk_global_ratio in C.TOPK_LAYER_NORMALIZED_RATIO_GRID:
        group_kept = topk_group_count(topk_global_ratio, group_channels)
        outside_kept = [topk_layer_count(topk_global_ratio, channels[name]) for name in outside]
        total = group_kept + sum(outside_kept)
        print(f"{topk_global_ratio:{len(ratio_column)}g}" + f" {f'{group_kept}/{group_channels}':>{width}s}"
              + "".join(f" {f'{kept}/{channels[name]}':>10s}" for kept, name in zip(outside_kept, outside))
              + f" {f'{total}/{M.NUM_COMPONENTS}':>11s} {total / M.NUM_COMPONENTS:9.1%}")


def print_tie_section(n_sections, ratio_column, tie_rates, baseline_circuit_ties, global_tie_rates,
                      baseline_circuit_global_ties, group_layers, group_channels, outside):
    """Section 5: how often a random tie-break decided a circuit, by TOPK_SELECTION_MODE."""
    tie_rates = np.mean(tie_rates, axis=0)
    if not utils.topk_selection_is_global():
        utils.section(
            f"[5/{n_sections}] Random tie-breaks: did exactly equal scores ever decide which channels are kept?",
            """
            Per top-K layer-normalized ratio and layer: % of the samples (mean over classes) in which the
            topk_layer_count-th and (topk_layer_count + 1)-th highest summed scores of the layer were exactly equal,
            so the random tie-break chose between them.
            In brackets: the number of classes whose baseline circuit A(D) had such a tie.
            0.0 % ( 0) everywhere = ties never decided any circuit.
            """)
        print(ratio_column + "".join(f" {name:>11s}" for name in M.LAYER_SLICES))
        baseline_circuit_ties = np.sum(baseline_circuit_ties, axis=0)
        for grid_idx, topk_layer_normalized_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
            print(f"{topk_layer_normalized_ratio:{len(ratio_column)}g}"
                  + "".join(f" {tie_rates[grid_idx, l]:6.1%} ({baseline_circuit_ties[grid_idx, l]:2d})"
                            for l in range(len(M.LAYER_SLICES))))
        return
    group_column = f"{C.LAYERS} (global)"
    width = max(len(group_column), 11)
    utils.section(
        f"[5/{n_sections}] Random tie-breaks: did exactly equal scores ever decide which channels are kept?",
        f"""
        {group_column}: % of the samples (mean over classes) in which the topk_group_count-th and
            (topk_group_count + 1)-th highest summed scores of the WHOLE group were exactly equal (one boundary for
            all {group_channels:,} channels of the group), so the random tie-break chose between them.
        {", ".join(outside) if outside else "(no layer outside the group)"}: the same per layer, for the layers outside the group (per-layer top-K, topk_layer_count).
        In brackets: the number of classes whose baseline circuit A(D) had such a tie.
        0.0 % ( 0) everywhere = ties never decided any circuit.
        """)
    global_tie_rates = np.mean(global_tie_rates, axis=0)
    baseline_circuit_global_ties = np.sum(baseline_circuit_global_ties, axis=0)
    outside_columns = [list(M.LAYER_SLICES).index(name) for name in outside]
    baseline_circuit_ties = np.nansum(baseline_circuit_ties, axis=0).astype(int)
    print(ratio_column + f" {group_column:>{width}s}" + "".join(f" {name:>11s}" for name in outside))
    for grid_idx, topk_global_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
        global_cell = f"{global_tie_rates[grid_idx]:6.1%} ({baseline_circuit_global_ties[grid_idx]:2d})"
        print(f"{topk_global_ratio:{len(ratio_column)}g}" + f" {global_cell:>{width}s}"
              + "".join(f" {tie_rates[grid_idx, l]:6.1%} ({baseline_circuit_ties[grid_idx, l]:2d})"
                        for l in outside_columns))


def print_layer_share_section(n_sections, ratio_column, layer_statistics, group_layers, group_channels, channels):
    """Section 6 (global_raw): per class, the share of each group layer's channels that A keeps."""
    def share(v):
        """A kept share: EMPTY for exactly 0, <0.1% for a share that rounds to 0.0 % but keeps a channel."""
        return "nan" if np.isnan(v) else "EMPTY" if v == 0 else "<0.1%" if v < 0.0005 else f"{v:.1%}"
    pct = lambda v: "nan" if np.isnan(v) else f"{v:.1%}"  # noqa: E731
    nonempty = {c: statistics["nonempty_sample_count"] for c, statistics in layer_statistics.items()}
    utils.section(
        f"[6/{n_sections}] WHICH LAYERS lose channels: share of each layer's channels that A keeps, per class "
        f"(global top-K over LAYERS = {C.LAYERS!r})",
        f"""
        One block per class. Rows: K = the {ratio_column} = the share of ALL {group_channels:,} channels of the group
        that A keeps. Per layer of the group ({", ".join(f"{name} {channels[name]:,}" for name in group_layers)} channels), three numbers:
          baseline   kept channels of the baseline circuit A(D_c) / the layer's channels
          samples    the same share for the sampled circuits A(D ⊙ ε): mean over the non-empty samples
          ± std      ... and its standard deviation over them (how much the split varies under RS-Del)
        Non-empty samples = the N0 + N = {C.N0 + C.N} samples minus the empty sub-datasets of table 2 (those circuits are
        empty by definition, not because of the global top-K), e.g. {min(nonempty.values())}–{max(nonempty.values())} per class here.
        A layer below K loses a larger share than the group as a whole, a layer above K a smaller one.
        EMPTY = no channel of the layer kept (baseline: in the baseline circuit; samples: in EVERY non-empty
        sample); table 7 counts these. <0.1% = at least one channel kept, but less than 0.05 % of the layer
        (e.g. 1 of layer4's 2,048 channels is 0.049 %).
        Check: the shares weighted by the layers' channel counts add up to the group's kept channels,
        Σ_l share_l · C_l = round(K · {group_channels:,}).
        The last block: the baseline share, mean over the {len(layer_statistics)} classes, with its range (min–max over the classes).
        """)
    cell_width = 24
    header_names = "".join(f" | {f'{name} ({channels[name]:,} channels)':^{cell_width}s}" for name in group_layers)
    header_columns = "".join(f" | {'baseline':>8s} {'samples':>7s} ± {'std':>5s}" for _ in group_layers)
    for c, statistics in layer_statistics.items():
        print(f"\nclass {c} {VOC_CLASSES[c]} ({statistics['nonempty_sample_count']} non-empty samples); rows: {ratio_column} K")
        print((f"{'K':>6s}" + header_names).rstrip())
        print(f"{'':6s}" + header_columns)
        for grid_idx, topk_global_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
            print(f"{topk_global_ratio:6g}" + "".join(
                f" | {share(statistics['baseline_layer_kept_share'][grid_idx, l]):>8s} "
                f"{share(statistics['sample_layer_kept_share_mean'][grid_idx, l]):>7s} ± "
                f"{pct(statistics['sample_layer_kept_share_std'][grid_idx, l]):>5s}"
                for l in range(len(group_layers))))
    baseline_share = np.stack([statistics["baseline_layer_kept_share"] for statistics in layer_statistics.values()])
    print(f"\nmean over the {len(layer_statistics)} classes of the baseline share (min–max over the classes); rows: {ratio_column} K")
    print((f"{'K':>6s}" + header_names).rstrip())
    for grid_idx, topk_global_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
        print((f"{topk_global_ratio:6g}" + "".join(
            f" | {share(baseline_share[:, grid_idx, l].mean()):>7s} "
            f"({share(baseline_share[:, grid_idx, l].min())}–{share(baseline_share[:, grid_idx, l].max())})".ljust(cell_width + 3)
            for l in range(len(group_layers)))).rstrip())


def print_empty_layer_section(n_sections, ratio_column, layer_statistics, group_layers):
    """Section 7 (global_raw): circuits in which a layer of the group keeps no channel."""
    total_nonempty = sum(statistics["nonempty_sample_count"] for statistics in layer_statistics.values())
    utils.section(
        f"[7/{n_sections}] EMPTY layers (no channel kept): the global top-K does not prevent them, so they are "
        f"counted here",
        f"""
        Per K (the {ratio_column}):
          baseline circuits   classes whose baseline circuit A(D_c) has at least one empty layer of the group, out
                              of the {len(layer_statistics)} classes (which class and layer: listed below the table).
          samples             non-empty samples A(D ⊙ ε) with at least one empty layer, out of all
                              {total_nonempty:,} non-empty samples of all classes.
          classes             classes with at least one such sample.
          {f"{group_layers[0]} … {group_layers[-1]}":20s}samples (all classes) in which that layer is empty; one sample can count in
                              several layers.
        An empty layer cuts the network: no skip connection goes around a stage, so the circuit's output no longer
        depends on the image. 0 everywhere = no circuit lost a whole layer.
        """)
    print(f"{'K':>6s} | {'baseline circuits':>17s} | {'samples':>17s} | {'classes':>7s} | "
          + " | ".join(f"{name:>7s}" for name in group_layers))
    listed = []
    for grid_idx, topk_global_ratio in enumerate(C.TOPK_LAYER_NORMALIZED_RATIO_GRID):
        baseline_empty = {c: [name for l, name in enumerate(group_layers)
                              if statistics["baseline_layer_kept_share"][grid_idx, l] == 0]
                          for c, statistics in layer_statistics.items()}
        baseline_empty = {c: names for c, names in baseline_empty.items() if names}
        samples = sum(int(statistics["samples_with_an_empty_layer"][grid_idx]) for statistics in layer_statistics.values())
        classes = sum(int(statistics["samples_with_an_empty_layer"][grid_idx] > 0) for statistics in layer_statistics.values())
        per_layer = [sum(int(statistics["sample_empty_layer_count"][grid_idx, l]) for statistics in layer_statistics.values())
                     for l in range(len(group_layers))]
        print(f"{topk_global_ratio:6g} | {f'{len(baseline_empty)}/{len(layer_statistics)}':>17s} | "
              f"{f'{samples}/{total_nonempty}':>17s} | {classes:7d} | " + " | ".join(f"{n:7d}" for n in per_layer))
        listed += [f"  K = {topk_global_ratio:g}: class {c} {VOC_CLASSES[c]} ({', '.join(names)})"
                   for c, names in baseline_empty.items()]
    print("Baseline circuits with an empty layer: " + ("none" if not listed else "\n" + "\n".join(listed)))


def main():
    utils.start_run("s03_sample_circuits", USED_PARAMS)
    utils.require(utils.score_dir() / "run_info.json", "s02_score_cache.py")
    out_dir = utils.sample_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = {name: s.stop - s.start for name, s in M.LAYER_SLICES.items()}
    n_samples = C.N0 + C.N
    grid_idxs = range(len(C.TOPK_LAYER_NORMALIZED_RATIO_GRID))
    ratio_column = utils.topk_ratio_label()
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    is_global = utils.topk_selection_is_global()
    group_layers, group_index, _ = M.certifiable_components(C.LAYERS)   # global_raw: the group of the global top-K
    group_channels = len(group_index)
    outside = [name for name in M.LAYER_SLICES if name not in group_layers]
    n_sections = 7 if is_global else 5

    # ---- 1. circuit size --------------------------------------------------------------------
    print_size_section(n_sections, ratio_column, channels, group_layers, group_channels, outside)

    # ---- sample every class ----------------------------------------------------------------
    sizes, shared, overlap, tie_rates, baseline_circuit_ties = {}, {}, {}, [], []
    global_tie_rates, baseline_circuit_global_ties, layer_statistics = [], [], {}
    overlap_index = group_index if is_global else slice(None)  # global_raw: tables 3 and 4 over the group's channels
    for c in C.CLASSES:
        result, circuits, statistics = sample_class(c, group_index, group_layers)
        np.savez_compressed(out_dir / f"class_{c:02d}.npz", **result)
        tie_rates.append(result["tie_rate"])
        baseline_circuit_ties.append(result["baseline_circuit_ties"])
        if is_global:
            global_tie_rates.append(result["global_boundary_tie_rate"])
            baseline_circuit_global_ties.append(result["baseline_circuit_global_boundary_tie"])
            layer_statistics[c] = statistics
        sizes[c] = result["eps"].shape[1], result["eps"].sum(axis=1)
        shared[c] = [shared_share(circuits[:, grid_idx][:, overlap_index], result["baseline_circuit"][grid_idx][overlap_index])
                     for grid_idx in grid_idxs]
        overlap[c] = [jaccard(circuits[:, grid_idx][:, overlap_index], result["baseline_circuit"][grid_idx][overlap_index])
                      for grid_idx in grid_idxs]
        print(f"  sampled class {c:2d} {VOC_CLASSES[c]}", flush=True)

    # ---- 2. sub-dataset sizes --------------------------------------------------------------
    utils.section(
        f"[2/{n_sections}] Sub-dataset sizes |D ⊙ ε|: how many of the |D_c| discovery images each of the {n_samples} samples keeps",
        f"""
        Every image of D_c is kept independently with probability 1 − P_DEL = {1 - C.P_DEL:.2f}, so the expected
        mean is |D_c| · {1 - C.P_DEL:.2f} (column "expected"). mean / min / max are over the N0 + N = {n_samples} samples.
        "empty" = samples that kept no image at all; their circuit is empty (no channel kept), which
        lowers tables 3 and 4.{" Tables 6 and 7 leave them out." if is_global else ""}
        """)
    print(f"{'':3s} {'class':16s} {'|D_c|':>6s} {'expected':>9s} {'mean':>6s} {'min':>5s} {'max':>5s} {'empty':>6s}")
    for c, (d, s) in sizes.items():
        print(f"{c:3d} {VOC_CLASSES[c]:16s} {d:6d} {d * (1 - C.P_DEL):9.1f} {s.mean():6.1f} {s.min():5d} "
              f"{s.max():5d} {(s == 0).sum():6d}")

    # ---- 3./4. stability of which channels are kept ----------------------------------------
    chance = [chance_overlap(topk_layer_normalized_ratio, group_channels)
              for topk_layer_normalized_ratio in C.TOPK_LAYER_NORMALIZED_RATIO_GRID]
    if not is_global:
        shared_how_to_read = f"""
        Per sample: |A(D ⊙ ε) ∩ A(D)| / |A(D)|, all {M.NUM_COMPONENTS:,} channels pooled; mean over the {n_samples} samples.
        100 % = the sample keeps exactly the baseline circuit's channels; 94 % = 6 % of the sample's channels are
        swapped for channels outside A(D). This measures STABILITY, NOT SIZE: the size is fixed by the top-K
        layer-normalized ratio K (table 1).
        "chance level" = the expected value for a circuit of the same size with RANDOM channels per layer
        (≈ K). A column only says something when the classes are well above it; at K = 1 every circuit is
        the whole model, so 100 % there is trivial.
        """
        jaccard_how_to_read = """
        The same information as table 3 on another scale, since all circuits of one top-K layer-normalized ratio
        have the same size: Jaccard J and shared share s convert as J = s / (2 − s), e.g. J = 0.89 ↔ s = 94 %,
        J = 0.83 ↔ s = 91 %. 1.00 = every sample equals A(D). The chance row is approximate (expected overlap
        of random channels).
        """
    else:
        outside_note = (f"; {', '.join(outside)} keep the per-layer top-K, are never masked in s05 and are left out "
                        f"here" if outside else "")
        shared_how_to_read = f"""
        Per sample: |A(D ⊙ ε) ∩ A(D)| / |A(D)|, counted over the {group_channels:,} channels of the group LAYERS = {C.LAYERS!r}
        ONLY (the channels the global top-K ranks{outside_note}); mean over the {n_samples} samples.
        100 % = the sample keeps exactly the baseline circuit's channels of the group; 94 % = 6 % of them are
        swapped for other channels of the group. This measures STABILITY, NOT SIZE: the group's total size is
        fixed by K (table 1); which layers the kept channels come from is in table 6.
        "chance level" = the expected value for a circuit that keeps the same number of channels, drawn at
        RANDOM over the whole group: round(K · {group_channels:,}) / {group_channels:,} ≈ K. A column only says something when the
        classes are well above it; at K = 1 every circuit is the whole model, so 100 % there is trivial.
        """
        jaccard_how_to_read = f"""
        The same information as table 3 on another scale, since all circuits of one K keep the same number of
        channels of the group: Jaccard J and shared share s convert as J = s / (2 − s), e.g. J = 0.89 ↔ s = 94 %,
        J = 0.83 ↔ s = 91 %. 1.00 = every sample equals A(D). Counted over the {group_channels:,} channels of the group
        only, as table 3. The chance row is approximate (random channels over the whole group, J = s / (2 − s)).
        """
    print_class_table(
        f"[3/{n_sections}] WHICH channels are kept: share of the baseline circuit A(D)'s channels that a sample also keeps",
        shared_how_to_read, shared, pct, chance=[s for s, _ in chance])
    print_class_table(
        f"[4/{n_sections}] The same comparison as a Jaccard index |A(D ⊙ ε) ∩ A(D)| / |A(D ⊙ ε) ∪ A(D)| (mean over samples)",
        jaccard_how_to_read, overlap, lambda v: f"{v:.2f}", chance=[j for _, j in chance])

    # ---- 5. ties ---------------------------------------------------------------------------
    print_tie_section(n_sections, ratio_column, tie_rates, baseline_circuit_ties, global_tie_rates,
                      baseline_circuit_global_ties, group_layers, group_channels, outside)

    # ---- 6./7. global_raw: which layers lose channels, empty layers --------------------------
    if is_global:
        print_layer_share_section(n_sections, ratio_column, layer_statistics, group_layers, group_channels, channels)
        print_empty_layer_section(n_sections, ratio_column, layer_statistics, group_layers)

    utils.write_run_info(out_dir, topk_selection_mode=utils.topk_selection_mode(),
                         topk_selection_group=utils.topk_selection_group())
    print(f"\nSaved {len(C.CLASSES)} class files to {out_dir}")


if __name__ == "__main__":
    main()
