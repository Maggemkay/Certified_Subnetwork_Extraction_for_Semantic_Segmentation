"""Pairwise overlap of the per-class circuits (not part of the codebase pipeline).

As in Gullstrand et al. (2026), Fig. 6: for every pair of classes (a, b), the overlap of their circuits over the
channels of U (backbone, |U| = 3,840), measured as
    IoU   = |a ∩ b| / |a ∪ b|
    IoMin = |a ∩ b| / min(|a|, |b|)   (overlap coefficient; 1 = the smaller circuit lies fully inside the larger one)
Both are symmetric, so one matrix holds both: upper-left triangle = IoU, lower-right triangle = IoMin, diagonal
left empty. Classes are ordered by semantic group (CLASS_GROUPS), starting in the bottom-left corner.

One figure per circuit type (3 columns): the key first, then one panel per top-k level, largest to smallest:
  subnetwork_overlap/overlap_matrices_baseline_circuit.png
  subnetwork_overlap/overlap_matrices_majority.png
  subnetwork_overlap/overlap_matrices_certified.png
Circuits come from s04's class_XX.npz: baseline_circuit, majority, certified = decision == 1 (abstain counts as out,
as in src/common/utils.py). Empty circuits give an undefined overlap (0/0), drawn as hatched cells.

Run:  python src/Extra_Material/Extra_images/make_overlap_matrices.py
"""

import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CERTIFICATION_DIR = ROOT / ("results/main/voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/s02_scores/"
                            "s03_pred_actgrad_pdel0.60_n100-1000_sseed0/s04_backbone_tau0.95_a0.001")
OUT = Path(__file__).resolve().parent / "subnetwork_overlap"
SUBTITLE = ("run Dbottleneck262 · 20 classes · U = backbone (3,840 channels) · τ = 0.95 · p_del = 0.6 · N = 1000 · "
            "target pred · actgrad")
CIRCUIT_NAMES = {"baseline_circuit": "baseline circuit", "majority": "majority vote", "certified": "certified circuit"}
COLUMNS = 3

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
       "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# Display order, by semantic group (first = bottom-left). Rows and columns use the same order, so the two triangles
# stay intact; thin lines in each panel mark the group boundaries.
CLASS_GROUPS = {
    "vehicles": ["aeroplane", "boat", "train", "bus", "car", "motorbike", "bicycle"],
    "animals": ["bird", "cat", "dog", "horse", "cow", "sheep"],
    "person": ["person"],
    "indoor": ["sofa", "chair", "diningtable", "bottle", "pottedplant", "tvmonitor"],
}
SURFACE, INK, INK_2, MUTED, AXIS, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#f0efec"
# Sequential, one hue (blue), light -> dark; the lightest step is near the surface so 0 recedes.
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab",
                                                        "#0d366b"])
SEQUENTIAL.set_bad(EMPTY)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
})


def load_circuits():
    """{circuit type: bool [|grid|, |classes|, |U|]}, the class list, the grid and the mean size per type and level."""
    order = [VOC.index(c) for group in CLASS_GROUPS.values() for c in group]
    files = [CERTIFICATION_DIR / f"class_{c:02d}.npz" for c in order if (CERTIFICATION_DIR / f"class_{c:02d}.npz").exists()]
    classes = [int(f.stem.split("_")[1]) for f in files]
    masks = {name: [] for name in CIRCUIT_NAMES}
    sizes = {name: [] for name in CIRCUIT_NAMES}
    for f in files:
        z = np.load(f)
        grid = z["topk_layer_normalized_ratio_grid"]
        masks["baseline_circuit"].append(z["baseline_circuit"])
        masks["majority"].append(z["majority"])
        masks["certified"].append(z["decision"] == 1)
        for name in CIRCUIT_NAMES:
            sizes[name].append(z[f"subnet_size_layer_normalized_ratio_mean_{name}"])
    masks = {name: np.stack(m, axis=1) for name, m in masks.items()}
    sizes = {name: np.mean(s, axis=0) for name, s in sizes.items()}  # mean over classes, per level
    return masks, classes, grid, sizes


def overlap(mask):
    """mask: bool [|classes|, |U|] -> (IoU, IoMin), each [|classes|, |classes|]; NaN where the denominator is 0."""
    m = mask.astype(np.float64)
    intersection = m @ m.T
    size = m.sum(axis=1)
    union = size[:, None] + size[None, :] - intersection
    smaller = np.minimum(size[:, None], size[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        return intersection / union, intersection / smaller


def draw_panel(ax, iou, iomin, names, show_y):
    """Row i drawn at height i (origin lower): cells with i > j sit above the diagonal = upper-left triangle."""
    n = len(names)
    upper_left = np.tril(np.ones((n, n), bool), k=-1)
    combined = np.where(upper_left, iou, iomin)
    np.fill_diagonal(combined, np.nan)
    image = ax.imshow(np.ma.masked_invalid(combined), cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest",
                      origin="lower")
    for i in range(n):
        for j in range(n):
            v = combined[i, j]
            if i == j:
                continue
            if np.isnan(v):  # undefined (empty circuit): hatched cell, no number
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor=MUTED,
                                           linewidth=0))
                continue
            ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center", fontsize=3.6,
                    color="white" if v > 0.55 else INK_2)
    boundary = -0.5
    for group in list(CLASS_GROUPS.values())[:-1]:  # group boundaries, same on both axes
        boundary += sum(c in names for c in group)
        ax.axhline(boundary, color="white", linewidth=0.9)
        ax.axvline(boundary, color="white", linewidth=0.9)
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color=INK_2, linewidth=0.6)  # IoU / IoMin divider
    ax.set_xticks(range(n), names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), names if show_y else [])
    ax.tick_params(axis="both", length=2.5, width=0.5, color=INK_2)
    return image


def mean_off_diagonal(matrix):
    n = len(matrix)
    return np.nanmean(matrix[~np.eye(n, dtype=bool)])


def draw_key(ax, image):
    """First slot: colour bar + which triangle holds which metric."""
    ax.axis("off")
    bar = ax.inset_axes([0.08, 0.80, 0.84, 0.045])
    plt.colorbar(image, cax=bar, orientation="horizontal").outline.set_visible(False)
    bar.tick_params(labelsize=7.5, length=0)
    bar.set_title("overlap (same scale in every panel)", fontsize=8.5, color=INK_2, pad=4)
    ax.text(0.08, 0.66,
            "◤ upper-left triangle   IoU   = |a ∩ b| / |a ∪ b|\n"
            "◢ lower-right triangle  IoMin = |a ∩ b| / min(|a|, |b|)\n\n"
            "classes grouped (bottom-left first):\n"
            "  vehicles · animals · person · indoor\n"
            "panels: top-k ratio, largest to smallest\n\n"
            "size = channels kept per layer of U,\n           mean over layers and classes\n"
            "mean = over all class pairs\n"
            "hatched = undefined (empty circuit)",
            transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top", family="monospace")


def figure(name, masks, classes, grid, sizes):
    names = [VOC[c] for c in classes]
    levels = list(range(len(grid)))[::-1]  # largest top-k first
    slots = len(levels) + 1  # + the key
    rows = math.ceil(slots / COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4.4 * COLUMNS, 4.9 * rows), squeeze=False)
    flat = list(axes.flat)
    image = None
    for slot, ax in enumerate(flat[1:], start=1):
        if slot > len(levels):
            ax.set_visible(False)
            continue
        level = levels[slot - 1]
        iou, iomin = overlap(masks[name][level])
        image = draw_panel(ax, iou, iomin, names, show_y=True)
        ax.set_title(f"top-k ratio = {grid[level]:.2f}  ·  size = {sizes[name][level]:.3f}\n"
                     f"mean IoU = {mean_off_diagonal(iou):.2f}  ·  mean IoMin = {mean_off_diagonal(iomin):.2f}",
                     fontsize=8.5, color=INK, pad=17)
        # Plain text, not loc="left"/"right" titles: matplotlib puts all three titles at the same height.
        ax.text(0, 1.008, "◤ IoU", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK_2)
        ax.text(1, 1.008, "IoMin ◢", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)
    draw_key(flat[0], image)
    fig.suptitle(f"Pairwise overlap of the per-class circuits: {CIRCUIT_NAMES[name]}\n{SUBTITLE}", y=1.0,
                 va="bottom", fontsize=10.5, color=INK)
    fig.tight_layout()
    path = OUT / f"overlap_matrices_{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    masks, classes, grid, sizes = load_circuits()
    for name in CIRCUIT_NAMES:
        figure(name, masks, classes, grid, sizes)
