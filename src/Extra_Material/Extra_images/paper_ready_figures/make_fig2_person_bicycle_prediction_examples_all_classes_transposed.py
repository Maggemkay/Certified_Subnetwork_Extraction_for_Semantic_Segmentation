"""Paper figure 1: segmentation masks of the CERTIFIED circuits of class 15 (person) and class 2 (bicycle) on the s07
voc_val image 2007_000129, at K = 0.6, 0.5 and 0.4.

Layout:
  columns    first column = image, ground truth, full model (top to bottom; independent of the rows);
             then one column per demonstration, labelled at the top:
               1  per-layer top-K, raw,         person subnetwork  (TOPK_SELECTION_MODE "per_layer")
               2  global top-K, raw,            person subnetwork  (TOPK_SELECTION_MODE "global_raw")
               3  global top-K, standardized,   person subnetwork  (logits after
                                                                    matched_to_full_model_on_full_model_prediction_pixels)
               4  global top-K, standardized,   bicycle subnetwork
  rows       K = 0.6, 0.5, 0.4 (label on the left of the demonstration columns)
  bottom     one-row class legend (background, person, bicycle, striped "other classes")
Every panel is cropped vertically to CROP_TOP_FRACTION / CROP_BOTTOM_FRACTION of the image height and then squeezed
vertically by VERTICAL_SQUEEZE_FACTOR (display only: the model always sees the whole, unsqueezed image). No captions; sizes are printed to the terminal.

  fig2_person_bicycle_prediction_examples_all_classes_transposed.png

Run:
  python src/Extra_Material/Extra_images/paper_ready_figures/make_fig2_person_bicycle_prediction_examples_all_classes_transposed.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import colorsys  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
from common import datasets, logit_standardization, model as M, utils  # noqa: E402
from common.datasets import VOC_CLASSES, VOID_LABEL  # noqa: E402

OUT = Path(__file__).resolve().parent
OUTPUT_NAME = "fig2_person_bicycle_prediction_examples_all_classes_transposed"
IMAGE_ID = "2007_000129"
DATASET = "voc_val"
K_VALUES = [0.6, 0.5, 0.4]
CIRCUIT = "certified"
STANDARDIZED = "matched_to_full_model_on_full_model_prediction_pixels"
PERSON, BICYCLE = 15, 2
# (column label at the top, TOPK_SELECTION_MODE, logit variant, subnetwork class)
COLUMNS = [("Per-layer\ntop-K, raw\n$\\bf{person}$ subnetwork", "per_layer", "raw", PERSON),
           ("Global\ntop-K, raw\n$\\bf{person}$ subnetwork", "global_raw", "raw", PERSON),
           ("Global top-K,\nstandardized\n$\\bf{person}$ subnetwork", "global_raw", STANDARDIZED, PERSON),
           ("Global top-K,\nstandardized\n$\\bf{bicycle}$ subnetwork", "global_raw", STANDARDIZED, BICYCLE)]
# Vertical crop of every panel (fractions of the image height removed at the top and at the bottom).
CROP_TOP_FRACTION = 0.07
CROP_BOTTOM_FRACTION = 0.15
# Vertical squeeze of every (cropped) panel: displayed height = factor x cropped height (1.0 = no squeeze).
# 0.77 shows the 390 cropped rows in the height of 300.
VERTICAL_SQUEEZE_FACTOR = 0.77
INK = "#0b0b0b"
KEPT_CLASSES = [0, 15, 2]                            # background, person, bicycle: named in the legend
OTHER_LABEL = "other classes"                        # striped legend entry for the remaining 18 classes


class StripedPatch:
    """Legend stand-in: a box filled with vertical stripes of the given RGB colours (0..1)."""

    def __init__(self, colours):
        self.colours = colours


class StripedPatchHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = len(orig_handle.colours)
        return [Rectangle((-xdescent + i * width / n, -ydescent), width / n, height, facecolor=c,
                          edgecolor="none", transform=trans)
                for i, c in enumerate(orig_handle.colours)]


def run_predictions(model, x, cls, mode, variants, certifiable_slices):
    """{(variant, K): (prediction, pooled size, empty layers)} for the certified circuit of class `cls` in the given
    top-K mode."""
    C.TOPK_SELECTION_MODE = mode
    circuits, _, grid = utils.load_circuits(cls, certifiable_slices, [CIRCUIT])
    full_statistics, circuit_statistics = logit_standardization.load_class_logit_statistics(cls, circuits)
    n_channels = sum(s.stop - s.start for s in certifiable_slices.values())
    out = {}
    for k in K_VALUES:
        grid_idx = int(np.flatnonzero(np.isclose(grid, k))[0])
        masks, _ = circuits[CIRCUIT][grid_idx]
        size = sum(int(m.sum()) for m in masks.values()) / n_channels
        empty = [name for name, m in masks.items() if not m.any()]
        logits = M.forward(model, x, masks)
        for v in variants:
            adjusted = logits if v == "raw" else logit_standardization.adjust_logits(
                logits, v, circuit_statistics[CIRCUIT][grid_idx], full_statistics)
            out[(v, k)] = (adjusted.argmax(dim=0).cpu().numpy(), size, empty)
        print(f"class {cls:2d} {VOC_CLASSES[cls]:8s} {mode:10s} K={k:g}: size {size:.3f}, "
              f"empty layers {empty or 'none'}")
    return out


@torch.no_grad()
def main():
    C.DEVICE = "cpu"
    eval_lists = utils.load_discovery_and_eval_lists()["eval"][DATASET]
    image = mask = None
    for cls in dict.fromkeys(c for *_, c in COLUMNS):
        lists = eval_lists[str(cls)]
        position = lists["ids"].index(IMAGE_ID)
        print(f"{DATASET} {IMAGE_ID}: class {cls} eval-list position {position}")
        if image is None:
            image, mask = utils.eval_dataset(DATASET)[lists["indices"][position]]
    print(f"{DATASET} {IMAGE_ID}: {image.size[0]}x{image.size[1]} px")
    model = M.load_model(C.MODEL_NAME, "cpu")
    _, _, certifiable_slices = M.certifiable_components(C.LAYERS)
    x = M.preprocess(image, "cpu")
    full = M.forward(model, x).argmax(dim=0).cpu().numpy()

    results = {}
    for cls, mode in dict.fromkeys((c, m) for _, m, _, c in COLUMNS):
        variants = list(dict.fromkeys(v for _, m, v, c in COLUMNS if (c, m) == (cls, mode)))
        results[(cls, mode)] = run_predictions(model, x, cls, mode, variants, certifiable_slices)

    # Full VOC palette: every class in its own colour; void (GT only) keeps its VOC colour.
    voc = datasets.voc_palette()
    palette = voc
    for name, prediction in [("ground truth", mask), ("full model", full)] + [
            (f"{VOC_CLASSES[c]} {m} {v} K={k:g}", results[(c, m)][(v, k)][0]) for _, m, v, c in COLUMNS for k in K_VALUES]:
        present = [VOC_CLASSES[i] for i in np.unique(np.asarray(prediction)) if i < len(VOC_CLASSES)]
        print(f"classes present in {name}: {present}")

    width, height = image.size
    crop_top = int(round(CROP_TOP_FRACTION * height))
    crop_bottom = height - int(round(CROP_BOTTOM_FRACTION * height))
    print(f"vertical crop: rows {crop_top}..{crop_bottom} of {height} (kept {crop_bottom - crop_top} px), "
          f"vertical squeeze factor {VERTICAL_SQUEEZE_FACTOR:g}")

    def crop(picture):
        return np.asarray(picture)[crop_top:crop_bottom]

    n_rows, n_columns = len(K_VALUES), 1 + len(COLUMNS)
    panel_w = 1.2
    panel_h = panel_w * VERTICAL_SQUEEZE_FACTOR * (crop_bottom - crop_top) / width
    label_w, legend_h, header_h, gap_w, gap_h = 0.2, 0.21, 0.37, 0.015, 0.015
    fig_w = label_w + n_columns * panel_w + (n_columns - 2) * gap_w
    grid_h = n_rows * panel_h + (n_rows - 1) * gap_h
    fig_h = legend_h + header_h + grid_h
    print(f"figure size {fig_w:.2f} x {fig_h:.2f} in (panel {panel_w:.2f} x {panel_h:.2f} in)")
    fig = plt.figure(figsize=(fig_w, fig_h))

    def column_left(j):  # K labels sit between the first column and the demonstration columns
        return j * panel_w + (label_w + (j - 1) * gap_w if j else 0)

    axes = np.empty((n_rows, n_columns), dtype=object)
    for r in range(n_rows):
        for j in range(n_columns):
            bottom = legend_h + (n_rows - 1 - r) * (panel_h + gap_h)
            axes[r, j] = ax = fig.add_axes([column_left(j) / fig_w, bottom / fig_h, panel_w / fig_w, panel_h / fig_h])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    def first_column(ax, picture, text):
        ax.imshow(crop(picture), aspect="auto", interpolation="nearest" if picture is not image else "antialiased")
        ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    first_column(axes[0, 0], image, "Image")
    first_column(axes[1, 0], palette[mask], "Ground truth")
    first_column(axes[2, 0], palette[full], "Full model")

    for j, (label, mode, variant, cls) in enumerate(COLUMNS, start=1):
        for r, k in enumerate(K_VALUES):
            pred, _, _ = results[(cls, mode)][(variant, k)]
            axes[r, j].imshow(crop(palette[pred]), aspect="auto", interpolation="nearest")
        fig.text((column_left(j) + 0.5 * panel_w) / fig_w, (legend_h + grid_h + 0.04) / fig_h, label,
                 ha="center", va="bottom", fontsize=6.6, color=INK, linespacing=1.1)
    for r, k in enumerate(K_VALUES):
        y_mid = axes[r, 0].get_position().y0 + 0.5 * panel_h / fig_h
        fig.text((panel_w + 0.5 * label_w) / fig_w, y_mid, f"K = {k:g}", rotation=90,
                 ha="center", va="center", fontsize=8, color=INK)

    # One-row class legend below the panels.
    handles = [Patch(color=voc[v] / 255, label=VOC_CLASSES[v]) for v in KEPT_CLASSES]
    other = [v for v in range(len(VOC_CLASSES)) if v not in KEPT_CLASSES]
    other_colours = sorted((voc[v] / 255 for v in other), key=lambda c: colorsys.rgb_to_hsv(*c)[0])
    handles.append(StripedPatch(other_colours))
    fig.legend(handles=handles, labels=[h.get_label() for h in handles[:-1]] + [OTHER_LABEL],
               loc="center", bbox_to_anchor=(0.5, 0.5 * legend_h / fig_h),
               ncol=len(handles), frameon=False, fontsize=7, handlelength=1.4, handleheight=1.0,
               columnspacing=1.8, handler_map={StripedPatch: StripedPatchHandler()})

    for ext in ("png",):  # add "pdf" for a vector version
        path = OUT / f"{OUTPUT_NAME}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print("saved", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
