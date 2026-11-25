"""s06c · Figures and tables of the evaluation on label-remapped datasets (from s05c's output).
A separate plot stage, so these figures can be drawn without re-drawing every s06 figure; it uses s06's
plotting functions and styles.

Every dataset subfolder found in s05c's folder (label_remapped_eval_dir() for the current config) is drawn; the log
names each one with the time, job and commit from its run_info.json. Means are over the dataset's evaluated classes
(the classes of CLASSES it can score), each class weighted equally. The x-axis of the curves is the subnetwork size, as
in s06. Every figure of a dataset carries its title note (GT remapped, N of 20 classes evaluable).

VOC val on the same classes ("like-for-like"): from s05's stored vs-GT confusion matrices on voc_val (the s05 folder
utils.eval_dir() gives for the current config: _Cap200 or _CapAll from MAX_EVAL_IMAGES), raw circuits. Per class c the
same circuits and the same metrics, with the mIoU over background + the dataset's evaluable classes. Left out with a
note when s05 has no voc_val counts for every class of the dataset.

Output in <label_remapped_plot_dir> (IMAGES_DIR + the s05c folder's path), format PLOT_FORMAT:
    overview_curves             cIoU and evaluable-class mIoU vs subnetwork size; per dataset two columns: the dataset,
                                then VOC val on the same classes. Raw circuits
    curves_mean.csv             the numbers behind every curve (means per dataset / VOC val column, variant, circuit, K)
    reference_table.csv / .png  the full-model reference of each dataset (s05c) next to VOC val on the same classes:
                                per-class IoU, evaluable-class mIoU, pixel accuracy
    <dataset>/curves                         as overview_curves, this dataset only
              curves_per_class               cIoU per evaluated class
              misclassification_of_class_c   where class c's GT pixels go, at
                                             LABEL_REMAPPED_PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO
              logit_standardization/         curves_<circuit type>, curves_per_class_<circuit type>,
                                             misclassification_of_class_c_<variant>
                                             (only with LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS)
"""

import csv
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd

import config as C
import s06_plots as s06
from common import datasets, metrics, utils
from common.datasets import NUM_CLASSES, VOC_CLASSES

plt = s06.plt
USED_PARAMS = [
    "RUN_NAME", "RESULTS_DIR", "IMAGES_DIR", "CLASSES",
    "DISCOVERY_DATASET", "DISCOVERY_SAMPLE_SIZE", "MIN_PIXELS", "REQUIRE_PRED_IN_DISCOVERY", "DISCOVERY_AND_EVAL_SEED",
    "TARGET", "CHANNEL_SCORING_METHOD", "P_DEL", "N0", "N", "SAMPLING_SEED",
    "TOPK_SELECTION_MODE", "LAYERS", "AGREEMENT_THRESHOLD_TAU", "FAILURE_PROB_ALPHA",
    "LOGIT_STATISTICS_IMAGES", "LOGIT_STATISTICS_MIN_PIXELS",
    "MAX_EVAL_IMAGES", "LOGIT_STANDARDIZATION_VARIANTS",          # where s05's VOC val counts are (eval_dir())
    "LABEL_REMAPPED_MAX_EVAL_IMAGES", "LABEL_REMAPPED_EVAL_SEED", "LABEL_REMAPPED_CIRCUIT_TYPES",
    "LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS", "LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE",
    "LABEL_REMAPPED_PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO", "PLOT_FORMAT", "SMOKE_TEST",
]
IMAGE_SET = "images_with_class_c"
MEAN_IOU_COLUMN = "evaluable_classes_mIoU_vs_ground_truth"
METRIC_COLUMNS = ["cIoU", MEAN_IOU_COLUMN, "class_c_predicted_pixel_ratio", "images_with_class_c_predicted_ratio"]
CURVE_ROWS = [("cIoU", "cIoU (vs GT)"), (MEAN_IOU_COLUMN, "mIoU, background + the\nevaluable classes (vs GT)")]
SELECTED_SETTING = "LABEL_REMAPPED_PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO"
VOC = "voc_val"


def variants():
    return s06.variants(C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS)


def circuit_types():
    return s06.circuit_types(C.LABEL_REMAPPED_CIRCUIT_TYPES)


def class_count(n):
    return f"{n} class" if n == 1 else f"{n} classes"


def voc_key(name):
    """Column key of VOC val on the classes of dataset `name` (in curves_mean.csv: eval_dataset)."""
    return f"{VOC} (classes of {name})"


def save(fig, path, what):
    s06.save(fig, path, what, utils.label_remapped_plot_dir())


def figure_path(*parts):
    """<label_remapped_plot_dir>/<parts...>.<PLOT_FORMAT>"""
    return utils.label_remapped_plot_dir().joinpath(*parts[:-1], f"{parts[-1]}.{C.PLOT_FORMAT}")


# ---- data ------------------------------------------------------------------------------
def load_dataset_folder(name, folder):
    """s05c's summary.csv (restricted to CLASSES and "raw" + LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS),
    reference.json and run_info.json of one dataset subfolder."""
    summary = pd.read_csv(folder / "summary.csv")
    missing = [v for v in variants() if v not in set(summary.logit_standardization_variant)]
    if missing:
        sys.exit(f"\n[missing input] {folder / 'summary.csv'} has no rows for the logit-standardization variant(s) "
                 f"{missing}\n-> re-run s05c_evaluate_label_remapped_datasets.py with the current config.py.")
    summary = summary[summary["class"].isin(C.CLASSES) & summary.logit_standardization_variant.isin(variants())]
    return summary, utils.load_json(folder / "reference.json"), utils.load_json(folder / "run_info.json")


def mean_curves(summary, key):
    """Mean over the classes per variant, circuit and K (the full model: one row per variant) of the metric and size
    columns; eval_dataset = key."""
    columns = METRIC_COLUMNS + ["subnet_size_layer_normalized_ratio_mean"]
    keys = ["logit_standardization_variant"]
    with warnings.catch_warnings():  # all-nan cells stay nan
        warnings.simplefilter("ignore", RuntimeWarning)
        pruned = (summary[summary.circuit != "full"].groupby(keys + ["circuit", "topk_layer_normalized_ratio"])[columns]
                  .mean().reset_index())
        full = summary[summary.circuit == "full"].groupby(keys)[columns].mean().reset_index()
    full["circuit"] = "full"
    out = pd.concat([pruned, full], ignore_index=True)
    out.insert(0, "eval_dataset", key)
    return out


def voc_like_for_like(name, summary, evaluable):
    """VOC val on the classes of dataset `name`, raw circuits: summary-like rows (cIoU, the mIoU over background +
    `evaluable`, c-pixels, c-images; size from s05c's summary, the same circuits) from s05's stored counts, or
    (None, reason) when s05 has no voc_val counts for every class or circuit type."""
    folder = utils.eval_dir() / "confusion_matrices_and_logit_histograms" / VOC
    classes = sorted(summary["class"].unique())
    missing = [c for c in classes if not (folder / f"class_{c:02d}_{IMAGE_SET}.npz").exists()]
    if missing:
        return None, (f"no s05 counts on {VOC} for {', '.join(VOC_CLASSES[c] for c in missing)} in {folder} "
                      f"(run s05_evaluate.py with {VOC} in EVAL_DATASETS)")
    raw = summary[summary.logit_standardization_variant == "raw"]
    mean_classes = [0, *evaluable]
    rows = []
    for c in classes:
        with np.load(folder / f"class_{c:02d}_{IMAGE_SET}.npz") as z:
            cms = z["confusion_matrix_vs_ground_truth"]
            stored_types = z["model_circuit_types"]
            ratios = z["model_topk_layer_normalized_ratio"]
            v_idx = z["logit_standardization_variants"].tolist().index("raw")
            images_predicted, num_images = z["images_with_class_c_predicted"], len(z["image_indices"])
        for _, row in raw[raw["class"] == c].iterrows():
            t, k = row.circuit, row.topk_layer_normalized_ratio
            hit = np.flatnonzero(stored_types == "full") if t == "full" else \
                np.flatnonzero((stored_types == t) & np.isclose(ratios, k))
            if not len(hit):
                return None, f"s05's {VOC} counts have no {t} circuit at K = {k} (s05 ran with other CIRCUIT_TYPES)"
            cm = cms[hit[0], v_idx]
            rows.append({"class": c, "circuit": t, "logit_standardization_variant": "raw",
                         "topk_layer_normalized_ratio": k,
                         "cIoU": metrics.class_scores(cm)["iou"][c],
                         MEAN_IOU_COLUMN: metrics.mean_iou(cm, mean_classes),
                         "class_c_predicted_pixel_ratio": metrics.predicted_class_share(cm, c),
                         "images_with_class_c_predicted_ratio":
                             images_predicted[hit[0], v_idx] / num_images if num_images else np.nan,
                         "subnet_size_layer_normalized_ratio_mean": row.subnet_size_layer_normalized_ratio_mean})
    return pd.DataFrame(rows), None


# ---- figures ----------------------------------------------------------------------------
def plot_curve_grid(means, columns, path, title, what, note, classes_text, t=None):
    """CURVE_ROWS against the subnetwork size, one column per (key, column title). t None: the raw circuits, one line
    per circuit type, grey = the full model; else circuit type t, one line per logit-standardization variant."""
    fig, axes = plt.subplots(len(CURVE_ROWS), len(columns), figsize=(3.4 * len(columns) + 1.0, 2.3 * len(CURVE_ROWS)),
                             sharex=True, squeeze=False)
    for col, (key, column_title) in enumerate(columns):
        d = means[means.eval_dataset == key]
        for row, (metric, label) in enumerate(CURVE_ROWS):
            ax = axes[row, col]
            if t is None:
                dr = d[d.logit_standardization_variant == "raw"]
                for circuit in circuit_types():
                    sub = dr[dr.circuit == circuit].sort_values("topk_layer_normalized_ratio")
                    s06.line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], circuit)
                full = dr[dr.circuit == "full"][metric]
                if len(full):
                    s06.full_line(ax, full.iloc[0])
            else:
                for v in variants():
                    sub = d[(d.circuit == t) & (d.logit_standardization_variant == v)].sort_values(
                        "topk_layer_normalized_ratio")
                    s06.variant_line(ax, sub["subnet_size_layer_normalized_ratio_mean"], sub[metric], v)
                s06.full_variant_lines(ax, d, metric, C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            if row == 0:
                ax.set_title(column_title, fontsize=9)
            if col == 0:
                ax.set_ylabel(label)
            if row == len(CURVE_ROWS) - 1:
                ax.set_xlabel(s06.SIZE_LABEL)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if t is None:
        s06.header(fig, title, handles, labels, note=note, classes_text=classes_text)
    else:
        full_handles, full_labels = s06.full_variant_handles(C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS)
        s06.header(fig, title, handles + full_handles, labels + full_labels, ncol=2, note=note,
                   classes_text=classes_text)
    save(fig, path, what)


def plot_reference_table(rows, path):
    """Per-class IoU, evaluable-class mIoU and pixel accuracy of the full model: per dataset its row and the VOC val
    row on the same classes; cells of classes the dataset cannot score are hatched."""
    columns = [VOC_CLASSES[c] for c in range(NUM_CLASSES)] + ["mIoU (evaluable)", "pixel accuracy"]
    values = np.array([[r[f"IoU_{c:02d}_{VOC_CLASSES[c]}"] for c in range(NUM_CLASSES)]
                       + [r["evaluable_classes_mIoU"], r["pixel_accuracy"]] for r in rows], dtype=float)
    labels = [r["row"] for r in rows]
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 3.0, 0.42 * len(rows) + 1.6))
    image = s06.heatmap(ax, values, labels, columns, s06.SEQUENTIAL, 0, 1)
    for r, c in zip(*np.nonzero(np.isfinite(values))):
        ax.text(c + 0.5, r + 0.5, f"{values[r, c]:.2f}", ha="center", va="center", fontsize=6.5,
                color=s06.SURFACE if values[r, c] > 0.6 else s06.INK)
    for r in range(2, len(rows), 2):                      # a gap line between the datasets
        ax.axhline(r, color=s06.INK_2, linewidth=0.8)
    s06.colorbar(fig, image, ax, "IoU / accuracy")
    fig.tight_layout()
    s06.header(fig, "Full-model reference on the label-remapped datasets, each next to VOC val on the same classes",
               note="GT remapped to VOC labels (merged / ignored classes); hatched = a class the dataset cannot score, or (rarely) "
                    "one with neither GT nor predicted pixels",
               classes_text="classes per dataset: its evaluable classes")
    save(fig, path, "per dataset: its full-model IoU per class (background + the evaluable classes), the mIoU over "
         "them and the pixel accuracy (all classes),\n      then VOC val on the same classes (like-for-like; VOC val's "
         "pixel accuracy over all its classes). Numbers: reference_table.csv.")


def reference_rows(name, reference):
    """reference_table.csv rows of one dataset: the dataset (s05c's reference) and VOC val on the same classes."""
    evaluable = set(reference["mean_iou_classes"])
    out = []
    for label, record, compared in ((name, reference, ""),
                                    (f"{VOC}, classes of {name}", reference["voc_val_like_for_like"], VOC)):
        row = {"row": label, "dataset": name, "compared_with": compared, "num_images": record["num_images"],
               "num_evaluable_classes": len(reference["evaluable_classes"]),
               "evaluable_classes_mIoU": record["evaluable_classes_miou"], "pixel_accuracy": record["pixel_accuracy"],
               "mIoU_all_21_classes": record.get("miou_all_21_classes", np.nan)}
        for c in range(NUM_CLASSES):
            row[f"IoU_{c:02d}_{VOC_CLASSES[c]}"] = record["iou"][c] if c in evaluable else np.nan
        out.append(row)
    return out


def load_counts(folder, classes):
    """s05c's counts of one dataset: {class: npz arrays} for the classes whose file exists."""
    out = {}
    for c in classes:
        path = folder / "confusion_matrices" / f"class_{c:02d}_{IMAGE_SET}.npz"
        if path.exists():
            with np.load(path) as stored:
                out[c] = {key: stored[key] for key in stored.files}
    return out


# ---- main --------------------------------------------------------------------------------
def main():
    utils.start_run("s06c_plots_label_remapped_datasets", USED_PARAMS)
    eval_dir = utils.label_remapped_eval_dir()
    out_dir = utils.label_remapped_plot_dir()
    names = [name for name in datasets.LABEL_REMAPPED_DATASETS if (eval_dir / name / "summary.csv").exists()]
    if not names:
        sys.exit(f"\n[missing input] no dataset subfolder with a summary.csv in {eval_dir}\n"
                 f"-> run s05c_evaluate_label_remapped_datasets.py first with the current config.py.")
    out_dir.mkdir(parents=True, exist_ok=True)
    types, variant_list = circuit_types(), C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS

    loaded = {}
    utils.section(
        f"Input: s05c's dataset subfolders in {eval_dir}",
        textwrap.dedent(f"""
        Every dataset subfolder found is drawn. time / job / commit come from its run_info.json, so a subfolder from
        an older s05c run is visible here. classes = the classes of CLASSES the dataset was evaluated on.
        VOC val on the same classes comes from s05's counts in {utils.eval_dir()}
        (MAX_EVAL_IMAGES = {C.MAX_EVAL_IMAGES!r}; raw circuits).
        """))
    for name in names:
        summary, reference, run_info = load_dataset_folder(name, eval_dir / name)
        classes = sorted(summary["class"].unique())
        voc_rows, voc_note = voc_like_for_like(name, summary, reference["evaluable_classes"]) if classes else (None, "")
        loaded[name] = (summary, reference, classes, voc_rows, voc_note)
        print(f"  {name:20s} time {run_info.get('timestamp')} · job {run_info.get('log_tag')} · commit "
              f"{run_info.get('git_commit')} · classes: {', '.join(VOC_CLASSES[c] for c in classes) or 'none'}")
        print(f"  {'':20s} VOC val on the same classes: {'yes' if voc_rows is not None else 'left out: ' + voc_note}")

    # ---- means, overview, reference table ------------------------------------------------------
    parts, columns_overview, notes = [], [], []
    for name in names:
        summary, reference, classes, voc_rows, _ = loaded[name]
        if not classes:
            continue
        n_evaluable = len(reference["evaluable_classes"])
        parts.append(mean_curves(summary, name))
        columns_overview.append((name, f"{name}\n{class_count(len(classes))}, {n_evaluable} of {NUM_CLASSES - 1} evaluable"))
        if voc_rows is not None:
            parts.append(mean_curves(voc_rows, voc_key(name)))
            columns_overview.append((voc_key(name), f"{VOC}\non the classes of {name}"))
        notes.append(f"{name} {n_evaluable}")
    means = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(means):
        means.to_csv(out_dir / "curves_mean.csv", index=False)
        print(f"\n  saved curves_mean.csv\n      the means behind every curve (eval_dataset = the dataset, or "
              f"\"{VOC} (classes of <dataset>)\").")
    utils.section(
        f"Figures in {out_dir}",
        textwrap.dedent(f"""
        overview_curves, reference_table   all datasets; <dataset>/   one dataset. Figures outside a
        logit_standardization/ folder show the RAW circuits.{" The logit_standardization/ folders show raw + " + ", ".join(variant_list) + " (the dataset only: s05's VOC val counts are not used there)." if variant_list else ""}
        Every figure names the title note of its dataset(s): GT remapped to VOC labels, N of 20 classes evaluable.
        """))
    if columns_overview:
        plot_curve_grid(means, columns_overview, figure_path("overview_curves"),
                        "Label-remapped datasets: accuracy vs. subnetwork size (mean over each dataset's classes), "
                        "each next to VOC val on the same classes",
                        "cIoU and the evaluable-class mIoU against the subnetwork size; per dataset two columns: the "
                        "dataset, then VOC val on the same classes\n      (s05's counts, same circuits); raw circuits; "
                        "the grey line is the full model.",
                        "GT remapped to VOC labels (merged / ignored classes) · evaluable classes of 20: "
                        + ", ".join(notes), "classes per column: see the column titles")
    rows = [row for name in names for row in reference_rows(name, loaded[name][1])]
    with open(out_dir / "reference_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved reference_table.csv\n      per dataset its full-model row (s05c) and VOC val on the same classes; "
          f"IoU columns empty for classes the dataset cannot score.")
    plot_reference_table(rows, figure_path("reference_table"))

    # ---- per dataset -------------------------------------------------------------------------
    for name in names:
        summary, reference, classes, voc_rows, voc_note = loaded[name]
        if not classes:
            print(f"\n[{name}] no evaluated class of CLASSES: only the reference table.")
            continue
        note = reference["title_note"]
        classes_text = class_count(len(classes))
        print(f"\n[{name}] {note}; classes: {', '.join(VOC_CLASSES[c] for c in classes)}")
        columns = [(name, name)] + ([(voc_key(name), f"{VOC} on the classes of {name}")] if voc_rows is not None else [])
        plot_curve_grid(means, columns, figure_path(name, "curves"),
                        f"Accuracy on {name} vs. subnetwork size (mean over its {class_count(len(classes))})"
                        + (", next to VOC val on the same classes" if voc_rows is not None else ""),
                        f"cIoU and the evaluable-class mIoU on {name} against the subnetwork size, raw circuits"
                        + (", next to VOC val on the same classes" if voc_rows is not None
                           else f" (VOC val left out: {voc_note})") + "; grey = the full model.",
                        note, classes_text)
        s06.plot_per_class(summary, name, IMAGE_SET, "cIoU", "cIoU (vs GT)", f"cIoU per class on {name}",
                           figure_path(name, "curves_per_class"),
                           f"cIoU against the subnetwork size on {name}, one panel per class; the grey line is the "
                           f"full model's cIoU of that class.", classes=classes, types=types, variant_list=variant_list,
                           note=note, base=out_dir)
        for t in types if variant_list else []:
            plot_curve_grid(means, [(name, name)], figure_path(name, "logit_standardization", f"curves_{t}"),
                            f"Logit standardization, circuit type {s06.STYLE[t][0]}: accuracy on {name} vs. "
                            f"subnetwork size (mean over its {class_count(len(classes))})",
                            f"as {name}/curves, for the {t} circuits only: one line per logit-standardization "
                            f"variant;\n      grey lines: the full model (raw; after the Jung variant if it is on).",
                            note, classes_text, t=t)
            s06.plot_per_class(summary, name, IMAGE_SET, "cIoU", "cIoU (vs GT)",
                               f"Logit standardization, circuit type {s06.STYLE[t][0]}: cIoU per class on {name}",
                               figure_path(name, "logit_standardization", f"curves_per_class_{t}"),
                               f"cIoU of the {t} circuits on {name}, one panel per class, one line per logit-"
                               f"standardization variant;\n      grey lines: the class's full-model cIoU (raw; after "
                               f"the Jung variant if it is on).", t=t, classes=classes, types=types,
                               variant_list=variant_list, note=note, base=out_dir)
        counts = load_counts(eval_dir / name, classes)
        if not counts:
            print(f"  NOTE: no confusion matrices in {eval_dir / name / 'confusion_matrices'}; the heatmaps are skipped.")
            continue
        for v in variants():
            sub = [] if v == "raw" else ["logit_standardization"]
            suffix = "" if v == "raw" else f"_{v}"
            s06.plot_confusion_heatmaps(counts, name, "misclassification_of_class_c", v,
                                        figure_path(name, *sub, "misclassification_of_class_c" + suffix), types=types,
                                        selected=C.LABEL_REMAPPED_PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO,
                                        selected_setting=SELECTED_SETTING, note=note, base=out_dir)
        del counts
    utils.write_run_info(out_dir)


if __name__ == "__main__":
    main()
