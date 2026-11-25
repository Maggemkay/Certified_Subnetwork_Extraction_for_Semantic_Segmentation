"""Shared boilerplate: log tag, parameter header, run_info.json, result paths, seeding, I/O."""

import json
import re
import socket
import subprocess
import sys
import textwrap
from datetime import datetime

import numpy as np
import torch
import torchvision

import config as C

# The job script passes "log_<id>" as the first argument; by hand it is "manual".
LOG_TAG = sys.argv[1] if len(sys.argv) > 1 else "manual"

_run = {}  # filled by start_run(), reused by write_run_info()


def get_device():
    if C.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(C.DEVICE)


def _git_commit():
    root = C.PROJECT_ROOT
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" (uncommitted changes)" if dirty else "")


def start_run(stage, used_params):
    """Print the job tag, environment and every config value the stage reads."""
    device = get_device()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    _run.update(
        stage=stage,
        log_tag=LOG_TAG,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        hostname=socket.gethostname(),
        device=f"{device} ({device_name})",
        git_commit=_git_commit(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        params={name: getattr(C, name) for name in used_params},
    )

    line = "=" * 72
    print(line)
    print(f"{stage} | job {LOG_TAG} | {_run['timestamp']}")
    print(f"host: {_run['hostname']} | device: {_run['device']}")
    print(f"git: {_run['git_commit']} | torch {_run['torch']} | torchvision {_run['torchvision']}")
    print("-" * 72)
    width = max(len(name) for name in used_params)
    for name, value in _run["params"].items():
        print(f"{name:<{width}} = {value!r}")
    print(line, flush=True)


def section(title, how_to_read=""):
    """Print a log section: a framed title, then a "How to read" block that says what every number
    below is, its denominator, and what is fixed by construction or trivial."""
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)
    if how_to_read:
        print("How to read:")
        print(textwrap.indent(textwrap.dedent(how_to_read).strip("\n"), "  "))
        print()


def write_run_info(folder, **extra):
    """Write run_info.json (job tag, timestamp, parameters, plus any `extra` entries) next to a stage's outputs."""
    save_json(folder / "run_info.json", {**_run, **extra})


def rng(*keys):
    """Independent random stream per (seed, class, ...) key, e.g. rng(C.DISCOVERY_AND_EVAL_SEED, c)."""
    return np.random.default_rng(list(keys))


# ------------------------------------------------------------------
# Eval lists WITH class c (s01; s05c) and their cap: MAX_EVAL_IMAGES, LABEL_REMAPPED_MAX_EVAL_IMAGES
# ------------------------------------------------------------------
def eval_image_cap(value, setting):
    """The cap of an eval-image setting: None for "all" (every image with c), else the positive int. Any other value
    (0, negative, float, bool, another string) stops with a message naming the setting."""
    if value == "all":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    sys.exit(f"\n[config] {setting} = {value!r}: allowed are \"all\" (every image with c) or an int > 0 (a cap: a "
             f"seeded random subset of the images with c).")


def cap_suffix(value, setting):
    """Folder-name token of an eval-image cap: "Cap200" for 200, "CapAll" for "all"."""
    cap = eval_image_cap(value, setting)
    return "CapAll" if cap is None else f"Cap{cap}"


def cap_text(value, setting):
    """Log text of an eval-image cap."""
    cap = eval_image_cap(value, setting)
    return (f"{setting} = \"all\": every image with c (no cap)" if cap is None
            else f"{setting} = {cap}: at most {cap} images with c (seeded random subset if there are more)")


def eval_list(c, counts, max_eval_images, setting, seed, exclude=()):
    """Indices (sorted) of the images whose GT contains class c (counts: [images, 21] GT pixels per class; any pixel
    counts), minus `exclude`, and how many such images there are. With a cap below that number, a random subset of
    cap images is drawn with rng(seed, c); with "all" every image is kept (no random draw)."""
    available = np.setdiff1d(np.flatnonzero(counts[:, c] > 0), exclude)
    cap = eval_image_cap(max_eval_images, setting)
    if cap is None or len(available) <= cap:
        return available, len(available)
    chosen = rng(seed, c).choice(available, cap, replace=False)
    return np.sort(chosen), len(available)


def check_eval_list_cap(discovery_and_eval_lists):
    """Stops if s01's eval lists were drawn with another MAX_EVAL_IMAGES than config.py's (s05, s05b, s07): the lists
    live in s01's file, which a new s01 run overwrites, while the eval folders are named by config.py's cap."""
    eval_image_cap(C.MAX_EVAL_IMAGES, "MAX_EVAL_IMAGES")
    path = discovery_dir() / "discovery_and_eval_lists.json"
    if "max_eval_images" not in discovery_and_eval_lists:
        sys.exit(f"\n[stale input] {path}\ndoes not record the MAX_EVAL_IMAGES its eval lists were drawn with (an older "
                 f"s01 output).\n-> re-run s01_discovery_samples_dataset.py with the current "
                 f"config.py (cheap, no circuits; D_c stays the same, so s02-s04b stay valid).")
    stored = discovery_and_eval_lists["max_eval_images"]
    if stored != C.MAX_EVAL_IMAGES:
        sys.exit(f"\n[config mismatch] the eval lists in {path}\nwere drawn with MAX_EVAL_IMAGES = {stored!r}, "
                 f"config.py has {C.MAX_EVAL_IMAGES!r}.\n-> re-run s01_discovery_samples_dataset.py with the current "
                 f"config.py (cheap; D_c stays the same), or set MAX_EVAL_IMAGES back to {stored!r}.")


# ------------------------------------------------------------------
# Top-K selection mode of the base algorithm A (s03): TOPK_SELECTION_MODE, and how K is named in logs and figures
# ------------------------------------------------------------------
TOPK_SELECTION_MODES = ("per_layer", "global_raw")


def topk_selection_mode():
    """TOPK_SELECTION_MODE, checked: stops with a message for any value not in TOPK_SELECTION_MODES."""
    if C.TOPK_SELECTION_MODE not in TOPK_SELECTION_MODES:
        sys.exit(f"\n[config] TOPK_SELECTION_MODE = {C.TOPK_SELECTION_MODE!r}: allowed are "
                 f"{', '.join(repr(mode) for mode in TOPK_SELECTION_MODES)}.")
    return C.TOPK_SELECTION_MODE


def topk_selection_is_global():
    """True in a global top-K mode (one top-K over all channels of LAYERS), False for "per_layer"."""
    return topk_selection_mode() != "per_layer"


def topk_selection_group():
    """The group whose channels one global top-K ranks: LAYERS in a global mode; "none" for "per_layer" (every layer
    is ranked on its own). Stored by s03 and checked by s04."""
    return C.LAYERS if topk_selection_is_global() else "none"


def topk_ratio_label():
    """The name of K in log columns and figure axes: "top-K layer-normalized ratio" (per_layer) or "top-K global ratio
    of <LAYERS>" (global, e.g. "top-K global ratio of backbone")."""
    return f"top-K global ratio of {C.LAYERS}" if topk_selection_is_global() else "top-K layer-normalized ratio"


def topk_ratio_meaning():
    """What K keeps, in a few words (axis labels, "How to read" blocks): "ratio kept per layer by A" (per_layer) or
    "share of all <n> <LAYERS> channels kept by A, per layer it varies" (global)."""
    if not topk_selection_is_global():
        return "ratio kept per layer by A"
    from common import model as M
    channels = len(M.certifiable_components(C.LAYERS)[1])
    return f"share of all {channels:,} {C.LAYERS} channels kept by A, per layer it varies"


def topk_ratio_axis_label():
    """Two-line axis label of K: the name, then what it keeps."""
    return f"{topk_ratio_label()} K\n({topk_ratio_meaning()})"


def topk_ratio_description():
    """One sentence on what K is, for the "How to read" blocks of the evaluation stages (no trailing period)."""
    if not topk_selection_is_global():
        return ("K, the ratio of channels per layer the per-layer top-K selection keeps (s03): the baseline circuit "
                "keeps round(K · C) channels per layer")
    from common import model as M
    channels = len(M.certifiable_components(C.LAYERS)[1])
    return (f"K, the share of all {channels:,} channels of LAYERS = {C.LAYERS!r} that the global top-K selection keeps "
            f"(s03, TOPK_SELECTION_MODE = {C.TOPK_SELECTION_MODE!r}): the baseline circuit keeps round(K · {channels:,}) "
            f"of them in total; per layer the share varies, a layer can even be empty")


def layer_channel_counts(mask, certifiable_slices):
    """Kept channels per layer of U: mask [..., |U|] (bool) -> [..., layers of U] int, in the order of
    certifiable_slices (M.certifiable_components)."""
    return np.stack([np.count_nonzero(mask[..., s], axis=-1) for s in certifiable_slices.values()], axis=-1)


# ------------------------------------------------------------------
# Result paths: one nested folder per stage, named by the parameters it adds.
# ------------------------------------------------------------------
def discovery_dir(bottleneck=None):
    """s01 output: results/<RUN_NAME>/<dataset>_D<size>_min<pixels>[_predreq]_DiscAndEvalSeed<seed>/
    With DISCOVERY_SAMPLE_SIZE = "bottleneck", <size> is "bottleneck<number>" (e.g. bottleneck262). s01 passes
    the number it resolved; the later stages find the folder s01 wrote (_find_bottleneck_dir)."""
    size = C.DISCOVERY_SAMPLE_SIZE
    if size == "bottleneck":
        if bottleneck is None:
            return _find_bottleneck_dir()
        size = f"bottleneck{bottleneck}"
    return C.RESULTS_DIR / C.RUN_NAME / _discovery_dir_name(size)


def _discovery_dir_name(size):
    name = f"{C.DISCOVERY_DATASET}_D{size}_min{C.MIN_PIXELS}"
    if C.REQUIRE_PRED_IN_DISCOVERY:
        name += "_predreq"
    return f"{name}_DiscAndEvalSeed{C.DISCOVERY_AND_EVAL_SEED}"


def _find_bottleneck_dir():
    """The s01 folder D_bottleneck<number> whose discovery_and_eval_lists.json holds every class in CLASSES.
    If several do (s01 run with different CLASSES), the one holding exactly CLASSES."""
    run_dir = C.RESULTS_DIR / C.RUN_NAME
    prefix, suffix = _discovery_dir_name("bottleneck<N>").split("<N>")
    pattern = re.escape(prefix) + r"\d+" + re.escape(suffix)
    folders = sorted(d for d in run_dir.iterdir() if re.fullmatch(pattern, d.name)) if run_dir.exists() else []

    def classes_in(folder):
        path = folder / "discovery_and_eval_lists.json"
        return set(load_json(path)["discovery_samples"]) if path.exists() else set()

    wanted = {str(c) for c in C.CLASSES}
    found = [d for d in folders if wanted <= classes_in(d)]
    if len(found) > 1:
        found = [d for d in found if classes_in(d) == wanted]
    if len(found) == 1:
        return found[0]
    if not found:
        sys.exit(f"\n[missing input] no s01 folder {prefix}<number>{suffix} in {run_dir} holds all CLASSES\n"
                 f"-> run s01_discovery_samples_dataset.py first with the current config.py.")
    sys.exit(f"\n[ambiguous input] several s01 folders hold exactly CLASSES: {[d.name for d in found]}\n"
             f"-> remove the outdated one(s).")


def score_dir():
    """s02 output: <discovery_dir>/s02_scores/"""
    return discovery_dir() / "s02_scores"


def sample_dir():
    """s03 output: <score_dir>/s03_<target>_<channel scoring method>_pdel<p>_n<N0>-<N>_sseed<seed>[_topk_<mode>_<LAYERS>]/
    The suffix only in a global top-K mode (TOPK_SELECTION_MODE other than "per_layer"), e.g. _topk_global_raw_backbone:
    there LAYERS is the group s03's global top-K ranks. "per_layer" adds no suffix."""
    name = f"s03_{C.TARGET}_{C.CHANNEL_SCORING_METHOD}_pdel{C.P_DEL:.2f}_n{C.N0}-{C.N}_sseed{C.SAMPLING_SEED}"
    if topk_selection_is_global():
        name += f"_topk_{C.TOPK_SELECTION_MODE}_{C.LAYERS}"
    return score_dir() / name


def certify_dir():
    """s04 output: <sample_dir>/s04_<layers>_tau<agreement threshold τ>_a<failure probability α>/"""
    return sample_dir() / f"s04_{C.LAYERS}_tau{C.AGREEMENT_THRESHOLD_TAU}_a{C.FAILURE_PROB_ALPHA}"


def logit_statistics_dir():
    """s04b output: <certify_dir>/s04b_logit_statistics_<images>_minpx<min pixels>/"""
    return certify_dir() / f"s04b_logit_statistics_{C.LOGIT_STATISTICS_IMAGES}_minpx{C.LOGIT_STATISTICS_MIN_PIXELS}"


def eval_dir():
    """s05 output: <certify_dir>/s05_eval_Cap<MAX_EVAL_IMAGES>/ (s05_eval_CapAll for "all"); with
    LOGIT_STANDARDIZATION_VARIANTS on, below <logit_statistics_dir> instead."""
    return ((logit_statistics_dir() if C.LOGIT_STANDARDIZATION_VARIANTS else certify_dir())
            / f"s05_eval_{cap_suffix(C.MAX_EVAL_IMAGES, 'MAX_EVAL_IMAGES')}")


def merged_circuits_eval_dir():
    """s05b output: <certify_dir>/s05b_merged_circuits_eval_Cap<MAX_EVAL_IMAGES>/ (raw circuits only, so never below
    the s04b folder)"""
    return certify_dir() / f"s05b_merged_circuits_eval_{cap_suffix(C.MAX_EVAL_IMAGES, 'MAX_EVAL_IMAGES')}"


def label_remapped_eval_dir():
    """s05c output: <certify_dir>/s05c_label_remapped_eval_Cap<LABEL_REMAPPED_MAX_EVAL_IMAGES>[_seed<seed>]
    [_cityscapes<W>x<H>]/, below <logit_statistics_dir> instead when LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS is
    non-empty. _seed only with an int cap (with "all" the seed draws nothing); _cityscapes only when
    LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE is set. One subfolder per dataset inside."""
    name = f"s05c_label_remapped_eval_{cap_suffix(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, 'LABEL_REMAPPED_MAX_EVAL_IMAGES')}"
    if eval_image_cap(C.LABEL_REMAPPED_MAX_EVAL_IMAGES, "LABEL_REMAPPED_MAX_EVAL_IMAGES") is not None:
        name += f"_seed{C.LABEL_REMAPPED_EVAL_SEED}"
    if C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE is not None:
        width, height = C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE
        name += f"_cityscapes{width}x{height}"
    return (logit_statistics_dir() if C.LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS else certify_dir()) / name


def plot_dir():
    """s06 output: IMAGES_DIR + the path of eval_dir() below RESULTS_DIR (same nesting as results/)."""
    return C.IMAGES_DIR / eval_dir().relative_to(C.RESULTS_DIR)


def label_remapped_plot_dir():
    """s06c output: IMAGES_DIR + the path of label_remapped_eval_dir() below RESULTS_DIR."""
    return C.IMAGES_DIR / label_remapped_eval_dir().relative_to(C.RESULTS_DIR)


def prediction_dir():
    """s07 output: IMAGES_DIR + the path of certify_dir() below RESULTS_DIR (of logit_statistics_dir() with
    PREDICT_LOGIT_STANDARDIZATION_VARIANTS or PREDICT_CONFIDENCE_MAPS on, which use s04b's statistics) +
    s07_<dataset>_Cap<MAX_EVAL_IMAGES>_pos<positions>_topkratio<top-K layer-normalized ratios>/ ("/" in the dataset
    name becomes "_"; CapAll for "all": the positions index s01's eval lists, which the cap changes)."""
    positions = ",".join(str(p) for p in C.PREDICT_POSITIONS)
    topk_layer_normalized_ratio_values = ",".join(  # in the figure-row order
        f"{topk_layer_normalized_ratio:g}"
        for topk_layer_normalized_ratio in sorted(set(C.PREDICT_TOPK_LAYER_NORMALIZED_RATIO), reverse=True))
    name = (f"s07_{C.PREDICT_DATASET.replace('/', '_')}_{cap_suffix(C.MAX_EVAL_IMAGES, 'MAX_EVAL_IMAGES')}"
            f"_pos{positions}_topkratio{topk_layer_normalized_ratio_values}")
    parent = logit_statistics_dir() if needs_logit_statistics_in_predictions() else certify_dir()
    return C.IMAGES_DIR / parent.relative_to(C.RESULTS_DIR) / name


def needs_logit_statistics_in_predictions():
    """Whether s07 uses s04b's statistics (logit-standardization panels or confidence maps)."""
    return bool(C.PREDICT_LOGIT_STANDARDIZATION_VARIANTS) or C.PREDICT_CONFIDENCE_MAPS


def require(path, stage):
    """Stop with a clear message if an upstream output is missing."""
    if not path.exists():
        sys.exit(f"\n[missing input] {path}\n"
                 f"-> run {stage} first with the current config.py.")


def require_datasets(names):
    """Stop with a clear message if any of the datasets is not on disk (checked before any work starts)."""
    from common import datasets
    missing = []
    for name in names:
        try:
            datasets.get_dataset(name)
        except (FileNotFoundError, RuntimeError, ImportError) as err:
            missing.append(f"  {name}: {err}")
    if missing:
        sys.exit("\n[missing dataset]\n" + "\n".join(missing) + "\n"
                 "-> voc_sbd_train / voc_val: set CHECK_DOWNLOAD = True in config.py and run setup_check_and_baseline.py\n"
                 "   (downloads VOC 2012 into data/voc and SBD into data/sbd).\n"
                 "-> coco_*: run download_scripts/coco_download.sh.\n"
                 "-> ade20k_val_voc: run download_scripts/ade20k_download.sh (no account needed).\n"
                 "-> cityscapes_val_voc: run download_scripts/cityscapes_download.sh (needs a cityscapes-dataset.com "
                 "account).")


def load_discovery_and_eval_lists():
    """discovery_and_eval_lists.json of the current config; stops if it or one of CLASSES is missing."""
    path = discovery_dir() / "discovery_and_eval_lists.json"
    require(path, "s01_discovery_samples_dataset.py")
    discovery_and_eval_lists = load_json(path)
    missing = [c for c in C.CLASSES if str(c) not in discovery_and_eval_lists["discovery_samples"]]
    if missing:
        sys.exit(f"\n[missing input] classes {missing} are not in {path}\n"
                 f"-> re-run s01_discovery_samples_dataset.py with them in CLASSES.")
    return discovery_and_eval_lists


def eval_dataset(name):
    """The dataset of an EVAL_DATASETS entry; "discovery_samples" images come from the discovery dataset."""
    from common import datasets
    return datasets.get_dataset(C.DISCOVERY_DATASET if name == "discovery_samples" else name)


# ------------------------------------------------------------------
# Circuits (s04 output)
# ------------------------------------------------------------------
def mask_key(masks):
    """Bytes that identify a {layer: 0/1 mask} circuit, so identical circuits are run only once."""
    return b"".join(np.packbits(m).tobytes() for m in masks.values())


def load_circuits(c, certifiable_slices, types):
    """Circuits of class c from s04, for each circuit type in `types` ("baseline_circuit", "majority",
    "certified"): {type: a list over the grid indices of ({layer: 0/1 mask}, key)},
    {type: subnet_size_layer_normalized_ratio_mean per grid index}, and the top-K layer-normalized ratio grid."""
    path = certify_dir() / f"class_{c:02d}.npz"
    require(path, "s04_certify.py")
    certification_results = np.load(path)
    selected = {"baseline_circuit": certification_results["baseline_circuit"],
                "majority": certification_results["majority"],
                "certified": certification_results["decision"] == 1}
    grid_idxs = range(len(certification_results["topk_layer_normalized_ratio_grid"]))
    circuits, subnet_size_layer_normalized_ratio_mean = {}, {}
    for t in types:
        circuits_per_grid_idx = [{name: selected[t][grid_idx, s] for name, s in certifiable_slices.items()}
                                 for grid_idx in grid_idxs]
        circuits[t] = [(masks, mask_key(masks)) for masks in circuits_per_grid_idx]
        subnet_size_layer_normalized_ratio_mean[t] = \
            certification_results[f"subnet_size_layer_normalized_ratio_mean_{t}"]
    return circuits, subnet_size_layer_normalized_ratio_mean, certification_results["topk_layer_normalized_ratio_grid"]


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
