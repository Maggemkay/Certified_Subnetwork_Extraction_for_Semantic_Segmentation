"""Global configuration of the certified-circuits pipeline.

Every stage script reads this file directly (`import config as C`). Each
parameter sits in the section of the stage where it first takes effect; later
stages may read it too. Every stage prints the parameters it reads at start-up
and stores them in run_info.json next to its outputs. Which stages to run again
after a change: README.md, "Re-running after a change".
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==================================================================
# GENERAL (all stages)
# ==================================================================
RUN_NAME    = "main"                      # results/<RUN_NAME>/
DEVICE      = "cuda"                      # falls back to cpu if unavailable
DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR  = PROJECT_ROOT / "images"
MODEL_NAME  = "deeplabv3_resnet50"        # torchvision DEFAULT weights (COCO_WITH_VOC_LABELS_V1)
CLASSES     = list(range(1, 21))          # VOC indices, 0 = background excluded

# ==================================================================
# SETUP CHECK & BASELINE (setup_check_and_baseline.py, before s01)
# ==================================================================
CHECK_DOWNLOAD      = False               # True: download VOC 2012 and SBD first if they are missing (COCO, ADE20K and
                                          # Cityscapes are manual: download_scripts/)
CHECK_DATASETS      = ["voc_sbd_train", "voc_val", "coco_val2017_voc", "ade20k_val_voc", "cityscapes_val_voc"]
                                          # stats + one example plot each; missing ones are skipped
CHECK_EXAMPLE_INDEX = 0                   # index of the example image plotted per dataset
PRINT_MODEL         = True                # architecture + parameters per layer
CHECK_LABEL_REMAPPING = True              # label check of the label-remapped datasets in CHECK_DATASETS (COCO, ADE20K,
                                          # Cityscapes): mapping table, pixels per class, example figures (check before s05c)
CHECK_LABEL_REMAPPING_NUM_EXAMPLES = 10   # example figures per dataset (image | original labels | remapped labels)
RUN_BASELINE        = False               # full-model metrics over every image (slow)
BASELINE_DATASETS   = ["voc_val", "voc_sbd_train", "coco_val2017_voc"]
BASELINE_MAX_IMAGES = None                # None = all images; an int caps each dataset (quick check)

# ==================================================================
# s01 · DISCOVERY SAMPLES AND EVAL LISTS
# ==================================================================
DISCOVERY_DATASET = "voc_sbd_train"       # "voc_sbd_train" (10,582 images) | "coco_val2017_voc" (5,000)
                                          # | "coco_train2017_voc" (118,287; the model's training split)
EVAL_DATASETS     = ["voc_val", "discovery_samples"]  # "discovery_samples" = D_c itself; also "coco_val2017_voc"
DISCOVERY_SAMPLE_SIZE = "bottleneck"                # |D_c| per class. int: that many images (Anani B.1: 50) | "all": every
                                          # candidate | float in (0, 1]: that fraction of each class's candidates
                                          # | "bottleneck": the fewest candidates of any class in CLASSES, for all
                                          #   (with REQUIRE_PRED_IN_DISCOVERY: the fewest usable ones)
MIN_PIXELS        = 500                   # our choice: min. GT pixels of class c for an image to enter D_c
MAX_EVAL_IMAGES   = "all"                 # eval images WITH c per class: int > 0 = cap (seeded random subset) | "all" =
                                          # every image with c. In the s05/s05b/s07 folder names (_Cap200 / _CapAll);
                                          # a change needs an s01 re-run first
REQUIRE_PRED_IN_DISCOVERY = False         # True: D_c only takes images where the model predicts class c
# "predicts class c" means at least one predicted class-c pixel anywhere in the image, with no overlap with the
# GT region required. Candidates are checked lazily in the seeded random order until |D_c| is full (with
# "bottleneck": all candidates first).
DISCOVERY_AND_EVAL_SEED = 0               # seeds the D_c draw and the eval-list subsets

# ==================================================================
# s02 · HAMBLIN SCORE CACHE   (all hook points, all targets cached)
# ==================================================================
CACHED_TARGETS = ["pred", "gt", "all"]    # "pred" | "gt" | "all"; each costs one backward pass per image

# ==================================================================
# s03 · MONTE CARLO SAMPLING (RS-Del)
# ==================================================================
TARGET        = "pred"                    # "pred" | "gt" | "all"
CHANNEL_SCORING_METHOD = "actgrad"        # "actgrad" | "activation"
P_DEL         = 0.6                       # Anani Table 2
N0, N         = 100, 1000                 # N from Anani; N0 our choice
TOPK_LAYER_NORMALIZED_RATIO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # per-layer top-K selection: ratio of each layer's channels kept (Anani §4–5 sweep)
                                          # in global mode (TOPK_SELECTION_MODE "global_raw"): share of the whole LAYERS
                                          # group kept (e.g. 0.1 of the backbone's 3,840 channels = 384), per layer it varies
TOPK_SELECTION_MODE = "global_raw"        # "per_layer" (Anani: round(K·C) in every layer) | "global_raw"
                                          # (Hamblin Eq. 3: one top-K over all channels of LAYERS, raw scores; can empty
                                          # a layer, logged). "global_raw" adds _topk_global_raw_<LAYERS> to the s03 folder
SAMPLING_SEED = 0

# ==================================================================
# s04 · CERTIFICATION
# ==================================================================
LAYERS = "backbone"                       # "backbone" | "backbone+head" | "head"; with TOPK_SELECTION_MODE "global_raw" also
                                          # the group s03's global top-K ranks (in the s03 folder name)
AGREEMENT_THRESHOLD_TAU = 0.95            # → r = 1 with P_DEL = 0.6
FAILURE_PROB_ALPHA = 0.001                # Bonferroni over |U| (U = the certifiable components)

# ==================================================================
# s04b · CLASS-WISE LOGIT STATISTICS (Jung et al. 2021, adapted; certification unchanged)
# ==================================================================
LOGIT_STANDARDIZATION_VARIANTS = ["matched_to_full_model_on_full_model_prediction_pixels"]       
                                          # [] = off. Any of
                                          # "matched_to_full_model_on_full_model_prediction_pixels",
                                          # "matched_to_full_model_on_all_pixels",
                                          # "jung_standardized_logits_on_own_prediction_pixels".
                                          # s05 evaluates each next to the raw circuits, from the same forward pass
LOGIT_STATISTICS_IMAGES = "discovery_samples"  # the images the statistics are taken on: D_c of each class
                                               # (only "discovery_samples" is implemented)
LOGIT_STATISTICS_MIN_PIXELS = 1000        # our choice: a class with fewer
                                          # pixels in its pixel set has no usable statistics (fallback)

# ==================================================================
# s05 · EVALUATION
# ==================================================================
CIRCUIT_TYPES = ["full", "baseline_circuit", "majority", "certified"]
EVALUATE_IMAGES_WITHOUT_CLASS = False      # also evaluate every image WITHOUT class c, reported apart from the images with c.
                                          # Real eval sets: every image whose GT has
                                          # no c (no cap); "discovery_samples": the other classes' D_c' (images that
                                          # contain c dropped). Costly: ≈ 12× the image passes on voc_val
STORE_LOGIT_HISTOGRAMS = True             # fixed-bin histograms of every class's logit (Wasserstein-1, box plots)
LOGIT_HISTOGRAM_LOWER_EDGE = -20.0        # logit units; the histogram range (one extra bin at each end counts the
LOGIT_HISTOGRAM_UPPER_EDGE = 20.0         # values outside it). The same bins for the Jung variant's z-scores
LOGIT_HISTOGRAM_BIN_WIDTH = 0.25          # 160 bins; adjacent bins can be merged later, finer ones cannot be recovered

# ==================================================================
# s06 · PLOTS
# ==================================================================
PLOT_FORMAT = "png"
PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO = [0.2, 0.4, 0.6, 0.8]  # the figures that show selected K only (logit box plots,
                                          # misclassification and false-positive heatmaps): one figure row per value;
                                          # each must be in TOPK_LAYER_NORMALIZED_RATIO_GRID

# ==================================================================
# s07 · PREDICTION EXAMPLES
# ==================================================================
PREDICT_DATASET   = "voc_val"             # one of EVAL_DATASETS ("discovery_samples" = the in-sample D_c images)
PREDICT_POSITIONS = [0]                   # 0-based positions in each class's s01 eval list, e.g. [0] or [0, 9]
PREDICT_TOPK_LAYER_NORMALIZED_RATIO = TOPK_LAYER_NORMALIZED_RATIO_GRID  # each must be in TOPK_LAYER_NORMALIZED_RATIO_GRID (e.g. [0.3]); one figure row per value
PREDICT_TOP_PANELS = ["image", "gt", "full"]     # top row, drawn once: any of "image", "gt", "full"
PREDICT_CIRCUITS  = ["baseline_circuit", "majority", "certified"]  # one column each in every figure row below the top row: any of "baseline_circuit", "majority", "certified"
PREDICT_LOGIT_STANDARDIZATION_VARIANTS = LOGIT_STANDARDIZATION_VARIANTS  # one extra panel per variant next to each circuit panel; [] = none (needs s04b)
PREDICT_CONFIDENCE_MAPS = True            # Jung's standardized max logit S as maps, a third figure per image (needs s04b):
                                          # the full model's own S; per circuit its own S and S with the full model's statistics

# ==================================================================
# s05c / s06c · EVALUATION ON LABEL-REMAPPED DATASETS (accuracy only; circuits from s04 unchanged)
# ==================================================================
LABEL_REMAPPED_EVAL_DATASETS = ["coco_val2017_voc", "ade20k_val_voc", "cityscapes_val_voc"]  # GT remapped to VOC labels
LABEL_REMAPPED_MAX_EVAL_IMAGES = "all"    # as MAX_EVAL_IMAGES: "all" = every image with at least one pixel of c
                                          # | int > 0: cap (seeded random subset)
LABEL_REMAPPED_EVAL_SEED = 0              # as DISCOVERY_AND_EVAL_SEED (the list draws only; unused with "all")
LABEL_REMAPPED_CIRCUIT_TYPES = ["full", "baseline_circuit", "majority", "certified"]   # as CIRCUIT_TYPES
LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS = ["matched_to_full_model_on_full_model_prediction_pixels"]
                                          # as LOGIT_STANDARDIZATION_VARIANTS (non-empty: needs s04b); [] = raw only
LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE = None  # None = native 2048×1024 (as VOC, no resize) | (width, height), e.g.
                                          # (1024, 512): image bilinear, labels nearest. Also read by the setup check
LABEL_REMAPPED_PLOT_SELECTED_TOPK_LAYER_NORMALIZED_RATIO = [0.2, 0.4, 0.6, 0.8]  # as PLOT_SELECTED_..._RATIO

# ==================================================================
# SMOKE TEST (overrides the values above when True)
# ==================================================================
SMOKE_TEST = False
if SMOKE_TEST:
    RUN_NAME, CLASSES = "smoke", [12]
    DISCOVERY_SAMPLE_SIZE, N0, N, MAX_EVAL_IMAGES = 10, 10, 50, 10
    LABEL_REMAPPED_MAX_EVAL_IMAGES = 3    # s05c: also its full-model baselines run on the first 3 images only
