# Certified Subnetwork Extraction for Semantic Segmentation

Code for the paper *Certified Subnetwork Extraction for Semantic Segmentation* by Marcus Gullstrand, Florian Westphal and Maria Riveiro.

We extend the certified-circuits framework of Anani et al. from classification to semantic segmentation. For each of the 20 PASCAL VOC object classes, we extract a class subnetwork from DeepLabV3-ResNet50 that is provably stable under edits to its discovery images. Channels are selected globally across network stages, and pruning-induced logit shifts are counteracted with class-wise logit standardization. 

For results and figures from our experiment runs, download the data this Zenodo repository and put it into the project root folder: [doi.org/10.5281/zenodo.24471710](https://doi.org/10.5281/zenodo.24471710).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24471710.svg)](https://doi.org/10.5281/zenodo.24471710)

## References

- A. Anani, T. Lorenz, B. Schiele, M. Fritz, and J. Fischer. "Certified Circuits: Stability Guarantees for Mechanistic Circuits." In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, PMLR vol. 306, pp. 2650–2676, 2026. https://proceedings.mlr.press/v306/anani26a.html

- C. Hamblin, T. Konkle, and G. Alvarez. "Pruning for Feature-Preserving Circuits in CNNs." arXiv preprint arXiv:2206.01627, 2023. https://arxiv.org/abs/2206.01627

- S. Jung, J. Lee, D. Gwak, S. Choi, and J. Choo. "Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation." In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 15405–15414, 2021. https://doi.org/10.1109/ICCV48922.2021.01514

We took inspiration for a few extra experiments from the following:

- M. Gullstrand, F. Westphal, and M. Riveiro. "Subnetwork Analysis for Neural Network Interpretability – A Survey and Experimental Study." Preprint, SSRN, 2026. https://doi.org/10.2139/ssrn.6942838

# Experiment

## Model

We use torchvision's pretrained `deeplabv3_resnet50` with the `COCO_WITH_VOC_LABELS_V1` weights. Its ResNet-50 backbone is pretrained on ImageNet. The full model is then trained for segmentation on COCO train2017, keeping only the 20 PASCAL VOC categories (the other 60 COCO categories count as background). The model has 21 outputs: background plus the 20 VOC classes. The auxiliary classifier is removed.

The components that can be pruned are the output channels of the four backbone stages (`layer1`–`layer4`: 256, 512, 1,024 and 2,048 channels, 3,840 in total). The decoder (ASPP head and classifier) is always kept whole.

## Data setup

We discover subnetworks on one dataset and evaluate them on four:

| Dataset | Images | Role | Description |
|---|---|---|---|
| VOC+SBD train | 10,582 | Discovery | The VOC 2012 training images together with the SBD training and validation images, with the VOC 2012 validation images removed. Also known as "VOC aug" or "SegmentationClassAug". |
| VOC 2012 val | 1,449 | Evaluation | Comes from the same distribution as the discovery data. |
| COCO val2017 | 5,000 | Evaluation | The held-out split of the dataset the model was trained on. |
| ADE20K val | 2,000 | Evaluation | Indoor and outdoor scenes, different from both the model's training data and the discovery data. 13 of the 20 classes can be evaluated. |
| Cityscapes val | 500 | Evaluation | Urban driving scenes, different from both the model's training data and the discovery data. 6 of the 20 classes can be evaluated. |

ADE20K and Cityscapes test whether a subnetwork captures the class itself rather than its discovery data, e.g. whether the person subnetwork found on VOC still segments persons in ADE20K.

### Discovery set per class

An image is eligible for a class if its ground truth contains at least 500 pixels of that class. Every class gets the same number of discovery images: 262, the number of eligible images for the least frequent class (cow).

### Label remapping

COCO, ADE20K and Cityscapes do not share VOC's 21 classes. Each of their classes is mapped to its closest VOC class, ambiguous classes (e.g. truck and van) are ignored, and the remaining classes become background. Only the subnetworks whose class has a counterpart in the dataset are evaluated. The mapping tables are in `src/common/datasets.py`.

### Metric

Each class subnetwork is scored by its IoU on its own class (cIoU). The counts are pooled over all evaluation images whose ground truth contains that class.

### No overlap

COCO and VOC+SBD share no images, and the VOC 2012 validation images are removed from the discovery set.

## Segmentation Adaptation

The certification of Anani et al. is kept unchanged. Each run of the discovery algorithm sees a random sub-dataset of the discovery set, in which every image is deleted independently with probability *p*<sub>del</sub>. The more frequent outcome over *n*<sub>0</sub> selection sub-datasets gives each channel its hypothesis (in or out of the subnetwork). Agreement with that hypothesis is then tested on *n* fresh certification sub-datasets. A channel is certified if the lower confidence bound on its agreement rate exceeds the threshold τ, and abstains otherwise. Only certified-in channels form the subnetwork. Every certified decision holds for any discovery set within edit distance *r* = ⌊log(1.5 − τ) / log *p*<sub>del</sub>⌋.

Moving from one prediction per image to one per pixel needs three changes:

1. **Channel scoring for segmentation.** Channels are scored with ActGrad (Hamblin et al.). Since the model outputs a logit map rather than one logit per class, the target is the class-*c* logit averaged over the pixels the full model predicts as *c*. Scores are summed over the images of each sub-dataset.
2. **Global top-K.** The top-K fraction of all 3,840 channels is kept in one pool of raw scores across stages, as in Hamblin et al., instead of Anani et al.'s per-layer top-K. The remaining channels are zeroed without fine-tuning. This lets the budget vary per stage, so early stages, whose removals propagate to every output pixel, need not be pruned as heavily as late ones.
3. **Class-wise logit standardization.** Zeroing channels shifts each class's logits by a different amount, and since every pixel is an argmax over all 21 classes, these shifts can relabel pixels even where the evidence for *c* is kept. Adapting Jung et al., we compute for each class *d* the mean and standard deviation of the class-*d* logit over the discovery-set pixels the full model predicts as *d*, for both the full model and the subnetwork. The subnetwork's logits are then mapped onto the full model's statistics before the argmax. This is applied after certification, without fine-tuning. Classes the full model rarely predicts keep their raw logits.

Settings used in the paper:

| Parameter | Value |
|---|---|
| Deletion probability *p*<sub>del</sub> | 0.6 |
| Selection / certification sub-datasets *n*<sub>0</sub> / *n* | 100 / 1,000 |
| Confidence level α | 0.001 (Bonferroni-corrected over all channels) |
| Agreement threshold τ | 0.95 (gives certified radius *r* = 1) |
| Kept fraction K | 0.1, 0.2, …, 1.0 (K = 1.0 is the full model) |


# Project Specifics

## Project structure

```
.
├── src/
│   ├── config.py                     All parameters of the pipeline
│   ├── run_all.py                    Runs the stages listed in it one after another, in one job
│   ├── setup_check_and_baseline.py   Checks the datasets and the model (optional, before the pipeline)
│   ├── s01_ … s07_*.py               The pipeline stages (see "Experiment Stages")
│   ├── Extra_Material/               Scripts and figures beyond the pipeline, incl. the paper's figures (see "More Plots")
│   └── common/                       Shared code
│       ├── datasets.py               Dataset loaders and the label-remapping tables
│       ├── model.py                  Model loading, preprocessing, activation hooks and channel masking
│       ├── logit_standardization.py  The class-wise logit standardization
│       ├── metrics.py                IoU and the other evaluation metrics
│       ├── circuit_size.py           Subnetwork size measures
│       └── utils.py                  Paths, logging and run bookkeeping
├── download_scripts/                 Download scripts for COCO, ADE20K and Cityscapes
├── RUN_ME.sh                         Entry point: runs LOCAL_JOB.sh, or submits a cluster job script if one exists
├── LOCAL_JOB.sh                      Runs one stage (or run_all.py) and writes the log
├── requirements.txt                  Python dependencies, with the versions used for the paper
└── LICENSE
```

The following folders are created when the code runs and are not part of the repository:

- `data/`: the datasets (see "Data layout").
- `results/<RUN_NAME>/`: the output of every stage (discovery lists, score cache, sampled and certified subnetworks, evaluation tables). Each stage writes into a subfolder whose name contains its key parameters, so runs with different settings do not overwrite each other.
- `images/`: the figures from s06, s06c and s07, in the same folder structure as `results/`.
- `LOCAL/`: the run logs.

## Quick setup

1. **Install the dependencies** (Python with PyTorch and torchvision; a GPU is strongly recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the datasets.** For VOC 2012 and SBD, set `CHECK_DOWNLOAD = True` in `src/config.py` and run the setup check once. COCO, ADE20K and Cityscapes have their own scripts (Cityscapes needs an account):

   ```bash
   python src/setup_check_and_baseline.py
   bash download_scripts/coco_download.sh
   bash download_scripts/ade20k_download.sh
   bash download_scripts/cityscapes_download.sh
   ```

   The COCO and Cityscapes scripts download the full releases, so they can also be used for other work: for COCO the train2017, val2017 and test2017 images with the instance, stuff and test-info annotations, and for Cityscapes all three splits (about 11 GB). The paper only uses the validation splits (for COCO, `images/val2017/` and `annotations/instances_val2017.json`). COCO train2017 is only read when discovering on it (`DISCOVERY_DATASET = "coco_train2017_voc"`).

3. **Run the pipeline.** First change to the Python environment where the dependencies are installed (`LOCAL_JOB.sh` runs whichever `python3` is on the PATH). Then, from the repository root:

   ```bash
   bash RUN_ME.sh
   ```

   This runs `src/run_all.py`, which runs every stage listed in `STAGES` at the top of the file, each as its own process. `src/config.py` is read once at the start, so editing it during a run does not affect the running job. The log is written to `LOCAL/logs/`, and the job stops at the first stage that fails.

The defaults in `src/config.py` reproduce the global top-K results of the paper. For the per-layer comparison, set `TOPK_SELECTION_MODE = "per_layer"` and run s03 onward again. The outputs go to separate folders.

## Data layout

The datasets are kept in `data/` (set by `DATA_DIR` in `src/config.py`). The code expects this layout:

```
data/
├── voc/VOCdevkit/VOC2012/              PASCAL VOC 2012 (JPEGImages/, SegmentationClass/, ImageSets/Segmentation/)
├── sbd/                                SBD (img/, cls/, train.txt, val.txt)
├── coco/
│   ├── images/val2017/                 COCO val2017 images
│   └── annotations/instances_val2017.json
├── ade20k/ADEChallengeData2016/
│   ├── images/validation/              ADE20K val images
│   ├── annotations/validation/         ADE20K val labels (150 classes)
│   └── objectInfo150.txt               ADE20K class names
└── cityscapes/
    ├── leftImg8bit/val/<city>/         Cityscapes val images
    └── gtFine/val/<city>/              Cityscapes val labels (*_gtFine_labelIds.png)
```

| Dataset | Name in `src/config.py` | How to get it | Needed for |
|---|---|---|---|
| PASCAL VOC 2012 | `voc_val`, and part of `voc_sbd_train` | Set `CHECK_DOWNLOAD = True` and run `src/setup_check_and_baseline.py`. torchvision then downloads [`VOCtrainval_11-May-2012.tar`](https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). | The whole pipeline |
| SBD | part of `voc_sbd_train` | Downloaded in the same run as VOC. torchvision downloads [`benchmark.tgz`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz). | The whole pipeline |
| COCO 2017 | `coco_val2017_voc` | `bash download_scripts/coco_download.sh` | s05c |
| ADE20K (SceneParsing) | `ade20k_val_voc` | `bash download_scripts/ade20k_download.sh` | s05c |
| Cityscapes (fine annotations) | `cityscapes_val_voc` | `bash download_scripts/cityscapes_download.sh` (needs a Cityscapes account) | s05c |

To download VOC and SBD by hand instead: extract the VOC archive into `data/voc/`, and move `cls/`, `img/`, `train.txt` and `val.txt` from `benchmark_RELEASE/dataset/` in the SBD archive into `data/sbd/`.

Notes:

- **Nothing is preprocessed or written to `data/`.** VOC+SBD train is assembled when it is loaded: VOC 2012 train, SBD train and SBD val, each image once, minus the VOC 2012 val images. For images that are in both VOC and SBD, the VOC annotation is used. COCO masks are built from the instance annotations with `pycocotools`, as in torchvision's reference scripts: only the 20 VOC categories are kept, and pixels covered by more than one instance are void. ADE20K and Cityscapes labels are remapped to the VOC classes when they are loaded.
- **Another location.** To keep the datasets elsewhere, change `DATA_DIR` or make `data/` a symbolic link. The download scripts always write into `data/` in the repository root.
- **Missing datasets.** A stage that needs a dataset that is not on disk stops before it does any work and says how to get the dataset. The setup check skips missing datasets.

## Experiment Stages

The pipeline is split into stages, with one script for each in `src/`. Each stage reads the outputs of earlier stages from `results/` and writes its own (see "Running the Pipeline"). In the code and its outputs, a subnetwork is called a *circuit*, as in Anani et al. (e.g. `baseline_circuit`), and the discovery set of class *c* is called `D_c` or `discovery_samples`.

| Stage | Script | What it does | In `run_all.py` by default |
|---|---|---|---|
| Setup check | `setup_check_and_baseline.py` | Checks the datasets and the model; optional full-model baseline | No |
| s01 | `s01_discovery_samples_dataset.py` | Draws the discovery set of every class and lists the evaluation images | Yes |
| s02 | `s02_score_cache.py` | Computes the ActGrad score of every channel on every discovery image | Yes |
| s03 | `s03_sample_circuits.py` | Runs the discovery algorithm on the random sub-datasets | Yes |
| s04 | `s04_certify.py` | Certifies every channel as in, out or abstain | Yes |
| s04b | `s04b_logit_statistics.py` | Computes the logit statistics for the class-wise logit standardization | Yes |
| s05 | `s05_evaluate.py` | Evaluates the subnetworks on VOC val and on the discovery images | Yes |
| s05b | `s05b_evaluate_merged_circuits.py` | Evaluates unions of two class subnetworks (not used in the paper) | No |
| s05c | `s05c_evaluate_label_remapped_datasets.py` | Evaluates the subnetworks on COCO val2017, ADE20K and Cityscapes | Yes |
| s06 | `s06_plots.py` | Figures and tables from s05 | Yes |
| s06c | `s06c_plots_label_remapped_datasets.py` | Figures and tables from s05c | Yes |
| s07 | `s07_predictions.py` | Example predictions of the subnetworks, as figures | Yes |

**Setup check** (optional). With `CHECK_DOWNLOAD = True`, it first downloads VOC 2012 and SBD if they are missing. For every dataset in `CHECK_DATASETS` that is on disk, it prints the number of images and the ground-truth pixels per class, and saves one example prediction of the full model. With `CHECK_LABEL_REMAPPING`, it prints which VOC class every COCO, ADE20K and Cityscapes class is mapped to, with pixel counts, and saves example figures of the original and remapped labels. With `RUN_BASELINE`, it evaluates the full model on `BASELINE_DATASETS`. Output: `images/setup_check/` and `results/baseline/`.

**s01 · Discovery sets and evaluation lists.** For every class *c*, it draws the discovery set D_c from `DISCOVERY_DATASET`: `DISCOVERY_SAMPLE_SIZE` images among those with at least `MIN_PIXELS` ground-truth pixels of *c* (262 per class with the paper's settings). For every dataset in `EVAL_DATASETS`, it also lists the images whose ground truth contains *c*. These are the images the class-*c* subnetwork is evaluated on. The evaluation set `discovery_samples` is D_c itself, i.e. an in-sample evaluation on the discovery images. Output: `discovery_and_eval_lists.json`.

**s02 · Score cache.** For every class *c* and every image in D_c, one forward pass records the activations, and one backward pass per target gives the ActGrad score of every channel (Hamblin et al., Eq. 5). The paper uses the target `pred`: the class-*c* logit averaged over the pixels the full model predicts as *c*. The targets `gt` (the ground-truth pixels of *c*) and `all` (every pixel) are cached as well. Since the scores are stored per image, s03 can score any sub-dataset by summing them, without running the model again (as in Anani et al., App. D). If the full model predicts no pixel of *c* in an image, that image's `pred` scores are zero. Such images are kept and listed in the log. The cache also holds the 512 decoder channels (4,352 channels in total), but with `LAYERS = "backbone"` only the 3,840 backbone channels are certified. Output: `s02_scores/class_XX.pt`.

**s03 · Sampled subnetworks.** For every class, it draws *n*<sub>0</sub> + *n* = 1,100 sub-datasets of D_c, deleting each image independently with probability `P_DEL`. For each sub-dataset and each K in `TOPK_LAYER_NORMALIZED_RATIO_GRID`, it sums the cached scores over the sub-dataset's images and keeps the top-K channels. With `TOPK_SELECTION_MODE = "global_raw"` (the paper), this is one top-K over all 3,840 backbone channels. With `"per_layer"` (Anani et al.), each stage keeps round(K · C) of its C channels. Channels with exactly equal scores are put in random order. The same selection on the whole D_c gives the baseline subnetwork. Output: the channels selected on every sub-dataset and how often each channel was selected (`class_XX.npz`).

**s04 · Certification.** For every class, K and backbone channel, the channel's hypothesis (in or out) is its more frequent outcome over the *n*<sub>0</sub> selection sub-datasets. A one-sided binomial test on how many of the *n* certification sub-datasets agree with it decides whether the channel is certified (p-value ≤ α / 3,840) or abstains. The certified-in channels form the certified subnetwork. s04 also builds the majority-vote subnetwork (the channels selected in more than half of the *n* certification sub-datasets, without a test) and computes the size of all three subnetworks. The log prints the certified radius and how many agreeing sub-datasets a channel needs (*r* = 1 and 982 of 1,000 with the paper's settings). Output: `class_XX.npz`, with the decision for every channel.

**s04b · Logit statistics.** For every class *c*, it runs the full model and every distinct subnetwork of *c* (baseline, majority-vote and certified, at every K) on D_c, without gradients. For each of the 21 classes *d*, it stores the mean and standard deviation of the class-*d* logit over three pixel sets: the pixels the full model predicts as *d* (used in the paper), the pixels the model itself predicts as *d* (as in Jung et al.), and all pixels. A class with fewer than `LOGIT_STATISTICS_MIN_PIXELS` pixels keeps its raw logits. The class-wise logit standardization in s05, s05c and s07 uses these statistics; the certification does not. Output: `class_XX.npz`.

**s05 · Evaluation.** For every dataset in `EVAL_DATASETS` (VOC val and the discovery images by default), class *c* and K, it runs the full model and the baseline, majority-vote and certified subnetworks of *c* on the evaluation images of *c*, with the channels outside the subnetwork zeroed. Every metric is computed for the raw subnetwork and, from the same forward pass, after each variant in `LOGIT_STANDARDIZATION_VARIANTS`. The pixels are pooled over the images of *c*. The metrics are:

- accuracy against the ground truth: cIoU (the paper's metric), the 21-class mIoU and others;
- faithfulness to the full model: faith (class-*c* IoU against the full model's prediction), |Pearson r| of the class-*c* logit as in Hamblin et al., and the share of pixels labelled the same as by the full model;
- logit distributions: the shift in mean and standard deviation of every class's logit relative to the full model;
- size: the share of channels kept per stage and in total, and the share of weights kept.

With `EVALUATE_IMAGES_WITHOUT_CLASS = True`, it also evaluates every image without *c*, which shows the false positives of *c*. This is off by default, as it takes about 12 times as many image passes on VOC val. Output: `summary.csv` (one row per class, dataset, subnetwork, variant and K), one JSON file per class, and the confusion matrices behind every metric, so that new metrics can be computed without running s05 again.

**s05b · Merged subnetworks** (optional, not used in the paper). A subnetwork evaluated only on its own class can look better than it is. For every pair of classes (190 pairs), s05b therefore evaluates the union of the two class subnetworks on the images of both classes, and the union of all class subnetworks on all their images. Raw subnetworks only. It runs up to about 19 times as many forward passes as s05.

**s05c · Label-remapped datasets.** It evaluates the subnetworks from s04 (discovered on VOC+SBD, unchanged) on COCO val2017, ADE20K and Cityscapes, with their ground truth remapped to the VOC classes. It measures accuracy only: cIoU, the mIoU over background and the classes the dataset can score, and the share of pixels predicted as *c*. Only the classes with a counterpart in the dataset are evaluated, and images whose remapped ground truth has no object class are left out. To make the numbers comparable, it also evaluates the full model on VOC val with the mIoU over the same classes. The logit-standardization statistics come from s04b, i.e. from VOC+SBD, and are applied unchanged. Output: one subfolder per dataset, each saved as soon as that dataset is done.

**s06 · Figures for s05.** It draws figures and tables from the output of s05 (and of s04, and s05b if it has run) into `images/`, under the same path as the s05 folder. They are grouped into `1_accuracy_vs_ground_truth/`, `2_faithfulness_to_full_model/`, `3_logit_distributions/` and `4_size_and_structure/`, and `curves_mean.csv` holds the numbers behind every curve. Means are over the classes, each class weighted equally. The full layout is in the docstring of `src/s06_plots.py`. The x-axis of the curves is the layer-normalized size, not the size used in the paper's figures (see "Known Caveats").

**s06c · Figures for s05c.** The same kind of figures for the datasets of s05c, with VOC val on the same classes next to each dataset (taken from s05's VOC val results). Output: `overview_curves`, `reference_table` and one folder per dataset.

**s07 · Prediction examples.** For every class *c*, it takes the image at each position in `PREDICT_POSITIONS` of the evaluation list of *c* in `PREDICT_DATASET`. It draws the predictions of the full model and of the subnetworks at every K in three figures: one with all 21 classes, one with class *c* against the rest, and, with `PREDICT_CONFIDENCE_MAPS`, one with Jung et al.'s standardized max logit as a confidence map. This is a visual check, not a metric.

## Running the Pipeline

### All stages or one at a time

`bash RUN_ME.sh` runs `LOCAL_JOB.sh`, which runs the script set by `python_file` in it.

- **All stages in one job.** Keep `python_file="run_all.py"` (the default), and choose the stages by commenting lines in or out of `STAGES` at the top of `src/run_all.py`. Each stage runs as its own process, `src/config.py` is read once at the start, and the job stops at the first stage that fails. The log ends with a table of every stage's outcome and run time.
- **One stage.** Set `python_file` in `LOCAL_JOB.sh` to the stage's script (each stage has a commented-out line there), and run `bash RUN_ME.sh`. A stage can also be run directly, e.g. `python src/s04_certify.py`; its output then goes to the terminal only.

Each run through `LOCAL_JOB.sh` gets a job number. Its output is written to `LOCAL/logs/log_output_<job>.txt` and appended to `LOCAL/log_all.txt`. Each stage prints the parameters it uses when it starts, and saves them in `run_info.json` next to its outputs, together with the job number, git commit and library versions.

### Which stages build on which

```
s01 ─► s02 ─► s03 ─► s04 ─┬───────────────────────► s05b
                          │
                          └─► s04b ─┬─► s05 ──────► s06
                                    ├─► s05c ─────► s06c
                                    └─► s07
```

| Stage | Needs the output of |
|---|---|
| s01 | The datasets only |
| s02 | s01 |
| s03 | s02 |
| s04 | s03 |
| s04b | s04 |
| s05 | s04 and s01's evaluation lists; s04b if `LOGIT_STANDARDIZATION_VARIANTS` is non-empty |
| s05b | s04 and s01's evaluation lists |
| s05c | s04; s04b if `LABEL_REMAPPED_LOGIT_STANDARDIZATION_VARIANTS` is non-empty. It does not need s05. |
| s06 | s05 and s04; s05b if it has run |
| s06c | s05c; for the VOC val columns, s05 on `voc_val` with the same `MAX_EVAL_IMAGES` |
| s07 | s04 and s01's evaluation lists; s04b if `PREDICT_LOGIT_STANDARDIZATION_VARIANTS` is non-empty or `PREDICT_CONFIDENCE_MAPS = True` |

s04b is only needed for the logit standardization and the confidence maps, which are on by default. Without them, s05, s05c and s07 read the s04 output directly. A stage whose input is missing stops with `[missing input]` and names the stage to run first.

s02, s04b, s05, s05b and s05c run the model over many images, s07 on a few images only. s01, s03, s04, s06 and s06c do not run the model (s01 only with `REQUIRE_PRED_IN_DISCOVERY = True`).

### Where the outputs go

Each stage writes into a folder inside the folder of the stage it builds on, named after the parameters it adds. With the paper's settings:

```
results/main/
└── voc_sbd_train_Dbottleneck262_min500_DiscAndEvalSeed0/                               s01
    └── s02_scores/                                                                     s02
        └── s03_pred_actgrad_pdel0.60_n100-1000_sseed0_topk_global_raw_backbone/        s03
            └── s04_backbone_tau0.95_a0.001/                                            s04
                ├── s05b_merged_circuits_eval_CapAll/                                   s05b (optional)
                └── s04b_logit_statistics_discovery_samples_minpx1000/                  s04b
                    ├── s05_eval_CapAll/                                                s05
                    └── s05c_label_remapped_eval_CapAll/                                s05c
```

The figures go to `images/` under the same path: s06 under the s05 folder, s06c under the s05c folder and s07 under the s04b folder. With the logit standardization and the confidence maps turned off, s05, s05c and s07 write into the s04 folder instead. With `TOPK_SELECTION_MODE = "per_layer"`, the s03 folder name has no `_topk_global_raw_backbone` suffix, so the two modes do not overwrite each other. The folder names are built by the `*_dir()` functions in `src/common/utils.py`.

### Re-running after a change

A parameter that is part of a folder name creates new folders when it changes, so nothing is overwritten. Only the stage that reads the parameter and the stages after it need to run again; the earlier outputs are reused. Some examples:

| Change | Parameters | Run again |
|---|---|---|
| Per-layer instead of global top-K (the paper's comparison) | `TOPK_SELECTION_MODE = "per_layer"` | s03 onward |
| Sub-datasets, K grid or channel scoring | `P_DEL`, `N0`, `N`, `TOPK_LAYER_NORMALIZED_RATIO_GRID`, `TARGET`, `CHANNEL_SCORING_METHOD`, `SAMPLING_SEED` | s03 onward |
| Certification threshold or confidence level | `AGREEMENT_THRESHOLD_TAU`, `FAILURE_PROB_ALPHA` | s04 onward |
| Discovery sets | `DISCOVERY_DATASET`, `DISCOVERY_SAMPLE_SIZE`, `MIN_PIXELS`, `REQUIRE_PRED_IN_DISCOVERY`, `DISCOVERY_AND_EVAL_SEED` | s01 onward |
| Logit statistics | `LOGIT_STATISTICS_IMAGES`, `LOGIT_STATISTICS_MIN_PIXELS` | s04b onward |
| Evaluation images per class | `MAX_EVAL_IMAGES` | s01 (fast, and the discovery sets stay the same), then s05 onward |
| Figures only | `PLOT_*`, `PREDICT_*`, `LABEL_REMAPPED_PLOT_*` | s06, s06c and s07 |

Parameters that are not part of a folder name, such as `EVALUATE_IMAGES_WITHOUT_CLASS`, the choice of logit-standardization variants and the plot settings, overwrite the earlier outputs of the stage that reads them.

The stages also guard against mixing outputs from different settings:

- s01 does not overwrite an existing discovery set with a different one. If the newly drawn D_c of a class differs from the stored one, it stops without writing.
- s05, s05b and s07 stop if s01's evaluation lists were drawn with a different `MAX_EVAL_IMAGES` than the one in `src/config.py`.
- s05, s05c and s07 stop if s04 was run again after s04b, since the statistics then belong to other subnetworks. Run s04b again first.

### Notes

- **Without COCO, ADE20K or Cityscapes.** s05c stops when one of the datasets in `LABEL_REMAPPED_EVAL_DATASETS` is missing. Since s05c and s06c come last in `STAGES`, every other stage has run by then. To avoid the failure, comment out s05c and s06c, or remove the missing datasets from `LABEL_REMAPPED_EVAL_DATASETS`.
- **Smoke test.** `SMOKE_TEST = True` runs the pipeline for one class (dog) with 10 discovery images, *n*<sub>0</sub> = 10, *n* = 50 and at most 10 evaluation images per dataset (3 in s05c, whose full-model references also use only the first 3 images), into `results/smoke/`. It only checks that the code runs: with *n* = 50 no channel can be certified (at τ = 0.95, with the Bonferroni correction over 3,840 channels, this needs *n* ≥ 296), so the certified subnetwork is empty. s05b needs at least two classes and stops under the smoke test.
- **Editing `src/config.py` during a run.** `run_all.py` reads it once at the start, so the running job is not affected. A stage started on its own reads the file when it starts.

## Parameter Values

These are the values used for the results in the paper. All of them are set in `src/config.py`, where each parameter has a short comment.

Images are used at their native resolution, without resizing or augmentation, and normalized with the ImageNet mean and standard deviation.

### Discovery data

| Parameter | Value | In `src/config.py` |
|---|---|---|
| Discovery dataset | VOC+SBD train | `DISCOVERY_DATASET = "voc_sbd_train"` |
| Classes | The 20 VOC object classes (background is not a subnetwork class) | `CLASSES = list(range(1, 21))` |
| Minimum ground-truth pixels of a class for an image to be eligible | 500 | `MIN_PIXELS = 500` |
| Discovery images per class | 262, the fewest eligible images of any class (cow), used for every class | `DISCOVERY_SAMPLE_SIZE = "bottleneck"` |
| Require the full model to predict the class in a discovery image | No | `REQUIRE_PRED_IN_DISCOVERY = False` |
| Seed for drawing the discovery images | 0 | `DISCOVERY_AND_EVAL_SEED = 0` |

### Channel scoring and selection

| Parameter | Value | In `src/config.py` |
|---|---|---|
| Prunable channels | Output channels of the four backbone stages (3,840) | `LAYERS = "backbone"` |
| Scoring criterion | ActGrad | `CHANNEL_SCORING_METHOD = "actgrad"` |
| Scoring target | The class logit averaged over the pixels the full model predicts as that class | `TARGET = "pred"` |
| Top-K selection | Global, over the raw scores of all backbone channels | `TOPK_SELECTION_MODE = "global_raw"` (`"per_layer"` for the per-layer comparison) |
| Kept fractions K | 0.1, 0.2, …, 1.0 | `TOPK_LAYER_NORMALIZED_RATIO_GRID` |
| Seed for the sub-dataset draws | 0 | `SAMPLING_SEED = 0` |

### Certification

| Parameter | Value | In `src/config.py` |
|---|---|---|
| Deletion probability *p*<sub>del</sub> | 0.6 | `P_DEL = 0.6` |
| Selection sub-datasets *n*<sub>0</sub> | 100 | `N0` |
| Certification sub-datasets *n* | 1,000 | `N` |
| Agreement threshold τ | 0.95 | `AGREEMENT_THRESHOLD_TAU = 0.95` |
| Confidence level α | 0.001, Bonferroni-corrected over the 3,840 channels (≈ 2.6·10⁻⁷ per channel) | `FAILURE_PROB_ALPHA = 0.001` |

With these values, the certified radius is *r* = 1: no certified decision changes if any single image is added to or removed from the discovery set. A channel needs at least 982 of the 1,000 certification sub-datasets to agree with its hypothesis to be certified. Each sub-dataset holds on average about 105 of the 262 discovery images.

### Logit standardization

| Parameter | Value | In `src/config.py` |
|---|---|---|
| Variant | The subnetwork's logits are mapped onto the full model's per-class mean and standard deviation, measured on the pixels the full model predicts as each class | `LOGIT_STANDARDIZATION_VARIANTS = ["matched_to_full_model_on_full_model_prediction_pixels"]` |
| Images the statistics are computed on | The discovery images of the subnetwork's class | `LOGIT_STATISTICS_IMAGES = "discovery_samples"` |
| Minimum pixels for a class's statistics to be used | 1,000 (classes with fewer keep their raw logits) | `LOGIT_STATISTICS_MIN_PIXELS = 1000` |

### Evaluation

| Parameter | Value | In `src/config.py` |
|---|---|---|
| Subnetworks evaluated | The full model, the baseline subnetwork (found on the whole discovery set), the majority-vote subnetwork and the certified subnetwork | `CIRCUIT_TYPES` |
| Evaluation images per class | Every image whose ground truth contains the class | `MAX_EVAL_IMAGES = "all"`, `LABEL_REMAPPED_MAX_EVAL_IMAGES = "all"` |
| Cityscapes resolution | Native (2048 × 1024) | `LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE = None` |

# Extra Material

## Experiment Hardware

All results were produced on a SLURM GPU server with an Intel Xeon Gold 6526Y CPU and NVIDIA L4 GPU (24 GB). The software was Python 3.12.16 and PyTorch 2.14.1 (CUDA 13.0 build). Every stage saves the host and GPU it ran on in its `run_info.json`.

## Extended testing

Beyond what the paper reports, the subnetworks were tested in several other ways. Most of these results are among the figures on Zenodo. Where it applies, each test covers both the raw and the logit-standardized subnetworks.

- **Baseline and majority-vote subnetworks.** Every evaluation also covers the baseline subnetwork (selected once on the whole discovery set, without certification) and the majority-vote subnetwork (the channels selected in more than half of the certification sub-datasets, without the binomial test), next to the certified one. Comparing the three separates the effect of the certification from that of the channel selection itself.
- **Faithfulness to the full model.** Besides accuracy against the ground truth, each subnetwork is compared with the full model's own output: the class-*c* IoU against the full model's prediction, the share of pixels labelled the same as by the full model, and |Pearson r| of the class-*c* logit across images (the preservation measure of Hamblin et al.). |r| ignores scale and offset and stayed high even for an empty subnetwork in a small test, so it only means something together with the other two.
- **The rest of the image.** cIoU only asks whether a pixel is *c* or not, so a class-*c* subnetwork can erase or relabel the other objects in an image and still score well. The evaluation therefore also computes 21-class confusion matrices against the ground truth and against the full model. From these come the 21-class mIoU and how far the share of each predicted class moves away from the full model's.
- **False positives.** cIoU is computed only on the images that contain *c*, so predicting *c* where it does not belong goes unnoticed. For per-layer top-K, the subnetworks were also evaluated on every VOC val image without *c*. This measures the share of pixels and of images predicted as *c*, and which ground-truth classes those false positives cover.
- **The discovery images.** The subnetworks were also evaluated on their own discovery images, as in Anani et al.'s in-distribution evaluation. These results are in-sample twice over, since both the channel selection and the logit-standardization statistics come from these images.
- **Logit distributions.** For every class, how far the subnetwork moves each of the 21 logits away from the full model: the shift in mean, the ratio of standard deviations and the Wasserstein-1 distance, before and after standardization. The paper's Fig. 3 shows one K; the figures cover all of them.
- **Subnetwork size.** Size is also measured as the share of channels kept in each stage, the share of all model weights kept, and the share of the weights tied to a prunable channel. An empty subnetwork still keeps about 49 % of all weights, because the decoder and the weights not tied to a prunable channel are never removed.
- **Overlap between classes.** For per-layer top-K, the channel sets of every pair of class subnetworks were compared (IoU and overlap coefficient) at every K. This shows how much of a subnetwork is shared with other classes and how much is specific to its own.
- **Score scale across stages.** Raw ActGrad scores were compared across the four backbone stages. A typical `layer1` or `layer2` channel scores about 17 times as high as a `layer4` channel, which is why global top-K keeps most early-stage channels and few late ones (see "More Plots").
- **Confidence maps.** Next to the prediction examples, s07 draws Jung et al.'s standardized max logit as written (only the winning logit, so the label does not change) as a per-pixel confidence map, for the full model and for the subnetworks. This is a visual check only.

## More Plots

`src/Extra_Material/` holds scripts and figures that are not part of the pipeline: the scripts that made the paper's figures, further figures, and an analysis of how ActGrad scores scale across the backbone stages. Each script reads the pipeline's output from `results/` and writes its figures next to itself; the figures made from the paper's runs are included. The docstring of each script describes its figure and its inputs. Run a script after the pipeline, e.g.:

```bash
python src/Extra_Material/Extra_images/paper_ready_figures/make_fig1_split_voc_global_vs_layer_normalized_and_all_datasets.py
```

Most scripts read the run with the paper's settings (global top-K). The ones that compare with per-layer top-K also need the per-layer run (`TOPK_SELECTION_MODE = "per_layer"`, s03 onward), and a few read only the per-layer run.

The figure numbers in the script names differ from the paper's:

| Paper | Script in `Extra_images/paper_ready_figures/` |
|---|---|
| Fig. 1 | `make_fig2_person_bicycle_prediction_examples_all_classes_transposed.py` |
| Fig. 2 | `make_fig1_split_voc_global_vs_layer_normalized_and_all_datasets.py` |
| Fig. 3 | `make_fig3_global_topk_logit_distributions_K0.5.py` (its `_by_median_shift` output) |

The rest of the folder:

| Path in `src/Extra_Material/` | Shows | Reads |
|---|---|---|
| `Extra_images/paper_ready_figures/make_fig2_cat_dog_prediction_examples_all_classes_transposed.py` | The paper's Fig. 1 for the cat and dog subnetworks, on a VOC val image with a cat and a dog | Both runs |
| `Extra_images/make_global_topk_images.py` | cIoU and mIoU of the certified subnetworks, raw and standardized, per class and averaged | The global run |
| `Extra_images/make_per_layer_vs_global_topk_images.py` | The same, per-layer against global top-K | Both runs |
| `Extra_images/make_extra_images.py` | Accuracy and faithfulness curves of all three subnetwork types, raw and standardized, including the images without the class | The per-layer run, with s05 run with `MAX_EVAL_IMAGES = 200` and `EVALUATE_IMAGES_WITHOUT_CLASS = True` |
| `Extra_images/make_accuracy_and_overlap.py` | The accuracy of the certified subnetworks next to the overlap between the class subnetworks | As `make_extra_images.py` |
| `Extra_images/make_overlap_matrices.py`, `make_overlap_curves.py` | The overlap (IoU and overlap coefficient) of the channels of every pair of class subnetworks, in `subnetwork_overlap/` | The per-layer run (s04) |
| `actgrad_layer_scale_comparison/` | Whether raw ActGrad scores are comparable across stages, and how much of each stage a global top-K keeps. Its tables are in `actgrad_layer_scale_comparison_report.txt`. | s02, and s03 of the per-layer run |

## Known Caveats

### What the certificate covers

Certification makes the channel selection stable: with confidence 1 − α, no certified decision changes if one discovery image is added or removed (*r* = 1). It says nothing about how well the subnetwork predicts, which is why the subnetwork is evaluated separately. The logit standardization is not covered either. Its statistics (each class's logit mean and standard deviation) are measured on the same discovery images, but they are not certified. A standardized subnetwork is therefore certified in which channels it keeps, not in how its logits are adjusted. The statistics come from VOC+SBD and are applied unchanged to COCO, ADE20K and Cityscapes, and nothing checks that they fit there. The paper traces the drop of standardized Cityscapes at K = 0.5 to this.

### Global top-K can empty a stage

Raw ActGrad scores are about 17 times larger in `layer1` and `layer2` than in `layer4`, so global top-K keeps most early-stage channels and few late ones. Nothing stops it from keeping no channel of a stage. A removed channel is set to zero in the stage output, which is the only path to the next stage (DeepLabV3 has no skip connection around a stage). An empty stage therefore makes the output independent of the image. BatchNorm stays in eval mode and nothing is re-normalized. In the score analysis (`src/Extra_Material/actgrad_layer_scale_comparison/`), the baseline subnetwork lost a whole stage only when fewer than about 3 % of the backbone channels were kept (7.6 % for one class), so it is not expected at K ≥ 0.1. Single sub-datasets can still be thinner. s03, s04 and s06 report per class which stages lose channels and whether a subnetwork has an empty stage.

### Subnetwork size

In the paper (Fig. 2), subnetwork size is the share of the 3,840 backbone channels kept (`subnet_size_group_pooled_ratio`). The s06 and s06c figures and the s07 panel titles instead show the layer-normalized size (`subnet_size_layer_normalized_ratio_mean`): the share kept in each stage, averaged over the four stages with equal weight. The two differ strongly under global top-K, which keeps most of the early-stage channels and few of the late ones. For example, at K = 0.2 the baseline subnetwork keeps about 78, 88, 10 and 1 % of `layer1`–`layer4`. That is 20 % of all channels, but a layer-normalized size of about 0.44. The s06 figures therefore do not look like the paper's Fig. 2.

The names in the code are older than the paper's terms: what the code calls "layer-normalized", the paper calls "per-layer" (e.g. `global_vs_layer_normalized` in the name of the Fig. 2 script compares global and per-layer top-K). With global top-K, the settings and columns named `TOPK_LAYER_NORMALIZED_RATIO…` or `topk_layer_normalized_ratio` hold the share of all 3,840 backbone channels kept, i.e. K as the paper uses it. The size measures `subnet_size_layer_normalized_ratio_…` hold the per-stage shares described above in both modes.

### Discovery images where the model misses the class

The scoring target is the class-*c* logit averaged over the pixels the full model predicts as *c*. If the full model predicts no pixel of *c* in a discovery image, that image's scores are all zero and it adds nothing to the summed scores. Such images are kept, so the effective discovery set of a class can be smaller than 262. With the paper's settings this applies to 389 of the 5,240 discovery images (20 classes × 262). s02 lists every such image in its log.

### Annotation styles differ

VOC masks have a void band around object borders, which every metric leaves out. SBD masks (most of VOC+SBD train) and COCO masks have no such band. On SBD images, the 500-pixel eligibility count therefore includes the border pixels. The setup check's full-model numbers on VOC+SBD train are also not directly comparable with those on VOC val.

### Logit standardization on near-empty subnetworks

At small K, a subnetwork can keep almost no variation in some logits. The matched standardization divides by the subnetwork's own standard deviation, so it stretches whatever is left (about 40–100× for an empty subnetwork in a test). In the s07 figures this can show up as thin bands of other classes along the image borders, where padding gives the only variation left. This is the method acting on a degenerate subnetwork, not a bug.

### Label-remapped datasets

- Images whose remapped ground truth holds no object class are left out completely, the full model included: 969 of the 5,000 COCO images, 710 of the 2,000 ADE20K images and 7 of the 500 Cityscapes images.
- The mIoU on these datasets is over background and the classes the dataset can score. It cannot be compared across datasets or with the 21-class mIoU on VOC val, so s05c and s06c show VOC val over the same classes next to each dataset.
- Predictions of classes the dataset cannot score are kept and count as errors, e.g. a dog predicted on a Cityscapes road.
- Only images that contain *c* are evaluated, so false positives are not measured on these datasets.
- Some mappings have a large effect. ADE20K "plant" includes shrubs and hedges, so it is ignored, while ADE20K pot, vase and flower become pottedplant. ADE20K has a single "animal" class, which is ignored, so none of the six VOC animal classes can be evaluated there. Cityscapes rider becomes person, since VOC labels riders as person. Trucks are ignored in both datasets, since the model's COCO training labels count trucks as background.
