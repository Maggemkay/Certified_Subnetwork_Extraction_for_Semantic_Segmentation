"""Confusion matrix accumulation and per-class IoU / precision / recall, plus the multi-class metrics
that follow from a confusion matrix."""

import numpy as np
import torch

from common.datasets import NUM_CLASSES, VOID_LABEL


def confusion_matrix(pred, gt, num_classes=NUM_CLASSES):
    """[num_classes, num_classes] pixel counts, rows = GT, columns = prediction.
    Void (255) and out-of-range GT pixels are ignored."""
    pred = np.asarray(pred, dtype=np.int64).ravel()
    gt = np.asarray(gt, dtype=np.int64).ravel()
    valid = (gt != VOID_LABEL) & (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[valid] + pred[valid]
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def confusion_matrix_torch(pred, reference, num_classes=NUM_CLASSES):
    """As confusion_matrix, for torch tensors on any device (on the GPU, a prediction need not be copied to the CPU):
    [num_classes, num_classes] int64 tensor of pixel counts, rows = reference (the GT, or the full model's prediction),
    columns = prediction. Reference pixels outside 0..num_classes-1 (void = 255) are ignored."""
    pred = pred.reshape(-1).long()
    reference = reference.reshape(-1).long()
    valid = (reference >= 0) & (reference < num_classes)
    idx = num_classes * reference[valid] + pred[valid]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def class_scores(cm):
    """Per-class IoU, precision and recall from a confusion matrix (nan where undefined)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_sum = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    union = gt_sum + pred_sum - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "iou": np.where(union > 0, tp / union, np.nan),
            "precision": np.where(pred_sum > 0, tp / pred_sum, np.nan),
            "recall": np.where(gt_sum > 0, tp / gt_sum, np.nan),
        }


# ------------------------------------------------------------------
# Multi-class metrics. cm is a [num_classes, num_classes] confusion matrix, rows =
# reference, columns = prediction; every metric is nan for an empty matrix (no pixels).
# ------------------------------------------------------------------
def mean_iou(cm, classes=None):
    """mIoU: the mean over every class (background included) of the IoU, or with `classes` over those classes only
    (e.g. background and the classes a label-remapped dataset can score); classes with an undefined IoU (neither in the
    reference nor predicted) are left out."""
    iou = class_scores(cm)["iou"]
    if classes is not None:
        iou = iou[list(classes)]
    return float(np.nanmean(iou)) if not np.isnan(iou).all() else np.nan


def pixel_agreement(cm):
    """Share of pixels whose predicted class equals the reference class: trace / total. For the circuit vs the full
    model this is pixel_agreement_with_full_model (1 = an identical segmentation)."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else np.nan


def class_share_total_variation(cm):
    """½ Σ_d |s_d^prediction − s_d^reference|, with s_d the share of pixels of class d: the column sums (prediction)
    and the row sums (reference) of cm over its total. For the circuit vs the full model this is
    class_share_total_variation_to_full_model: 0 = the same class mix, 1 = no class in common. It compares the class
    MIX only, not where the pixels are."""
    total = cm.sum()
    if total == 0:
        return np.nan
    return float(0.5 * np.abs(cm.sum(axis=0) - cm.sum(axis=1)).sum() / total)


def predicted_class_share(cm, c):
    """Share of the matrix's pixels predicted as class c: column c over the total. On images whose GT has no c (vs-GT
    matrix), every such pixel is a false positive of c."""
    total = cm.sum()
    return float(cm[:, c].sum() / total) if total > 0 else np.nan
