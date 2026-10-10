"""Segmentation and uncertainty-quality metrics (numpy only, no torch)."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- segmentation
def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    """Rows = ground truth, cols = prediction. Ignored GT pixels are dropped."""
    gt = gt.ravel()
    pred = pred.ravel()
    keep = gt != ignore_index
    gt, pred = gt[keep].astype(np.int64), pred[keep].astype(np.int64)
    return np.bincount(gt * num_classes + pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def segmentation_scores(cm: np.ndarray, class_names: list[str]) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    present = cm.sum(1) > 0  # classes that appear in GT
    out = {
        "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "mIoU": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "mean_dice": float(np.nanmean(dice[present])) if present.any() else float("nan"),
        "per_class": {},
        "confusion_matrix": cm.astype(np.int64).tolist(),
        "class_names": list(class_names),
    }
    for i, n in enumerate(class_names):
        out["per_class"][n] = {
            "IoU": _f(iou[i]), "dice": _f(dice[i]), "precision": _f(precision[i]),
            "recall": _f(recall[i]), "gt_pixels": int(cm[i].sum()),
        }
    return out


def _f(x) -> float | None:
    return None if not np.isfinite(x) else float(x)


# ------------------------------------------------------------------ uncertainty
def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    """ECE over max-probability confidence. Returns ECE and the reliability-diagram bins."""
    confidence = confidence.ravel().astype(np.float64)
    correct = correct.ravel().astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    conf_sum = np.bincount(idx, weights=confidence, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_conf = conf_sum / count
        mean_acc = acc_sum / count
    n = max(count.sum(), 1)
    nz = count > 0
    ece = float(np.sum(count[nz] / n * np.abs(mean_acc[nz] - mean_conf[nz])))
    return {
        "ECE": ece,
        "bin_edges": edges.tolist(),
        "bin_confidence": [_f(v) for v in mean_conf],
        "bin_accuracy": [_f(v) for v in mean_acc],
        "bin_count": count.astype(np.int64).tolist(),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC for `scores` predicting binary `labels` (1 = positive), via rank statistic (ties averaged)."""
    scores = scores.ravel().astype(np.float64)
    labels = labels.ravel().astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    # average ranks for ties
    _, first, counts = np.unique(s, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def risk_coverage(uncertainty: np.ndarray, error: np.ndarray, n_points: int = 50) -> dict:
    """Selective prediction: keep the most-certain `coverage` fraction, report error rate (risk) on it.

    AURC = area under risk-coverage curve (lower is better). Also returns the
    oracle AURC (perfect ranking) and random AURC (= overall error) for reference.
    """
    u = uncertainty.ravel()
    e = error.ravel().astype(np.float64)
    order = np.argsort(u, kind="mergesort")
    cum_err = np.cumsum(e[order])
    n = len(e)
    k = np.arange(1, n + 1)
    risk_full = cum_err / k
    cov_full = k / n
    pick = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    oracle = np.cumsum(np.sort(e)) / k
    return {
        "coverage": cov_full[pick].tolist(),
        "risk": risk_full[pick].tolist(),
        "AURC": float(np.mean(risk_full)),
        "AURC_oracle": float(np.mean(oracle)),
        "AURC_random": float(e.mean()),
    }


def entropy_error_summary(entropy: np.ndarray, error: np.ndarray) -> dict:
    entropy = entropy.ravel()
    error = error.ravel().astype(bool)
    return {
        "mean_entropy_correct": float(entropy[~error].mean()) if (~error).any() else None,
        "mean_entropy_incorrect": float(entropy[error].mean()) if error.any() else None,
        "error_detection_AUROC": auroc(entropy, error),
        "n_pixels": int(entropy.size),
        "error_rate": float(error.mean()),
    }
