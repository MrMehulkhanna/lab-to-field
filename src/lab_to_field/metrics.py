"""Evaluation metrics.

Accuracy alone hides problems, so every model is scored with:
- accuracy          : fraction of images classified correctly
- macro-F1          : F1 per class, then averaged, so rare classes count as much as common ones
- balanced accuracy : recall per class, then averaged
- ECE               : Expected Calibration Error, the gap between how confident the
                      model *says* it is and how often it is actually right
"""
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

# A model can predict a class that has no images in a small test set; that is expected.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Split predictions into confidence bins and average |accuracy - confidence| per bin.

    A perfectly calibrated model that says "80% sure" is right 80% of the time (ECE = 0).
    """
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Per-bin (mean confidence, accuracy, count) for drawing a reliability diagram."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    return np.array(rows)


def summarize(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    """All headline numbers for one model on one test set."""
    pred = probs.argmax(1)
    present = np.unique(labels)  # classes that actually appear in this test set
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, labels=present, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(labels, pred)),
        "mean_confidence": float(probs.max(1).mean()),
        "ece": expected_calibration_error(probs, labels),
    }


def confusion(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(labels, probs.argmax(1), labels=list(range(n_classes)))


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, metric="accuracy", n_boot: int = 1000, seed: int = 0):
    """95% confidence interval by resampling the test set with replacement.

    Small test sets (PlantDoc has ~230 images) give noisy numbers; the interval shows how noisy.
    """
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if metric == "accuracy":
            vals.append((pred[idx] == labels[idx]).mean())
        else:
            vals.append(f1_score(labels[idx], pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
