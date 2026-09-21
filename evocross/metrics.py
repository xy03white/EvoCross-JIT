from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def search_threshold(
    y_true: List[int],
    y_prob: List[float],
    coarse_step: float = 0.05,
    fine_step: float = 0.001,
) -> Tuple[float, float, float, float]:

    if len(y_prob) == 0:
        return 0.5, 0.0, 0.0, 0.0

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    low = np.percentile(y_prob, 2)
    high = np.percentile(y_prob, 98)
    if low >= high:
        low, high = 0.0, 1.0

    best_j = -1.0
    best_th_coarse = 0.5
    thresholds_coarse = np.arange(low, high + coarse_step, coarse_step)
    for th in thresholds_coarse:
        pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_th_coarse = th

    search_low = max(low, best_th_coarse - coarse_step)
    search_high = min(high, best_th_coarse + coarse_step)
    thresholds_fine = np.arange(search_low, search_high + fine_step, fine_step)

    best_f1 = -1.0
    best_th = best_th_coarse
    best_precision = 0.0
    best_recall = 0.0
    for th in thresholds_fine:
        pred = (y_prob >= th).astype(int)
        try:
            f1 = f1_score(y_true, pred, zero_division=0)
            precision = precision_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
        except Exception:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_precision = precision
            best_recall = recall

    return best_th, best_f1, best_precision, best_recall


def compute_metrics(
    y_true: List[int], y_prob: List[float], threshold: float = 0.5
) -> Dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty split")

    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else None,
        "auc_pr": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics
