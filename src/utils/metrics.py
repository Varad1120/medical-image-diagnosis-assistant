"""
Evaluation metrics for binary classification (Normal vs Pneumonia).
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> Dict[str, float]:
    """
    Compute all binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (PNEUMONIA).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
    }
    return metrics


def print_classification_report(y_true, y_pred, class_names=("NORMAL", "PNEUMONIA")):
    """Print a full classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return confusion matrix as numpy array."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(y_true, y_prob):
    """Return FPR, TPR, thresholds and AUC for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score
