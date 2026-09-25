"""
Clinical Evaluation Metrics for Medical Imaging and ASD Classification.
Calculates Accuracy, Sensitivity, Specificity, Precision, F1-Score, and ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Computes a comprehensive dictionary of binary medical diagnosis metrics.
    y_true: array-like of ground truth labels (0: TC, 1: ASD)
    y_pred: array-like of binary predicted class labels
    y_probs: optional array-like of predicted probabilities for positive class (ASD)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / Recall
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix to derive Specificity: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC AUC
    auc = None
    if y_probs is not None:
        try:
            # If probabilities are 2D (N, 2), take column 1 (ASD)
            if hasattr(y_probs, "ndim") and y_probs.ndim == 2:
                y_probs = y_probs[:, 1]
            auc = roc_auc_score(y_true, y_probs)
        except Exception:
            auc = 0.5

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),  # True Positive Rate
        "specificity": float(spec),  # True Negative Rate
        "precision": float(prec),    # Positive Predictive Value
        "f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def print_evaluation_summary(metrics, prefix="Evaluation"):
    """Prints a clean tabular summary of model performance."""
    print(f"\n==================== {prefix} Summary ====================")
    print(f"  Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity: {metrics['sensitivity'] * 100:.2f}%  (Recall for ASD)")
    print(f"  Specificity: {metrics['specificity'] * 100:.2f}%  (Ability to rule out ASD)")
    print(f"  Precision:   {metrics['precision'] * 100:.2f}%")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print("=========================================================\n")
