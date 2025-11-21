"""
Evaluation metrics for stress detection
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive evaluation metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Prediction probabilities for ROC-AUC

    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # ROC-AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = None

    return metrics


def print_metrics(metrics):
    """
    Pretty print evaluation metrics.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_metrics()
    """
    print("="*50)
    print("Evaluation Metrics")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*50)
