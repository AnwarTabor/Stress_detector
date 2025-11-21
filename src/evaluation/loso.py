"""
Leave-One-Subject-Out (LOSO) cross-validation
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def leave_one_subject_out(X, y, subjects, train_fn, predict_fn):
    """
    Perform Leave-One-Subject-Out cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    subjects : np.ndarray
        Subject IDs for each sample
    train_fn : callable
        Training function that takes (X_train, y_train) and returns a model
    predict_fn : callable
        Prediction function that takes (model, X_test) and returns predictions

    Returns:
    --------
    dict : Dictionary containing:
        - 'predictions': all predictions
        - 'true_labels': all true labels
        - 'subject_scores': per-subject accuracy scores
    """
    logo = LeaveOneGroupOut()

    all_predictions = []
    all_true_labels = []
    subject_scores = {}

    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_subject = subjects[test_idx][0]

        # Train model
        model = train_fn(X_train, y_train)

        # Make predictions
        predictions = predict_fn(model, X_test)

        # Store results
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)

        # Calculate subject-specific accuracy
        accuracy = np.mean(predictions == y_test)
        subject_scores[test_subject] = accuracy

        print(f"Subject {test_subject}: Accuracy = {accuracy:.4f}")

    return {
        'predictions': np.array(all_predictions),
        'true_labels': np.array(all_true_labels),
        'subject_scores': subject_scores
    }
