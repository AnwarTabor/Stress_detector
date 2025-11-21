"""
Training script for baseline (classical ML) models
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle


def train_baseline_model(X_train, y_train, model_type='rf', **kwargs):
    """
    Train a baseline machine learning model.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    model_type : str
        Type of model ('rf', 'svm', etc.)
    **kwargs : dict
        Additional arguments for the model

    Returns:
    --------
    tuple : (trained_model, scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize model
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            random_state=42
        )
    elif model_type == 'svm':
        model = SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def save_model(model, scaler, filepath):
    """
    Save trained model and scaler.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    scaler : sklearn scaler
        Fitted scaler
    filepath : str
        Path to save the model
    """
    model_data = {
        'model': model,
        'scaler': scaler
    }

    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model(filepath):
    """
    Load trained model and scaler.

    Parameters:
    -----------
    filepath : str
        Path to saved model

    Returns:
    --------
    tuple : (model, scaler)
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    return model_data['model'], model_data['scaler']
