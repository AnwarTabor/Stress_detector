"""
Accelerometer signal feature extraction
"""
import numpy as np


def extract_acc_features(acc_x, acc_y, acc_z):
    """
    Extract accelerometer features from 3-axis data.

    Parameters:
    -----------
    acc_x : np.ndarray
        X-axis acceleration
    acc_y : np.ndarray
        Y-axis acceleration
    acc_z : np.ndarray
        Z-axis acceleration

    Returns:
    --------
    dict : Dictionary of accelerometer features
    """
    features = {}

    # Magnitude
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    # Statistical features for each axis
    for axis_name, axis_data in [('x', acc_x), ('y', acc_y), ('z', acc_z)]:
        features[f'acc_{axis_name}_mean'] = np.mean(axis_data)
        features[f'acc_{axis_name}_std'] = np.std(axis_data)
        features[f'acc_{axis_name}_range'] = np.max(axis_data) - np.min(axis_data)

    # Magnitude features
    features['acc_mag_mean'] = np.mean(magnitude)
    features['acc_mag_std'] = np.std(magnitude)
    features['acc_mag_max'] = np.max(magnitude)
    features['acc_mag_min'] = np.min(magnitude)

    # Signal Magnitude Area (SMA)
    sma = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / len(acc_x)
    features['acc_sma'] = sma

    # Energy
    features['acc_energy'] = np.sum(magnitude**2) / len(magnitude)

    return features


def calculate_movement_intensity(acc_x, acc_y, acc_z):
    """
    Calculate overall movement intensity.

    Parameters:
    -----------
    acc_x, acc_y, acc_z : np.ndarray
        3-axis acceleration data

    Returns:
    --------
    float : Movement intensity score
    """
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    intensity = np.std(magnitude)

    return intensity
