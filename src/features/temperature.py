"""
Temperature signal feature extraction
"""
import numpy as np


def extract_temp_features(temp_signal):
    """
    Extract temperature features.

    Parameters:
    -----------
    temp_signal : np.ndarray
        Temperature signal

    Returns:
    --------
    dict : Dictionary of temperature features
    """
    features = {}

    # Statistical features
    features['temp_mean'] = np.mean(temp_signal)
    features['temp_std'] = np.std(temp_signal)
    features['temp_min'] = np.min(temp_signal)
    features['temp_max'] = np.max(temp_signal)
    features['temp_range'] = features['temp_max'] - features['temp_min']

    # Trend features
    features['temp_slope'] = calculate_trend(temp_signal)

    # Change rate
    temp_diff = np.diff(temp_signal)
    features['temp_change_mean'] = np.mean(temp_diff)
    features['temp_change_std'] = np.std(temp_diff)

    return features


def calculate_trend(signal_data):
    """
    Calculate linear trend (slope) of signal.

    Parameters:
    -----------
    signal_data : np.ndarray
        Input signal

    Returns:
    --------
    float : Slope of linear fit
    """
    x = np.arange(len(signal_data))
    coefficients = np.polyfit(x, signal_data, 1)
    slope = coefficients[0]

    return slope
