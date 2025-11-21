"""
Respiration signal feature extraction
"""
import numpy as np
from scipy import signal


def extract_resp_features(resp_signal, fs=700):
    """
    Extract respiration features.

    Parameters:
    -----------
    resp_signal : np.ndarray
        Respiration signal
    fs : float
        Sampling frequency (Hz)

    Returns:
    --------
    dict : Dictionary of respiration features
    """
    features = {}

    # Statistical features
    features['resp_mean'] = np.mean(resp_signal)
    features['resp_std'] = np.std(resp_signal)

    # Breathing rate
    peaks, _ = signal.find_peaks(resp_signal, distance=fs*2)  # Minimum 2 seconds between breaths
    breathing_rate = len(peaks) / (len(resp_signal) / fs) * 60  # breaths per minute
    features['breathing_rate'] = breathing_rate

    # Breath amplitude
    if len(peaks) > 1:
        breath_amplitudes = resp_signal[peaks]
        features['breath_amp_mean'] = np.mean(breath_amplitudes)
        features['breath_amp_std'] = np.std(breath_amplitudes)
    else:
        features['breath_amp_mean'] = 0
        features['breath_amp_std'] = 0

    # Breathing variability
    if len(peaks) > 1:
        breath_intervals = np.diff(peaks) / fs
        features['breath_interval_std'] = np.std(breath_intervals)
    else:
        features['breath_interval_std'] = 0

    return features
