"""
Heart Rate Variability (HRV) feature extraction
"""
import numpy as np
from scipy import signal


def extract_hrv_features(rr_intervals):
    """
    Extract HRV features from RR intervals.

    Parameters:
    -----------
    rr_intervals : np.ndarray
        RR intervals in milliseconds

    Returns:
    --------
    dict : Dictionary of HRV features
    """
    features = {}

    # Time-domain features
    features['mean_rr'] = np.mean(rr_intervals)
    features['std_rr'] = np.std(rr_intervals)
    features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    features['sdnn'] = np.std(rr_intervals)

    # NN50 and pNN50
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
    features['nn50'] = nn50
    features['pnn50'] = (nn50 / len(rr_intervals)) * 100

    # Frequency-domain features (simplified)
    # TODO: Implement proper FFT-based frequency analysis
    features['hr_mean'] = 60000 / np.mean(rr_intervals)  # beats per minute
    features['hr_std'] = np.std(60000 / rr_intervals)

    return features


def detect_r_peaks(ecg_signal, fs=700):
    """
    Detect R peaks in ECG signal.

    Parameters:
    -----------
    ecg_signal : np.ndarray
        ECG signal
    fs : float
        Sampling frequency (Hz)

    Returns:
    --------
    np.ndarray : Indices of R peaks
    """
    # Simple R-peak detection using scipy
    # For production, use more robust methods (e.g., Pan-Tompkins, NeuroKit2)
    peaks, _ = signal.find_peaks(ecg_signal, distance=fs*0.5, height=np.mean(ecg_signal))

    return peaks


def calculate_rr_intervals(r_peaks, fs=700):
    """
    Calculate RR intervals from R peak locations.

    Parameters:
    -----------
    r_peaks : np.ndarray
        Indices of R peaks
    fs : float
        Sampling frequency (Hz)

    Returns:
    --------
    np.ndarray : RR intervals in milliseconds
    """
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to milliseconds

    return rr_intervals
