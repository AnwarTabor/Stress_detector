"""
Electrodermal Activity (EDA) feature extraction
"""
import numpy as np
from scipy import signal


def extract_eda_features(eda_signal, fs=700):
    """
    Extract EDA features.

    Parameters:
    -----------
    eda_signal : np.ndarray
        EDA/GSR signal
    fs : float
        Sampling frequency (Hz)

    Returns:
    --------
    dict : Dictionary of EDA features
    """
    features = {}

    # Decompose into tonic and phasic components
    tonic, phasic = decompose_eda(eda_signal, fs)

    # Statistical features
    features['eda_mean'] = np.mean(eda_signal)
    features['eda_std'] = np.std(eda_signal)
    features['eda_min'] = np.min(eda_signal)
    features['eda_max'] = np.max(eda_signal)
    features['eda_range'] = features['eda_max'] - features['eda_min']

    # Tonic features (SCL - Skin Conductance Level)
    features['scl_mean'] = np.mean(tonic)
    features['scl_std'] = np.std(tonic)

    # Phasic features (SCR - Skin Conductance Response)
    features['scr_mean'] = np.mean(phasic)
    features['scr_std'] = np.std(phasic)

    # Peak detection in phasic component
    peaks, _ = signal.find_peaks(phasic, height=0.01)
    features['scr_peaks'] = len(peaks)
    features['scr_rate'] = len(peaks) / (len(eda_signal) / fs)  # peaks per second

    return features


def decompose_eda(eda_signal, fs=700, cutoff=0.05):
    """
    Decompose EDA into tonic and phasic components.

    Parameters:
    -----------
    eda_signal : np.ndarray
        Raw EDA signal
    fs : float
        Sampling frequency (Hz)
    cutoff : float
        Cutoff frequency for decomposition (Hz)

    Returns:
    --------
    tuple : (tonic, phasic) components
    """
    # Use low-pass filter for tonic component
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = signal.butter(4, normal_cutoff, btype='low')
    tonic = signal.filtfilt(b, a, eda_signal)

    # Phasic is the residual
    phasic = eda_signal - tonic

    return tonic, phasic
