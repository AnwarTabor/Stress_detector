"""
Signal filtering utilities for physiological signals
"""
import numpy as np
from scipy import signal


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply bandpass filter to signal.

    Parameters:
    -----------
    data : np.ndarray
        Input signal
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order

    Returns:
    --------
    np.ndarray : Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply lowpass filter to signal.

    Parameters:
    -----------
    data : np.ndarray
        Input signal
    cutoff : float
        Cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order

    Returns:
    --------
    np.ndarray : Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def smooth_eda(eda_signal, window_size=10):
    """
    Smooth EDA signal using moving average.

    Parameters:
    -----------
    eda_signal : np.ndarray
        Raw EDA signal
    window_size : int
        Size of smoothing window

    Returns:
    --------
    np.ndarray : Smoothed EDA signal
    """
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(eda_signal, kernel, mode='same')

    return smoothed


def remove_baseline_drift(signal_data, fs, cutoff=0.5):
    """
    Remove baseline drift from signal using high-pass filter.

    Parameters:
    -----------
    signal_data : np.ndarray
        Input signal
    fs : float
        Sampling frequency (Hz)
    cutoff : float
        High-pass cutoff frequency (Hz)

    Returns:
    --------
    np.ndarray : Signal with baseline removed
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = signal.butter(4, normal_cutoff, btype='high')
    filtered_data = signal.filtfilt(b, a, signal_data)

    return filtered_data
