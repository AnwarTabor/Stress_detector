"""
Signal resampling utilities
"""
import numpy as np
from scipy import signal


def resample_signal(data, original_fs, target_fs):
    """
    Resample signal to target frequency.

    Parameters:
    -----------
    data : np.ndarray
        Input signal
    original_fs : float
        Original sampling frequency (Hz)
    target_fs : float
        Target sampling frequency (Hz)

    Returns:
    --------
    np.ndarray : Resampled signal
    """
    num_samples = int(len(data) * target_fs / original_fs)
    resampled_data = signal.resample(data, num_samples)

    return resampled_data


def synchronize_signals(signals_dict, target_fs=32):
    """
    Synchronize multiple signals to the same sampling rate.

    Parameters:
    -----------
    signals_dict : dict
        Dictionary of signals with their sampling rates
        Format: {'signal_name': (data, fs), ...}
    target_fs : float
        Target sampling frequency (Hz)

    Returns:
    --------
    dict : Dictionary of synchronized signals
    """
    synchronized = {}

    for name, (data, fs) in signals_dict.items():
        if fs != target_fs:
            synchronized[name] = resample_signal(data, fs, target_fs)
        else:
            synchronized[name] = data

    return synchronized
