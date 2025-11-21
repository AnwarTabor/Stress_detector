"""
Signal windowing utilities
"""
import numpy as np


def create_windows(data, window_size, overlap=0.5, labels=None):
    """
    Create sliding windows from signal data.

    Parameters:
    -----------
    data : np.ndarray
        Input signal data (samples x features)
    window_size : int
        Size of each window in samples
    overlap : float
        Overlap between windows (0-1)
    labels : np.ndarray, optional
        Labels corresponding to each sample

    Returns:
    --------
    tuple : (windows, window_labels) if labels provided, else just windows
        windows: np.ndarray of shape (n_windows, window_size, n_features)
        window_labels: np.ndarray of shape (n_windows,)
    """
    step_size = int(window_size * (1 - overlap))
    n_samples = len(data)

    windows = []
    window_labels = [] if labels is not None else None

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])

        if labels is not None:
            # Use majority voting for window label
            window_label = np.bincount(labels[start:end]).argmax()
            window_labels.append(window_label)

    windows = np.array(windows)

    if labels is not None:
        window_labels = np.array(window_labels)
        return windows, window_labels

    return windows


def sliding_window_1d(signal, window_size, step_size):
    """
    Create sliding windows from 1D signal.

    Parameters:
    -----------
    signal : np.ndarray
        1D input signal
    window_size : int
        Size of each window
    step_size : int
        Step size between windows

    Returns:
    --------
    np.ndarray : Array of windows
    """
    n_windows = (len(signal) - window_size) // step_size + 1
    windows = np.array([signal[i*step_size:i*step_size + window_size]
                       for i in range(n_windows)])

    return windows
