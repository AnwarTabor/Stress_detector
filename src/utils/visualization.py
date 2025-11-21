"""
Visualization utilities for physiological signals
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_signals(signals_dict, duration=30, fs=700, title="Physiological Signals"):
    """
    Plot multiple physiological signals.

    Parameters:
    -----------
    signals_dict : dict
        Dictionary of signals {name: signal_array}
    duration : float
        Duration to plot in seconds
    fs : float
        Sampling frequency
    title : str
        Plot title
    """
    n_signals = len(signals_dict)
    fig, axes = plt.subplots(n_signals, 1, figsize=(14, 2*n_signals))

    if n_signals == 1:
        axes = [axes]

    samples = int(duration * fs)
    time = np.arange(samples) / fs

    for idx, (name, signal) in enumerate(signals_dict.items()):
        axes[idx].plot(time, signal[:samples])
        axes[idx].set_ylabel(name)
        axes[idx].grid(alpha=0.3)

        if idx == n_signals - 1:
            axes[idx].set_xlabel('Time (s)')

    axes[0].set_title(title)
    plt.tight_layout()
    plt.show()


def plot_signal_comparison(signal1, signal2, labels=None, fs=700, duration=30):
    """
    Compare two signals side by side.

    Parameters:
    -----------
    signal1, signal2 : np.ndarray
        Signals to compare
    labels : tuple, optional
        Labels for signals
    fs : float
        Sampling frequency
    duration : float
        Duration to plot in seconds
    """
    if labels is None:
        labels = ('Signal 1', 'Signal 2')

    samples = int(duration * fs)
    time = np.arange(samples) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

    ax1.plot(time, signal1[:samples])
    ax1.set_ylabel(labels[0])
    ax1.grid(alpha=0.3)

    ax2.plot(time, signal2[:samples])
    ax2.set_ylabel(labels[1])
    ax2.set_xlabel('Time (s)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(features_df, label_col='label', save_path=None):
    """
    Plot distributions of features by class.

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing features and labels
    label_col : str
        Name of label column
    save_path : str, optional
        Path to save the plot
    """
    feature_cols = [col for col in features_df.columns if col != label_col]

    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        for label in features_df[label_col].unique():
            data = features_df[features_df[label_col] == label][feature]
            axes[idx].hist(data, alpha=0.5, bins=30, label=f'Class {label}')

        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
