"""
WESAD (Wearable Stress and Affect Detection) dataset loader
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def load_wesad_subject(subject_id, data_dir='data/raw/WESAD'):
    """
    Load WESAD data for a specific subject.

    Parameters:
    -----------
    subject_id : str or int
        Subject identifier (e.g., 'S2', 'S3', etc.)
    data_dir : str
        Path to WESAD dataset directory

    Returns:
    --------
    dict : Dictionary containing subject data with keys:
        - 'signal': physiological signals
        - 'label': stress labels
        - 'subject': subject identifier
    """
    if isinstance(subject_id, int):
        subject_id = f'S{subject_id}'

    subject_path = Path(data_dir) / subject_id / f'{subject_id}.pkl'

    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data


def extract_chest_signals(data):
    """
    Extract chest sensor signals from WESAD data.

    Parameters:
    -----------
    data : dict
        WESAD subject data

    Returns:
    --------
    pd.DataFrame : DataFrame containing chest sensor signals
    """
    chest = data['signal']['chest']

    # WESAD chest signals are sampled at 700 Hz
    signals_df = pd.DataFrame({
        'ACC_X': chest[:, 0],
        'ACC_Y': chest[:, 1],
        'ACC_Z': chest[:, 2],
        'ECG': chest[:, 3],
        'EMG': chest[:, 4],
        'EDA': chest[:, 5],
        'TEMP': chest[:, 6],
        'RESP': chest[:, 7]
    })

    return signals_df


def extract_wrist_signals(data):
    """
    Extract wrist sensor signals from WESAD data.

    Parameters:
    -----------
    data : dict
        WESAD subject data

    Returns:
    --------
    pd.DataFrame : DataFrame containing wrist sensor signals
    """
    wrist = data['signal']['wrist']

    # WESAD wrist signals
    signals_df = pd.DataFrame({
        'ACC_X': wrist[:, 0],
        'ACC_Y': wrist[:, 1],
        'ACC_Z': wrist[:, 2],
        'BVP': wrist[:, 3],
        'EDA': wrist[:, 4],
        'TEMP': wrist[:, 5]
    })

    return signals_df
