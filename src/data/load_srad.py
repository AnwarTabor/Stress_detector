"""
SRAD (Stress Recognition in Automobile Drivers) dataset loader
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_srad_subject(subject_id, data_dir='data/raw/SRAD'):
    """
    Load SRAD data for a specific subject.

    Parameters:
    -----------
    subject_id : str or int
        Subject identifier
    data_dir : str
        Path to SRAD dataset directory

    Returns:
    --------
    dict : Dictionary containing subject data
    """
    # TODO: Implement based on SRAD data format
    raise NotImplementedError("SRAD loader to be implemented based on dataset format")


def extract_signals(data):
    """
    Extract physiological signals from SRAD data.

    Parameters:
    -----------
    data : dict
        SRAD subject data

    Returns:
    --------
    pd.DataFrame : DataFrame containing signals
    """
    # TODO: Implement based on SRAD data format
    raise NotImplementedError("SRAD signal extraction to be implemented")
