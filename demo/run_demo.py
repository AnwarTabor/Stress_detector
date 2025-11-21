"""
Demo script for stress detection system
"""
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import project modules
from src.preprocessing.filters import bandpass_filter
from src.preprocessing.windowing import create_windows
from src.features.hrv import extract_hrv_features
from src.features.eda import extract_eda_features
from src.utils.visualization import plot_signals


def run_demo(data_path='sample_data/sample.pkl'):
    """
    Run a demonstration of the stress detection pipeline.

    Parameters:
    -----------
    data_path : str
        Path to sample data file
    """
    print("="*60)
    print("Stress Detection System Demo")
    print("="*60)

    # TODO: Load sample data
    print("\n1. Loading sample data...")
    # data = load_sample_data(data_path)

    # TODO: Preprocess signals
    print("\n2. Preprocessing signals...")
    # preprocessed = preprocess_pipeline(data)

    # TODO: Extract features
    print("\n3. Extracting features...")
    # features = extract_features(preprocessed)

    # TODO: Make prediction
    print("\n4. Making stress prediction...")
    # prediction = predict_stress(features)

    # TODO: Visualize results
    print("\n5. Visualizing results...")
    # plot_results(data, prediction)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == '__main__':
    run_demo()
