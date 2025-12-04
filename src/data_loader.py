"""
Data Loader for WESAD Dataset
Downloads and prepares the WESAD (Wearable Stress and Affect Detection) dataset
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

class WESADDataLoader:
    """
    Loads and preprocesses WESAD dataset
    Dataset contains physiological signals from wearable sensors
    Labels: 0=not defined, 1=baseline, 2=stress, 3=amusement, 4=meditation
    """
    
    def __init__(self, data_dir='data/raw/WESAD'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subject IDs in WESAD dataset
        self.subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        
        # Signal sampling rates (Hz)
        self.chest_fs = 700  # Chest device sampling rate
        self.wrist_fs = 64   # Wrist device sampling rate
        
    def download_dataset(self):
        """
        Download WESAD dataset from UCI ML Repository
        Note: For actual use, you need to manually download from:
        https://archive.ics.uci.edu/ml/datasets/WESAD+(Wearable+Stress+and+Affect+Detection)
        """
        print("=" * 60)
        print("WESAD Dataset Download Instructions")
        print("=" * 60)
        print("\nPlease download the dataset manually from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/00465/")
        print("\nDownload: WESAD.zip")
        print(f"Extract to: {self.data_dir}")
        print("\nThe dataset structure should be:")
        print(f"{self.data_dir}/S2/S2.pkl")
        print(f"{self.data_dir}/S3/S3.pkl")
        print("... etc for subjects S2-S17 (15 subjects total)")
        print("=" * 60)
        
    def load_subject_data(self, subject_id):
        """
        Load data for a single subject
        
        Args:
            subject_id: Subject ID (2-17, excluding 12)
            
        Returns:
            dict: Contains 'chest' signals, 'wrist' signals, and 'labels'
        """
        subject_path = self.data_dir / f'S{subject_id}' / f'S{subject_id}.pkl'
        
        if not subject_path.exists():
            raise FileNotFoundError(f"Subject data not found at {subject_path}")
            
        with open(subject_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        return data
    
    def extract_features_simple(self, signal, window_size=60, overlap=30):
        """
        Extract simple statistical features from physiological signals
        
        Args:
            signal: 1D numpy array of physiological signal
            window_size: Window size in seconds
            overlap: Overlap in seconds
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        # Calculate window parameters
        step_size = window_size - overlap
        n_samples_window = int(window_size * self.wrist_fs)
        n_samples_step = int(step_size * self.wrist_fs)
        
        for i in range(0, len(signal) - n_samples_window, n_samples_step):
            window = signal[i:i + n_samples_window]
            
            # Statistical features
            features = {
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'median': np.median(window),
                'q25': np.percentile(window, 25),
                'q75': np.percentile(window, 75),
                'range': np.max(window) - np.min(window)
            }
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)
    
    def prepare_dataset(self, use_wrist=True, binary_classification=True):
        """
        Prepare complete dataset for ML models
        
        Args:
            use_wrist: Use wrist device data (simpler, fewer sensors)
            binary_classification: Binary (stress vs non-stress) or multi-class
            
        Returns:
            X: Feature matrix
            y: Labels
            subject_ids: Subject ID for each sample (for LOSO CV)
        """
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        print("Loading WESAD dataset...")
        
        for subject_id in tqdm(self.subject_ids, desc="Processing subjects"):
            try:
                data = self.load_subject_data(subject_id)
                
                if use_wrist:
                    # Use wrist device data (Empatica E4)
                    # Contains: BVP, EDA, TEMP (simpler than chest)
                    signal_data = data['signal']['wrist']
                    eda = signal_data['EDA']  # Electrodermal activity
                    bvp = signal_data['BVP']  # Blood volume pulse
                    temp = signal_data['TEMP']  # Temperature
                    
                    # Labels
                    labels = data['label']
                    
                    # Downsample labels to match wrist sampling rate
                    labels_downsampled = labels[::int(self.chest_fs/self.wrist_fs)]
                    
                    # Ensure same length
                    min_len = min(len(eda), len(bvp), len(temp), len(labels_downsampled))
                    eda = eda[:min_len]
                    bvp = bvp[:min_len]
                    temp = temp[:min_len]
                    labels_downsampled = labels_downsampled[:min_len]
                    
                    # Extract features in windows
                    window_size = 60  # 60 seconds
                    overlap = 30  # 30 seconds overlap
                    n_samples_window = int(window_size * self.wrist_fs)
                    n_samples_step = int((window_size - overlap) * self.wrist_fs)
                    
                    for i in range(0, min_len - n_samples_window, n_samples_step):
                        # Get window
                        eda_window = eda[i:i + n_samples_window]
                        bvp_window = bvp[i:i + n_samples_window]
                        temp_window = temp[i:i + n_samples_window]
                        label_window = labels_downsampled[i:i + n_samples_window]
                        
                        # Get most common label in window
                        label = np.bincount(label_window.astype(int)).argmax()
                        
                        # Skip undefined labels (0)
                        if label == 0:
                            continue
                        
                        # Convert labels to binary BEFORE filtering
                        # 1=baseline, 2=stress, 3=amusement, 4=meditation
                        if binary_classification:
                            # Binary: stress (2) vs non-stress (1,3,4)
                            binary_label = 1 if label == 2 else 0
                        else:
                            binary_label = label
                        
                        # Extract features
                        features = {}
                        
                        # EDA features
                        features['eda_mean'] = np.mean(eda_window)
                        features['eda_std'] = np.std(eda_window)
                        features['eda_min'] = np.min(eda_window)
                        features['eda_max'] = np.max(eda_window)
                        features['eda_range'] = np.max(eda_window) - np.min(eda_window)
                        
                        # BVP features
                        features['bvp_mean'] = np.mean(bvp_window)
                        features['bvp_std'] = np.std(bvp_window)
                        features['bvp_min'] = np.min(bvp_window)
                        features['bvp_max'] = np.max(bvp_window)
                        
                        # TEMP features
                        features['temp_mean'] = np.mean(temp_window)
                        features['temp_std'] = np.std(temp_window)
                        
                        all_features.append(features)
                        all_labels.append(binary_label)
                        all_subject_ids.append(subject_id)
                        
            except FileNotFoundError:
                print(f"Warning: Subject {subject_id} data not found, skipping...")
                continue
        
        # Convert to arrays
        X = pd.DataFrame(all_features).values
        y = np.array(all_labels)
        subject_ids = np.array(all_subject_ids)
        
        print(f"\nDataset prepared!")
        print(f"Total samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y, subject_ids
    
    def create_sample_dataset(self):
        """
        Create a small sample dataset for demo purposes
        (in case full dataset is not available)
        """
        print("Creating sample dataset for demo...")
        
        np.random.seed(42)
        n_samples = 500
        n_features = 11  # Same as real feature count
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Add some patterns for stress vs non-stress
        # Stress samples (label=1) have higher EDA and BVP values
        stress_mask = np.random.rand(n_samples) > 0.6
        X[stress_mask, 0:5] += np.random.randn(stress_mask.sum(), 5) * 0.5 + 1.0  # Higher EDA
        X[stress_mask, 5:9] += np.random.randn(stress_mask.sum(), 4) * 0.3 + 0.5  # Higher BVP
        
        y = stress_mask.astype(int)
        subject_ids = np.random.randint(2, 18, size=n_samples)
        
        print(f"Sample dataset created: {n_samples} samples")
        print(f"Features: {n_features}")
        print(f"Stress samples: {stress_mask.sum()}, Non-stress: {(~stress_mask).sum()}")
        
        return X, y, subject_ids


if __name__ == "__main__":
    # Example usage
    loader = WESADDataLoader()
    
    # Show download instructions
    loader.download_dataset()
    
    # Try to load real data, fallback to sample data
    try:
        X, y, subject_ids = loader.prepare_dataset()
        
        # Save processed data
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'X.npy', X)
        np.save(output_dir / 'y.npy', y)
        np.save(output_dir / 'subject_ids.npy', subject_ids)
        
        print(f"\nData saved to {output_dir}")
        
    except FileNotFoundError:
        print("\nReal data not available, creating sample dataset...")
        X, y, subject_ids = loader.create_sample_dataset()
        
        # Save sample data
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'X_sample.npy', X)
        np.save(output_dir / 'y_sample.npy', y)
        np.save(output_dir / 'subject_ids_sample.npy', subject_ids)
        
        print(f"\nSample data saved to {output_dir}")