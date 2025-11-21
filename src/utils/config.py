"""
Configuration management
"""
import yaml
from pathlib import Path


class Config:
    """
    Configuration manager for stress detection project.
    """

    def __init__(self, config_path=None):
        """
        Initialize configuration.

        Parameters:
        -----------
        config_path : str, optional
            Path to YAML config file
        """
        self.config = self._load_default_config()

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)

    def _load_default_config(self):
        """Load default configuration."""
        return {
            'data': {
                'raw_dir': 'data/raw',
                'interim_dir': 'data/interim',
                'processed_dir': 'data/processed',
                'datasets': ['WESAD', 'SRAD', 'SWELL']
            },
            'preprocessing': {
                'target_fs': 32,  # Target sampling frequency
                'window_size': 60,  # Window size in seconds
                'overlap': 0.5  # Window overlap
            },
            'models': {
                'baseline': {
                    'rf': {
                        'n_estimators': 100,
                        'max_depth': None
                    },
                    'svm': {
                        'C': 1.0,
                        'kernel': 'rbf'
                    }
                },
                'deep_learning': {
                    'batch_size': 32,
                    'epochs': 50,
                    'learning_rate': 0.001
                }
            },
            'evaluation': {
                'cv_method': 'loso',  # Leave-One-Subject-Out
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            }
        }

    def get(self, key, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def save(self, filepath):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
