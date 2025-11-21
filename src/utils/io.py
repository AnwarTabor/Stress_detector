"""
Input/Output utilities
"""
import pickle
import json
import numpy as np
import h5py


def save_pickle(obj, filepath):
    """
    Save object to pickle file.

    Parameters:
    -----------
    obj : any
        Object to save
    filepath : str
        Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """
    Load object from pickle file.

    Parameters:
    -----------
    filepath : str
        Path to pickle file

    Returns:
    --------
    any : Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data, filepath):
    """
    Save data to JSON file.

    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to JSON file

    Returns:
    --------
    dict : Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_hdf5(data_dict, filepath):
    """
    Save data to HDF5 file.

    Parameters:
    -----------
    data_dict : dict
        Dictionary of arrays to save
    filepath : str
        Path to save file
    """
    with h5py.File(filepath, 'w') as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)


def load_hdf5(filepath, keys=None):
    """
    Load data from HDF5 file.

    Parameters:
    -----------
    filepath : str
        Path to HDF5 file
    keys : list, optional
        Specific keys to load

    Returns:
    --------
    dict : Loaded data
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        if keys is None:
            keys = list(f.keys())

        for key in keys:
            data[key] = f[key][:]

    return data
