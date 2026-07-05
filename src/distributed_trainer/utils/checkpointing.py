import numpy as np
import os

def save_checkpoint(params: dict, path: str):
    """Safely saves model parameters to disk using NPZ format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **params)

def load_checkpoint(path: str) -> dict:
    """Loads model parameters from disk."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
