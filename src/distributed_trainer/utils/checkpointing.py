import numpy as np
import os
import pickle

def save_checkpoint(path: str, params: dict, optimizer_state: dict, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'params': params, 'optimizer_state': optimizer_state, 'step': step}, f)

def load_checkpoint(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
