import numpy as np

class DataPreprocessor:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean = None
        self.std = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return x
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize or self.mean is None:
            return x
        return (x - self.mean) / (self.std + 1e-8)
