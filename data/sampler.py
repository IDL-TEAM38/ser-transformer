
import numpy as np, torch
from torch.utils.data import WeightedRandomSampler

def create_samplers(y_train, y_val):
    def _make(y):
        counts = np.bincount(y)
        w = 1.0 / counts
        return WeightedRandomSampler(w[y], num_samples=len(y), replacement=True)
    return _make(y_train), _make(y_val)
