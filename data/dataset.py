
import numpy as np, torch, soundfile as sf, librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """Generic speech‐emotion dataset."""
    def __init__(self, file_paths, labels, max_len=300, augment=False,
                 feature_fn=None, augment_fn=None):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len
        self.augment = augment
        self.feature_fn = feature_fn
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = self.feature_fn(self.file_paths[idx])
        if self.augment and self.augment_fn:
            x = self.augment_fn(x)
        if x.shape[1] > self.max_len:
            start = 0 if not self.augment else np.random.randint(0, x.shape[1]-self.max_len+1)
            x = x[:, start:start+self.max_len]
        else:
            pad = np.zeros((x.shape[0], self.max_len - x.shape[1]))
            x = np.concatenate([x, pad], axis=1)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
