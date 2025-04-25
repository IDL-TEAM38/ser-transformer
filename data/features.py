
import numpy as np, soundfile as sf, librosa

def extract_features(path, n_mels=128, n_fft=1024, hop_length=256):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.max(np.abs(y)) < 1e-10:
        y = np.random.normal(0, 1e-6, len(y))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for f in (contrast, chroma):
        if f.shape[1] != log_mel.shape[1]:
            f[:] = librosa.util.fix_length(f, size=log_mel.shape[1], axis=1)
    return np.vstack([log_mel, contrast, chroma])

def apply_augmentation(feat, time_ratio=0.2, freq_ratio=0.2, mode='both'):
    if mode in ('time_mask', 'both'):
        t = int(feat.shape[1]*time_ratio)
        start = np.random.randint(0, feat.shape[1]-t+1)
        feat[:, start:start+t] = 0
    if mode in ('freq_mask', 'both'):
        f = int(feat.shape[0]*freq_ratio)
        start = np.random.randint(0, feat.shape[0]-f+1)
        feat[start:start+f, :] = 0
    return feat
