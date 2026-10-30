import librosa
import numpy as np

SR = 4000        # sample rate
DURATION = 5     # har audio ko 5 second ka banayenge
N_MELS = 64
N_FFT = 512
HOP = 128

def wav_to_mel(path):
    y, _ = librosa.load(path, sr=SR)
    y = librosa.util.fix_length(y, size=SR * DURATION)   # chhota ho toh pad, bada ho toh cut
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)   # normalize
    return mel_db.astype(np.float32)