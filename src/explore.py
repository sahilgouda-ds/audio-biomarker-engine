import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/training-a"

# Labels padho (header nahi hota, 2 columns: filename, label)
labels = pd.read_csv(f"{DATA_DIR}/REFERENCE.csv", header=None, names=["file", "label"])
print(labels.head())
print("\nTotal files:", len(labels))
print(labels["label"].value_counts())   # -1 = normal, 1 = abnormal

# Ek normal aur ek abnormal file uthao
normal_file = labels[labels["label"] == -1].iloc[0]["file"]
abnormal_file = labels[labels["label"] == 1].iloc[0]["file"]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for row, (name, title) in enumerate([(normal_file, "Normal"), (abnormal_file, "Abnormal")]):
    y, sr = librosa.load(f"{DATA_DIR}/{name}.wav", sr=4000)
    print(f"{title}: {name}.wav, sample rate={sr}, duration={len(y)/sr:.1f}s")

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[row][0])
    axes[row][0].set_title(f"{title} waveform ({name})")

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=512, hop_length=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=128, x_axis="time", y_axis="mel", ax=axes[row][1])
    axes[row][1].set_title(f"{title} mel-spectrogram")

plt.tight_layout()
plt.show()