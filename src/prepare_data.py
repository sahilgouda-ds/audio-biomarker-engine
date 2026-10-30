import os
import numpy as np
import pandas as pd
from features import wav_to_mel

ROOT = "data"
FOLDERS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
OUT_DIR = f"{ROOT}/processed"
os.makedirs(OUT_DIR, exist_ok=True)

X, y, src = [], [], []
for folder in FOLDERS:
    ref_path = f"{ROOT}/{folder}/REFERENCE.csv"
    if not os.path.exists(ref_path):
        print(f"{folder}: nahi mila, skip")
        continue

    labels = pd.read_csv(ref_path, header=None, names=["file", "label"])
    done = 0
    for _, row in labels.iterrows():
        try:
            X.append(wav_to_mel(f"{ROOT}/{folder}/{row['file']}.wav"))
        except Exception as e:
            print("Skip:", row["file"], e)
            continue
        y.append(0 if row["label"] == -1 else 1)
        src.append(folder[-1])
        done += 1
    print(f"{folder}: {done} files done")

X = np.stack(X)
y = np.array(y)
src = np.array(src)

np.save(f"{OUT_DIR}/X.npy", X)
np.save(f"{OUT_DIR}/y.npy", y)
np.save(f"{OUT_DIR}/src.npy", src)

print("X shape:", X.shape)
print("Normal:", (y == 0).sum(), "| Abnormal:", (y == 1).sum())