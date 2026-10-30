import os
import sys
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.append("src")
from features import wav_to_mel
from model import CNNLSTM
from analysis import analyze_file

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Audio Biomarker Engine")

device = "cpu"
model = CNNLSTM()
model.load_state_dict(torch.load("best.pt", map_location=device))
model.eval()

LABELS = {0: "normal", 1: "abnormal"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    # Windows pe temp file ko dobara kholne ke liye delete=False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        mel = wav_to_mel(tmp_path)
        x = torch.tensor(mel)[None, None]          # (1, 1, 64, T)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax())
        label = LABELS[pred]
        confidence = float(probs[pred])

        try:
            analysis = analyze_file(tmp_path, label, confidence)
        except Exception:
            analysis = None        # report na bane toh bhi result milega
    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't read this audio file. Try another .wav recording.")
    finally:
        os.remove(tmp_path)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "normal": round(float(probs[0]), 4),
            "abnormal": round(float(probs[1]), 4),
        },
        "analysis": analysis,
    }