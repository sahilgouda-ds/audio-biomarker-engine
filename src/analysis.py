import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, find_peaks

SR = 4000
DURATION = 5
N_MELS = 64
N_FFT = 512
HOP = 128
ENV_FS = 200          # envelope sampling rate (Hz)


def _load(path):
    y, _ = librosa.load(path, sr=SR)
    return librosa.util.fix_length(y, size=SR * DURATION)


def _envelope(y):
    """Heart-sound energy envelope (Shannon energy), sampled at ENV_FS."""
    sos = butter(3, [25, 400], btype="band", fs=SR, output="sos")
    f = sosfiltfilt(sos, y)
    f = f / (np.max(np.abs(f)) + 1e-9)
    e = -(f ** 2) * np.log(f ** 2 + 1e-9)
    block = SR // ENV_FS
    n = len(e) // block
    e = e[: n * block].reshape(n, block).mean(axis=1)
    e = np.convolve(e, np.ones(8) / 8, mode="same")
    return (e - e.min()) / (e.max() - e.min() + 1e-9)


def _rhythm(env):
    """Estimate heart rate and how regular the repeating pattern is."""
    x = env - env.mean()
    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = acf / (acf[0] + 1e-9)
    lo, hi = int(0.4 * ENV_FS), int(1.6 * ENV_FS)
    lag = lo + int(np.argmax(acf[lo:hi]))
    bpm = 60.0 / (lag / ENV_FS)
    return float(bpm), float(max(acf[lag], 0.0))


def _between_beats(env, peaks):
    """How loud the sound is between beats compared with the beats themselves."""
    mask = np.zeros(len(env), dtype=bool)
    w = int(0.07 * ENV_FS)
    for p in peaks:
        mask[max(0, p - w): p + w + 1] = True
    if mask.any() and (~mask).any():
        return float(env[~mask].mean() / (env[mask].mean() + 1e-9))
    return 0.0


def _bands(y):
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)

    def share(lo, hi):
        return float(spec[(freqs >= lo) & (freqs < hi)].sum())

    low, mid, high = share(20, 100), share(100, 250), share(250, 600)
    total = low + mid + high + 1e-12
    return {"low": low / total, "mid": mid / total, "high": high / total}


def _wave_pairs(y, n=320):
    block = max(1, len(y) // n)
    peak = np.max(np.abs(y)) + 1e-9
    out = []
    for i in range(n):
        seg = y[i * block:(i + 1) * block]
        if len(seg) == 0:
            break
        out.append([round(float(seg.min() / peak), 3), round(float(seg.max() / peak), 3)])
    return out


def compute_metrics(y):
    env = _envelope(y)
    peaks, _ = find_peaks(env, height=0.35, distance=int(0.12 * ENV_FS))
    bpm, regularity = _rhythm(env)
    return {
        "bpm": round(bpm),
        "regularity": round(regularity, 2),
        "between": round(_between_beats(env, peaks), 2),
        "bands": {k: round(v, 3) for k, v in _bands(y).items()},
        "beats": [round(float(p / ENV_FS), 3) for p in peaks],
    }


def build_reasons(label, confidence, m):
    """Plain-language observations. Thresholds are rule-of-thumb, not clinical cut-offs."""
    abnormal = label == "abnormal"
    pct = f"{confidence * 100:.0f}%"
    reasons = []

    reasons.append({
        "tone": "watch" if abnormal else "ok",
        "title": "What the model concluded",
        "text": (
            f"The model labelled this recording {label} with {pct} confidence. It learned this by comparing "
            "the sound pattern of the recording with thousands of labelled heart sound recordings."
        ),
    })

    bpm, reg = m["bpm"], m["regularity"]
    if bpm < 60:
        rate = f"about {bpm} beats per minute, slower than the typical resting adult range of 60 to 100"
    elif bpm <= 100:
        rate = f"about {bpm} beats per minute, within the typical resting adult range of 60 to 100"
    else:
        rate = f"about {bpm} beats per minute, faster than the typical resting adult range of 60 to 100 (children normally run faster)"
    if reg >= 0.5:
        rhythm_tone, rhythm = "ok", "The lub-dub pattern repeats clearly and evenly."
    elif reg >= 0.25:
        rhythm_tone, rhythm = "info", "The beat pattern is somewhat uneven or partly hidden by noise."
    else:
        rhythm_tone, rhythm = "watch", "The beat pattern is hard to pick out. Background noise, movement or an irregular rhythm can cause this."
    reasons.append({
        "tone": rhythm_tone,
        "title": "Heart rate and rhythm",
        "text": f"Estimated rate is {rate}. {rhythm} The rate is estimated from the repeating pattern, so treat it as approximate.",
    })

    q = m["between"]
    if q < 0.2:
        q_tone, q_text = "ok", "The recording is quiet between beats, which is what a clean, normal recording usually looks like."
    elif q < 0.4:
        q_tone, q_text = "info", "There is some sound between beats. This is common and often just background noise."
    else:
        q_tone, q_text = "watch", "There is noticeable sound between beats. Heart murmurs can look like this, but so can breathing, rubbing of the stethoscope or room noise."
    reasons.append({
        "tone": q_tone,
        "title": "Sound between beats",
        "text": f"{q_text} The gap between beats carries about {q * 100:.0f}% of the loudness of the beats themselves.",
    })

    h = m["bands"]["high"]
    if h >= 0.25:
        f_tone, f_text = "watch", "A fairly large share of the energy is high-pitched. Murmurs and noise both add high-pitched sound."
    else:
        f_tone, f_text = "ok", "Most of the energy is low-pitched, which is where normal heart sounds sit."
    reasons.append({
        "tone": f_tone,
        "title": "Pitch of the sound",
        "text": f"{f_text} About {h * 100:.0f}% of the energy is in the high band (250 to 600 Hz).",
    })

    flags = (q >= 0.4) + (reg < 0.25) + (h >= 0.25)
    if abnormal and flags:
        text = "The signal checks above also show unusual features, which fits the model's result."
        tone = "watch"
    elif abnormal:
        text = "The simple signal checks above don't show an obvious marker, so the model is probably reacting to subtler patterns in the spectrogram."
        tone = "info"
    elif flags:
        text = "The model said normal, but some signal checks above look unusual. A cleaner repeat recording would help confirm."
        tone = "info"
    else:
        text = "The signal checks above also look ordinary, which fits the model's result."
        tone = "ok"
    reasons.append({"tone": tone, "title": "Do the checks agree?", "text": text})

    if abnormal:
        nxt = "An abnormal pattern is a reason to have the recording reviewed by a doctor, who can listen with a stethoscope and order tests if needed."
    else:
        nxt = "A normal pattern here doesn't rule out a heart condition. If you have symptoms, see a doctor."
    reasons.append({"tone": "info", "title": "What to do next", "text": nxt})
    return reasons


def analyze_file(path, label, confidence):
    y = _load(path)
    m = compute_metrics(y)

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    spec = np.clip((mel_db + 80.0) / 80.0, 0, 1) * 255
    spec = spec.astype(np.uint8).tolist()

    return {
        "duration": DURATION,
        "wave": _wave_pairs(y),
        "spec": spec,
        "metrics": m,
        "reasons": build_reasons(label, confidence, m),
    }