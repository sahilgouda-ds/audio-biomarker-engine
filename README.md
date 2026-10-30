# Heart Sound Analyzer

A deep learning web app that listens to a heart sound recording and says whether the pattern looks **normal** or **abnormal**, then shows the spectrogram, the detected beats and plain-language notes about why.

> **Live demo:** _add your Render link here_ (the first load can take about a minute because the free server sleeps when idle)

> **Not a medical device.** This is a learning project. Its output is not a diagnosis.

![Upload screen](docs/ui-upload.png)
![Result](docs/ui-result.png)
![Charts and reasons](docs/ui-charts.png)

## Demo video

_Add your video link here_

## What it does

1. You upload a `.wav` heart sound recording.
2. The audio is converted into a mel-spectrogram (an image of sound over time).
3. A CNN finds frequency patterns in the spectrogram and a bidirectional LSTM follows how they change over time.
4. The app returns the label, the confidence and the probability of each class.
5. A report shows the waveform with detected beats, the spectrogram, the energy per pitch band and short notes on heart rate, rhythm and sound between beats.

The notes in the report come from simple signal measurements with rule-of-thumb limits. They describe the audio and do not explain the model's inner workings.

## Model

| Part | Details |
|---|---|
| Features | Librosa mel-spectrogram, 64 mel bands, 5 second clips at 4 kHz |
| Network | 2 convolution blocks (BatchNorm, ReLU, MaxPool) followed by a bidirectional LSTM (128 units) and a linear layer |
| Regularisation | Dropout 0.3, class-weighted loss for the imbalanced classes |
| Training | PyTorch, Adam (lr 1e-3), learning-rate scheduler, early stopping on validation F1 |
| Serving | FastAPI with a single-page web UI |

## Dataset

[PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/) heart sound recordings, training sets a to f: 3,240 recordings (2,575 normal, 665 abnormal). The data is not stored in this repository.

## Results

Random stratified split (70% train, 15% validation, 15% test), best checkpoint chosen on validation F1:

| Metric | Test set (486 recordings) |
|---|---|
| Accuracy | 92.6% |
| Macro F1 | 0.891 |
| ROC-AUC | 0.977 |
| Abnormal recall | 0.879 |
| Abnormal precision | 0.784 |

Held-out source test (one whole recording folder kept out of training): _results to be added_

## Limitations

- The dataset has no patient IDs, so recordings from the same patient may appear in both the training and test splits. The random-split numbers above can be optimistic.
- The recordings come from different hospitals and devices, and the model can partly learn the device rather than the condition. That is why a held-out folder test is reported separately.
- Only the first 5 seconds of a recording are analysed.
- The heart rate and rhythm notes are estimates from a simple method and are not clinical measurements.

## Run it locally

```bash
git clone https://github.com/sahilgouda-ds/audio-biomarker-engine.git
cd audio-biomarker-engine
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

Open http://127.0.0.1:8000 and drop in a `.wav` file. The trained model (`best.pt`) is included.

## Retrain the model

1. Download the training sets from PhysioNet and place the folders as `data/training-a`, `data/training-b` and so on.
2. Install the extra training libraries: `pip install pandas scikit-learn matplotlib`
3. Build the features: `python src/prepare_data.py`
4. Train: `python src/train.py`
5. Optional source-generalisation check: `python src/eval_heldout.py`

## Project structure

```
app/
  main.py          FastAPI app and prediction endpoint
  index.html       Web UI
src/
  features.py      Mel-spectrogram extraction
  prepare_data.py  Builds the processed dataset
  model.py         CNN + LSTM network
  train.py         Training and evaluation
  eval_heldout.py  Held-out folder evaluation
  analysis.py      Signal checks and report text
best.pt            Trained weights
requirements.txt
```

## Tech stack

Python, PyTorch, Librosa, SciPy, FastAPI, Uvicorn, HTML/CSS/JavaScript