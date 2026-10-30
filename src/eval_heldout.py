import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from model import CNNLSTM

HOLDOUT = "b"        # jo folder test ke liye alag rakhna hai (a, b, e meaningful hain)
SEED = 42
EPOCHS = 30
PATIENCE = 8
BATCH = 16
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)

X = np.load("data/processed/X.npy")[:, None, :, :]
y = np.load("data/processed/y.npy")
src = np.load("data/processed/src.npy")

# Holdout folder = test, baaki = train + val
test_mask = src == HOLDOUT
X_test, y_test = X[test_mask], y[test_mask]
X_rest, y_rest = X[~test_mask], y[~test_mask]
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.15, stratify=y_rest, random_state=SEED
)
print(f"Holdout folder: training-{HOLDOUT}")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Test normal:", (y_test == 0).sum(), "| abnormal:", (y_test == 1).sum())


def loader(Xs, ys, shuffle):
    ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


train_dl = loader(X_train, y_train, True)
val_dl = loader(X_val, y_val, False)
test_dl = loader(X_test, y_test, False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNLSTM().to(device)

counts = np.bincount(y_train)
weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


def evaluate(dl):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(yb)
            all_probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    return {
        "loss": total_loss / len(labels),
        "f1": f1_score(labels, preds, average="macro"),
        "auc": roc_auc_score(labels, probs),
        "preds": preds,
        "labels": labels,
    }


best_f1, wait = 0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * len(yb)

    val = evaluate(val_dl)
    scheduler.step(val["f1"])
    print(f"Epoch {epoch:02d} | train loss {run_loss/len(y_train):.4f} | "
          f"val loss {val['loss']:.4f} | val F1 {val['f1']:.3f} | val AUC {val['auc']:.3f}")

    if val["f1"] > best_f1:
        best_f1, wait = val["f1"], 0
        torch.save(model.state_dict(), "best_heldout.pt")   # best.pt ko touch nahi karte
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

model.load_state_dict(torch.load("best_heldout.pt", map_location=device))
test = evaluate(test_dl)
print(f"\n=== HELD-OUT TEST: training-{HOLDOUT} (model ne ye folder kabhi nahi dekha) ===")
print(f"F1 (macro): {test['f1']:.3f} | ROC-AUC: {test['auc']:.3f}")
print("Confusion matrix [rows=true, cols=pred] (0=normal, 1=abnormal):")
print(confusion_matrix(test["labels"], test["preds"]))
print(classification_report(test["labels"], test["preds"],
                            target_names=["normal", "abnormal"], digits=3))