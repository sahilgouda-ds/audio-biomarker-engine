import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from model import CNNLSTM

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

EPOCHS = 40
PATIENCE = 10      # itne epoch tak improvement na mile toh ruk jao
BATCH = 16
LR = 1e-3

# ---------- Data ----------
X = np.load("data/processed/X.npy")[:, None, :, :]    # channel dimension add: (N, 1, 64, T)
y = np.load("data/processed/y.npy")

# 70% train, 15% val, 15% test (stratified = dono classes ka ratio same rahega)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

def loader(Xs, ys, shuffle):
    ds = TensorDataset(torch.tensor(Xs), torch.tensor(ys, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_dl = loader(X_train, y_train, True)
val_dl = loader(X_val, y_val, False)
test_dl = loader(X_test, y_test, False)

# ---------- Model, loss, optimizer ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = CNNLSTM().to(device)

# Class imbalance: kam wali class (normal) ko zyada weight
counts = np.bincount(y_train)
weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# ---------- Evaluation helper ----------
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

# ---------- Training loop ----------
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
        torch.save(model.state_dict(), "best.pt")     # best model save
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

# ---------- Final test ----------
model.load_state_dict(torch.load("best.pt", map_location=device))
test = evaluate(test_dl)
print("\n=== TEST RESULTS (best model) ===")
print(f"F1 (macro): {test['f1']:.3f} | ROC-AUC: {test['auc']:.3f}")
print("Confusion matrix [rows=true, cols=pred] (0=normal, 1=abnormal):")
print(confusion_matrix(test["labels"], test["preds"]))
print(classification_report(test["labels"], test["preds"],
                            target_names=["normal", "abnormal"], digits=3))