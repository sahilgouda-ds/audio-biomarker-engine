import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_mels=64, hidden=128, n_classes=2, dropout=0.3):
        super().__init__()

        # CNN: spectrogram image se frequency patterns nikalta hai
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),      # frequency half, time same

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Dropout(dropout),
        )

        # LSTM: time ke saath patterns ka sequence samajhta hai
        cnn_out = 32 * (n_mels // 4)   # channels x remaining frequency bins
        self.lstm = nn.LSTM(cnn_out, hidden, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):                      # x: (batch, 1, 64, time)
        x = self.cnn(x)                        # (batch, 32, 16, time)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)              # (batch, time, 32, 16)
        x = x.reshape(b, t, c * f)             # (batch, time, 512)
        out, _ = self.lstm(x)                  # (batch, time, 256)
        out = out.mean(dim=1)                  # time pe average
        return self.fc(self.dropout(out))      # (batch, 2)


if __name__ == "__main__":
    model = CNNLSTM()
    dummy = torch.randn(8, 1, 64, 157)         # 8 fake spectrograms
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))