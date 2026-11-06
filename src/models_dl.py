"""Deep-learning models (PyTorch, CPU).

Character-level URL models (dataset D2):
  CharCNN      embedding -> 3 parallel Conv1d (k=3,5,7) -> global max-pool -> MLP
  CharLSTM     embedding -> 1-layer LSTM -> last hidden state -> MLP
  CharCNNLSTM  embedding -> Conv1d (k=5) -> max-pool(2) -> LSTM -> MLP

Feature-vector model (dataset D1), a re-implementation of the best 1D-CNN
configuration reported by Yerima & Alzaylaee [7]: two Conv1d blocks with 64
filters of length 12 and 6, max-pool 2, dense 32, sigmoid output.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

MAX_LEN = 128
# printable ASCII -> 1..95, 0 = padding, 96 = other
VOCAB = 97


def encode_urls(urls, max_len: int = MAX_LEN) -> torch.Tensor:
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, u in enumerate(urls):
        codes = [(ord(c) - 31) if 32 <= ord(c) < 127 else 96 for c in u[:max_len]]
        out[i, :len(codes)] = codes
    return torch.from_numpy(out)


class CharCNN(nn.Module):
    def __init__(self, emb=32, filters=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k, padding=k // 2) for k in (3, 5, 7))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * filters, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(h)).amax(dim=2) for c in self.convs], dim=1)
        return self.head(h).squeeze(1)


class CharLSTM(nn.Module):
    def __init__(self, emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(1)


class CharCNNLSTM(nn.Module):
    def __init__(self, emb=32, filters=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, filters, 5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(filters, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.pool(torch.relu(self.conv(self.emb(x).transpose(1, 2)))).transpose(1, 2)
        lengths = ((x != 0).sum(1) // 2).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.head(hn[-1]).squeeze(1)


class FeatureCNN(nn.Module):
    """1D-CNN over the 30 UCI features, after [7] (CNN2, 64 filters, 12/6, FC 32)."""

    def __init__(self, n_features=30, filters=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, filters, 12, padding=6), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(filters, filters, 6, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.LazyLinear(32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.net(torch.zeros(2, 1, n_features))  # materialise lazy layer

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


DL_MODELS = {"CNN (char)": CharCNN, "LSTM (char)": CharLSTM, "CNN+LSTM (char)": CharCNNLSTM}


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, batch: int = 1024) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(X[i:i + batch])).numpy()
                           for i in range(0, len(X), batch)])


def train(model, Xtr, ytr, Xva, yva, *, epochs=10, patience=2, batch=128, lr=1e-3, seed=42,
          log=print):
    """Adam + BCE; keeps the weights with the lowest validation loss (early stopping)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    yva_t = torch.as_tensor(yva, dtype=torch.float32)
    best, best_state, bad, hist = float("inf"), None, 0, []
    g = torch.Generator().manual_seed(seed)
    t_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(model(Xva[i:i + 2048]), yva_t[i:i + 2048]).item()
                                for i in range(0, len(Xva), 2048)]))
        hist.append(vl)
        log(f"    epoch {ep + 1}: val_loss={vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"epochs_run": len(hist), "best_val_loss": best,
            "train_s": time.perf_counter() - t_start, "val_history": hist}
