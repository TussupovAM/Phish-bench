"""Text classifiers for e-mail: ANN (MLP on TF-IDF), CNN and LSTM on word sequences (PyTorch, CPU)."""
from __future__ import annotations

import copy
import re
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

TOKEN = re.compile(r"[a-z]+|urltoken|emailtoken|numtoken|[!$%]")
MAX_LEN = 300


def tokenize(t: str) -> list[str]:
    return TOKEN.findall(t)


class Vocab:
    def __init__(self, docs, max_size=20000, min_count=2):
        c = Counter(w for d in docs for w in set(tokenize(d)))
        words = [w for w, n in c.most_common(max_size) if n >= min_count]
        self.idx = {w: i + 2 for i, w in enumerate(words)}  # 0 pad, 1 unknown

    def __len__(self):
        return len(self.idx) + 2

    def encode(self, docs, max_len=MAX_LEN) -> torch.Tensor:
        out = np.zeros((len(docs), max_len), dtype=np.int64)
        for i, d in enumerate(docs):
            ids = [self.idx.get(w, 1) for w in tokenize(d)[:max_len]]
            out[i, :len(ids)] = ids
        return torch.from_numpy(out)


class ANN(nn.Module):
    """MLP over a TF-IDF vector."""

    def __init__(self, n_in, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    """Kim-style CNN: embedding -> Conv1d (3,4,5) -> global max-pool -> dense."""

    def __init__(self, vocab, emb=100, filters=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, k, padding=k // 2) for k in (3, 4, 5))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * filters, 1))

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        return self.head(torch.cat([torch.relu(c(h)).amax(2) for c in self.convs], 1)).squeeze(1)


class LSTM(nn.Module):
    def __init__(self, vocab, emb=100, hidden=128, bidirectional=False, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * (2 if bidirectional else 1), 1))

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], 1) if self.lstm.bidirectional else h[-1]
        return self.head(h).squeeze(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _batch(X, idx):
    if isinstance(X, torch.Tensor):
        return X[idx]
    return torch.from_numpy(X[idx.numpy()].toarray().astype(np.float32))  # scipy CSR -> dense


@torch.no_grad()
def predict(model, X, batch=256):
    model.eval()
    n = X.shape[0]
    return np.concatenate([torch.sigmoid(model(_batch(X, torch.arange(i, min(i + batch, n))))).numpy()
                           for i in range(0, n, batch)])


def train(model, Xtr, ytr, Xva, yva, *, epochs=15, patience=3, batch=64, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, state, bad, hist = float("inf"), None, 0, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xtr.shape[0], generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(_batch(Xtr, idx)), ytr_t[idx]).backward()
            opt.step()
        p = np.clip(predict(model, Xva), 1e-7, 1 - 1e-7)
        vl = float(-np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))
        hist.append(vl)
        if vl < best - 1e-4:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return {"epochs_run": len(hist), "best_val_loss": best, "train_s": time.perf_counter() - t0}
