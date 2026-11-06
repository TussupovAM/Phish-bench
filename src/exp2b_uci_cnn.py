"""Experiment 2b — 1D-CNN of Yerima & Alzaylaee [7] on the UCI dataset.

Same 10 folds and both protocols (A: all rows, B: deduplicated) as experiment 1,
so the CNN is directly comparable to the seven ML models.
Within each training fold 10% is held out for early stopping, as in [7]
(max 200 epochs, patience 20, Adam lr 1e-3, batch 64).

Outputs: results/exp2b_uci_cnn_folds.csv, results/exp2b_uci_cnn_summary.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import RESULTS, SEED, load_uci, metrics
from models_dl import FeatureCNN, n_params, predict_proba, train

torch.set_num_threads(1)
rows = []
for protocol, dedup in [("A: all rows (as in [3])", False), ("B: deduplicated", True)]:
    X, y = load_uci(deduplicate=dedup)
    X = torch.tensor(X.to_numpy(), dtype=torch.float32)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        fit, val = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
        torch.manual_seed(SEED + fold)
        model = FeatureCNN()
        log = train(model, X[fit], y[fit], X[val], y[val], epochs=200, patience=20,
                    batch=64, seed=SEED + fold, log=lambda *_: None)
        s = predict_proba(model, X[te])
        r = metrics(y[te], (s >= 0.5).astype(int), s)
        r.update(protocol=protocol, model="1D-CNN (features, after [7])", fold=fold,
                 fit_s=log["train_s"], epochs_run=log["epochs_run"], params=n_params(model))
        rows.append(r)
        print(f"{protocol} fold {fold}: acc={r['accuracy']:.4f} epochs={log['epochs_run']} "
              f"t={log['train_s']:.0f}s", flush=True)

folds = pd.DataFrame(rows)
folds.to_csv(RESULTS / "exp2b_uci_cnn_folds.csv", index=False)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "fit_s", "params"]
summ = folds.groupby(["protocol", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(RESULTS / "exp2b_uci_cnn_summary.csv", index=False)
print(summ.round(4).to_string())
