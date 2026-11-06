"""Experiment 1 — classical ML on the UCI Phishing Websites dataset.

Protocol A ("as in [3]"): stratified 10-fold CV on all 11,055 rows.
Protocol B ("deduplicated"): identical, after removing the 5,206 exact
duplicate rows, so that no feature vector appears in both train and test.

Outputs:
  results/exp1_uci_ml_folds.csv   one row per (protocol, model, fold)
  results/exp1_uci_ml_summary.csv mean and std over folds
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from common import RESULTS, SEED, load_uci, metrics
from models_ml import ml_models

rows = []
for protocol, dedup in [("A: all rows (as in [3])", False), ("B: deduplicated", True)]:
    X, y = load_uci(deduplicate=dedup)
    X = X.to_numpy(dtype=np.float32)
    print(f"{protocol}: n={len(y)}, phishing={y.mean():.3f}", flush=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for name, est in ml_models(n_jobs=1).items():
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            m = clone(est)
            t0 = time.perf_counter(); m.fit(X[tr], y[tr]); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); score = m.predict_proba(X[te])[:, 1]
            pred_s = time.perf_counter() - t0
            r = metrics(y[te], (score >= 0.5).astype(int), score)
            r.update(protocol=protocol, model=name, fold=fold, n_train=len(tr),
                     n_test=len(te), fit_s=fit_s, predict_s=pred_s)
            rows.append(r)
        print(f"  {name:20s} acc={np.mean([r['accuracy'] for r in rows[-10:]]):.4f}", flush=True)

folds = pd.DataFrame(rows)
folds.to_csv(RESULTS / "exp1_uci_ml_folds.csv", index=False)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "fit_s"]
summ = folds.groupby(["protocol", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(RESULTS / "exp1_uci_ml_summary.csv", index=False)
print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())
