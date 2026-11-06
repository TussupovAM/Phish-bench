"""Experiment 2 — ML vs DL on raw phishing URLs (dataset D2, Ebbu2017).

Split: stratified 80/20 train/test, repeated for seeds 42, 43, 44.
  * ML models (7): fit on the 80% training part, using the 20 lexical features
    from common.lexical_features (computed from the URL string only).
  * DL models (3): character-level, input = URL string truncated/padded to 128
    chars. 10% of the training part is held out for early stopping
    (max 10 epochs, patience 2, Adam lr 1e-3, batch 128).
  * The test part is never used for training, tuning or early stopping.

Usage: python exp2_urls.py [seed ...]   (default: 42 43 44)

Outputs:
  results/exp2_urls_runs_seed{S}.csv one row per model for seed S
  results/exp2_urls_runs.csv         all seeds run so far
  results/exp2_urls_summary.csv      mean and std over seeds
  results/scores/exp2_seed{S}.npz    test labels + every model's test scores
  results/models/                    fitted seed-42 models (for experiment 4)
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from common import RESULTS, environment, lexical_matrix, load_ebbu, metrics
from models_dl import DL_MODELS, encode_urls, n_params, predict_proba, train
from models_ml import ml_models, score_and_predict

torch.set_num_threads(2)
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43, 44]
(RESULTS / "scores").mkdir(exist_ok=True)
(RESULTS / "models").mkdir(exist_ok=True)

df = load_ebbu()
info = {"n_raw": df.attrs["n_raw"], "n_after_dedup": len(df),
        "n_phishing": int(df.label.sum()), "n_legitimate": int((df.label == 0).sum()),
        "median_len": float(df.url.str.len().median()),
        "p95_len": float(df.url.str.len().quantile(0.95)),
        "share_longer_than_128": float((df.url.str.len() > 128).mean()),
        "environment": environment()}
json.dump(info, open(RESULTS / "exp2_dataset_info.json", "w"), indent=2)
print(json.dumps({k: v for k, v in info.items() if k != "environment"}), flush=True)

t0 = time.perf_counter()
X_lex = lexical_matrix(df.url)
print(f"lexical features: {time.perf_counter() - t0:.1f}s", flush=True)
X_chr = encode_urls(df.url.tolist())
y = df.label.to_numpy()

rows = []
for seed in SEEDS:
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    scores = {"y_test": y[idx_te], "idx_test": idx_te}

    for name, est in ml_models(n_jobs=2, svm_probability=False).items():
        m = clone(est)
        t = time.perf_counter(); m.fit(X_lex[idx_tr], y[idx_tr]); fit_s = time.perf_counter() - t
        s, p = score_and_predict(m, X_lex[idx_te])
        r = metrics(y[idx_te], p, s)
        r.update(seed=seed, family="ML", model=name, train_s=fit_s, epochs_run=None, params=None)
        rows.append(r); scores[name] = s
        print(f"seed {seed} {name:20s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} fit={fit_s:.1f}s", flush=True)
        if seed == 42:
            joblib.dump(m, RESULTS / "models" / f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.joblib")

    i_fit, i_val = train_test_split(idx_tr, test_size=0.1, stratify=y[idx_tr], random_state=seed)
    for name, cls in DL_MODELS.items():
        torch.manual_seed(seed)
        model = cls()
        print(f"seed {seed} {name} params={n_params(model)}", flush=True)
        log = train(model, X_chr[i_fit], y[i_fit], X_chr[i_val], y[i_val], seed=seed)
        s = predict_proba(model, X_chr[idx_te])
        r = metrics(y[idx_te], (s >= 0.5).astype(int), s)
        r.update(seed=seed, family="DL", model=name, train_s=log["train_s"],
                 epochs_run=log["epochs_run"], params=n_params(model))
        rows.append(r); scores[name] = s
        print(f"seed {seed} {name:20s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"train={log['train_s']:.0f}s epochs={log['epochs_run']}", flush=True)
        if seed == 42:
            torch.save(model.state_dict(), RESULTS / "models" / f"{cls.__name__}.pt")
    np.savez_compressed(RESULTS / "scores" / f"exp2_seed{seed}.npz", **scores)
    pd.DataFrame([r for r in rows if r["seed"] == seed]).to_csv(
        RESULTS / f"exp2_urls_runs_seed{seed}.csv", index=False)

# summary over every seed run so far (seeds may be run in separate processes)
runs = pd.concat([pd.read_csv(f) for f in sorted(RESULTS.glob("exp2_urls_runs_seed*.csv"))])
runs.to_csv(RESULTS / "exp2_urls_runs.csv", index=False)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "train_s"]
summ = runs.groupby(["family", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(RESULTS / "exp2_urls_summary.csv", index=False)
print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())
