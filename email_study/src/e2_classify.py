"""E2 (weeks 4-5) — spam classification with ANN, CNN, LSTM; hyper-parameter selection; comparison.

Corpora: SA and CSDMC separately (duplicates removed within each corpus).
Split: stratified 80/20 train/test, seeds 42/43/44; 15% of the training part is the
validation set (early stopping and hyper-parameter selection). Test is never used for either.

Step 1 (tuning, seed 42 only): a small grid per model, chosen by validation F1
(ties -> lower validation loss). Grid:
  ANN   hidden {64, 256} x dropout {0.2, 0.5}       input: TF-IDF 1-2 grams, 20,000 features
  CNN   filters {64, 128} x dropout {0.3, 0.5}       embedding 100, kernels 3/4/5, 300 tokens
  LSTM  hidden {64, 128} x bidirectional {no, yes}   embedding 100, 300 tokens
All: Adam lr 1e-3, batch 64, <= 15 epochs, patience 3 on validation loss.
Step 2 (final): the chosen configuration is trained for every seed and evaluated on test.
Reference baselines (no tuning): Multinomial Naive Bayes and Logistic Regression on TF-IDF.

Outputs: results/e2_tuning.csv, results/e2_runs.csv, results/e2_summary.csv,
         results/e2_models/<corpus>_<model>.* (seed 42, reused by E4)
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from features import RESULTS, SEEDS, environment, load, metrics, split, texts
from textmodels import ANN, CNN, LSTM, Vocab, n_params, predict, train

torch.set_num_threads(2)
MODELS_DIR = RESULTS / "e2_models"
MODELS_DIR.mkdir(exist_ok=True)
GRID = {
    "ANN": [dict(hidden=h, dropout=d) for h in (64, 256) for d in (0.2, 0.5)],
    "CNN": [dict(filters=f, dropout=d) for f in (64, 128) for d in (0.3, 0.5)],
    "LSTM": [dict(hidden=h, bidirectional=b) for h in (64, 128) for b in (False, True)],
}
CORPORA = sys.argv[1:] or ["SA", "CSDMC"]


def prepare(T, fit_idx):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    tf.fit([T[i] for i in fit_idx])
    vocab = Vocab([T[i] for i in fit_idx])
    return tf, tf.transform(T).tocsr(), vocab, vocab.encode(T)


def build(name, cfg, n_tfidf, n_vocab):
    if name == "ANN":
        return ANN(n_tfidf, **cfg)
    if name == "CNN":
        return CNN(n_vocab, **cfg)
    return LSTM(n_vocab, **cfg)


def run_dl(name, cfg, Xtf, Xseq, y, fit, val, te, seed):
    X = Xtf if name == "ANN" else Xseq
    torch.manual_seed(seed)
    m = build(name, cfg, Xtf.shape[1], int(Xseq.max()) + 1)
    log = train(m, X[fit], y[fit], X[val], y[val], seed=seed)
    return m, log, predict(m, X[val]), predict(m, X[te])


# Tuning results already on disk are reused (e.g. when a run was interrupted after tuning),
# so that the configuration choice is not repeated; selection rule is the same.
prev_t = pd.read_csv(RESULTS / "e2_tuning.csv") if (RESULTS / "e2_tuning.csv").exists() else pd.DataFrame()
tuning, rows = [], []
for ds in CORPORA:
    df = load([ds])
    y = df.label.to_numpy()
    T = texts(df)
    best = {}
    done = prev_t[prev_t.dataset == ds] if len(prev_t) else prev_t
    if len(done) == sum(len(g) for g in GRID.values()):
        tuning += done.to_dict("records")
        for name in GRID:
            t = done[done.model == name]
            best[name] = json.loads(t.sort_values(["val_f1", "val_loss"], ascending=[False, True])
                                    .iloc[0].config)
            print(f"[best, reused] {ds} {name}: {best[name]}", flush=True)
    # ---- step 1: tuning on seed 42 ---------------------------------------------------
    tr, te = split(y, 42)
    fit, val = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42)
    tf, Xtf, vocab, Xseq = prepare(T, fit)
    for name, grid in GRID.items():
        if name in best:
            continue
        for cfg in grid:
            m, log, pv, _ = run_dl(name, cfg, Xtf, Xseq, y, fit, val, te, 42)
            r = metrics(y[val], (pv >= 0.5).astype(int), pv)
            tuning.append(dict(dataset=ds, model=name, config=json.dumps(cfg), val_f1=r["f1"],
                               val_loss=log["best_val_loss"], epochs=log["epochs_run"],
                               train_s=log["train_s"], params=n_params(m)))
            print(f"[tune] {ds} {name} {cfg} valF1={r['f1']:.4f} loss={log['best_val_loss']:.4f} "
                  f"ep={log['epochs_run']} t={log['train_s']:.0f}s", flush=True)
        t = pd.DataFrame([x for x in tuning if x["dataset"] == ds and x["model"] == name])
        best[name] = json.loads(t.sort_values(["val_f1", "val_loss"], ascending=[False, True])
                                .iloc[0].config)
        print(f"[best] {ds} {name}: {best[name]}", flush=True)
    keep = prev_t[~prev_t.dataset.isin([x["dataset"] for x in tuning])] if len(prev_t) else prev_t
    pd.concat([keep, pd.DataFrame(tuning)]).to_csv(RESULTS / "e2_tuning.csv", index=False)

    # ---- step 2: final runs over seeds ------------------------------------------------
    scores = {}
    for seed in SEEDS:
        tr, te = split(y, seed)
        fit, val = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
        tf, Xtf, vocab, Xseq = prepare(T, fit)
        sc = {"y_test": y[te], "idx_test": te}
        for name, est in [("Naive Bayes", MultinomialNB(alpha=0.1)),
                          ("Logistic Regression", LogisticRegression(C=10, max_iter=3000))]:
            t0 = time.perf_counter(); est.fit(Xtf[tr], y[tr]); fit_s = time.perf_counter() - t0
            s = est.predict_proba(Xtf[te])[:, 1]
            r = metrics(y[te], (s >= 0.5).astype(int), s)
            r.update(dataset=ds, seed=seed, model=name, family="baseline", train_s=fit_s,
                     config="", params=None, epochs_run=None)
            rows.append(r); sc[name] = s
            if seed == 42:
                joblib.dump(est, MODELS_DIR / f"{ds}_{name.replace(' ', '_')}.joblib")
        for name in ["ANN", "CNN", "LSTM"]:
            m, log, _, s = run_dl(name, best[name], Xtf, Xseq, y, fit, val, te, seed)
            r = metrics(y[te], (s >= 0.5).astype(int), s)
            r.update(dataset=ds, seed=seed, model=name, family="DL", train_s=log["train_s"],
                     config=json.dumps(best[name]), params=n_params(m), epochs_run=log["epochs_run"])
            rows.append(r); sc[name] = s
            print(f"[final] {ds} seed{seed} {name} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f}", flush=True)
            if seed == 42:
                torch.save(m.state_dict(), MODELS_DIR / f"{ds}_{name}.pt")
        if seed == 42:
            joblib.dump({"tfidf": tf, "vocab": vocab, "best": best}, MODELS_DIR / f"{ds}_prep.joblib")
        np.savez_compressed(RESULTS / f"e2_scores_{ds}_seed{seed}.npz", **sc)
        pd.DataFrame(rows).to_csv(RESULTS / f"e2_runs_partial.csv", index=False)

runs = pd.DataFrame(rows)
old = RESULTS / "e2_runs.csv"
if old.exists() and set(CORPORA) != {"SA", "CSDMC"}:  # merge with corpora run earlier
    prev = pd.read_csv(old)
    runs = pd.concat([prev[~prev.dataset.isin(CORPORA)], runs])
runs.to_csv(old, index=False)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "train_s"]
summ = runs.groupby(["dataset", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(RESULTS / "e2_summary.csv", index=False)
json.dump(environment(), open(RESULTS / "environment.json", "w"), indent=2)
print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())
