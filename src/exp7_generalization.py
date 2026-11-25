"""Experiment 7 — generalisation on D2 (Ebbu2017) and model vs. representation controls.

Splits (seeds 42, 43, 44; about 20% of URLs in test):
  random  stratified 80/20 split of URLs, as in experiment 2
  domain  StratifiedGroupKFold(5), group = registered domain: no registered domain occurs in
          both training and test (checked). Validation data are likewise domain-disjoint.
Inside the training part 10% (random) or one of 10 group folds (domain) is held out as
validation data for early stopping, the choice of C for the linear model, and (experiment 8)
probability calibration and threshold selection. All models are fitted on the remaining 90%.

Models: 2 x 2 design, family (classical / DL) x input (20 lexical features / raw characters)
  lexical  Logistic Regression, Random Forest, XGBoost (classical) | MLP (DL)
  chars    TF-IDF of character 1-5-grams + Logistic Regression (classical) | CNN (char) (DL)

Independent sources (prep_external.py; registered domains disjoint from all of Ebbu2017):
  PhiUSIIL phishing (TPR), PhiUSIIL legitimate home pages (FPR), Phishing.Database 2026 (TPR).

Outputs: results/exp7_runs.csv, results/exp7_summary.csv, results/exp7_external.csv,
         results/scores/exp7_{split}_seed{S}.npz (validation, test and external scores)
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from common import DATA, RESULTS, lexical_matrix, load_ebbu, metrics, reg_domain
from models_dl import CharCNN, LexMLP, encode_urls, n_params, predict_proba, train
from models_ml import ml_models

torch.set_num_threads(2)
SEEDS = [42, 43, 44]
SPLITS = [s for s in sys.argv[1:]] or ["random", "domain"]
(RESULTS / "scores").mkdir(exist_ok=True)

df = load_ebbu()
df["domain"] = df.url.map(reg_domain)
y = df.label.to_numpy()
X_lex = lexical_matrix(df.url)
X_chr = encode_urls(df.url.tolist())
ext = {"phiusiil": pd.read_csv(DATA / "external" / "phiusiil_sample.csv"),
       "phishdb": pd.read_csv(DATA / "external" / "phishdb_sample.csv")}
E_lex = {k: lexical_matrix(v.url) for k, v in ext.items()}
E_chr = {k: encode_urls(v.url.tolist()) for k, v in ext.items()}
print(f"D2: {len(df)} URLs, {df.domain.nunique()} registered domains", flush=True)


def split_indices(split, seed):
    idx = np.arange(len(y))
    if split == "random":
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        fit, val = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=seed)
    else:
        g = df.domain.to_numpy()
        tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, g))
        f_, v_ = next(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(tr, y[tr], g[tr]))
        fit, val = tr[f_], tr[v_]
        assert not set(g[tr]) & set(g[te]) and not set(g[fit]) & set(g[val])
    return fit, val, te


def run_ml(est, Xf, yf, Xs):
    m = clone(est)
    t = time.perf_counter(); m.fit(Xf, yf); fit_s = time.perf_counter() - t
    return [m.predict_proba(X)[:, 1] for X in Xs], fit_s, None


def run_char_tfidf(urls_fit, yf, urls_val, yv, urls_list):
    t = time.perf_counter()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                          lowercase=False, dtype=np.float32)
    Xf = vec.fit_transform(urls_fit)
    Xv = vec.transform(urls_val)
    best = None
    for C in (1.0, 10.0, 100.0):  # chosen on validation log-loss only
        m = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Xf, yf)
        ll = log_loss(yv, m.predict_proba(Xv)[:, 1])
        if best is None or ll < best[0]:
            best = (ll, C, m)
    fit_s = time.perf_counter() - t
    m = best[2]
    return [m.predict_proba(vec.transform(u))[:, 1] for u in urls_list], fit_s, {
        "C": best[1], "n_features": Xf.shape[1]}


rows, ext_rows = [], []
for split in SPLITS:
    for seed in SEEDS:
        fit, val, te = split_indices(split, seed)
        print(f"== {split} seed {seed}: fit={len(fit)} val={len(val)} test={len(te)} "
              f"test phishing={y[te].mean():.3f}", flush=True)
        store = {"y_val": y[val], "y_test": y[te], "idx_val": val, "idx_test": te}
        sets_lex = [X_lex[val], X_lex[te], E_lex["phiusiil"], E_lex["phishdb"]]
        sets_chr = [X_chr[val], X_chr[te], E_chr["phiusiil"], E_chr["phishdb"]]
        url_sets = [df.url.values[val], df.url.values[te], ext["phiusiil"].url.values, ext["phishdb"].url.values]
        results = {}
        ml = ml_models(n_jobs=2)
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            results[name] = ("ML", "lexical") + run_ml(ml[name], X_lex[fit], y[fit], sets_lex)
        results["TF-IDF char n-grams + LR"] = ("ML", "characters") + run_char_tfidf(
            df.url.values[fit], y[fit], df.url.values[val], y[val], url_sets)
        # DL on lexical features: MLP on standardised features
        sc = StandardScaler().fit(X_lex[fit])
        T = lambda X: torch.tensor(sc.transform(X), dtype=torch.float32)
        torch.manual_seed(seed)
        mlp = LexMLP()
        log = train(mlp, T(X_lex[fit]), y[fit], T(X_lex[val]), y[val], epochs=60, patience=5,
                    batch=256, seed=seed, log=lambda *_: None)
        results["MLP"] = ("DL", "lexical", [predict_proba(mlp, T(X)) for X in sets_lex], log["train_s"],
                          {"epochs": log["epochs_run"], "params": n_params(mlp)})
        torch.manual_seed(seed)
        cnn = CharCNN()
        log = train(cnn, X_chr[fit], y[fit], X_chr[val], y[val], seed=seed, log=lambda *_: None)
        results["CNN (char)"] = ("DL", "characters", [predict_proba(cnn, X) for X in sets_chr], log["train_s"],
                                 {"epochs": log["epochs_run"], "params": n_params(cnn)})

        for name, (family, rep, (s_val, s_te, s_pu, s_pd), fit_s, extra) in results.items():
            r = metrics(y[te], (s_te >= 0.5).astype(int), s_te)
            r.update(split=split, seed=seed, family=family, input=rep, model=name, train_s=fit_s,
                     extra=json.dumps(extra) if extra else "")
            rows.append(r)
            key = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("-", "")
            store[f"{key}__val"], store[f"{key}__test"] = s_val, s_te
            store[f"{key}__phiusiil"], store[f"{key}__phishdb"] = s_pu, s_pd
            pu_y = ext["phiusiil"].label.to_numpy()
            e = {"split": split, "seed": seed, "family": family, "input": rep, "model": name,
                 "phiusiil_tpr": float((s_pu[pu_y == 1] >= 0.5).mean()),
                 "phiusiil_fpr": float((s_pu[pu_y == 0] >= 0.5).mean()),
                 "phishdb_tpr": float((s_pd >= 0.5).mean())}
            ext_rows.append(e)
            print(f"  {name:26s} F1={r['f1']:.4f} FPR={r['fpr']:.4f} AUC={r['roc_auc']:.4f} "
                  f"| PhiUSIIL TPR={e['phiusiil_tpr']:.3f} FPR={e['phiusiil_fpr']:.3f} "
                  f"PhishDB TPR={e['phishdb_tpr']:.3f} | {fit_s:.0f}s", flush=True)
        np.savez_compressed(RESULTS / "scores" / f"exp7_{split}_seed{seed}.npz", **store)
        pd.DataFrame(rows).to_csv(RESULTS / "exp7_runs.csv", index=False)
        pd.DataFrame(ext_rows).to_csv(RESULTS / "exp7_external_runs.csv", index=False)

runs = pd.DataFrame(rows)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "train_s"]
summ = runs.groupby(["split", "family", "input", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(RESULTS / "exp7_summary.csv", index=False)
ext_s = pd.DataFrame(ext_rows).groupby(["split", "family", "input", "model"], sort=False).agg(["mean", "std"])
ext_s.columns = [f"{a}_{b}" for a, b in ext_s.columns]
ext_s.drop(columns=[c for c in ext_s.columns if c.startswith("seed")]).reset_index().to_csv(
    RESULTS / "exp7_external.csv", index=False)
print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())
