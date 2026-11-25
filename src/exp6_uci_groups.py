"""Experiment 6 — UCI Phishing Websites: grouped cross-validation and conflicting labels.

Protocol B of experiment 1 drops exact duplicate ROWS (features + label). Two rows with the
same feature vector but different labels are not duplicates, so they survive and can fall into
different folds: the model then sees in training the exact vector it is tested on.

Here every distinct 30-feature vector is a group, and folds never split a group:
  C: all 11,055 rows, StratifiedGroupKFold(10), group = feature vector
  D: as C, after removing every row whose feature vector occurs with both labels

Diagnostics (seed-42 folds of protocols A and B, as in experiment 1):
  share of test rows whose feature vector also occurs in the training folds, and
  share whose vector occurs in training with the opposite label.
For C we also report accuracy separately on rows with conflicting and with consistent vectors,
and the best accuracy any classifier can reach on the conflicting rows (majority label per vector).

Outputs: results/exp6_uci_groups_folds.csv, results/exp6_uci_groups_summary.csv,
         results/exp6_uci_groups_info.json
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from common import RESULTS, SEED, load_uci, metrics
from models_dl import FeatureCNN, predict_proba, train
from models_ml import ml_models

torch.set_num_threads(1)

Xdf, y_all = load_uci(deduplicate=False)
key_all = Xdf.astype(str).agg(",".join, axis=1).to_numpy()
lab_per_key = pd.Series(y_all).groupby(key_all).agg(["nunique", "size", "mean"])
conflict_keys = set(lab_per_key.index[lab_per_key["nunique"] > 1])
conf_rows = np.isin(key_all, list(conflict_keys))
# best possible accuracy on conflicting rows: predict the majority label of each vector
cl = lab_per_key.loc[list(conflict_keys)]
ceiling = float((np.maximum(cl["mean"], 1 - cl["mean"]) * cl["size"]).sum() / cl["size"].sum())

info = {"n_rows": int(len(y_all)), "n_distinct_vectors": int(len(lab_per_key)),
        "n_conflicting_vectors": int(len(conflict_keys)),
        "n_rows_with_conflicting_vectors": int(conf_rows.sum()),
        "best_accuracy_on_conflicting_rows": ceiling}

# ---- leakage diagnostics for protocols A and B (same folds as experiment 1) --------------
Xb, yb = load_uci(deduplicate=True)
key_b = Xb.astype(str).agg(",".join, axis=1).to_numpy()
for name, keys, yy in [("A", key_all, y_all), ("B", key_b, yb)]:
    seen, opposite = [], []
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(keys, yy):
        tr_lab = pd.Series(yy[tr]).groupby(keys[tr]).agg(set)
        for k, lab in zip(keys[te], yy[te]):
            s = tr_lab.get(k)
            seen.append(s is not None)
            opposite.append(s is not None and (1 - lab) in s)
    info[f"protocol_{name}_test_rows_vector_in_train"] = float(np.mean(seen))
    info[f"protocol_{name}_test_rows_vector_in_train_with_other_label"] = float(np.mean(opposite))
    info[f"protocol_{name}_n_rows"] = int(len(yy))
print(json.dumps(info, indent=2), flush=True)
json.dump(info, open(RESULTS / "exp6_uci_groups_info.json", "w"), indent=2)

# ---- grouped cross-validation ----------------------------------------------------------
rows = []
for protocol in ["C: grouped by feature vector", "D: grouped, conflicting vectors removed"]:
    keep = np.ones(len(y_all), bool) if protocol.startswith("C") else ~conf_rows
    X = Xdf.to_numpy(dtype=np.float32)[keep]
    y, groups, conf = y_all[keep], key_all[keep], conf_rows[keep]
    print(f"{protocol}: n={len(y)}, groups={len(set(groups))}, phishing={y.mean():.3f}", flush=True)
    folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(X, y, groups))
    for fold, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te])
    models = dict(ml_models(n_jobs=1))
    models["1D-CNN"] = None
    for name, est in models.items():
        for fold, (tr, te) in enumerate(folds):
            t0 = time.perf_counter()
            if est is not None:
                m = clone(est).fit(X[tr], y[tr])
                s = m.predict_proba(X[te])[:, 1]
            else:  # 1D-CNN of experiment 2b; early stopping on a group-disjoint 10% of training
                gtr = groups[tr]
                ug = np.unique(gtr)
                g_fit, g_val = train_test_split(ug, test_size=0.1, random_state=SEED + fold)
                fit, val = tr[np.isin(gtr, g_fit)], tr[np.isin(gtr, g_val)]
                Xt = torch.tensor(X)
                torch.manual_seed(SEED + fold)
                net = FeatureCNN()
                train(net, Xt[fit], y[fit], Xt[val], y[val], epochs=200, patience=20, batch=64,
                      seed=SEED + fold, log=lambda *_: None)
                s = predict_proba(net, Xt[te])
            fit_s = time.perf_counter() - t0
            p = (s >= 0.5).astype(int)
            r = metrics(y[te], p, s)
            ct = conf[te]
            r.update(protocol=protocol, model=name, fold=fold, n_train=len(tr), n_test=len(te), fit_s=fit_s,
                     acc_conflicting_rows=float((p[ct] == y[te][ct]).mean()) if ct.any() else np.nan,
                     acc_consistent_rows=float((p[~ct] == y[te][~ct]).mean()),
                     n_conflicting_test=int(ct.sum()))
            rows.append(r)
        last = [r for r in rows if r["protocol"] == protocol and r["model"] == name]
        print(f"  {name:20s} acc={np.mean([r['accuracy'] for r in last]):.4f}", flush=True)

folds_df = pd.DataFrame(rows)
folds_df.to_csv(RESULTS / "exp6_uci_groups_folds.csv", index=False)
cols = ["accuracy", "precision", "recall", "f1", "fpr", "mcc", "roc_auc", "pr_auc", "fit_s"]
summ = folds_df.groupby(["protocol", "model"], sort=False)[cols].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.reset_index()
# accuracy on conflicting rows pooled over folds (row-weighted)
pool = folds_df.assign(c_ok=folds_df.acc_conflicting_rows.fillna(0) * folds_df.n_conflicting_test)
pool = pool.groupby(["protocol", "model"], sort=False)[["c_ok", "n_conflicting_test"]].sum()
summ = summ.merge((pool.c_ok / pool.n_conflicting_test).rename("acc_conflicting_pooled").reset_index(),
                  on=["protocol", "model"])
summ.to_csv(RESULTS / "exp6_uci_groups_summary.csv", index=False)
print(summ[["protocol", "model", "accuracy_mean", "accuracy_std", "f1_mean", "mcc_mean",
            "acc_conflicting_pooled"]].round(4).to_string())
