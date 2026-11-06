"""Experiment 3 — the same trained models evaluated at realistic phishing prevalence.

Models are those of experiment 2 (trained on the balanced 80% split). For every
seed and every model we keep all legitimate test URLs and randomly draw phishing
test URLs so that phishing makes up pi of the test set, pi in {full test set (~45%), 10%, 2%, 1%}.
Each reduced prevalence is drawn 50 times; we report the mean. Decision thresholds stay
at their default (0.5 probability, 0 for the SVM decision value) — nothing is
re-tuned on the test data.

We also compare the measured precision with the analytic value
  precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))
using each model's TPR and FPR on the full balanced test set.

Outputs: results/exp3_imbalance.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import RESULTS, metrics

PREVALENCES = [None, 0.1, 0.02, 0.01]  # None = full balanced test set (~45% phishing)
DRAWS = 50
rows = []
for f in sorted((RESULTS / "scores").glob("exp2_seed*.npz")):
    seed = int(f.stem.split("seed")[1])
    z = np.load(f)
    y = z["y_test"]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for model in [k for k in z.files if k not in ("y_test", "idx_test")]:
        s = z[model]
        thr = 0.0 if model.startswith("SVM") else 0.5
        full = metrics(y, (s >= thr).astype(int), s)
        rng = np.random.default_rng(seed)
        for target in PREVALENCES:
            pi = target
            k = len(pos) if pi is None else int(round(pi * len(neg) / (1 - pi)))
            draws = 1 if pi is None else DRAWS
            pi = k / (k + len(neg))  # actual prevalence of the evaluated set
            acc = []
            for _ in range(draws):
                idx = np.concatenate([neg, rng.choice(pos, size=k, replace=False)])
                m = metrics(y[idx], (s[idx] >= thr).astype(int), s[idx])
                acc.append(m)
            mean = {key: float(np.mean([a[key] for a in acc]))
                    for key in ("accuracy", "precision", "recall", "f1", "fpr", "mcc", "pr_auc")}
            tpr, fpr = full["recall"], full["fpr"]
            mean.update(seed=seed, model=model, target="full" if target is None else f"{target:g}",
                        prevalence=round(pi, 4), n_phishing=k, n_legit=len(neg),
                        precision_analytic=tpr * pi / (tpr * pi + fpr * (1 - pi)),
                        trivial_accuracy=1 - pi)
            rows.append(mean)

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "exp3_imbalance_runs.csv", index=False)
summ = out.groupby(["model", "target"], sort=False).mean(numeric_only=True).drop(columns="seed")
summ.reset_index().to_csv(RESULTS / "exp3_imbalance.csv", index=False)
print(summ[["accuracy", "precision", "precision_analytic", "recall", "f1", "mcc", "pr_auc",
            "trivial_accuracy"]].round(4).to_string())
