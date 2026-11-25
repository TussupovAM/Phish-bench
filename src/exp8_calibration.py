"""Experiment 8 — calibration, prevalence-aware thresholds, confidence intervals, significance.

Input: the validation and test scores saved by experiment 7 (both splits, three seeds).
Nothing here is fitted or chosen on test data.

1. Calibration. Platt scaling (logistic regression on the logit of the score) fitted on the
   validation part. Brier score and expected calibration error (ECE, 15 equal-width bins) on
   the test part, before and after.
2. Thresholds for a target phishing prevalence pi = 1%. Chosen on validation data:
     t_0.5   the default 0.5 used in experiments 2-3 and 7
     t_F1    maximises F1 at prevalence pi, computed from validation TPR(t) and FPR(t) with
             Eq. (1) (validation data are balanced, so no imbalanced validation set is needed)
     t_FPR   largest recall subject to validation FPR <= 0.5%
   Evaluated on test data re-sampled to pi = 1% (all legitimate test URLs, phishing drawn).
3. Uncertainty. 95% percentile bootstrap intervals (B = 1000): each replicate resamples the
   legitimate and the phishing test URLs of every seed with replacement (at pi = 1%: draws
   round(pi*n_legit/(1-pi)) phishing URLs) and averages the metric over the three seeds.
   The intervals cover test-set sampling; the seed standard deviation is reported separately.
4. Paired comparisons on the same bootstrap replicates: CNN (char) vs. TF-IDF char n-grams + LR,
   and CNN (char) vs. XGBoost, for F1 on the full test set and F1 at pi = 1% with t_F1.
   Two-sided bootstrap p-values, Holm correction over all comparisons.

Outputs: results/exp8_calibration.csv, results/exp8_prevalence.csv, results/exp8_ci.csv,
         results/exp8_tests.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from common import RESULTS

PI, B, FPR_BUDGET = 0.01, 1000, 0.005
MODELS = {"Logistic_Regression": "Logistic Regression", "Random_Forest": "Random Forest",
          "XGBoost": "XGBoost", "MLP": "MLP", "TFIDF_char_ngrams__LR": "TF-IDF char n-grams + LR",
          "CNN_char": "CNN (char)"}
COMPARE = [("CNN (char)", "TF-IDF char n-grams + LR"), ("CNN (char)", "XGBoost")]


def logit(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def ece(y, p, bins=15):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def rates(y, s, thr):
    """TPR and FPR for an array of thresholds."""
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tpr = 1 - np.searchsorted(pos, thr, side="left") / len(pos)
    fpr = 1 - np.searchsorted(neg, thr, side="left") / len(neg)
    return tpr, fpr


def choose_thresholds(y, s):
    cand = np.unique(s)
    tpr, fpr = rates(y, s, cand)
    prec = np.where(tpr + fpr > 0, tpr * PI / (tpr * PI + fpr * (1 - PI) + 1e-12), 0)
    f1 = np.where(prec + tpr > 0, 2 * prec * tpr / (prec + tpr + 1e-12), 0)
    ok = fpr <= FPR_BUDGET
    return {"t_0.5": 0.5, "t_F1": float(cand[np.argmax(f1)]),
            "t_FPR": float(cand[ok][np.argmax(tpr[ok])]) if ok.any() else float(cand.max())}


def prf(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    return p, r, np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)


cal_rows, prev_rows = [], []
boot = {}  # (split, model, metric) -> array over B of seed-averaged values
rng = np.random.default_rng(2026)
for split in ["random", "domain"]:
    files = sorted(RESULTS.glob(f"scores/exp7_{split}_seed*.npz"))
    zs = [np.load(f) for f in files]
    seeds = [int(f.stem.split("seed")[1]) for f in files]
    # one set of bootstrap index draws per seed, shared by all models (paired comparisons)
    draws = []
    for z in zs:
        y = z["y_test"]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = int(round(PI * len(neg) / (1 - PI)))
        draws.append({"pos": pos, "neg": neg, "k": k,
                      "full": [(rng.choice(neg, len(neg)), rng.choice(pos, len(pos))) for _ in range(B)],
                      "low": [(rng.choice(neg, len(neg)), rng.choice(pos, k)) for _ in range(B)]})
    for key, name in MODELS.items():
        per_seed = {m: np.zeros((len(zs), B)) for m in ["f1_full", "prauc_full", "f1_pi_tF1", "prec_pi_tF1",
                                                        "rec_pi_tF1", "f1_pi_t0.5", "prec_pi_t0.5"]}
        for i, (z, seed) in enumerate(zip(zs, seeds)):
            yv, sv, yt, st = z["y_val"], z[f"{key}__val"], z["y_test"], z[f"{key}__test"]
            # 1. calibration (Platt on validation)
            platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(sv)[:, None], yv)
            qt = platt.predict_proba(logit(st)[:, None])[:, 1]
            cal_rows.append({"split": split, "seed": seed, "model": name,
                             "brier_raw": float(np.mean((st - yt) ** 2)), "brier_platt": float(np.mean((qt - yt) ** 2)),
                             "ece_raw": float(ece(yt, st)), "ece_platt": float(ece(yt, qt))})
            # 2. thresholds chosen on validation, evaluated at pi = 1% on test (50 draws as in exp. 3)
            th = choose_thresholds(yv, sv)
            d = draws[i]
            r2 = np.random.default_rng(seed)
            for tname, t in th.items():
                vals = []
                for _ in range(50):
                    idx = np.concatenate([d["neg"], r2.choice(d["pos"], d["k"], replace=False)])
                    pr = st[idx] >= t
                    tp, fp = int((pr & (yt[idx] == 1)).sum()), int((pr & (yt[idx] == 0)).sum())
                    fn = int(((~pr) & (yt[idx] == 1)).sum())
                    p, r, f = prf(tp, fp, fn)
                    vals.append((p, r, f, fp / len(d["neg"])))
                v = np.mean(vals, axis=0)
                tv, fv = rates(yv, sv, np.array([t]))
                prev_rows.append({"split": split, "seed": seed, "model": name, "threshold": tname, "t": t,
                                  "val_tpr": float(tv[0]), "val_fpr": float(fv[0]),
                                  "precision": v[0], "recall": v[1], "f1": v[2], "fpr": v[3],
                                  "alerts_per_10k_legit": v[3] * 1e4})
            # 3. bootstrap replicates
            hit_t = {t: st >= th[t] for t in ("t_F1", "t_0.5")}
            for b in range(B):
                n_, p_ = d["full"][b]
                h = hit_t["t_0.5"]
                tp, fp, fn = h[p_].sum(), h[n_].sum(), (~h[p_]).sum()
                per_seed["f1_full"][i, b] = prf(tp, fp, fn)[2]
                yy = np.r_[np.zeros(len(n_)), np.ones(len(p_))]
                per_seed["prauc_full"][i, b] = average_precision_score(yy, np.r_[st[n_], st[p_]])
                n_, p_ = d["low"][b]
                for t, tag in (("t_F1", "tF1"), ("t_0.5", "t0.5")):
                    h = hit_t[t]
                    tp, fp, fn = h[p_].sum(), h[n_].sum(), (~h[p_]).sum()
                    p, r, f = prf(tp, fp, fn)
                    per_seed[f"f1_pi_{tag}"][i, b] = f
                    per_seed[f"prec_pi_{tag}"][i, b] = p
                    if tag == "tF1":
                        per_seed["rec_pi_tF1"][i, b] = r
        for m, arr in per_seed.items():
            boot[(split, name, m)] = arr.mean(axis=0)
        print(f"{split:6s} {name:26s} done", flush=True)

cal = pd.DataFrame(cal_rows)
cal.to_csv(RESULTS / "exp8_calibration_runs.csv", index=False)
cal.groupby(["split", "model"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index().to_csv(
    RESULTS / "exp8_calibration.csv", index=False)
prev = pd.DataFrame(prev_rows)
prev.to_csv(RESULTS / "exp8_prevalence_runs.csv", index=False)
ps = prev.groupby(["split", "model", "threshold"], sort=False)[
    ["t", "val_fpr", "precision", "recall", "f1", "fpr", "alerts_per_10k_legit"]].agg(["mean", "std"])
ps.columns = [f"{a}_{b}" for a, b in ps.columns]
ps.reset_index().to_csv(RESULTS / "exp8_prevalence.csv", index=False)

ci = [{"split": s, "model": n, "metric": m, "boot_mean": float(v.mean()),
       "ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5))}
      for (s, n, m), v in boot.items()]
pd.DataFrame(ci).to_csv(RESULTS / "exp8_ci.csv", index=False)

tests = []
for split in ["random", "domain"]:
    for a, b in COMPARE:
        for m in ["f1_full", "f1_pi_tF1"]:
            dlt = boot[(split, a, m)] - boot[(split, b, m)]
            p = min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean()))
            tests.append({"split": split, "a": a, "b": b, "metric": m, "diff_mean": float(dlt.mean()),
                          "ci_low": float(np.percentile(dlt, 2.5)), "ci_high": float(np.percentile(dlt, 97.5)),
                          "p_boot": max(p, 1 / B)})
t = pd.DataFrame(tests).sort_values("p_boot").reset_index(drop=True)
m = len(t)
t["p_holm"] = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(t.p_boot)])
t.to_csv(RESULTS / "exp8_tests.csv", index=False)
print(ps[[c for c in ps.columns if c.endswith("_mean")]].round(4).to_string())
print(t.round(4).to_string())
print(cal.groupby(["split", "model"], sort=False).mean(numeric_only=True).round(4).to_string())
