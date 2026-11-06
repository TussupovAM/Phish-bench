"""Experiment 5 — a measured two-tier cascade (fast ML -> DL on uncertain URLs).

Tier 1 = XGBoost on lexical features; Tier 2 = the best character-level DL model
of experiment 2 (chosen by mean F1 over seeds, computed from experiment 2's
results file, not from these test scores). A URL goes to Tier 2 only if the
Tier-1 probability p satisfies |p - 0.5| < delta. The delta grid
{0, 0.1, 0.2, 0.3, 0.4, 0.5} is fixed in advance; delta = 0 means "Tier 1 only"
and delta = 0.5 means "Tier 2 for every URL". Nothing is tuned on test data.

Expected mean latency per URL = t1 + escalation_rate * t2, with t1 and t2 the
median single-URL latencies measured in experiment 4.

Output: results/exp5_cascade.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, metrics

summ = pd.read_csv(RESULTS / "exp2_urls_summary.csv")
tier2 = summ[summ.family == "DL"].sort_values("f1_mean", ascending=False).iloc[0]["model"]
lat = pd.read_csv(RESULTS / "exp4_latency.csv").set_index("model")["latency_median_ms"]
t1, t2 = lat["XGBoost"], lat[tier2]
print(f"Tier 2 = {tier2}; t1 = {t1:.3f} ms, t2 = {t2:.3f} ms")

rows = []
for f in sorted((RESULTS / "scores").glob("exp2_seed*.npz")):
    z = np.load(f)
    y, p1, p2 = z["y_test"], z["XGBoost"], z[tier2]
    for delta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        esc = np.abs(p1 - 0.5) < delta if delta < 0.5 else np.ones_like(y, dtype=bool)
        p = np.where(esc, p2, p1)
        m = metrics(y, (p >= 0.5).astype(int), p)
        m.update(seed=int(f.stem.split("seed")[1]), delta=delta, tier2=tier2,
                 escalation_rate=float(esc.mean()),
                 expected_latency_ms=t1 + float(esc.mean()) * t2 if delta < 0.5 else t2)
        rows.append(m)

runs = pd.DataFrame(rows)
runs.to_csv(RESULTS / "exp5_cascade_runs.csv", index=False)
out = runs.groupby("delta")[["escalation_rate", "accuracy", "precision", "recall", "f1", "fpr",
                             "expected_latency_ms"]].agg(["mean", "std"])
out.columns = [f"{a}_{b}" for a, b in out.columns]
out.reset_index().to_csv(RESULTS / "exp5_cascade.csv", index=False)
print(out.round(4).to_string())
