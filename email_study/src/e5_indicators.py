"""E5 (week 9) — how well-founded are the chosen risk indicators?

For each of the 21 indicators (features.INDICATORS) and each corpus:
  binary   share in ham and in spam, odds ratio (Haldane +0.5), 95% CI, Fisher exact p
  numeric  medians, AUC = P(spam value > ham value) (Mann-Whitney), its p-value
Corpora: SA; CSDMC restricted to messages not in SA (independent evidence);
"PP vs SA ham" compares modern phishing with SA ham (direction check only).
An indicator is called WELL-FOUNDED if, in both SA and CSDMC, it is significant after a
Bonferroni correction (alpha = 0.05 / 42) with the same direction and a non-trivial effect
(OR >= 2 or <= 0.5; AUC >= 0.6 or <= 0.4).

Group-level check: logistic regression on the standardised indicators of one group
(links / headers / attachments / content form / all), trained on the SA 80% split and
evaluated on the SA test part and on CSDMC-not-in-SA (seeds 42-44, mean ROC-AUC).

Outputs: results/e5_indicators.csv, results/e5_groups.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import BINARY, GROUPS, INDICATORS, RESULTS, SEEDS, indicators, load, meta_matrix, split

ALL = load(["SA", "CSDMC", "PP"])
sets = {
    "SA": ALL[ALL.dataset == "SA"],
    "CSDMC": ALL[(ALL.dataset == "CSDMC") & ~ALL.dup_of_other],
    "PP vs SA ham": pd.concat([ALL[ALL.dataset == "PP"], ALL[(ALL.dataset == "SA") & (ALL.label == 0)]]),
}
ALPHA = 0.05 / (2 * len(INDICATORS))
rows = []
for sname, d in sets.items():
    X = indicators(d)
    y = d.label.to_numpy()
    for f in INDICATORS:
        a1, a0 = X[f].to_numpy()[y == 1], X[f].to_numpy()[y == 0]
        r = dict(indicator=f, name=INDICATORS[f], set=sname, kind="binary" if f in BINARY else "numeric")
        if f in BINARY:
            t = [[a1.sum(), len(a1) - a1.sum()], [a0.sum(), len(a0) - a0.sum()]]
            orr = ((t[0][0] + .5) * (t[1][1] + .5)) / ((t[0][1] + .5) * (t[1][0] + .5))
            se = np.sqrt(sum(1 / (v + .5) for row in t for v in row))
            p = fisher_exact(t)[1] if (a1.sum() + a0.sum()) > 0 else 1.0
            r.update(spam_value=a1.mean(), ham_value=a0.mean(), effect=orr,
                     ci_low=np.exp(np.log(orr) - 1.96 * se), ci_high=np.exp(np.log(orr) + 1.96 * se), p=p,
                     nontrivial=(orr >= 2) or (orr <= 0.5), direction=np.sign(np.log(orr)))
        else:
            u, p = mannwhitneyu(a1, a0) if np.ptp(np.r_[a1, a0]) > 0 else (len(a1) * len(a0) / 2, 1.0)
            auc = u / (len(a1) * len(a0))
            r.update(spam_value=np.median(a1), ham_value=np.median(a0), effect=auc, p=p,
                     nontrivial=(auc >= 0.6) or (auc <= 0.4), direction=np.sign(auc - 0.5))
        r["significant"] = r["p"] < ALPHA
        rows.append(r)
ind = pd.DataFrame(rows)
w = ind.pivot(index="indicator", columns="set", values=["significant", "nontrivial", "direction"])
ok = (w["significant"]["SA"] & w["significant"]["CSDMC"] & w["nontrivial"]["SA"] & w["nontrivial"]["CSDMC"]
      & (w["direction"]["SA"] == w["direction"]["CSDMC"]))
ind["well_founded"] = ind.indicator.map(ok)
ind["pp_same_direction"] = ind.indicator.map(w["direction"]["PP vs SA ham"] == w["direction"]["SA"])
ind.to_csv(RESULTS / "e5_indicators.csv", index=False)

# ---- group-level discriminative power -------------------------------------------------
cols = list(INDICATORS)
sa, cs = sets["SA"].reset_index(drop=True), sets["CSDMC"].reset_index(drop=True)
Xsa, Xcs = meta_matrix(sa), meta_matrix(cs)
ysa, ycs = sa.label.to_numpy(), cs.label.to_numpy()
grows = []
for g, feats in list(GROUPS.items()) + [("Все индикаторы", cols)]:
    j = [cols.index(f) for f in feats]
    for seed in SEEDS:
        tr, te = split(ysa, seed)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xsa[tr][:, j], ysa[tr])
        grows.append(dict(group=g, n_indicators=len(j), seed=seed,
                          auc_sa_test=roc_auc_score(ysa[te], m.predict_proba(Xsa[te][:, j])[:, 1]),
                          auc_csdmc=roc_auc_score(ycs, m.predict_proba(Xcs[:, j])[:, 1])))
grp = pd.DataFrame(grows).groupby(["group", "n_indicators"], sort=False)[["auc_sa_test", "auc_csdmc"]] \
    .agg(["mean", "std"])
grp.columns = [f"{a}_{b}" for a, b in grp.columns]
grp.reset_index().to_csv(RESULTS / "e5_groups.csv", index=False)
pd.set_option("display.width", 250)
print(ind[ind.set != "PP vs SA ham"].pivot(index="indicator", columns="set", values="effect").round(3)
      .join(ind.drop_duplicates("indicator").set_index("indicator")[["well_founded", "pp_same_direction"]]))
print(grp.round(3))
