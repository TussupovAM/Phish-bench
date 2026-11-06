"""Figure: precision vs. phishing prevalence (measured points + analytic curve).

Reads results/exp3_imbalance.csv; writes results/fig_precision_prevalence.pdf/.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS

e3 = pd.read_csv(RESULTS / "exp3_imbalance.csv")
series = [("XGBoost", "#2a78d6", "o", "XGBoost (best ML)"),
          ("CNN (char)", "#eb6834", "s", "CNN (char) (best DL)")]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.3))
pi = np.logspace(np.log10(0.005), np.log10(0.5), 200)
for model, color, marker, label in series:
    d = e3[e3.model == model].sort_values("prevalence")
    full = d[d.target == "full"].iloc[0]
    tpr, fpr = full.recall, full.fpr
    ax.plot(pi * 100, 100 * tpr * pi / (tpr * pi + fpr * (1 - pi)), color=color, lw=1.5)
    ax.plot(d.prevalence * 100, d.precision * 100, marker, color=color, ms=5, mec="white",
            mew=1, label=label, linestyle="none")
    last = d[d.target == "0.01"].iloc[0]
    ax.annotate(f"{last.precision * 100:.0f}%", (last.prevalence * 100, last.precision * 100),
                textcoords="offset points", xytext=(6, -2), fontsize=7, color="#333333")
ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 45])
ax.set_xticklabels(["1", "2", "5", "10", "20", "45"])
ax.set_xlabel("Phishing share of test traffic (%)")
ax.set_ylabel("Precision (%)")
ax.set_ylim(0, 100)
ax.grid(True, which="major", color="#e5e5e5", lw=0.6)
ax.legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(RESULTS / "fig_precision_prevalence.pdf")
fig.savefig(RESULTS / "fig_precision_prevalence.png", dpi=200)
