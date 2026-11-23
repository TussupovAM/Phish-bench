"""Figure: communities of the e-mail interaction graph — size vs. spam share (E3)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from features import RESULTS

c = pd.read_csv(RESULTS / "e3_communities.csv")
c = c[(c.community != -1) & (c.messages >= 2)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.0, 3.0))
other = c[~c.suspicious & ~c.legit_dominated]
ax.scatter(other.messages, other.spam_share * 100, s=14, color="#9aa5b1", label="Прочие", zorder=2)
ax.scatter(c[c.legit_dominated].messages, c[c.legit_dominated].spam_share * 100, s=18, color="#2a78d6",
           label="Легитимные (≥5 писем, спам ≤10%)", zorder=3)
ax.scatter(c[c.suspicious].messages, c[c.suspicious].spam_share * 100, s=18, color="#c0392b",
           label="Подозрительные (≥5 писем, спам ≥90%)", zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Число писем в сообществе (лог. шкала)")
ax.set_ylabel("Доля спама, %")
ax.set_ylim(-5, 105)
ax.grid(True, color="#e5e5e5", lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(RESULTS / "fig_graph_communities.png", dpi=220)
print(len(c))
