"""Experiment 9 — why do the domain-split models transfer poorly to PhiUSIIL?

PhiUSIIL's legitimate URLs are bare home pages (www.<domain>, no path). We measure how common
such URLs are in each class of Ebbu2017 and of the external sets, and the threshold-free ROC-AUC
of every domain-split model on PhiUSIIL: all phishing vs. legitimate, and only phishing URLs
that are also bare host names vs. legitimate (same URL form in both classes).
Output: results/exp9_shift.csv, results/exp9_url_form.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, load_ebbu

MODELS = {"Logistic_Regression": "Logistic Regression", "Random_Forest": "Random Forest",
          "XGBoost": "XGBoost", "MLP": "MLP", "TFIDF_char_ngrams__LR": "TF-IDF char n-grams + LR",
          "CNN_char": "CNN (char)"}


def bare(u):
    return u.rstrip("/").find("/") < 0 and "?" not in u


eb = load_ebbu()
pu = pd.read_csv(DATA / "external" / "phiusiil_sample.csv")
pd_ = pd.read_csv(DATA / "external" / "phishdb_sample.csv")
form = {"ebbu_bare_share_legit": float(eb[eb.label == 0].url.map(bare).mean()),
        "ebbu_bare_share_phish": float(eb[eb.label == 1].url.map(bare).mean()),
        "phiusiil_bare_share_legit": float(pu[pu.label == 0].url.map(bare).mean()),
        "phiusiil_bare_share_phish": float(pu[pu.label == 1].url.map(bare).mean()),
        "phishdb_bare_share": float(pd_.url.map(bare).mean()),
        "ebbu_www_share_legit": float(eb[eb.label == 0].url.str.startswith("www.").mean()),
        "ebbu_www_share_phish": float(eb[eb.label == 1].url.str.startswith("www.").mean()),
        "phiusiil_www_share_legit": float(pu[pu.label == 0].url.str.startswith("www.").mean())}
json.dump(form, open(RESULTS / "exp9_url_form.json", "w"), indent=2)
print(json.dumps(form, indent=2))

y = pu.label.to_numpy()
b = pu.url.map(bare).to_numpy()
rows = []
for f in sorted(RESULTS.glob("scores/exp7_domain_seed*.npz")):
    z = np.load(f)
    for key, name in MODELS.items():
        s = z[f"{key}__phiusiil"]
        same_form = (y == 0) | b
        rows.append({"seed": int(f.stem.split("seed")[1]), "model": name,
                     "auc_phiusiil": roc_auc_score(y, s),
                     "auc_phiusiil_bare_only": roc_auc_score(y[same_form], s[same_form]),
                     "n_bare_phish": int((b & (y == 1)).sum())})
out = pd.DataFrame(rows).groupby("model", sort=False).agg(["mean", "std"])
out.columns = [f"{a}_{c}" for a, c in out.columns]
out = out.drop(columns=["seed_mean", "seed_std", "n_bare_phish_std"]).reset_index()
out.to_csv(RESULTS / "exp9_shift.csv", index=False)
print(out.round(4).to_string())
