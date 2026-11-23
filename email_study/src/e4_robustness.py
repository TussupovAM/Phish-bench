"""E4 (week 8) — robustness of the threat-detection models on control data.

Uses the seed-42 models of E2 (trained on 80% of a corpus) and, for metadata anomaly
detection, One-Class SVM and DBSCAN refitted exactly as in E1 (seed 42, meta features).
Decision thresholds are never re-tuned on control data.

(a) Cross-corpus: SA-trained models on CSDMC messages whose text does not occur in SA,
    and CSDMC-trained models on SA messages whose text does not occur in CSDMC.
(b) Out-of-distribution phishing: 1,868 unique Phishing Pot messages (2022-2026) — only
    recall (detection rate) can be measured, all messages are phishing.
(c) Adversarial text perturbation of the SA test spam (seed 42 split; ham unchanged):
      obf-10 / obf-30 : 10% / 30% of words with >= 4 letters obfuscated (look-alike
                        characters a->@, o->0, i->1, e->3, s->$ or dots between letters)
      good-words      : 50 words frequent in training ham appended to each spam message
    Recall on the perturbed spam is compared with recall on the original spam.
    Only the text is perturbed, so the metadata detectors (One-Class SVM, DBSCAN on the
    risk indicators) give identical decisions by construction; they are listed for contrast.

Outputs: results/e4_cross.csv, results/e4_perturb.csv
"""
import random
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from e1_anomaly import DBSCANNovelty
from features import RESULTS, load, meta_matrix, metrics, split, texts
from textmodels import ANN, CNN, LSTM, predict, tokenize

torch.set_num_threads(2)
MD = RESULTS / "e2_models"
ALL = load(["SA", "CSDMC", "PP"])


class Suite:
    """All models trained on one corpus (seed 42 split), with a common predict()."""

    def __init__(self, ds):
        prep = joblib.load(MD / f"{ds}_prep.joblib")
        self.tf, self.vocab, best = prep["tfidf"], prep["vocab"], prep["best"]
        self.text = {n: joblib.load(MD / f"{ds}_{n.replace(' ', '_')}.joblib")
                     for n in ["Naive Bayes", "Logistic Regression"]}
        for n, cls in [("ANN", ANN), ("CNN", CNN), ("LSTM", LSTM)]:
            m = cls(len(self.tf.vocabulary_), **best[n]) if n == "ANN" else cls(len(self.vocab), **best[n])
            m.load_state_dict(torch.load(MD / f"{ds}_{n}.pt")); m.eval()
            self.text[n] = m
        df = ALL[ALL.dataset == ds].reset_index(drop=True)
        y = df.label.to_numpy()
        tr, te = split(y, 42)
        self.test_idx, self.df, self.y = te, df, y
        ham = tr[y[tr] == 0]
        fit, val = train_test_split(ham, test_size=0.2, random_state=42)
        meta = meta_matrix(df)
        self.sc = StandardScaler().fit(meta[fit])
        Xf, Xv = self.sc.transform(meta[fit]), self.sc.transform(meta[val])
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Xf)
        db = DBSCANNovelty().fit(Xf)
        self.anom = {"One-Class SVM (meta)": (lambda Z, m=oc: -m.score_samples(Z)),
                     "DBSCAN (meta)": db.score}
        self.thr = {k: np.percentile(f(Xv), 95) for k, f in self.anom.items()}
        self.ham_words = [w for w, _ in Counter(
            w for t in texts(df.iloc[tr][df.iloc[tr].label == 0]) for w in tokenize(t)
            if len(w) > 3 and w.isalpha()).most_common(300)]

    def predict(self, frame, T=None):
        T = T if T is not None else texts(frame)
        Xtf = self.tf.transform(T).tocsr()
        out = {}
        for n in ["Naive Bayes", "Logistic Regression"]:
            out[n] = (self.text[n].predict_proba(Xtf)[:, 1] >= 0.5).astype(int)
        out["ANN"] = (predict(self.text["ANN"], Xtf) >= 0.5).astype(int)
        Xs = self.vocab.encode(T)
        for n in ["CNN", "LSTM"]:
            out[n] = (predict(self.text[n], Xs) >= 0.5).astype(int)
        Z = self.sc.transform(meta_matrix(frame))
        for n, f in self.anom.items():
            out[n] = (f(Z) > self.thr[n]).astype(int)
        return out


suites = {ds: Suite(ds) for ds in ["SA", "CSDMC"]}
rows = []
for train_ds, other in [("SA", "CSDMC"), ("CSDMC", "SA")]:
    s = suites[train_ds]
    in_test = s.df.iloc[s.test_idx]
    other_df = ALL[(ALL.dataset == other) & ~ALL.hash.isin(set(s.df.hash))]
    pp = ALL[ALL.dataset == "PP"]
    for setname, frame in [(f"{train_ds} test (in-corpus)", in_test),
                           (f"{other} (unseen texts)", other_df), ("Phishing Pot 2022-26", pp)]:
        preds = s.predict(frame)
        for model, p in preds.items():
            r = metrics(frame.label.to_numpy(), p)
            r.update(trained_on=train_ds, test_set=setname, model=model)
            rows.append(r)
        print(train_ds, setname, len(frame), {m: round(float(p[frame.label.to_numpy() == 1].mean()), 3)
                                             for m, p in preds.items()}, flush=True)
cross = pd.DataFrame(rows)
cross.to_csv(RESULTS / "e4_cross.csv", index=False)

# ---- (c) perturbation --------------------------------------------------------------
LOOK = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$"}


def obfuscate(text, p, rng):
    def f(m):
        w = m.group(0)
        if len(w) < 4 or w.endswith("token") or rng.random() >= p:
            return w
        if rng.random() < 0.5 and any(c in LOOK for c in w.lower()):
            return "".join(LOOK.get(c.lower(), c) if c.lower() in LOOK and rng.random() < 0.6 else c for c in w)
        return ".".join(w)
    return re.sub(r"[A-Za-z]+", f, text)


s = suites["SA"]
te = s.df.iloc[s.test_idx]
spam = te[te.label == 1]
rows = []
base = s.predict(spam)
variants = {"original": None}
rng = random.Random(42)
variants["obf-10"] = [obfuscate(t, 0.10, rng) for t in texts(spam)]
variants["obf-30"] = [obfuscate(t, 0.30, rng) for t in texts(spam)]
gw = " ".join(s.ham_words[:50])
variants["good-words"] = [t + " " + gw for t in texts(spam)]
for name, T in variants.items():
    preds = base if T is None else s.predict(spam, T)
    for model, p in preds.items():
        rows.append(dict(variant=name, model=model, n_spam=len(spam), recall=float(p.mean())))
pert = pd.DataFrame(rows)
pert.to_csv(RESULTS / "e4_perturb.csv", index=False)
print(pert.pivot(index="model", columns="variant", values="recall").round(3).to_string())
