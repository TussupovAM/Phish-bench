"""E1 (week 3) — anomaly detection on e-mail traffic: Isolation Forest, One-Class SVM, DBSCAN.

Setting: novelty detection. Each detector is fitted on LEGITIMATE (ham) messages only and
must flag spam as anomalous; labels are used only for evaluation.

Corpora: SA (5,916 unique) and CSDMC (4,080 unique), separately.
Split: stratified 80/20, seeds 42/43/44. From the ham of the 80% part, 20% is held out as
validation ham; the decision threshold is the 95th percentile of the anomaly score on this
validation ham (target false-positive rate 5%). The test part is never used for fitting
or thresholds.

Feature sets:
  meta       21 risk indicators (features.INDICATORS), log for counts, standardised
  meta+text  meta + 50 SVD components of TF-IDF (1-2 grams, fitted on training ham)

DBSCAN (min_samples=10) has no predict(); a new message's anomaly score is its distance to
the nearest core sample of the clusters found in training ham; eps = 95th percentile of
the 10-NN distance in training ham. We report the number of clusters and the noise share.

Output: results/e1_anomaly_runs.csv, results/e1_anomaly_summary.csv
"""
import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import RESULTS, SEEDS, load, meta_matrix, metrics, split, texts


def features(df, idx_fit, which):
    meta = meta_matrix(df)
    sc = StandardScaler().fit(meta[idx_fit])
    X = sc.transform(meta)
    if which == "meta+text":
        T = texts(df)
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
        A = tf.fit_transform([T[i] for i in idx_fit])
        svd = TruncatedSVD(50, random_state=42).fit(A)
        Z = svd.transform(tf.transform(T))
        Z = StandardScaler().fit(Z[idx_fit]).transform(Z)
        X = np.hstack([X, Z])
    return X.astype(np.float32)


class DBSCANNovelty:
    def __init__(self, min_samples=10, q=95):
        self.min_samples, self.q = min_samples, q

    def fit(self, X):
        d, _ = NearestNeighbors(n_neighbors=self.min_samples).fit(X).kneighbors(X)
        self.eps = float(np.percentile(d[:, -1], self.q))
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        self.n_clusters = int(len(set(db.labels_)) - (-1 in db.labels_))
        self.noise_share = float((db.labels_ == -1).mean())
        self.core = NearestNeighbors(n_neighbors=1).fit(X[db.core_sample_indices_])
        return self

    def score(self, X):  # higher = more anomalous
        return self.core.kneighbors(X)[0][:, 0]


def detectors(seed):
    return {
        "Isolation Forest": lambda X: (lambda m: (m, lambda Z: -m.score_samples(Z)))(
            IsolationForest(n_estimators=300, random_state=seed, n_jobs=2).fit(X)),
        "One-Class SVM": lambda X: (lambda m: (m, lambda Z: -m.score_samples(Z)))(
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X)),
        "DBSCAN": lambda X: (lambda m: (m, m.score))(DBSCANNovelty().fit(X)),
    }


if __name__ == "__main__":
    rows = []
    for ds in ["SA", "CSDMC"]:
        df = load([ds])
        y = df.label.to_numpy()
        for seed in SEEDS:
            tr, te = split(y, seed)
            ham = tr[y[tr] == 0]
            fit, val = train_test_split(ham, test_size=0.2, random_state=seed)
            for fs in ["meta", "meta+text"]:
                X = features(df, fit, fs)
                for name, make in detectors(seed).items():
                    t0 = time.perf_counter()
                    model, score = make(X[fit])
                    fit_s = time.perf_counter() - t0
                    thr = np.percentile(score(X[val]), 95)
                    s = score(X[te])
                    r = metrics(y[te], (s > thr).astype(int), s)
                    r.update(dataset=ds, seed=seed, features=fs, model=name, fit_s=fit_s,
                             n_fit=len(fit), n_features=X.shape[1])
                    if name == "DBSCAN":
                        r.update(dbscan_clusters=model.n_clusters, dbscan_noise=model.noise_share,
                                 dbscan_eps=model.eps)
                    rows.append(r)
                    print(f"{ds} seed{seed} {fs:9s} {name:16s} AUC={r['roc_auc']:.3f} "
                          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} FPR={r['fpr']:.3f}",
                          flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "e1_anomaly_runs.csv", index=False)
    cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "fpr", "mcc", "fit_s"]
    summ = runs.groupby(["dataset", "features", "model"], sort=False)[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    db = runs[runs.model == "DBSCAN"].groupby(["dataset", "features"])[
        ["dbscan_clusters", "dbscan_noise"]].mean().reset_index()
    summ = summ.merge(db.assign(model="DBSCAN"), how="left", on=["dataset", "features", "model"])
    summ.to_csv(RESULTS / "e1_anomaly_summary.csv", index=False)
    print(summ[["dataset", "features", "model", "roc_auc_mean", "precision_mean", "recall_mean",
                "f1_mean", "fpr_mean"]].round(3).to_string())
