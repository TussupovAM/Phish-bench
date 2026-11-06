"""Experiment 4 — inference latency, model size and memory on CPU.

For each seed-42 model from experiment 2, run in a fresh Python process
(so memory figures are not polluted by other models):
  * end-to-end single-URL latency: raw URL string -> feature extraction or
    character encoding -> model -> probability; batch size 1, ONE CPU thread,
    50 warm-up calls, then 1,000 timed calls on test URLs; median, p95, mean.
  * batched throughput: the same 1,000 URLs in one call (1 thread), per URL.
  * model size: serialized file on disk (joblib / XGBoost / torch state_dict).
  * memory: peak resident set size (RSS) of the process after loading the model
    and running inference, minus RSS after importing the libraries only.

Usage:  python exp4_latency.py            (driver; runs every model)
        python exp4_latency.py <model>    (worker; prints one JSON line)
Output: results/exp4_latency.csv
"""
import json
import os
import resource
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost", "SVM (RBF)",
          "KNN (k=5)", "XGBoost", "CNN (char)", "LSTM (char)", "CNN+LSTM (char)"]
DL_FILE = {"CNN (char)": "CharCNN", "LSTM (char)": "CharLSTM", "CNN+LSTM (char)": "CharCNNLSTM"}


def ml_file(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "") + ".joblib"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024


def worker(name):
    import numpy as np
    import joblib  # noqa: F401
    import torch
    from common import RESULTS, lexical_matrix, load_ebbu
    from models_dl import DL_MODELS, encode_urls
    torch.set_num_threads(1)

    z = np.load(RESULTS / "scores" / "exp2_seed42.npz")
    urls = load_ebbu().url.to_numpy()[z["idx_test"][:1050]].tolist()
    base = rss_mb()

    if name in DL_FILE:
        path = RESULTS / "models" / f"{DL_FILE[name]}.pt"
        model = DL_MODELS[name]()
        model.load_state_dict(torch.load(path))
        model.eval()

        def infer(batch):
            with torch.no_grad():
                return torch.sigmoid(model(encode_urls(batch))).numpy()
    else:
        path = RESULTS / "models" / ml_file(name)
        model = joblib.load(path)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        if hasattr(model, "set_params") and name == "XGBoost":
            model.set_params(n_jobs=1)

        def infer(batch):
            X = lexical_matrix(batch)
            return model.decision_function(X) if name.startswith("SVM") else model.predict_proba(X)[:, 1]

    for u in urls[:50]:
        infer([u])
    times = []
    for u in urls[50:1050]:
        t = time.perf_counter(); infer([u]); times.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); infer(urls[50:1050]); batch_ms = (time.perf_counter() - t) * 1000 / 1000
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    times = np.array(times)
    print(json.dumps({
        "model": name, "latency_median_ms": float(np.median(times)),
        "latency_p95_ms": float(np.percentile(times, 95)), "latency_mean_ms": float(times.mean()),
        "batched_ms_per_url": batch_ms, "model_size_mb": os.path.getsize(path) / 2**20,
        "peak_rss_mb": peak, "rss_after_imports_mb": base, "memory_increment_mb": peak - base,
    }))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        import pandas as pd
        from common import RESULTS
        rows = []
        for m in MODELS:
            out = subprocess.run([sys.executable, __file__, m], capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(m, "FAILED\n", out.stderr[-2000:]); continue
            r = json.loads(line[-1]); rows.append(r)
            print(f"{m:20s} median={r['latency_median_ms']:.3f}ms p95={r['latency_p95_ms']:.3f}ms "
                  f"size={r['model_size_mb']:.3f}MB mem+={r['memory_increment_mb']:.1f}MB", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "exp4_latency.csv", index=False)
