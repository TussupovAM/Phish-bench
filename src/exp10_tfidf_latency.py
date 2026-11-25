"""Experiment 10 — latency, size and memory of the TF-IDF character n-gram + LR model.

The model is refitted exactly as in experiment 7 (random split, seed 42, 90% fit part, C chosen
there on validation log-loss) and saved; its single-URL latency is then measured with the
worker of experiment 4 (fresh process, one thread, 50 warm-up + 1,000 timed URLs). XGBoost and
CNN (char) are re-measured in the same session as a reference for comparing with Table V.
Output: results/exp10_latency.csv
"""
import json
import os
import subprocess
import sys

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
NAME = "TF-IDF char n-grams + LR"

if len(sys.argv) > 1 and sys.argv[1] == "fit":
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from common import RESULTS, load_ebbu
    df = load_ebbu(); y = df.label.to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    fit, val = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=42)
    runs = pd.read_csv(RESULTS / "exp7_runs.csv")
    C = json.loads(runs[(runs.split == "random") & (runs.seed == 42) & (runs.model == NAME)].extra.iloc[0])["C"]
    pipe = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2, sublinear_tf=True,
                                         lowercase=False, dtype=np.float32),
                         LogisticRegression(C=C, max_iter=3000, solver="liblinear"))
    pipe.fit(df.url.values[fit], y[fit])
    z = np.load(RESULTS / "scores" / "exp7_random_seed42.npz")
    ref = z["TFIDF_char_ngrams__LR__test"]
    assert np.allclose(pipe.predict_proba(df.url.values[te])[:, 1], ref, atol=1e-6)
    joblib.dump(pipe, RESULTS / "models" / "TFIDF_char_LR.joblib", compress=3)
    print("saved, C =", C)
elif len(sys.argv) > 1 and sys.argv[1] == "worker":
    import resource
    import time
    import joblib
    import numpy as np
    from common import RESULTS, load_ebbu
    from exp4_latency import rss_mb
    z = np.load(RESULTS / "scores" / "exp2_seed42.npz")
    urls = load_ebbu().url.to_numpy()[z["idx_test"][:1050]].tolist()
    base = rss_mb()
    path = RESULTS / "models" / "TFIDF_char_LR.joblib"
    model = joblib.load(path)
    infer = lambda b: model.predict_proba(b)[:, 1]
    for u in urls[:50]:
        infer([u])
    times = []
    for u in urls[50:1050]:
        t = time.perf_counter(); infer([u]); times.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); infer(urls[50:1050]); batch_ms = (time.perf_counter() - t)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    times = np.array(times)
    print(json.dumps({"model": NAME, "latency_median_ms": float(np.median(times)),
                      "latency_p95_ms": float(np.percentile(times, 95)), "latency_mean_ms": float(times.mean()),
                      "batched_ms_per_url": batch_ms, "model_size_mb": os.path.getsize(path) / 2**20,
                      "peak_rss_mb": peak, "rss_after_imports_mb": base, "memory_increment_mb": peak - base}))
else:
    import pandas as pd
    from common import RESULTS
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, __file__, "fit"], check=True, cwd=here)
    rows = []
    for cmd in ([__file__, "worker"], ["exp4_latency.py", "XGBoost"], ["exp4_latency.py", "CNN (char)"]):
        out = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, cwd=here)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(cmd, "FAILED", out.stderr[-1500:]); continue
        r = json.loads(line[-1]); r["session"] = "exp10"; rows.append(r)
        print(f"{r['model']:26s} median={r['latency_median_ms']:.3f}ms p95={r['latency_p95_ms']:.3f}ms "
              f"batched={r['batched_ms_per_url']:.3f} size={r['model_size_mb']:.3f}MB mem+={r['memory_increment_mb']:.1f}MB")
    pd.DataFrame(rows).to_csv(RESULTS / "exp10_latency.csv", index=False)
