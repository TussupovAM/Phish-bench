"""Shared helpers: data loading, URL preprocessing, lexical features, metrics.

All experiments in this repository import from here so that every table in the
paper is produced by the same preprocessing and the same metric definitions.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42


# --------------------------------------------------------------------------- #
# Dataset D1: UCI Phishing Websites (used by Shahrivari et al. [3] and
# Yerima & Alzaylaee [7]). 11,055 rows, 30 ternary features, label in {-1, 1}.
# Positive class (1) = phishing in all our metrics.
# --------------------------------------------------------------------------- #
def load_uci(deduplicate: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA / "uci_phishing_websites.csv").drop(columns=["index"])
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)
    y = (df.pop("Result").to_numpy() == -1).astype(int)  # -1 = phishing in UCI
    return df, y


# --------------------------------------------------------------------------- #
# Dataset D2: Ebbu2017 raw URLs (Sahingoz et al., 2019), PhishTank phishing
# vs. legitimate URLs. We (1) strip the scheme, because 37% of legitimate but
# only 1.8% of phishing URLs start with "https://" in this corpus, which lets a
# model shortcut on a collection artefact; (2) drop duplicates after stripping.
# --------------------------------------------------------------------------- #
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def normalise_url(u: str) -> str:
    return SCHEME.sub("", u.strip())


def load_ebbu() -> pd.DataFrame:
    p = json.load(open(DATA / "data_phishing_37175.json"))
    l = json.load(open(DATA / "data_legitimate_36400.json"))
    df = pd.DataFrame({"raw": p + l, "label": [1] * len(p) + [0] * len(l)})
    df["url"] = df["raw"].map(normalise_url)
    n0 = len(df)
    df = df.drop_duplicates(subset="url", keep="first")
    # a URL string that appears with both labels would be unlearnable noise
    conflict = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflict[conflict > 1].index)].reset_index(drop=True)
    df.attrs["n_raw"] = n0
    return df


# 20 lexical features computed from the URL string only (no network access),
# so they are available at mail-delivery time.
SUSPICIOUS_WORDS = ("login", "signin", "verify", "account", "update", "secure",
                    "bank", "confirm", "password", "webscr", "ebay", "paypal")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


LEXICAL_NAMES = [
    "len_url", "len_host", "len_path", "n_dots", "n_hyphens", "n_digits",
    "n_slash", "n_at", "n_question", "n_equal", "n_underscore", "n_percent",
    "n_subdomains", "has_ip", "has_www", "digit_ratio", "entropy",
    "n_suspicious_words", "tld_len", "longest_token",
]

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")


def lexical_features(u: str) -> list[float]:
    host, _, path = u.partition("/")
    host = host.split("@")[-1].split(":")[0].lower()
    labels = [x for x in host.split(".") if x]
    tokens = re.split(r"[./?=&_\-:@]", u)
    low = u.lower()
    return [
        len(u), len(host), len(path), u.count("."), u.count("-"),
        sum(ch.isdigit() for ch in u), u.count("/"), u.count("@"),
        u.count("?"), u.count("="), u.count("_"), u.count("%"),
        max(len(labels) - 2, 0), float(bool(IP_RE.match(host))),
        float(host.startswith("www.")),
        sum(ch.isdigit() for ch in u) / max(len(u), 1), _entropy(u),
        sum(w in low for w in SUSPICIOUS_WORDS),
        len(labels[-1]) if labels else 0,
        max((len(t) for t in tokens), default=0),
    ]


def lexical_matrix(urls) -> np.ndarray:
    return np.asarray([lexical_features(u) for u in urls], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Metrics. Positive class = phishing. All values returned as fractions.
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out


def environment() -> dict:
    import sklearn
    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                     if l.startswith("model name")), platform.processor()),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "gpu": "none",
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
    }
    for mod in ("xgboost", "torch"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return info
