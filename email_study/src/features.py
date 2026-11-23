"""Shared feature extraction, text normalisation, metrics and splits for the e-mail study."""
from __future__ import annotations

import os
import platform
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from parse import RESULTS, RISKY_EXT, reg_domain

SEEDS = [42, 43, 44]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
              "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc"}
IPHOST = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def load(datasets=("SA", "CSDMC", "PP")) -> pd.DataFrame:
    df = pd.read_pickle(RESULTS / "emails.pkl")
    return df[df.dataset.isin(datasets)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Risk indicators (metadata features). Each is computable from one message.
# --------------------------------------------------------------------------- #
INDICATORS = {
    # links
    "n_urls": "Число ссылок",
    "url_density": "Плотность ссылок (на 100 слов)",
    "n_url_domains": "Число различных доменов в ссылках",
    "ip_url": "Ссылка на IP-адрес",
    "shortener_url": "Сокращённая ссылка",
    "anchor_mismatch": "Текст ссылки не совпадает с адресом",
    # headers
    "from_returnpath_mismatch": "From ≠ Return-Path (домен)",
    "from_replyto_mismatch": "From ≠ Reply-To (домен)",
    "msgid_from_mismatch": "Message-ID ≠ From (домен)",
    "missing_msgid": "Нет Message-ID",
    "missing_date": "Нет Date",
    "no_mailer": "Нет X-Mailer / User-Agent",
    "n_received": "Число Received-переходов",
    "n_rcpt": "Число получателей (To+Cc)",
    # attachments
    "has_attachment": "Есть вложение",
    "risky_attachment": "Опасное расширение вложения",
    # content form
    "html_only": "Только HTML, без текстовой части",
    "subj_upper_ratio": "Доля заглавных в теме",
    "subj_exclaim": "«!» в теме",
    "body_upper_ratio": "Доля заглавных в тексте",
    "log_body_len": "Длина текста (log)",
}
BINARY = {"ip_url", "shortener_url", "anchor_mismatch", "from_returnpath_mismatch", "from_replyto_mismatch",
          "msgid_from_mismatch", "missing_msgid", "missing_date", "no_mailer",
          "has_attachment", "risky_attachment", "html_only", "subj_exclaim"}
GROUPS = {
    "Ссылки": ["n_urls", "url_density", "n_url_domains", "ip_url", "shortener_url", "anchor_mismatch"],
    "Заголовки": ["from_returnpath_mismatch", "from_replyto_mismatch", "msgid_from_mismatch",
                  "missing_msgid", "missing_date", "no_mailer", "n_received", "n_rcpt"],
    "Вложения": ["has_attachment", "risky_attachment"],
    "Форма содержимого": ["html_only", "subj_upper_ratio", "subj_exclaim", "body_upper_ratio",
                          "log_body_len"],
}


def _upper_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def _mismatch(a: str, b: str) -> float:
    return float(bool(a) and bool(b) and reg_domain(a) != reg_domain(b))


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    words = df.body.str.split().map(len).clip(lower=1)
    out = pd.DataFrame({
        "n_urls": df.urls.map(len),
        "url_density": 100 * df.urls.map(len) / words,
        "n_url_domains": df.url_hosts.map(lambda h: len({reg_domain(x) for x in h if x})),
        "ip_url": df.url_hosts.map(lambda h: float(any(IPHOST.match(x) for x in h))),
        "shortener_url": df.url_hosts.map(lambda h: float(any(reg_domain(x) in SHORTENERS for x in h))),
        "anchor_mismatch": (df.anchor_mismatch > 0).astype(float),
        "from_returnpath_mismatch": [_mismatch(a, b) for a, b in zip(df.from_domain, df.return_path_domain)],
        "from_replyto_mismatch": [_mismatch(a, b) for a, b in zip(df.from_domain, df.reply_to_domain)],
        "msgid_from_mismatch": [_mismatch(a, b) for a, b in zip(df.from_domain, df.msgid_domain)],
        "missing_msgid": (~df.has_msgid).astype(float),
        "missing_date": (~df.has_date).astype(float),
        "no_mailer": (~df.has_xmailer).astype(float),
        "n_received": df.n_received.astype(float),
        "n_rcpt": df.n_rcpt.astype(float),
        "has_attachment": (df.attachments.map(len) > 0).astype(float),
        "risky_attachment": df.attachments.map(
            lambda a: float(any(os.path.splitext(f.lower())[1] in RISKY_EXT for f, _ in a))),
        "html_only": ((df.n_html > 0) & (df.n_plain == 0)).astype(float),
        "subj_upper_ratio": df.subject.map(_upper_ratio),
        "subj_exclaim": df.subject.str.contains("!", regex=False).astype(float),
        "body_upper_ratio": df.body.map(_upper_ratio),
        "log_body_len": np.log1p(df.body.str.len()),
    }, index=df.index)
    return out[list(INDICATORS)]


def meta_matrix(df: pd.DataFrame) -> np.ndarray:
    """Indicators with counts log-transformed, for scale-sensitive models."""
    X = indicators(df).copy()
    for c in ["n_urls", "url_density", "n_url_domains", "n_received", "n_rcpt"]:
        X[c] = np.log1p(X[c])
    return X.to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- #
# Text normalisation: subject + body; URLs, e-mails and numbers replaced by tokens.
# --------------------------------------------------------------------------- #
URL_T = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MAIL_T = re.compile(r"\S+@\S+\.\w+")
NUM_T = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def clean_text(subject: str, body: str, max_chars: int = 5000) -> str:
    t = (subject + " \n " + body[:max_chars]).lower()
    t = URL_T.sub(" urltoken ", t)
    t = MAIL_T.sub(" emailtoken ", t)
    t = NUM_T.sub(" numtoken ", t)
    return t


def texts(df: pd.DataFrame) -> list[str]:
    return [clean_text(s, b) for s, b in zip(df.subject, df.body)]


# --------------------------------------------------------------------------- #
def metrics(y, pred, score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"accuracy": (tp + tn) / len(y), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "fpr": fp / (fp + tn) if fp + tn else float("nan"),
           "mcc": matthews_corrcoef(y, pred) if len(set(y)) > 1 else float("nan"),
           "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "n": int(len(y))}
    if score is not None and len(set(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
    return out


def split(y, seed, test_size=0.2):
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)


def environment() -> dict:
    import sklearn
    info = {"python": platform.python_version(), "os": platform.platform(),
            "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                         if l.startswith("model name")), ""),
            "logical_cpus": os.cpu_count(),
            "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
            "gpu": "none", "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__}
    for m in ("torch", "xgboost", "networkx", "statsmodels"):
        try:
            info[m] = __import__(m).__version__
        except ImportError:
            pass
    return info
