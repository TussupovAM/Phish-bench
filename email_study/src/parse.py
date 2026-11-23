"""Parse raw RFC 822 e-mails of the three corpora into one table (results/emails.pkl).

Corpora (fetched by scripts/fetch_email_data.sh):
  SA     SpamAssassin public corpus (npm @stdlib/datasets-spam-assassin@0.2.3):
         easy-ham-1/2, hard-ham-1 = ham; spam-1/2 = spam. 6,046 messages.
  CSDMC  CSDMC2010 SPAM corpus, TRAINING part (4,327 messages). In SPAMTrain.label
         the label 1 means HAM and 0 means SPAM (2,949 ham / 1,378 spam, as in readme).
  PP     Phishing Pot (real phishing e-mails, 2022+). Used only as out-of-distribution
         control data; a fixed random sample of 2,000 messages (seed 42).

Positive class (label = 1) is always spam/phishing.
Duplicates: messages with identical normalised text (subject + body) are dropped within
each corpus (first occurrence kept); `dup_of_other` marks CSDMC messages whose text also
occurs in SA, so cross-corpus tests can exclude them.
"""
from __future__ import annotations

import email
import hashlib
import ipaddress
import os
import random
import re
import sys
import warnings
from email import policy
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

URL_RE = re.compile(r"""(?i)\b(?:https?://|www\.)[^\s<>"'()]+""")
RECV_FROM = re.compile(r"from\s+(\S+)(?:\s*\(([^)]*)\))?", re.I | re.S)
IP_RE = re.compile(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?")
RISKY_EXT = {".exe", ".scr", ".zip", ".rar", ".js", ".vbs", ".bat", ".com", ".pif",
             ".docm", ".xlsm", ".doc", ".xls", ".html", ".htm", ".iso", ".img", ".lnk"}


def domain_of(addr: str) -> str:
    a = parseaddr(addr or "")[1].lower().strip("<>")
    return a.rsplit("@", 1)[1].strip(".") if "@" in a else ""


def reg_domain(host: str) -> str:
    """Crude registered domain: last two labels (three for 2-letter ccTLD + short SLD)."""
    host = (host or "").lower().strip(".[]")
    if not host or IP_RE.fullmatch(host):
        return host
    p = [x for x in host.split(".") if x]
    if len(p) >= 3 and len(p[-1]) == 2 and len(p[-2]) <= 3:
        return ".".join(p[-3:])
    return ".".join(p[-2:])


def is_public_ip(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
        return not (ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local
                    or ip.is_unspecified or ip.is_multicast)
    except ValueError:
        return False


def received_hops(msg):
    """Received headers, oldest hop first: (claimed HELO host, connecting IP)."""
    hops = []
    for h in reversed(msg.get_all("Received") or []):
        h = str(h)
        m = RECV_FROM.search(h)
        if not m:
            continue
        helo, paren = m.group(1), m.group(2) or ""
        ips = IP_RE.findall(paren) or IP_RE.findall(helo)
        hops.append((helo.lower().strip("[]"), ips[0] if ips else ""))
    return hops


def text_parts(msg):
    plain, html, attachments = [], [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        fname = part.get_filename()
        disp = str(part.get("Content-Disposition") or "").lower()
        if fname or "attachment" in disp:
            attachments.append((fname or "", ctype))
            continue
        if part.get_content_maintype() != "text":
            if part.get_content_maintype() in ("application", "image", "audio", "video"):
                attachments.append((fname or "", ctype))
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            cs = part.get_content_charset() or "latin-1"
            txt = payload.decode(cs, errors="replace")
        except (LookupError, AssertionError):
            txt = (part.get_payload(decode=True) or b"").decode("latin-1", errors="replace")
        (html if ctype == "text/html" else plain).append(txt)
    return plain, html, attachments


def parse_message(raw: bytes) -> dict:
    msg = email.message_from_bytes(raw, policy=policy.compat32)
    plain, html, attachments = text_parts(msg)
    anchors_mismatch = 0
    html_text = []
    hrefs = []
    for h in html:
        soup = BeautifulSoup(h, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            hrefs.append(href)
            shown = a.get_text(" ", strip=True)
            m = URL_RE.search(shown)
            if m and href.lower().startswith("http"):
                d_shown = reg_domain(re.sub(r"^(https?://)?", "", m.group(0)).split("/")[0])
                d_href = reg_domain(re.sub(r"^https?://", "", href).split("/")[0].split("@")[-1])
                anchors_mismatch += int(bool(d_shown) and d_shown != d_href)
        html_text.append(soup.get_text(" ", strip=True))
    body = "\n".join(plain) if plain else "\n".join(html_text)
    urls = URL_RE.findall("\n".join(plain + html_text)) + [h for h in hrefs if h.lower().startswith(("http", "www"))]
    urls = list(dict.fromkeys(u.rstrip(".,;:!?") for u in urls))
    url_hosts = [re.sub(r"^https?://", "", u, flags=re.I).split("/")[0].split("@")[-1].split(":")[0].lower()
                 for u in urls]
    hops = received_hops(msg)
    origin_ip = next((ip for _, ip in hops if is_public_ip(ip)), "")
    origin_helo = next((helo for helo, ip in hops if is_public_ip(ip)), "")
    try:
        date = parsedate_to_datetime(str(msg.get("Date"))) if msg.get("Date") else None
        date = date.replace(tzinfo=None) if date else None
    except (TypeError, ValueError, IndexError):
        date = None
    subj = str(msg.get("Subject") or "")
    try:
        subj = str(email.header.make_header(email.header.decode_header(subj)))
    except Exception:
        pass
    rcpts = getaddresses([str(x) for x in (msg.get_all("To") or []) + (msg.get_all("Cc") or [])])
    return {
        "subject": subj,
        "body": body,
        "from_domain": domain_of(str(msg.get("From") or "")),
        "reply_to_domain": domain_of(str(msg.get("Reply-To") or "")),
        "return_path_domain": domain_of(str(msg.get("Return-Path") or "")),
        "msgid_domain": (re.findall(r"@([^>\s]+)", str(msg.get("Message-ID") or "")) or [""])[0].lower(),
        "rcpt_domains": sorted({a.rsplit("@", 1)[1].lower() for _, a in rcpts if "@" in a}),
        "n_rcpt": len([a for _, a in rcpts if a]),
        "has_date": msg.get("Date") is not None,
        "has_msgid": msg.get("Message-ID") is not None,
        "has_xmailer": msg.get("X-Mailer") is not None or msg.get("User-Agent") is not None,
        "date": date,
        "hops": hops,
        "n_received": len(msg.get_all("Received") or []),
        "origin_ip": origin_ip,
        "origin_helo": origin_helo,
        "urls": urls,
        "url_hosts": url_hosts,
        "anchor_mismatch": anchors_mismatch,
        "n_plain": len(plain),
        "n_html": len(html),
        "attachments": attachments,
    }


def norm_hash(subject: str, body: str) -> str:
    t = re.sub(r"\s+", " ", (subject + " " + body).lower()).strip()
    return hashlib.md5(t.encode("utf-8", "replace")).hexdigest()


def load_raw():
    rows = []
    sa = DATA / "spamassassin"
    for group in ["easy-ham-1", "easy-ham-2", "hard-ham-1", "spam-1", "spam-2"]:
        for f in sorted((sa / group).glob("*.txt")):
            rows.append(("SA", group, f.name, int(group.startswith("spam")), f.read_bytes()))
    cs = DATA / "csdmc2010"
    for line in (cs / "SPAMTrain.label").read_text().split("\n"):
        if line.strip():
            lab, name = line.split()
            rows.append(("CSDMC", "train", name, 1 - int(lab), (cs / "TRAINING" / name).read_bytes()))
    pp = sorted((DATA / "phishing_pot" / "email").glob("*.eml"))
    random.Random(42).shuffle(pp)
    for f in pp[:2000]:
        rows.append(("PP", "phishing", f.name, 1, f.read_bytes()))
    return rows


if __name__ == "__main__":
    recs = []
    raw = load_raw()
    for i, (ds, group, name, label, b) in enumerate(raw):
        try:
            r = parse_message(b)
            r["parse_error"] = False
        except Exception as e:  # keep the row so failures are counted, not hidden
            r = {"subject": "", "body": "", "parse_error": True, "error": repr(e)[:200]}
        r.update(dataset=ds, group=group, file=name, label=label, raw_bytes=len(b))
        recs.append(r)
        if i % 2000 == 0:
            print(i, "/", len(raw), flush=True)
    df = pd.DataFrame(recs)
    df["hash"] = [norm_hash(s, b) for s, b in zip(df.subject, df.body)]
    n0 = df.groupby("dataset").size().to_dict()
    df["dup_within"] = df.duplicated(subset=["dataset", "hash"], keep="first")
    sa_hashes = set(df[(df.dataset == "SA")].hash)
    df["dup_of_other"] = (df.dataset != "SA") & df.hash.isin(sa_hashes)
    stats = {
        "raw_counts": n0,
        "parse_errors": df.groupby("dataset").parse_error.sum().to_dict(),
        "dups_within": df.groupby("dataset").dup_within.sum().to_dict(),
        "csdmc_also_in_sa": int(df[(df.dataset == "CSDMC") & ~df.dup_within].dup_of_other.sum()),
    }
    df = df[~df.dup_within & ~df.parse_error].reset_index(drop=True)
    stats["after_dedup"] = df.groupby(["dataset", "label"]).size().unstack().to_dict()
    print(stats)
    pd.Series(stats).to_json(RESULTS / "e0_dataset_stats.json", indent=2, force_ascii=False)
    df.to_pickle(RESULTS / "emails.pkl")
