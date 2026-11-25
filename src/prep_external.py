"""Builds the two independent test sets used in experiment 7 (run after scripts/fetch_external.sh).

PhiUSIIL  Prasad & Chandra (2024): 235,795 URLs, label 1 = legitimate, 0 = phishing.
          Its legitimate URLs are site home pages only (https://www.<domain>), so it is used
          for TPR on phishing and FPR on legitimate home pages, never for one pooled accuracy.
PhishDB   Phishing.Database (commit 12a20bf, 2 Oct 2026): links listed as ACTIVE phishing.
          Phishing only, so only TPR (recall) can be measured; the list is community-maintained
          and may contain some mislabelled pages.

Processing, identical for both: strip the scheme as for Ebbu2017; drop duplicate URLs;
drop every URL whose registered domain occurs anywhere in Ebbu2017 (so the test is
domain-disjoint from all training data); draw 20,000 URLs per class with seed 42.
Outputs: data/external/phiusiil_sample.csv, data/external/phishdb_sample.csv, results/exp7_external_info.json
"""
import json
import sys

import numpy as np
import pandas as pd

from common import DATA, RESULTS, load_ebbu, normalise_url, reg_domain

N = 20000
ext = DATA.parent / "_ext"
ebbu_domains = set(load_ebbu().url.map(reg_domain))
info = {"ebbu_registered_domains": len(ebbu_domains)}


def clean(urls, labels, name):
    d = pd.DataFrame({"url": [normalise_url(str(u)) for u in urls], "label": labels})
    d = d[d.url.str.len() > 0]
    n0 = len(d)
    d = d.drop_duplicates("url")
    d = d[d.groupby("url").label.transform("nunique") == 1]
    d["domain"] = d.url.map(reg_domain)
    n1 = len(d)
    d = d[~d.domain.isin(ebbu_domains)]
    info[name] = {"rows": int(n0), "unique": int(n1), "after_removing_ebbu_domains": int(len(d)),
                  "by_label": {int(k): int(v) for k, v in d.label.value_counts().items()}}
    out = pd.concat([g.sample(min(N, len(g)), random_state=42) for _, g in d.groupby("label")])
    info[name]["sampled"] = {int(k): int(v) for k, v in out.label.value_counts().items()}
    info[name]["distinct_domains_sampled"] = int(out.domain.nunique())
    return out.reset_index(drop=True)


p = pd.read_csv(ext / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv", usecols=["URL", "label"],
                encoding="utf-8-sig")
clean(p.URL, 1 - p.label.to_numpy(), "phiusiil").to_csv(DATA / "external" / "phiusiil_sample.csv", index=False)

links = [l.strip() for l in open(ext / "phishdb" / "phishing-links-ACTIVE.txt", encoding="utf-8", errors="replace")
         if l.strip()]
clean(links, np.ones(len(links), int), "phishdb").to_csv(DATA / "external" / "phishdb_sample.csv", index=False)

json.dump(info, open(RESULTS / "exp7_external_info.json", "w"), indent=2)
print(json.dumps(info, indent=2))
