"""E3 (weeks 6-7) — graph of e-mail interactions, NMF communities and an ERGM.

Messages: SA + CSDMC (unique; CSDMC messages already present in SA are dropped so no
message is counted twice). Phishing Pot is not used here (different era and mailboxes).

Graph "sender -> route -> recipient" (undirected, weighted by message count):
  S:<registered domain of From>  --  N:<origin network /24>  --  R:<registered domain of each To/Cc>
The origin network is the /24 of the first public IP in the Received chain (oldest hop),
i.e. where the message entered the Internet. If it is unknown, S is linked to R directly.

NMF communities: symmetric NMF on log(1+weights) adjacency; each node joins the component
with the largest loading. The rank k in {10, 20, 40, 80} is chosen by weighted modularity
(labels are not used). A message belongs to its sender node's community.

Evaluation of the structure (labels used only here):
  * community spam share and "suspicious subgraphs": communities with >= 20 messages and
    >= 90% spam;
  * predictive check: for stratified 80/20 splits (seeds 42-44) the spam share of each
    community is estimated on training messages only (Laplace-smoothed) and used as a score
    for test messages; ROC-AUC on all test messages and on "cold-start" test messages
    whose sender domain never appears in training. Baseline: sender-domain spam share.

ERGM: one-mode graph of sender domains, an edge if two domains share an origin network.
Node attribute spam = majority of the domain's messages are spam. Terms: edges,
nodefactor(spam), nodematch(spam), triangles. Fitted by maximum pseudo-likelihood (MPLE,
Strauss & Ikeda 1990) as a logistic regression of dyad states on change statistics, with
all edges and a 1:20 random sample of non-edges (case-control; only the edges coefficient
is shifted, by log of the sampling rate, and is corrected). MPLE standard errors are
known to be optimistic, so they are reported but not over-interpreted.

Outputs: results/e3_graph_stats.json, e3_communities.csv, e3_predictive.csv, e3_ergm.csv
"""
import ipaddress
import json

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score

from features import RESULTS, SEEDS, load, split
from parse import reg_domain

df = load(["SA", "CSDMC"])
df = df[~df.dup_of_other].reset_index(drop=True)
y = df.label.to_numpy()


def net24(ip):
    return str(ipaddress.ip_network(ip + "/24", strict=False)) if ip else ""


df["S"] = "S:" + df.from_domain.map(reg_domain).replace("", "unknown")
df["N"] = df.origin_ip.map(net24).map(lambda x: "N:" + x if x else "")
df["R"] = df.rcpt_domains.map(lambda ds: sorted({"R:" + reg_domain(d) for d in ds if d}))

edges = {}
for s, n, rs in zip(df.S, df.N, df.R):
    route = [s] + ([n] if n else [])
    for a, b in zip(route[:-1], route[1:]):
        edges[(a, b)] = edges.get((a, b), 0) + 1
    for r in rs:
        edges[(route[-1], r)] = edges.get((route[-1], r), 0) + 1
G = nx.Graph()
for (a, b), w in edges.items():
    if a != b:
        G.add_edge(a, b, weight=w)
nodes = list(G.nodes())
ix = {n: i for i, n in enumerate(nodes)}
A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr").astype(float)
A.data = np.log1p(A.data)

stats = {"messages": int(len(df)), "spam": int(y.sum()), "nodes": G.number_of_nodes(),
         "edges": G.number_of_edges(),
         "node_types": {t: sum(n.startswith(t + ":") for n in nodes) for t in "SNR"},
         "messages_with_origin_ip": float((df.N != "").mean()),
         "connected_components": nx.number_connected_components(G),
         "largest_component_nodes": len(max(nx.connected_components(G), key=len))}

# ---- NMF communities ------------------------------------------------------------------
best = None
for k in [10, 20, 40, 80]:
    W = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=500).fit_transform(A)
    lab = np.where(W.max(1) > 0, W.argmax(1), -1)
    parts = [set(np.array(nodes)[lab == c]) for c in np.unique(lab)]
    q = nx.community.modularity(G, parts, weight="weight")
    print(f"NMF k={k}: modularity={q:.4f}", flush=True)
    stats.setdefault("nmf_modularity", {})[k] = q
    if best is None or q > best[0]:
        best = (q, k, lab)
q, k, lab = best
stats.update(nmf_k=k, nmf_modularity_best=q)
comm_of = {n: int(lab[ix[n]]) for n in nodes}
df["community"] = df.S.map(lambda s: comm_of.get(s, -1))

c = df.groupby("community").agg(messages=("label", "size"), spam=("label", "sum"),
                                 senders=("S", "nunique"), origin_nets=("N", lambda s: s[s != ""].nunique()))
c["nodes"] = pd.Series(lab).value_counts().reindex(c.index).fillna(0).astype(int)
c["spam_share"] = c.spam / c.messages
c["suspicious"] = (c.messages >= 20) & (c.spam_share >= 0.9)
c["legit_dominated"] = (c.messages >= 20) & (c.spam_share <= 0.1)
c.sort_values("messages", ascending=False).to_csv(RESULTS / "e3_communities.csv")
stats["suspicious_subgraphs"] = int(c.suspicious.sum())
stats["suspicious_messages"] = int(c[c.suspicious].messages.sum())
stats["suspicious_spam_captured_share"] = float(c[c.suspicious].spam.sum() / y.sum())
stats["suspicious_ham_in_them"] = int((c[c.suspicious].messages - c[c.suspicious].spam).sum())
stats["legit_dominated_subgraphs"] = int(c.legit_dominated.sum())
stats["mixed_communities_ge20"] = int(((c.messages >= 20) & ~c.suspicious & ~c.legit_dominated).sum())

# ---- predictive check ------------------------------------------------------------------
rows = []
for seed in SEEDS:
    tr, te = split(y, seed)
    prior = y[tr].mean()
    trd = df.iloc[tr]
    comm_rate = ((trd.groupby("community").label.sum() + prior) / (trd.groupby("community").size() + 1))
    dom_rate = ((trd.groupby("S").label.sum() + prior) / (trd.groupby("S").size() + 1))
    ted = df.iloc[te]
    s_comm = ted.community.map(comm_rate).fillna(prior).to_numpy()
    s_dom = ted.S.map(dom_rate).fillna(prior).to_numpy()
    cold = ~ted.S.isin(set(trd.S)).to_numpy()
    for name, s in [("NMF community spam share", s_comm), ("Sender-domain spam share", s_dom)]:
        rows.append(dict(seed=seed, score=name, subset="all test", n=len(te),
                         roc_auc=roc_auc_score(y[te], s)))
        rows.append(dict(seed=seed, score=name, subset="cold-start senders", n=int(cold.sum()),
                         spam_share=float(y[te][cold].mean()),
                         roc_auc=roc_auc_score(y[te][cold], s[cold])))
pred = pd.DataFrame(rows)
pred.to_csv(RESULTS / "e3_predictive_runs.csv", index=False)
pred.groupby(["score", "subset"]).agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                                      n=("n", "mean")).reset_index().to_csv(RESULTS / "e3_predictive.csv", index=False)
print(pd.read_csv(RESULTS / "e3_predictive.csv").round(4).to_string(), flush=True)

# ---- ERGM by MPLE on the sender projection ------------------------------------------
senders = sorted(df.loc[df.N != "", "S"].unique())
si = {s: i for i, s in enumerate(senders)}
nets = sorted(df.loc[df.N != "", "N"].unique())
ni = {n: i for i, n in enumerate(nets)}
sub = df[df.N != ""].drop_duplicates(["S", "N"])
B = sp.csr_matrix((np.ones(len(sub)), (sub.S.map(si), sub.N.map(ni))), shape=(len(senders), len(nets)))
P = (B @ B.T).tocsr()
P.setdiag(0); P.eliminate_zeros()
P.data[:] = 1
n = len(senders)
spam_node = df[df.N != ""].groupby("S").label.mean().reindex(senders).to_numpy() >= 0.5
CN = (P @ P).tocsr()  # common neighbours = change statistic of the triangle count
rng = np.random.default_rng(42)
iu, ju = sp.triu(P, k=1).nonzero()
n_dyads = n * (n - 1) // 2
n_edges = len(iu)
m = min(20 * n_edges, n_dyads - n_edges)
ri = rng.integers(0, n, size=int(m * 1.3)); rj = rng.integers(0, n, size=int(m * 1.3))
keep = (ri < rj) & (np.asarray(P[ri, rj]).ravel() == 0)
ri, rj = ri[keep], rj[keep]
pairs = np.unique(np.stack([ri, rj], 1), axis=0)[:m]
ri, rj = pairs[:, 0], pairs[:, 1]
I = np.concatenate([iu, ri]); J = np.concatenate([ju, rj])
Y = np.concatenate([np.ones(len(iu)), np.zeros(len(ri))])
Xd = pd.DataFrame({
    "edges": 1.0,
    "nodefactor.spam": spam_node[I].astype(float) + spam_node[J].astype(float),
    "nodematch.spam": (spam_node[I] == spam_node[J]).astype(float),
    "triangle": np.asarray(CN[I, J]).ravel().astype(float),
})
fit = sm.Logit(Y, Xd).fit(disp=0)
rate = len(ri) / (n_dyads - n_edges)
coef = fit.params.copy(); coef["edges"] += np.log(rate)
ergm = pd.DataFrame({"coef": coef, "se_mple": fit.bse, "z": fit.params / fit.bse,
                     "odds_ratio": np.exp(coef)})
ergm.to_csv(RESULTS / "e3_ergm.csv")
stats["ergm"] = {"senders": n, "spam_senders": int(spam_node.sum()), "edges": int(n_edges),
                 "density": n_edges / n_dyads, "sampled_non_edges": int(len(ri)),
                 "nonedge_sampling_rate": rate}
# homophily in plain numbers: share of edges between two spam / two ham / mixed senders
same_spam = (spam_node[iu] & spam_node[ju]).sum(); same_ham = (~spam_node[iu] & ~spam_node[ju]).sum()
stats["ergm"]["edge_types"] = {"spam-spam": int(same_spam), "ham-ham": int(same_ham),
                               "mixed": int(n_edges - same_spam - same_ham)}
json.dump(stats, open(RESULTS / "e3_graph_stats.json", "w"), indent=2, default=float)
print(json.dumps(stats, indent=1, default=float))
print(ergm.round(4).to_string())
