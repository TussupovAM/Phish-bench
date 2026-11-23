# E-mail study (research practice, weeks 2–9)

Experiments on full e-mails (headers + body) for the research-practice report.

| Script | Plan week | What it does | Main output |
|--------|-----------|--------------|-------------|
| `src/parse.py` | 2 | parses SpamAssassin, CSDMC2010 and a 2,000-message sample of Phishing Pot; removes duplicates; marks CSDMC messages also present in SpamAssassin | `results/emails.pkl` (not versioned), `results/e0_dataset_stats.json` |
| `src/e1_anomaly.py` | 3 | Isolation Forest, One-Class SVM, DBSCAN fitted on ham only (novelty detection) | `results/e1_anomaly_summary.csv` |
| `src/e2_classify.py` | 4–5 | ANN (TF-IDF), CNN, LSTM + Naive Bayes / Logistic Regression baselines; grid search on validation data | `results/e2_tuning.csv`, `results/e2_summary.csv` |
| `src/e3_graph.py` | 6–7 | sender – origin network – recipient graph; SymNMF communities; ERGM fitted by MPLE | `results/e3_*.csv`, `results/e3_graph_stats.json` |
| `src/e4_robustness.py` | 8 | cross-corpus transfer, modern phishing (Phishing Pot), text perturbations | `results/e4_cross.csv`, `results/e4_perturb.csv` |
| `src/e5_indicators.py` | 9 | statistical check of 21 risk indicators on both corpora | `results/e5_indicators.csv`, `results/e5_groups.csv` |

```bash
bash scripts/fetch_email_data.sh      # pinned corpus versions
cd src
python3 parse.py && python3 e1_anomaly.py && python3 e2_classify.py && python3 e3_graph.py \
  && python3 e4_robustness.py && python3 e5_indicators.py && python3 fig_smtp.py && python3 fig_graph.py
```

Notes
- Positive class = spam/phishing. In CSDMC2010's `SPAMTrain.label`, 1 = ham and 0 = spam.
- 2,052 of 4,080 unique CSDMC2010 messages have the same text as a SpamAssassin message;
  cross-corpus tests exclude texts present in the training corpus.
- Phishing Pot contains only phishing, so only recall can be measured on it. Its repository
  also contains a CSV with personal data; it is not used or copied.
- `e2_classify.py` reuses finished tuning results from `results/e2_tuning.csv` if a run is
  interrupted (the CSDMC2010 part was resumed this way; results were identical).
- Environment: `results/environment.json` (Intel Xeon 2.10 GHz, 2 vCPU, 7.8 GB RAM, no GPU,
  Python 3.13, scikit-learn 1.9.1, PyTorch 2.15.0 CPU).
