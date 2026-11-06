#!/usr/bin/env bash
# Runs every experiment in order; each writes its CSV/JSON into results/.
set -euo pipefail
cd "$(dirname "$0")/../src"
python3 exp1_uci_ml.py      | tee ../results/exp1_log.txt
python3 exp2b_uci_cnn.py    | tee ../results/exp2b_log.txt
python3 exp2_urls.py        | tee ../results/exp2_log.txt
python3 exp3_imbalance.py   | tee ../results/exp3_log.txt
python3 exp4_latency.py     | tee ../results/exp4_log.txt
python3 exp5_cascade.py     | tee ../results/exp5_log.txt
python3 make_tables.py
