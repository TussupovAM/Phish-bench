#!/usr/bin/env bash
# Re-creates data/ exactly as used in the paper (pinned commits + SHA-256 check).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data _ext

git clone -q https://github.com/fafal-abnir/phishing_detection _ext/uci
git -C _ext/uci checkout -q 57c2cef086e85f102a943b259324fe0be8b52ade
cp _ext/uci/dataset.csv data/uci_phishing_websites.csv

git clone -q https://github.com/ebubekirbbr/pdd _ext/ebbu
git -C _ext/ebbu checkout -q 90b394d8f9fb5d5dc48400e8bf46b001ce42999f
cp _ext/ebbu/input/data_phishing_37175.json _ext/ebbu/input/data_legitimate_36400.json data/

sha256sum -c <<'EOF'
83853a3ddf39a77e4bda0c8c907c5a7f521f16ed844299de75d215c22d5bd379  data/uci_phishing_websites.csv
c513d1f11b0ce159d43a23b465af2a65f752c6f65a816c2a8084f2a2adf2ebf6  data/data_phishing_37175.json
5e8fbcc88d6ba25f46bd85ed48d8aed9f56e2afc9deb712d13496112b8158ab4  data/data_legitimate_36400.json
EOF
