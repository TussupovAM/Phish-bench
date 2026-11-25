#!/usr/bin/env bash
# Independent test sources for experiment 7 (pinned versions, SHA-256 checked).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p _ext/phiusiil

# PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024; UCI id 967), Git-LFS copy on GitHub
curl -sL -o _ext/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv \
  "https://media.githubusercontent.com/media/ManojK1104/Phishing_Website_Detection/8e23e98787fbc63cf062109ff3bc5ecaea1991da/data/PhiUSIIL_Phishing_URL_Dataset.csv"

# Phishing.Database, ACTIVE phishing links as of 2 Oct 2026
git clone -q --filter=blob:none --no-checkout https://github.com/Phishing-Database/Phishing.Database _ext/phishdb
git -C _ext/phishdb checkout -q 12a20bf8dd2b7e78d9d38fc12b3e5c970f79d1a7 -- phishing-links-ACTIVE.txt

sha256sum -c <<'SUMS'
a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6  _ext/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv
be4a179b3ac92b7135595b459c2c9867fe681c12209489dd99517db8407a67c6  _ext/phishdb/phishing-links-ACTIVE.txt
SUMS
(cd src && python3 prep_external.py)
