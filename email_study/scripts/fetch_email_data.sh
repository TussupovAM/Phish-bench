#!/usr/bin/env bash
# Downloads the three e-mail corpora into email_study/data/ from pinned versions.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data _ext

# 1) SpamAssassin public corpus, as packaged on npm by stdlib (raw .txt messages)
(cd _ext && npm pack @stdlib/datasets-spam-assassin@0.2.3 -q && tar xzf stdlib-datasets-spam-assassin-0.2.3.tgz)
mkdir -p data/spamassassin
for g in easy-ham-1 easy-ham-2 hard-ham-1 spam-1 spam-2; do
  mkdir -p data/spamassassin/$g && cp _ext/package/data/$g/*.txt data/spamassassin/$g/
done

# 2) CSDMC2010 SPAM corpus (TRAINING part + labels), mirrored in hexgnu/spam_filter
git clone -q https://github.com/hexgnu/spam_filter _ext/csdmc
git -C _ext/csdmc checkout -q f0cfe3440d93ad2a2354224ed10d095a31bfa224
mkdir -p data/csdmc2010 && cp -r _ext/csdmc/data/TRAINING _ext/csdmc/data/SPAMTrain.label data/csdmc2010/

# 3) Phishing Pot (real phishing e-mails) -- only the .eml samples are used
GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/rf-peixoto/phishing_pot _ext/pp
git -C _ext/pp fetch -q --depth 1 origin 78be141545ed9e56eefb857808ac78e83eb81b81
git -C _ext/pp checkout -q 78be141545ed9e56eefb857808ac78e83eb81b81
mkdir -p data/phishing_pot/email && cp _ext/pp/email/*.eml data/phishing_pot/email/

echo "SA: $(find data/spamassassin -name '*.txt' | wc -l) (expect 6046)"
echo "CSDMC: $(ls data/csdmc2010/TRAINING | wc -l) (expect 4327)"
echo "PP: $(ls data/phishing_pot/email | wc -l) (expect 12864)"
