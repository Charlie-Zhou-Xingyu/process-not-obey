#!/usr/bin/env bash
# Clone the benchmark / reference repos we read data and constants from, at pinned commits.
# Nothing here is imported as code: SecFid's evaluate.py is run unmodified in a subprocess,
# the other repos are only parsed for literals (attack templates, delimiter sets, datasets).
#
#   bash scripts/fetch_third_party.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

fetch() {  # name url commit
  local name=$1 url=$2 commit=$3 dir="third_party/$1"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$commit"
  echo "$name @ $(git -C "$dir" rev-parse --short HEAD)"
}

# SecFid benchmark + official scorer (code MIT, data CC BY-SA 4.0)
fetch sec-fid       https://github.com/mhermon/sec-fid.git                               c316d5d32cad1413de22b4c55fce6ad1fe2eb4e3
# SEP dataset
fetch SEP           https://github.com/egozverev/Should-It-Be-Executed-Or-Processed.git  7606c0696f20f5aa433169fd2221f76852d1d4f5
# Meta-SecAlign: test-time "ignore" templates and fake-delimiter sets (CC BY-NC 4.0)
fetch Meta_SecAlign https://github.com/facebookresearch/Meta_SecAlign.git               2031502bdf5a046b437ec835c6b05ff468aa0699
# SecPO: seed-template pool used to tag released injections as refined vs unrefined (Apache-2.0)
fetch SecPO         https://github.com/Sizhe-Chen/SecPO.git                              da0372a0e598badf4e51c98c175a962147cac2ac
