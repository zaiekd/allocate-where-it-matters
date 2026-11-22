#!/usr/bin/env bash
# Official EvalPlus release files (already included in data/; re-download if missing).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -sSL -o data/HumanEvalPlus.jsonl.gz \
  https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz
curl -sSL -o data/MbppPlus.jsonl.gz \
  https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz
ls -la data
