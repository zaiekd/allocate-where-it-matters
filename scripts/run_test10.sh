#!/usr/bin/env bash
# End-to-end smoke test on 10 problems with the real API:  bash scripts/run_test10.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY=${PYTHON:-python3}
TS=$(date +%Y%m%d_%H%M%S)
echo "== python: $($PY --version)"
echo "== 1) Check API connectivity"
$PY scripts/check_api.py || exit 1
echo "== 2) Build candidate pools for the first 10 HumanEval+ problems"
$PY scripts/build_pool.py --benchmark humaneval --n-problems 10 --parallel 2 2>&1 | tee "logs/test10_humaneval_$TS.log"
echo "== 3) Replay the original paper's fixed configurations offline"
POOLDIR=$(ls -td pools/humaneval/*/ | grep -v mock | head -1)   # most recently built pool
$PY scripts/inspect_pool.py "$POOLDIR" 2>&1 | tee "logs/test10_inspect_$TS.txt"
echo "== Done. Results are in $POOLDIR and logs/*_$TS.*"
