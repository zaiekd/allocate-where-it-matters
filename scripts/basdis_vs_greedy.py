"""Paired bootstrap: BAS-Dis (and BAS-Ex) vs greedy and vs flat10 (both svex)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pool import run, boot
from pool.policies import BreadthSketch, Flat
from pool.replay import load_pool
pols = {"greedy": (Flat(1), False), "BAS-Dis": (BreadthSketch(), True),
        "BAS-Ex": (BreadthSketch(gate="examples"), True), "flat10 ex": (Flat(10, selector="sv_ex"), False)}
pairs = [("BAS-Dis", "greedy"), ("BAS-Ex", "greedy"), ("BAS-Dis", "flat10 ex")]
for d in sys.argv[1:]:
    pools = load_pool(d); ids = [p["task_id"] for p in pools]
    res = {k: run(pol, pools, 10, dep) for k, (pol, dep) in pols.items()}
    print("==", d)
    for k, (acc, cov, tok) in res.items():
        print(f"  {k:10s} pass@1={100*sum(acc.values())/len(ids):5.1f} tok={sum(tok.values())/len(ids):8.0f}")
    for a, b in pairs:
        m, lo, hi, p = boot(res[a][0], res[b][0], ids)
        print(f"  {a} vs {b}: {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] p={p:.4f}")
    sys.stdout.flush()
