"""Paired bootstrap: proposed pipelines vs the original Sketch-and-Verify protocol
(fixed K=10,M=3 with plain semantic vote), and vs flat sampling under both selectors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pool import run, boot
from pool.policies import BreadthSketch, FixedSketch, Flat, GatedFlat
from pool.replay import load_pool

pols = {
    "S&V plain": (FixedSketch(10, 3, selector="sv"), True),
    "BAS-Ex": (BreadthSketch(gate="examples"), True),
    "BAS-Dis": (BreadthSketch(), True),
    "GFS": (GatedFlat(n_extra=10), False),
    "BAS-Ex plain": (BreadthSketch(gate="examples", selector="sv"), True),
    "flat10 plain": (Flat(10, selector="sv"), False),
}
pairs = [("BAS-Ex", "S&V plain"), ("BAS-Dis", "S&V plain"), ("GFS", "S&V plain"),
         ("BAS-Ex plain", "S&V plain"), ("BAS-Ex", "flat10 plain"), ("BAS-Ex plain", "flat10 plain")]
for d in sys.argv[1:]:
    pools = load_pool(d)
    ids = [p["task_id"] for p in pools]
    res = {k: run(pol, pools, 10, dep) for k, (pol, dep) in pols.items()}
    print("==", d)
    for k, (acc, cov, tok) in res.items():
        print(f"  {k:14s} pass@1={100*sum(acc.values())/len(ids):5.1f}  tok={sum(tok.values())/len(ids):8.0f}")
    for a, b in pairs:
        m, lo, hi, p = boot(res[a][0], res[b][0], ids)
        print(f"  {a:14s} vs {b:13s}: {m:+5.1f} [{lo:+5.1f},{hi:+5.1f}] p={p:.4f}")
    sys.stdout.flush()
