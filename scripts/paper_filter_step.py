"""Paired bootstrap for the selector step: Fixed(10,3) with SV-Ex vs with SV (original)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pool import run, boot
from pool.policies import FixedSketch
from pool.replay import load_pool
for d in sys.argv[1:]:
    pools = load_pool(d); ids = [p["task_id"] for p in pools]
    a = run(FixedSketch(10, 3, selector="sv_ex"), pools, 10, True)[0]
    b = run(FixedSketch(10, 3, selector="sv"), pools, 10, True)[0]
    m, lo, hi, p = boot(a, b, ids)
    print(f"{d}: {m:+.1f} [{lo:+.1f},{hi:+.1f}] p={p:.4f}", flush=True)
