"""Sensitivity of the smoothing solver (certification time and final accuracy) to the initial eta, the iterations per stage
and the L-BFGS memory, on g05 n=30, seeds 1-3, order 2.   usage: python sensitivity.py"""
import sys, os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import numpy as np
from instances import instance, local_search_cut
from msos import SpectralDual, problems as pb, solvers
BASE = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
out = open(os.path.join(HERE, "..", "results", "sensitivity.jsonl"), "w")
for seed in (1, 2, 3):
    n, edges, w = instance("g05", 30, seed); pop = pb.maxcut(n, edges, w); d = SpectralDual(pop, 2); LB = local_search_cut(n, edges, w)
    grid = [(e0, ips, 30) for e0 in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4) for ips in (200, 500, 1000)] + [(1e-3, 500, mc) for mc in (10, 100)]
    for e0, ips, mc in grid:
        etas = tuple(e for e in BASE if e <= e0 * 1.0001)
        t = time.perf_counter(); U, th, h = solvers.lbfgs_continuation(d, etas=etas, iters_per_stage=ips, maxcor=mc, time_limit=300); el = time.perf_counter() - t
        ok = np.where(np.floor(h[:, 3] + 1e-9) <= LB + 1e-9)[0]
        out.write(json.dumps(dict(seed=seed, eta0=e0, iters=ips, maxcor=mc, LB=LB, U=U, gap=U - LB, t_cert=float(h[ok[0], 1]) if len(ok) else None, evals_cert=int(h[ok[0], 0]) if len(ok) else None, total=el, evals=len(h), load0=os.getloadavg()[0])) + "\n"); out.flush()
print("done")
