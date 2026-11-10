"""Sparse (correlative-sparsity) order-2 MAX-CUT on real Biq Mac torus instances.
python run_torus.py file [time_limit] [mosek 0/1].  Prints RESULT json."""
import sys, os, time, json, resource
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "src")); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments", "bench"))
import numpy as np, moment_sos as po
from instances import local_search_cut
f = sys.argv[1]; tl = float(sys.argv[2]) if len(sys.argv) > 2 else 600; do_mosek = int(sys.argv[3]) if len(sys.argv) > 3 else 0
L = open(f).read().split("\n"); n, m = map(int, L[0].split()); E, W = [], []
for l in L[1:m + 1]:
    i, j, w = l.split(); E.append((int(i) - 1, int(j) - 1)); W.append(float(w))
s = po.variables(n); cut = sum(w * (1 - s[i] * s[j]) / 2 for (i, j), w in zip(E, W)); P = po.Problem(cut, boolean=True)
LB = local_search_cut(n, E, W)
rec = dict(file=os.path.basename(f), n=n, edges=m, LB=LB)
t0 = time.time(); r1 = po.solve(P, order=1, cliques="auto", iters_per_stage=300, time_limit=tl); rec["U1"], rec["t1"] = float(r1.bound), time.time() - t0
t0 = time.time(); R = po.Relaxation(P, 2, "auto"); rec["blocks"] = len(R.blocks); rec["max_clique"] = max(len(c) for c in R.cliques)
r = po.solve(P, order=2, cliques="auto", iters_per_stage=int(os.environ.get("ITERS", 500)), time_limit=tl); rec["U2"], rec["t2"] = float(r.bound), time.time() - t0
rec["rel_gap"] = (rec["U2"] - LB) / abs(LB); rec["rel_gap1"] = (rec["U1"] - LB) / abs(LB)
rec["rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
print("RESULT " + json.dumps(rec), flush=True)
if do_mosek:
    t0 = time.time(); v, _ = po.solve_dual_ipm(R); rec["mosek"] = float(P.f_off + P.f_scale * v); rec["t_mosek"] = time.time() - t0
print("RESULT " + json.dumps(rec), flush=True)
