"""One run on a real Biq Mac / rudy file: python run_real.py path.  Prints one JSON line."""
import sys, os, json, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from instances import instance, local_search_cut
from msos import SpectralDual, problems as pb, solvers, reference

path = sys.argv[1]; do_mosek = 0
L = open(path).read().split("\n"); n, m = map(int, L[0].split())
edges, w = [], []
for l in L[1:m + 1]:
    i, j, x = l.split(); edges.append((int(i) - 1, int(j) - 1)); w.append(float(x))
family, seed = os.path.basename(path), 0
pop = pb.maxcut(n, edges, w)
load0 = os.getloadavg()[0]
d = SpectralDual(pop, 2)
t0 = time.perf_counter(); c0 = time.process_time()
U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=1500)
wall, cpu = time.perf_counter() - t0, time.process_time() - c0
LB = local_search_cut(n, edges, w)
hist = h[:, [1, 3]]                                    # (time, best U)
integer = all(float(x).is_integer() for x in w)
t_cert = None
if integer:
    target = np.floor(U + 1e-9)
    ok = np.where(np.floor(hist[:, 1] + 1e-9) <= LB + 1e-9)[0]            # floor(U_k) == best cut (cert. optimality)
    t_cert = float(hist[ok[0], 0]) if (len(ok) and LB >= np.floor(U + 1e-9) - 1e-9) else None
rec = dict(family=family, n=n, seed=seed, edges=len(edges), s=d.sizes()[0], U=float(U), LB=LB, gap_int=float(np.floor(U + 1e-9) - LB) if integer else None,
           certified=bool(integer and LB >= np.floor(U + 1e-9) - 1e-9), t_cert=t_cert, wall=wall, cpu=cpu, evals=len(h), load0=load0,
           rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, hist=hist[::max(1, len(hist) // 200)].tolist())
if do_mosek:
    t1 = time.perf_counter(); v, _ = reference.moment_relaxation(pop, 2); rec["mosek_val"], rec["mosek_wall"] = v, time.perf_counter() - t1
print("RESULT " + json.dumps(rec))
