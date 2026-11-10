"""Structured POP benchmark: python run_struct.py kind n seed   kind in {boxqp, chain_sparse, chain_dense}
Order 2.  Reports the interior-point value (MOSEK, independent primal moment model), the smoothing solver with the default
schedule and the dual ALM (rho0=1000): final certified gap U - IPM and wall time.  One JSON line."""
import sys, os, json, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
kind, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if kind == "boxqp":
    pop, cl = pb.box_qp(n, seed), None
else:
    pop, cl = pb.chain_quartic(n, seed)
    if kind == "chain_dense": cl = None
d = SpectralDual(pop, 2, cliques=cl)
load0 = os.getloadavg()[0]
t = time.perf_counter(); ipm, _ = reference.moment_relaxation(pop, 2, cliques=cl); t_ipm = time.perf_counter() - t
t = time.perf_counter(); U1, _, h1 = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=900); t_sm = time.perf_counter() - t
t = time.perf_counter(); U2, _, h2, info = solvers.alm(d, time_limit=900); t_alm = time.perf_counter() - t
def t_to(h, tol):
    ok = np.where(h[:, 3] - ipm <= tol)[0]; return float(h[ok[0], 1]) if len(ok) else None
rec = dict(kind=kind, n=n, seed=seed, blocks=len(d.blocks), maxblock=max(d.sizes()), mult=d.dim, ipm=ipm, t_ipm=t_ipm,
           smooth=U1, gap_smooth=U1 - ipm, t_smooth=t_sm, t1e3_smooth=t_to(h1, 1e-3), alm=U2, gap_alm=U2 - ipm, t_alm=t_alm, t1e3_alm=t_to(h2, 1e-3),
           load0=load0, rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)
print("RESULT " + json.dumps(rec))
