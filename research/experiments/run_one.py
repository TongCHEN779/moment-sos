"""Run one solver on one MAX-CUT instance in its own process; print a JSON line with value, time, peak RSS."""
import sys, json, time, resource; sys.path.insert(0, ".")
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
solver, n, r = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 1800
pop = pb.maxcut(n, pb.graph("reg3", n, seed=1))
t = time.perf_counter()
if solver == "fo":
    d = SpectralDual(pop, r)
    U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=tl)
    val, evals = U, len(h)
    # time to first reach within 1e-2 / 1e-3 of final is analysed offline from h
    hist = h[:, [1, 3]].tolist()
else:
    val, _ = reference.moment_relaxation(pop, r, solver=solver)
    evals, hist = 0, []
dt = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30     # GiB (macOS reports bytes)
print("RESULT " + json.dumps(dict(solver=solver, n=n, r=r, val=val, time=dt, rss_gib=rss, evals=evals, hist=hist)))
