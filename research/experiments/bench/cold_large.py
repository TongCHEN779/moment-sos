import sys, os, time
sys.path.insert(0, "/Users/tchen/Documents/Python/moment-sos")
import numpy as np
from msos import SpectralDual, problems as pb, solvers
n, LB = int(sys.argv[1]), int(sys.argv[2])
pop = pb.maxcut(n, pb.graph("reg3", n, seed=1)); d = SpectralDual(pop, 2)
print("load", os.getloadavg()[0], "block", d.sizes()[0], flush=True)
t = time.perf_counter()
U, th, h = solvers.lbfgs_continuation(d, etas=(1e-3, 3e-4, 1e-4), iters_per_stage=300, time_limit=900)
ok = np.where(np.floor(h[:, 3] + 1e-9) <= LB + 1e-9)[0]
print(f"n={n}: certified optimal (floor U = {LB}) after {h[ok[0],1] if len(ok) else None}s ({int(h[ok[0],0]) if len(ok) else None} evals); best U {U:.4f}; total {time.perf_counter()-t:.0f}s; recheck {d.certificate(th):.4f}")
