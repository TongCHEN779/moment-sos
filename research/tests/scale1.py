import sys, time; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
n = int(sys.argv[1]); r = int(sys.argv[2])
pop = pb.maxcut(n, pb.graph("reg3", n, seed=1))
d = SpectralDual(pop, r)
print("blocks", d.sizes(), "dim", d.dim, flush=True)
t = time.time(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=400, verbose=True, time_limit=float(sys.argv[3])); print("FO", U, time.time() - t)
ref = reference.moment_relaxation(pop, r); print("MOSEK", ref)
