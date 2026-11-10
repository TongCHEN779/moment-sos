"""E2: convergence of subgradient/target-level (original) vs entropic-smoothing + L-BFGS (new).
MAX-CUT, random 3-regular graph, order 2."""
import sys, json, time; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
pop = pb.maxcut(n, pb.graph("reg3", n, seed=1))
ref, tm = reference.moment_relaxation(pop, 2)
d = SpectralDual(pop, 2)
U1, _, h1 = solvers.lbfgs_continuation(d, iters_per_stage=500)
U0, _, h0 = solvers.subgradient_target(d, iters=20000)
json.dump(dict(n=n, ref=ref, t_mosek=tm, lbfgs=h1.tolist(), subgrad=h0.tolist()), open(f"results/e2_n{n}.json", "w"))
print("ref", ref, "lbfgs", U1, "subgrad", U0)
