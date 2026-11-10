import sys, time; sys.path.insert(0, "..")
import numpy as np
from msos import POP, SpectralDual, problems as pb, solvers, reference
for name, pop, r in [("ball", pb.ball(), 2), ("disk", pb.disk(), 2),
                     ("K3 r1", pb.maxcut(3, pb.graph("K", 3)), 1), ("K3 r2", pb.maxcut(3, pb.graph("K", 3)), 2)]:
    d = SpectralDual(pop, r)
    U, th, h = solvers.lbfgs_continuation(d)
    ref = reference.moment_relaxation(pop, r)[0]
    print(name, "dim", d.dim, "blocks", d.sizes(), "FO", round(U, 6), "MOSEK", round(ref, 6), "evals", len(h))
