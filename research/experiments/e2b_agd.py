import sys, json, time; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers
for n in (16, 24):
    pop = pb.maxcut(n, pb.graph("reg3", n, seed=1)); d = SpectralDual(pop, 2)
    U, _, h = solvers.agd_continuation(d, iters_per_stage=1500, time_limit=240)
    json.dump(dict(n=n, agd=h.tolist()), open(f"results/e2b_n{n}.json", "w"))
    print(n, "AGD best U", U, "evals", len(h), "time", h[-1, 1], flush=True)
