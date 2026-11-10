"""E5: correlative sparsity (chain quartic) and dense box QP -- FO vs MOSEK."""
import sys, json, time, resource; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
rows = []
def run(name, pop, r, cl, mosek=True):
    d = SpectralDual(pop, r, cliques=cl)
    t = time.perf_counter(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=1800); tf = time.perf_counter() - t
    row = dict(name=name, n=pop.n, r=r, sparse=cl is not None, maxblock=max(d.sizes()), nblocks=len(d.sizes()), dim=d.dim, FO=U, t_fo=tf)
    if mosek:
        v, tm = reference.moment_relaxation(pop, r, cliques=cl); row.update(mosek=v, t_mosek=tm)
    rows.append(row); print(row, flush=True)
    json.dump(rows, open("results/e5.json", "w"))
for n in (8, 12, 16, 20, 24):
    pop, cl = pb.chain_quartic(n, 1)
    run(f"chain{n}", pop, 2, cl)
    run(f"chain{n}", pop, 2, None, mosek=(n <= 16))
for n in (6, 8, 10, 12):
    run(f"boxqp{n}", pb.box_qp(n, 1), 2, None)
