"""E1: first-order certified bound vs independent exact moment relaxation (MOSEK)."""
import sys, time, json; sys.path.insert(0, "..")
import numpy as np
from msos import POP, SpectralDual, problems as pb, solvers, reference

sph = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")
pc, cl = pb.chain_quartic(8, 1)
cases = [("ball", pb.ball(), 2, None, 1.0), ("disk", pb.disk(), 2, None, 2 ** .5),
         ("sphere (eq.)", sph, 2, None, 2 ** .5)]
for nm, n, kind in [("K3", 3, "K"), ("C5", 5, "C"), ("K5", 5, "K"), ("Petersen", 10, "petersen"), ("K7", 7, "K"), ("reg3-12", 12, "reg3")]:
    e = pb.graph(kind, n, seed=2); opt = pb.brute_maxcut(n, e)
    for r in (1, 2, 3):
        if n >= 10 and r == 3: continue
        cases.append((f"MAX-CUT {nm}", pb.maxcut(n, e), r, None, opt))
for s in (1, 2):
    cases.append((f"box QP n=5 (seed {s})", pb.box_qp(5, s), 2, None, None))
cases.append(("chain quartic n=8 (sparse)", pc, 2, cl, None))
cases.append(("chain quartic n=8 (dense)", pc, 2, None, None))
rows = []
for name, pop, r, cliq, opt in cases:
    d = SpectralDual(pop, r, cliques=cliq)
    t = time.time(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500); tf = time.time() - t
    v1, t1 = reference.moment_relaxation(pop, r, cliques=cliq)
    v0, _ = reference.moment_relaxation(pop, r, cliques=cliq, trace_cuts=False)
    rows.append(dict(problem=name, n=pop.n, r=r, sizes=d.sizes()[:1], nblocks=len(d.blocks), dim=d.dim, opt=opt, FO=U, tf=tf,
                     mosek_cuts=v1, mosek_plain=v0, tm=t1, evals=len(h)))
    print(f"{name:28s} r={r} FO={U:.6f} MOSEK(cuts)={v1:.6f} MOSEK(plain)={v0:.6f} opt={opt}", flush=True)
json.dump(rows, open("results/e1.json", "w"), indent=1)
