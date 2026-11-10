"""E7: rigorous a posteriori certificates (Theorem thm:rigor) on the E1 instances and on order-2
MAX-CUT for 3-regular random graphs.  Results are appended to results/e7.json after every instance.
Run from experiments/:  python e7_rigor.py  [--skip40]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "3"
import sys, time, json, math; sys.path.insert(0, "..")
import numpy as np
import networkx as nx
from msos import POP, SpectralDual, problems as pb, solvers
from msos import verify as V

OUT = "results/e7.json"
SKIP40 = "--skip40" in sys.argv


def local_search_maxcut(n, edges, restarts=2000, seed=0):
    """Single-flip local search with random restarts (lower bound on the max cut)."""
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    best = 0
    for _ in range(restarts):
        x = rng.choice([-1, 1], n)
        improved = True
        while improved:
            improved = False
            for i in rng.permutation(n):
                gain = sum(x[i] * x[j] for j in adj[i])      # cut gain of flipping i = sum_j x_i x_j
                if gain > 0:
                    x[i] = -x[i]; improved = True
        cut = sum(1 for i, j in edges if x[i] != x[j])
        best = max(best, cut)
    return int(best)


# ---------------------------------------------------------------- case list (E1, reused)
sph = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")
pc, cl = pb.chain_quartic(8, 1)
cases = [("ball", pb.ball(), 2, None, 1.0, None), ("disk", pb.disk(), 2, None, 2 ** .5, None),
         ("sphere (eq.)", sph, 2, None, 2 ** .5, None)]
for nm, n, kind in [("K3", 3, "K"), ("C5", 5, "C"), ("K5", 5, "K"), ("Petersen", 10, "petersen"), ("K7", 7, "K"), ("reg3-12", 12, "reg3")]:
    e = pb.graph(kind, n, seed=2); opt = pb.brute_maxcut(n, e)
    for r in (1, 2, 3):
        if n >= 10 and r == 3: continue
        cases.append((f"MAX-CUT {nm}", pb.maxcut(n, e), r, None, opt, "brute"))
for s in (1, 2):
    cases.append((f"box QP n=5 (seed {s})", pb.box_qp(5, s), 2, None, None, None))
cases.append(("chain quartic n=8 (sparse)", pc, 2, cl, None, None))
cases.append(("chain quartic n=8 (dense)", pc, 2, None, None, None))
# order-2 MAX-CUT on 3-regular random graphs (seed 1, as in E3)
e3 = json.load(open("results/e3_bounds.json"))
for n in (10, 14, 20, 26, 30, 40):
    if n == 40 and SKIP40:
        continue
    edges = pb.graph("reg3", n, seed=1)
    if n <= 20:
        opt, src = pb.brute_maxcut(n, edges), "brute"
    else:
        ls = local_search_maxcut(n, edges)
        lb = e3[str(n)]["lb"]
        opt, src = max(ls, lb), f"max(local search {ls}, e3 lb {lb})"
    cases.append((f"MAX-CUT reg3 n={n}", pb.maxcut(n, edges), 2, None, opt, src))

rows = []
if os.path.exists(OUT):
    rows = json.load(open(OUT))
done = {(r_["problem"], r_["r"]) for r_ in rows}

for name, pop, r, cliq, opt, src in cases:
    if (name, r) in done:
        print(f"skip {name} r={r} (done)"); continue
    d = SpectralDual(pop, r, cliques=cliq)
    tl = 900 if pop.n >= 40 else None
    t = time.perf_counter()
    U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=tl)
    t_solve = time.perf_counter() - t
    out = V.rigorous_bound(d, th)
    row = dict(problem=name, n=pop.n, r=r, s=out["s_max"], nblocks=len(d.blocks), dim=d.dim, nmom=int(d.nmom),
               boolean=pop.boolean, opt=opt, opt_source=src,
               U=out["U"], U_rig=out["U_rig"], diff=out["U_rig"] - out["U"], E=out["E"],
               max_slack=out["max_slack"], slack=out["slack"], ell=out["ell"], lmin_hat=out["lmin_hat"],
               doublings=out["doublings"], chol_ok=out["all_ok"], t_chol=out["t_chol"], t_res=out["t_res"],
               t_eig=out["t_eig"], t_solve=t_solve, evals=len(h))
    if pop.boolean:
        fU, fR = math.floor(out["U"]), math.floor(out["U_rig"])
        row.update(floor_U=fU, floor_Urig=fR, certified=bool(fU == fR == opt))
    rows.append(row)
    json.dump(rows, open(OUT, "w"), indent=1)
    cert = f" floor(U)={row.get('floor_U')} floor(U_rig)={row.get('floor_Urig')} opt={opt} certified={row.get('certified')}" if pop.boolean else ""
    print(f"{name:28s} r={r} s={out['s_max']:4d} U={out['U']:.8f} U_rig-U={row['diff']:.2e} E={out['E']:.1e} "
          f"slack={out['max_slack']:.1e} t_chol={out['t_chol']:.2f}s t_res={out['t_res']:.2f}s t_solve={t_solve:.1f}s{cert}", flush=True)
print("done")
