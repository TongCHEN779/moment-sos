"""E9: trace cuts, diagonal cuts and the box (Proposition prop:diag, Example ex:boxfree).

For every instance: interior-point (Clarabel) values rho_r (no cuts), rho_r^tr (trace cuts), rho_r^d (diagonal cuts)
with solve times, the first-order values  U  (msos smoothing + L-BFGS, uniform shift -> rho^tr side) and
U^d (moment_sos dual ALM with shift='diag' -> rho^d side), plus the ALM with the uniform shift for reference.
Part B compares, for the box QP n = 4..10, the two formulations 'with box constraints + uniform shift' (n+1 blocks)
and 'without box constraints + diagonal shift' (1 block): block / multiplier counts, bounds and times.
Results are written incrementally to results/e9.json.  Run from experiments/:  python e9_cuts.py
"""
import os, sys, json, time, warnings
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "3"
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np
import torch; torch.set_num_threads(3)
import moment_sos as po
from msos import POP, SpectralDual, reference, solvers, problems as pb

OUT = "results/e9.json"
FO_TIME = 120.0          # time limit per first-order solve (s)
ITERS = 500              # smoothing iterations per eta stage (as in E1)


# ----------------------------------------------------------------------------- helpers
def to_polyopt(pop, box=False, cliques=None):
    """msos.POP -> moment_sos.Problem with exactly the same polynomials ([-1,1]^n box or Boolean).  The box constraints
    1 - x_i^2 >= 0 are taken from pop.ineq when present (msos.problems.box_qp / chain_quartic include them), so
    moment_sos's automatic add_box_constraints is always off; `box` is informational only."""
    x = po.variables(pop.n)
    def mk(poly):
        out = 0
        for ex, c in poly.items():
            m = c
            for i, e in enumerate(ex):
                if e:
                    m = m * x[i] ** e
            out = out + m
        return out
    cons = [mk(g) >= 0 for g in pop.ineq] + [mk(h) == 0 for h in pop.eq]
    if pop.boolean:
        return po.Problem(mk(pop.f), cons, boolean=True, add_box_constraints=False)
    return po.Problem(mk(pop.f), cons, box=(-1, 1), add_box_constraints=False)


def ipm(pop, r, cuts, cliques):
    try:
        v, t = reference.moment_relaxation(pop, r, cliques=cliques, cuts=cuts)
        return (None if np.isinf(v) else float(v)), float(t)        # None <-> +inf in JSON
    except Exception as exc:
        print(f"    ! Clarabel/SCS failed ({cuts}): {exc!r}")
        return "failed", None


def smoothing(pop, r, cliques):
    d = SpectralDual(pop, r, cliques)
    t = time.perf_counter()
    U, th, hist = solvers.lbfgs_continuation(d, iters_per_stage=ITERS, time_limit=FO_TIME)
    return dict(U=float(U), time=time.perf_counter() - t, nev=int(np.asarray(hist)[-1, 0]) if len(hist) else 0,
                blocks=len(d.blocks), dim=int(d.dim))


def alm(P, r, cliques, shift):
    cl = "dense" if cliques is None else cliques
    t = time.perf_counter()
    res = po.solve_alm(P, r, cliques=cl, shift=shift, time_limit=FO_TIME)
    out = dict(U=float(res.bound), time=time.perf_counter() - t, nev=int(res.n_eval),
               blocks=len(res.relaxation.blocks), dim=int(res.relaxation.dim),
               sizes=[int(s) for s in res.relaxation.sizes()])
    out["n_shift_vars"] = int(sum(out["sizes"])) if shift == "diag" else out["blocks"]
    if shift == "diag":
        out.update(outer=int(res.extra["outer_iters"]), pinf=float(res.extra["pinf"]), dinf=float(res.extra["dinf"]),
                   U_uniform_along_path=float(res.problem.f_off + res.problem.f_scale * res.extra["U_uniform"]))
    return out


def run_instance(rows, name, pop, r, box, cliques=None, do_fo=True):
    key = (name, r)
    if any((row["problem"], row["r"]) == key for row in rows):
        print(f"[skip] {name} r={r} already done"); return
    print(f"[{time.strftime('%H:%M:%S')}] {name}  r={r}  n={pop.n}  box_constraints={box}")
    row = dict(problem=name, n=pop.n, r=r, box_constraints=box, boolean=pop.boolean,
               n_ineq=len(pop.ineq), n_eq=len(pop.eq), cliques=(None if cliques is None else len(cliques)))
    for c in ("none", "trace", "diag"):
        v, t = ipm(pop, r, c, cliques)
        row[f"rho_{c}"], row[f"t_{c}"] = v, t
        print(f"    IPM {c:5s}: {('inf' if v is None else v):>14}  ({'-' if t is None else f'{t:.2f}s'})")
    if do_fo:
        sm = smoothing(pop, r, cliques)
        row["smooth"] = sm
        print(f"    smoothing U     = {sm['U']:.8f}  ({sm['time']:.1f}s, {sm['nev']} evals, {sm['blocks']} blocks, dim {sm['dim']})")
        P = to_polyopt(pop, box, cliques)
        ad = alm(P, r, cliques, "diag")
        row["alm_diag"] = ad
        print(f"    ALM-diag  U^d   = {ad['U']:.8f}  ({ad['time']:.1f}s, {ad['nev']} evals, {ad['outer']} outer, pinf {ad['pinf']:.1e}, dinf {ad['dinf']:.1e})")
        au = alm(P, r, cliques, "uniform")
        row["alm_uniform"] = au
        print(f"    ALM-unif  U     = {au['U']:.8f}  ({au['time']:.1f}s, {au['nev']} evals)")
    rows.append(row)
    save()


def save():
    json.dump(RES, open(OUT, "w"), indent=1)


# ----------------------------------------------------------------------------- instances
def cubic_on_ball(n=3, seed=1):
    """max  (random cubic + quadratic)  s.t.  1 - |x|^2 >= 0  (no box constraints; the ball implies the box)."""
    import itertools
    rng = np.random.default_rng(seed)
    f = {}
    for deg in (2, 3):
        for c in itertools.combinations_with_replacement(range(n), deg):
            e = [0] * n
            for i in c:
                e[i] += 1
            f[tuple(e)] = float(rng.standard_normal())
    g = {tuple([0] * n): 1.0}
    for i in range(n):
        e = [0] * n; e[i] = 2
        g[tuple(e)] = -1.0
    return POP(n, f, ineq=[g], name=f"cubic_ball{n}_s{seed}")


def sphere():
    """max x1 + x2  s.t.  x1^2 + x2^2 = 1  (equality only; the sphere implies the box)."""
    return POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    RES = json.load(open(OUT)) if os.path.exists(OUT) else dict(meta={}, A=[], B=[])
    RES["meta"] = dict(solver_ipm=reference.default_solver(), fo_time_limit=FO_TIME, smoothing_iters_per_stage=ITERS,
                       threads=3, note="shared machine (three other tracks running concurrently); timings indicative",
                       torch=torch.__version__, date=time.strftime("%Y-%m-%d"))
    rows = RES["A"]

    # --- box-free example (Example ex:boxfree)
    bf = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(1, 1): 1.0}], name="boxfree")
    for r in (1, 2, 3):
        run_instance(rows, "boxfree", bf, r, box=False)
    # --- box QP n=4,5,6 with and without the explicit box constraints
    for n in (4, 5, 6):
        q = pb.box_qp(n, 1); qn = POP(n, q.f, name=f"boxqp{n}_nobox")
        for r in (1, 2):
            run_instance(rows, f"boxqp{n}+box", q, r, box=True)
            run_instance(rows, f"boxqp{n}-nobox", qn, r, box=False)
    # --- cubic on the ball (Archimedean without box constraints)
    cb = cubic_on_ball(3, 1)
    for r in (2, 3):
        run_instance(rows, "cubic_ball3", cb, r, box=False)
    # --- sphere equality
    sp = sphere()
    for r in (1, 2):
        run_instance(rows, "sphere", sp, r, box=False)
    # --- chain quartic n=6, sparse cliques, with and without box constraints
    pc, cl = pb.chain_quartic(6, 1); pcn = POP(6, pc.f, name="chain6_nobox")
    run_instance(rows, "chain6+box", pc, 2, box=True, cliques=cl)
    run_instance(rows, "chain6-nobox", pcn, 2, box=False, cliques=cl)
    # --- Petersen MAX-CUT (Boolean)
    pet = pb.maxcut(10, pb.graph("petersen", 10), name="petersen")
    run_instance(rows, "petersen", pet, 2, box=False)

    # ----------------------------------------------------------------- Part B: formulation comparison, box QP n=4..10, r=2
    rowsB = RES["B"]
    for n in range(4, 11):
        if any(row["n"] == n for row in rowsB):
            print(f"[skip] B n={n}"); continue
        print(f"[{time.strftime('%H:%M:%S')}] Part B: box QP n={n} r=2")
        q = pb.box_qp(n, 1); qn = POP(n, q.f)
        row = dict(n=n, r=2)
        row["ipm_box_trace"], row["t_ipm_box_trace"] = ipm(q, 2, "trace", None)
        row["ipm_nobox_diag"], row["t_ipm_nobox_diag"] = ipm(qn, 2, "diag", None)
        d = SpectralDual(q, 2)
        row["msos_box_blocks"], row["msos_box_dim"] = len(d.blocks), int(d.dim)
        row["smooth_box"] = smoothing(q, 2, None)
        row["alm_uniform_box"] = alm(to_polyopt(q, True), 2, None, "uniform")       # q already contains the n box constraints
        row["alm_diag_nobox"] = alm(to_polyopt(qn, False), 2, None, "diag")
        print(f"    IPM box+trace {row['ipm_box_trace']:.6f} ({row['t_ipm_box_trace']:.2f}s)   IPM nobox+diag {row['ipm_nobox_diag']:.6f} ({row['t_ipm_nobox_diag']:.2f}s)")
        for k in ("smooth_box", "alm_uniform_box", "alm_diag_nobox"):
            s = row[k]
            print(f"    {k:16s} U={s['U']:.6f}  t={s['time']:.1f}s  blocks={s['blocks']}  dim={s['dim']}  shift_vars={s.get('n_shift_vars', s['blocks'])}")
        rowsB.append(row)
        save()
    print("done ->", OUT)
