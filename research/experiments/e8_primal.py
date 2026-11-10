"""E8: primal recovery and two-sided certificates along the eta-continuation (Prop. `primal recovery').

(a) order-2 MAX-CUT on random 3-regular graphs (seed 1), n in 12..30: at every function evaluation of the
    L-BFGS continuation we record the certified bound U, the recovered primal value L_ybar(f), the Boolean
    lower bound LB (Prop. (d)), |g|, the Hankel infeasibility and lmin M(ybar); reference rho_2 from Clarabel
    for n <= CLARABEL_MAX_N (=16; Clarabel needed > 27 min at n=20 on the shared machine), else the stored e3 floor
    value (n=20,30) or exact enumeration of the maximum cut (n=24).
(b) non-Boolean problems (box QP n=5 seeds 1,2; chain quartic n=8 sparse; disk), r=2: Lf, y0, infeasibilities
    and |g| along the stages; reference rho^tr (trace cuts) from Clarabel.
Run from experiments/:   python e8_primal.py [maxcut|nonbool|all]
Results are written incrementally to results/e8.json.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
import sys, json, time
sys.path.insert(0, "..")
import numpy as np
from scipy.optimize import minimize
from msos import SpectralDual, problems as pb
from msos.reference import moment_relaxation

R = "results/"
OUT = R + "e8.json"
ETAS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)      # as solvers.lbfgs_continuation
ITERS = 500
TOLS = (1e-1, 1e-2, 1e-3, 1e-4)


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"maxcut": {}, "nonbool": {}, "meta": {}}


def save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def continuation(dual, boolean_lb, etas=ETAS, iters=ITERS, maxcor=30, verbose=True):
    """Same loop as solvers.lbfgs_continuation, but every oracle call goes through primal_moments so that
    the primal quantities are available at every evaluation.  Returns (stage records, per-eval trajectory)."""
    th = dual.theta0()
    t0 = time.perf_counter()
    best = dict(U=np.inf, LB=-np.inf)
    traj = []                     # per evaluation: [t, eta, U, bestU, Lf, LB, bestLB, |g|, hankel_rss, lmin_M, y0]
    stages = []
    cur_eta = [None]

    def fg(x, eta):
        pm = dual.primal_moments(x, eta)
        best["U"] = min(best["U"], pm["U"])
        LB = np.nan
        if boolean_lb:
            LB = dual.boolean_lower_bound(x, eta, pm=pm)["LB"]
            best["LB"] = max(best["LB"], LB)
        traj.append([time.perf_counter() - t0, eta, pm["U"], best["U"], pm["Lf"], LB, best["LB"], pm["grad_norm"],
                     pm["infeas"]["hankel_rss"], min(pm["lmin_M"]), pm["y0"]])
        return pm["Us"], pm["g"]

    for eta in etas:
        res = minimize(fg, th, args=(eta,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=iters, maxcor=maxcor, ftol=1e-9, gtol=1e-10, maxfun=int(iters * 1.3)))
        th = res.x
        pm = dual.primal_moments(th, eta)
        rec = dict(eta=eta, time=time.perf_counter() - t0, nev=len(traj), nit=int(res.nit), U=best["U"], U_here=pm["U"],
                   Us=pm["Us"], Lf=pm["Lf"], y0=pm["y0"], grad_norm=pm["grad_norm"], g_Theta_norm=pm["g_Theta_norm"],
                   infeas=pm["infeas"], lmin_M=pm["lmin_M"], entropy_term=eta * pm["entropy"],
                   lagr_residual=pm["lagr_residual"], message=str(res.message))
        if boolean_lb:
            lb = dual.boolean_lower_bound(th, eta, pm=pm)
            rec.update(LB=best["LB"], LB_here=lb["LB"], delta=lb["delta"], tau=lb["tau"])
        stages.append(rec)
        if verbose:
            extra = f" LB={best['LB']:.6f} U-LB={best['U']-best['LB']:.2e}" if boolean_lb else f" y0={pm['y0']:.6f}"
            print(f"  eta={eta:.0e} t={rec['time']:6.1f}s nev={rec['nev']:5d} U={best['U']:.6f} Lf={pm['Lf']:.6f}{extra}"
                  f" |g|={pm['grad_norm']:.1e} hankel={pm['infeas']['hankel_rss']:.1e} lminM={min(pm['lmin_M']):.1e}", flush=True)
    return stages, traj


def first_time(traj, col_fn, tol):
    """first time at which col_fn(row) <= tol (monotone quantities use best-so-far columns)."""
    for row in traj:
        if col_fn(row) <= tol:
            return float(row[0])
    return None


def brute_maxcut_np(n, edges, chunk=1 << 20):
    """exact max cut by vectorised enumeration of 2^(n-1) sign patterns."""
    best = 0
    E = np.array(edges)
    for start in range(0, 1 << (n - 1), chunk):
        m = np.arange(start, min(start + chunk, 1 << (n - 1)), dtype=np.int64)
        bits = ((m[:, None] >> np.arange(n)) & 1).astype(np.int8)
        best = max(best, int((bits[:, E[:, 0]] != bits[:, E[:, 1]]).sum(1).max()))
    return best


CLARABEL_MAX_N = 16     # Clarabel (3 threads, shared machine) needed > 27 min at n=20 and was stopped


def run_maxcut(res, ns=(12, 16, 20, 24, 30)):
    e3 = json.load(open(R + "e3_bounds.json"))
    for n in ns:
        key = str(n)
        if key in res["maxcut"] and res["maxcut"][key].get("done"):
            print(f"n={n}: already done"); continue
        edges = pb.graph("reg3", n, seed=1)
        pop = pb.maxcut(n, edges)
        dual = SpectralDual(pop, 2)
        s = dual.sizes()[0]
        print(f"== MAX-CUT reg3 n={n} (m={len(edges)} edges), r=2, s={s} ==", flush=True)
        if key in res["maxcut"] and res["maxcut"][key].get("stages"):
            row = res["maxcut"][key]; stages, traj = row["stages"], row["traj"]
            print("  (continuation already recorded; computing the reference only)")
        else:
            stages, traj = continuation(dual, boolean_lb=True)
            row = dict(n=n, m=len(edges), s=s, stages=stages, traj=traj, done=False)
            res["maxcut"][key] = row; save(res)
        # reference: Clarabel for small n; otherwise the integer max cut (rho_2 = max cut on these graphs:
        # the certified bracket LB <= rho_2 <= U recorded below confirms it to the width of the bracket)
        if n <= CLARABEL_MAX_N:
            rho, t_ref = moment_relaxation(pop, 2, trace_cuts=False)
            row.update(rho=rho, rho_source="Clarabel", t_ref=t_ref)
        elif key in e3:
            row.update(rho=float(e3[key]["floor_ub"]), rho_source=f"e3_bounds.json floor_ub (lb={e3[key]['lb']}, ub={e3[key]['ub']})", t_ref=None)
        else:
            t = time.perf_counter(); mcut = brute_maxcut_np(n, edges)
            row.update(rho=float(mcut), rho_source=f"exact max cut by enumeration ({time.perf_counter()-t:.0f}s)", t_ref=None)
        rho = row["rho"]
        T = np.array(traj)
        fin = stages[-1]
        row.update(final=dict(U_minus_rho=fin["U"] - rho, rho_minus_LB=rho - fin["LB"], Lf_minus_rho=fin["Lf"] - rho,
                              U_minus_LB=fin["U"] - fin["LB"], grad_norm=fin["grad_norm"], hankel=fin["infeas"]["hankel_rss"],
                              lmin_M=min(fin["lmin_M"]), time=fin["time"], nev=fin["nev"]),
                   t_gap={f"{tol:g}": first_time(traj, lambda r: r[3] - r[6], tol) for tol in TOLS},          # U - LB (best-so-far)
                   t_dual={f"{tol:g}": first_time(traj, lambda r: r[3] - rho, tol) for tol in TOLS},          # U - rho
                   t_primal={f"{tol:g}": first_time(traj, lambda r: abs(r[4] - rho), tol) for tol in TOLS},   # |Lf - rho|
                   t_lb={f"{tol:g}": first_time(traj, lambda r: rho - r[6], tol) for tol in TOLS},            # rho - LB
                   bracket_ok=bool(np.all(T[:, 5] <= rho + 1e-6) and np.all(T[:, 2] >= rho - 1e-6)),
                   done=True)
        f = row["final"]
        print(f"  rho={rho:.6f} [{row['rho_source']}]  final: U-rho={f['U_minus_rho']:.2e}  rho-LB={f['rho_minus_LB']:.2e}  "
              f"Lf-rho={f['Lf_minus_rho']:+.2e}  U-LB={f['U_minus_LB']:.2e}  t(U-LB<1e-2)={row['t_gap']['0.01']}  t(<1e-3)={row['t_gap']['0.001']}"
              f"  t(U-rho<1e-3)={row['t_dual']['0.001']}  t(|Lf-rho|<1e-3)={row['t_primal']['0.001']}  bracket_ok={row['bracket_ok']}", flush=True)
        save(res)


def run_nonbool(res):
    cases = []
    for sd in (1, 2):
        cases.append((f"boxqp5_s{sd}", pb.box_qp(5, sd), None))
    pc, cl = pb.chain_quartic(8, 1)
    cases.append(("chain8_sparse", pc, cl))
    cases.append(("disk", pb.disk(), None))
    for name, pop, cl in cases:
        if name in res["nonbool"] and res["nonbool"][name].get("done"):
            print(f"{name}: already done"); continue
        dual = SpectralDual(pop, 2, cl)
        print(f"== {name}: n={pop.n} r=2 sizes={dual.sizes()} ==", flush=True)
        stages, traj = continuation(dual, boolean_lb=False)
        row = dict(name=name, n=pop.n, sizes=dual.sizes(), nblocks=len(dual.blocks), stages=stages, traj=traj, done=False)
        res["nonbool"][name] = row; save(res)
        rho, t_ref = moment_relaxation(pop, 2, cliques=cl, trace_cuts=True)
        fin = stages[-1]
        inf = fin["infeas"]
        row.update(rho_tr=rho, t_ref=t_ref,
                   final=dict(U_minus_rho=fin["U"] - rho, Lf_minus_rho=fin["Lf"] - rho, y0=fin["y0"], grad_norm=fin["grad_norm"],
                              hankel=inf["hankel_rss"], loc=inf["loc"], y0_err=inf["y0"], infeas_total=inf["hankel_rss"] + inf["loc"] + inf["y0"],
                              lmin_M=min(fin["lmin_M"]), time=fin["time"], nev=fin["nev"]),
                   t_primal={f"{tol:g}": first_time(traj, lambda r: abs(r[4] - rho), tol) for tol in TOLS},
                   t_dual={f"{tol:g}": first_time(traj, lambda r: r[3] - rho, tol) for tol in TOLS},
                   done=True)
        f = row["final"]
        print(f"  rho^tr={rho:.6f} (Clarabel {t_ref:.1f}s)  final: U-rho={f['U_minus_rho']:.2e}  Lf-rho={f['Lf_minus_rho']:+.2e}  y0={f['y0']:.8f}"
              f"  hankel={f['hankel']:.1e} loc={f['loc']:.1e} |g|={f['grad_norm']:.1e}  t(|Lf-rho|<1e-3)={row['t_primal']['0.001']} t(U-rho<1e-3)={row['t_dual']['0.001']}", flush=True)
        save(res)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = load()
    res["meta"].update(etas=list(ETAS), iters_per_stage=ITERS, threads=3, note="shared machine (3 BLAS threads); timings indicative",
                       traj_columns=["t", "eta", "U", "bestU", "Lf", "LB", "bestLB", "grad_norm", "hankel_rss", "lmin_M", "y0"])
    save(res)
    if what in ("nonbool", "all"):
        run_nonbool(res)
    if what in ("maxcut", "all"):
        run_maxcut(res)
    print("done")
