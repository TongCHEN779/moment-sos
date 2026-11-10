"""Certification pipelines on the Biq Mac-style MAX-CUT benchmark (order 2).  usage: hybrid_cert.py family n seed mode
  mode = bm      : Burer-Monteiro ALM; its own multipliers are turned into a dual point (the usual 'certifiable low-rank' recipe)
  mode = hybrid  : BM until its max constraint violation <= 2e-3 (cap 60 s), then OUR dual solver (L-BFGS on the smoothed dual,
                   eta = 1e-3 ... 1e-5) started from the dual point implied by BM's multipliers
  mode = cold    : ablation: our solver from theta = 0 with the same eta schedule as 'hybrid' (1e-3 ... 1e-5)
Every time includes everything the pipeline does (BM, certificate evaluations, refinement).  The best-cut lower bound is computed
beforehand and is not timed (same convention as run_bench.py).  Certified := floor(best U) <= best cut."""
import sys, os, json, time, resource
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bm_cert import *            # noqa: F401,F403  (np, Slots, SpectralDual, pb, instance, local_search_cut, bm_certified)
from msos import solvers

ETAS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def theta_from_z(d, S, z):
    """Dual point of the spectral dual implied by BM's multiplier vector z: U(theta) == sum z_diag - s*lmin(S)."""
    ns = S.nslots
    q = (-S.c + S.resid_T(z)) / S.w
    sigma = float(z[ns:].sum())
    r = -d.fvec.copy(); r[d.empty_id] += sigma
    th = d.theta0(); th[0] = sigma
    th[d.sl_Theta] = q - r[d.slot_g] / d.W[d.slot_g]
    return th


if __name__ == "__main__":
    family, n, seed, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    n, edges, w = instance(family, n, seed)
    pop = pb.maxcut(n, edges, w); d = SpectralDual(pop, 2); S = Slots(d)
    LB = local_search_cut(n, edges, w)
    rec = dict(family=family, n=n, seed=seed, mode=mode, LB=LB, s=S.s, load0=os.getloadavg()[0])
    t0 = time.perf_counter()
    if mode == "bm":
        res = bm_certified(S, budget=100.0, stop_when=lambda U, inf: np.floor(U + 1e-9) <= LB + 1e-9)
        h = np.array(res["hist"]); T = h[:, 0]; Ubest = np.minimum.accumulate(h[:, 1])
        rec.update(outer=res["outer"], evals=res["evals"])
    else:
        th0 = None; t_bm = 0.0
        if mode == "hybrid":
            res = bm_certified(S, budget=60.0, stop_when=lambda U, inf: inf <= 2e-3)
            t_bm = time.perf_counter() - t0
            th0 = theta_from_z(d, S, res["z"])
            rec.update(t_bm=t_bm, bm_outer=res["outer"], bm_infeas=float(res["hist"][-1][3]), bm_value=float(res["hist"][-1][2]),
                       U_after_bm=float(d.certificate(th0)))
        U, th, h = solvers.lbfgs_continuation(d, theta0=th0, etas=ETAS, iters_per_stage=500, time_limit=1500)
        T = t_bm + h[:, 1]; Ubest = h[:, 3]                  # refinement time is measured from the start of the refinement
        if mode == "hybrid":
            T = np.concatenate([[t_bm], T]); Ubest = np.concatenate([[rec["U_after_bm"]], Ubest]); Ubest = np.minimum.accumulate(Ubest)
    ok = np.where(np.floor(Ubest + 1e-9) <= LB + 1e-9)[0]
    rec.update(certified=bool(len(ok)), t_cert=float(T[ok[0]]) if len(ok) else None, U_final=float(Ubest[-1]), total_time=float(T[-1]),
               rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)
    print("RESULT " + json.dumps(rec))
