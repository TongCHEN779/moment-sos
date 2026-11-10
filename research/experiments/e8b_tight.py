"""E8 addendum: tight stationarity for the Boolean two-sided certificate.

Standard eta-continuation (as in e8_primal.py), then from its final theta a 'tight' phase: L-BFGS-B on U_eta at
fixed eta = 1e-3, 3e-4, 1e-4 (warm-started in sequence), maxiter 3000, maxfun 4000, ftol 1e-15, gtol 1e-8.
U (best so far) and LB (best so far) are tracked at every evaluation from the start of the whole run; total time
limit 900 s per instance.  Results -> results/e8.json under key 'tight'.   Run from experiments/.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
import sys, json, time
sys.path.insert(0, ".."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from msos import SpectralDual, problems as pb
from e8_primal import ETAS, ITERS, TOLS, load, save, first_time, R

TIGHT_ETAS = (1e-3, 3e-4, 1e-4)
TIME_LIMIT = 900.0


class Run:
    def __init__(self, dual):
        self.dual = dual
        self.t0 = time.perf_counter()
        self.bestU, self.bestLB = np.inf, -np.inf
        self.traj = []

    def elapsed(self):
        return time.perf_counter() - self.t0

    def fg(self, x, eta):
        pm = self.dual.primal_moments(x, eta)
        lb = self.dual.boolean_lower_bound(x, eta, pm=pm)
        self.bestU = min(self.bestU, pm["U"]); self.bestLB = max(self.bestLB, lb["LB"])
        self.traj.append([self.elapsed(), eta, pm["U"], self.bestU, pm["Lf"], lb["LB"], self.bestLB, pm["grad_norm"],
                          pm["infeas"]["hankel_rss"], pm["lmin_M"][0], pm["y0"]])
        return pm["Us"], pm["g"]

    def stage(self, th, eta, opts):
        def cb(xk):
            if self.elapsed() > TIME_LIMIT:
                raise StopIteration
        res = minimize(self.fg, th, args=(eta,), jac=True, method="L-BFGS-B", options=opts, callback=cb)
        return res

    def record(self, th, eta, res, phase):
        pm = self.dual.primal_moments(th, eta)
        lb = self.dual.boolean_lower_bound(th, eta, pm=pm)
        return dict(phase=phase, eta=eta, nit=int(res.nit), nfev=int(res.nfev), time=self.elapsed(), nev=len(self.traj),
                    grad_norm=pm["grad_norm"], g_Theta_norm=pm["g_Theta_norm"], U=self.bestU, U_here=pm["U"], Lf=pm["Lf"],
                    LB=self.bestLB, LB_here=lb["LB"], delta=lb["delta"], tau=lb["tau"], lmin_M=pm["lmin_M"][0],
                    hankel=pm["infeas"]["hankel_rss"], bracket=self.bestU - self.bestLB, message=str(res.message))


def run_instance(n):
    edges = pb.graph("reg3", n, seed=1)
    pop = pb.maxcut(n, edges)
    dual = SpectralDual(pop, 2)
    run = Run(dual)
    th = dual.theta0()
    stages = []
    print(f"== n={n} s={dual.sizes()[0]} ==", flush=True)
    for eta in ETAS:                                   # standard continuation (identical options to e8_primal.continuation)
        res = run.stage(th, eta, dict(maxiter=ITERS, maxcor=30, ftol=1e-9, gtol=1e-10, maxfun=int(ITERS * 1.3)))
        th = res.x
        stages.append(run.record(th, eta, res, "standard"))
        if run.elapsed() > TIME_LIMIT:
            break
    s = stages[-1]
    print(f"  standard done: t={s['time']:.0f}s |g|={s['grad_norm']:.1e} U-LB={s['bracket']:.2e}", flush=True)
    for eta in TIGHT_ETAS:
        if run.elapsed() > TIME_LIMIT:
            print("  time limit reached before tight stage", eta); break
        res = run.stage(th, eta, dict(maxiter=3000, maxcor=30, ftol=1e-15, gtol=1e-8, maxfun=4000))
        th = res.x
        rec = run.record(th, eta, res, "tight")
        stages.append(rec)
        print(f"  tight eta={eta:.0e}: nit={rec['nit']} t={rec['time']:.0f}s |g|={rec['grad_norm']:.1e} U={rec['U']:.6f} Lf={rec['Lf']:.6f} "
              f"LB={rec['LB']:.6f} delta={rec['delta']:.1e} U-LB={rec['bracket']:.2e}  [{rec['message']}]", flush=True)
    return dict(n=n, s=dual.sizes()[0], stages=stages, traj=run.traj,
                t_gap={f"{tol:g}": first_time(run.traj, lambda r: r[3] - r[6], tol) for tol in TOLS},
                total_time=run.elapsed(), nev=len(run.traj))


if __name__ == "__main__":
    ns = [int(a) for a in sys.argv[1:]] or [12, 16, 20, 24]
    for n in ns:
        res = load()
        res.setdefault("tight", {})
        row = run_instance(n)
        rho = res["maxcut"][str(n)]["rho"]
        row["rho"] = rho; row["rho_source"] = res["maxcut"][str(n)]["rho_source"]
        fin = row["stages"][-1]
        row["final"] = dict(U_minus_rho=fin["U"] - rho, rho_minus_LB=rho - fin["LB"], Lf_minus_rho=fin["Lf"] - rho, U_minus_LB=fin["bracket"])
        row["bracket_ok"] = bool(all(r[5] <= rho + 1e-6 and r[2] >= rho - 1e-6 for r in row["traj"]))
        res = load(); res.setdefault("tight", {})[str(n)] = row; save(res)
        print(f"  n={n}: U-rho={row['final']['U_minus_rho']:.2e} rho-LB={row['final']['rho_minus_LB']:.2e} U-LB={row['final']['U_minus_LB']:.2e} "
              f"t(U-LB<1e-2)={row['t_gap']['0.01']} t(<1e-3)={row['t_gap']['0.001']} total={row['total_time']:.0f}s bracket_ok={row['bracket_ok']}", flush=True)
    print("done")
