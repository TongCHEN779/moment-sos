"""Burer-Monteiro + dual certificate baseline for order-2 MAX-CUT.

Pipeline (the 'certifiable low-rank' recipe): run the BM augmented Lagrangian (as in experiments/baselines/bm.py),
and after every outer iteration turn its multiplier estimate z into a dual point of the SAME spectral dual used by
our method.  With  S = D(-c + A^T z)  (the dual slack matrix, an s x s symmetric matrix) the identity
sigma - f = v^T S v  holds with  sigma = sum_i z_diag,i , and the certified bound of Theorem 1 (constant-trace case) is

        U_BM = sum_i z_diag,i - s * lambda_min(S).

One dense eigenvalue computation per certificate; time spent on it is counted.
usage: python bm_cert.py family n seed [k] [beta0]
"""
import sys, os, json, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "baselines")); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import minimize
from instances import instance, local_search_cut
from common import Slots
from msos import SpectralDual, problems as pb


class _Stop(Exception):
    pass


def bm_certified(S, k=10, beta0=10.0, inner=200, growth=1.0, drop=0.25, budget=600.0, stop_when=None, seed=0, beta_max=1e5):
    """Returns dict(hist=[(time, U_cert, value, infeas), ...], ...).  stop_when(U, infeasibility) -> bool ends the run."""
    s, ns = S.s, S.nslots
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((s, k)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    z = np.zeros(ns + s); beta = beta0
    D = np.zeros((s, s)); hist = []; t0 = time.perf_counter(); state = dict(val=0.0, inf=np.inf)

    def fg(vflat):
        Vm = vflat.reshape(s, k)
        x = np.einsum("pk,pk->p", Vm[S.I], Vm[S.J]); r = S.resid(x)
        state.update(val=S.value(x), inf=float(np.abs(r).max()))
        if time.perf_counter() - t0 > budget:
            raise _Stop
        F = -(S.c @ x) + z @ r + 0.5 * beta * (r @ r)
        S.to_dense(-S.c + S.resid_T(z + beta * r), D)
        return F, (2.0 * D @ Vm).ravel()

    def certificate(zz):
        Smat = S.to_dense(-S.c + S.resid_T(zz))
        lam = np.linalg.eigvalsh(Smat)[0]
        return float(zz[ns:].sum() - s * lam)

    vflat, prev_inf, outer, nev = V.ravel(), np.inf, 0, 0
    try:
        while True:
            res = minimize(fg, vflat, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=inner, maxfun=int(1.5 * inner), maxcor=20, ftol=1e-15, gtol=1e-9))
            vflat = res.x; nev += res.nfev
            Vm = vflat.reshape(s, k); x = np.einsum("pk,pk->p", Vm[S.I], Vm[S.J]); r = S.resid(x)
            z += beta * r
            inf = float(np.abs(r).max())
            U = certificate(z)                                   # certificate evaluated at every outer iteration
            hist.append((time.perf_counter() - t0, U, S.value(x), inf))
            if inf > drop * prev_inf:
                beta = min(beta * growth, beta_max)
            prev_inf = inf; outer += 1
            if stop_when is not None and stop_when(U, inf):
                break
    except _Stop:
        pass
    return dict(hist=hist, outer=outer, evals=nev, beta=beta, z=z, V=vflat.reshape(s, k))


if __name__ == "__main__":
    family, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 10; beta0 = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    n, edges, w = instance(family, n, seed)
    pop = pb.maxcut(n, edges, w); d = SpectralDual(pop, 2); S = Slots(d)
    LB = local_search_cut(n, edges, w)                           # not timed (same convention as run_bench.py)
    load0 = os.getloadavg()[0]
    res = bm_certified(S, k=k, beta0=beta0, budget=100.0, stop_when=lambda U, inf: np.floor(U + 1e-9) <= LB + 1e-9)
    h = np.array(res["hist"])
    cert = bool(len(h) and np.floor(h[:, 1].min() + 1e-9) <= LB + 1e-9)
    ok = np.where(np.floor(h[:, 1] + 1e-9) <= LB + 1e-9)[0]
    rec = dict(family=family, n=n, seed=seed, k=k, beta0=beta0, s=S.s, LB=LB, certified=cert, t_cert=float(h[ok[0], 0]) if len(ok) else None,
               U_min=float(h[:, 1].min()), final_value=float(h[-1, 2]), final_infeas=float(h[-1, 3]), total_time=float(h[-1, 0]), outer=res["outer"],
               evals=res["evals"], load0=load0, rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30,
               hist=h[:, :2].tolist())
    print("RESULT " + json.dumps(rec))
