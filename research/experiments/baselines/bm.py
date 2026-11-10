"""Burer-Monteiro low-rank augmented Lagrangian for the same SDP.

X = V V^T, V in R^{s x k} (rows initialised on the unit sphere).  With x_p = <V_I, V_J> and the
stacked equality residual r(x) = [dev(x); diag(x)-1] (common.py) we minimise
    F(V) = -c.x + z.r + (beta/2)|r|^2         (scipy L-BFGS-B, maxcor 20, `inner` iterations / stage)
then update z <- z + beta r and beta <- beta*growth if |r|_inf did not drop by `drop`.
Nonconvex in V: no guarantee of reaching the global optimum, and no bound whatsoever.
"""
import time
import numpy as np
from scipy.optimize import minimize
from common import History


class _Stop(Exception):
    pass


def run(S, k=8, beta0=1.0, inner=200, growth=2.0, drop=0.25, budget=1200.0, ref=None, tol_stop=1e-3,
        seed=0, beta_max=1e5, **_):
    s, ns = S.s, S.nslots
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((s, k)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    z = np.zeros(ns + s)
    beta = beta0
    D = np.zeros((s, s))
    hist = History(); t0 = time.perf_counter()
    cnt = [0]; status = ["budget"]
    state = dict(val=0.0, inf=np.inf)

    def fg(vflat):
        V = vflat.reshape(s, k)
        x = np.einsum("pk,pk->p", V[S.I], V[S.J])
        r = S.resid(x)
        val = S.value(x); inf = float(np.abs(r).max())
        state.update(val=val, inf=inf)
        cnt[0] += 1
        el = time.perf_counter() - t0
        hist.add(el, val, inf)
        if ref is not None and abs(val - ref) <= tol_stop and inf <= tol_stop:
            status[0] = "reached"; raise _Stop
        if el > budget:
            raise _Stop
        F = -(S.c @ x) + z @ r + 0.5 * beta * (r @ r)
        gx = -S.c + S.resid_T(z + beta * r)
        S.to_dense(gx, D)
        return F, (2.0 * D @ V).ravel()

    outer = 0; vflat = V.ravel(); prev_inf = np.inf
    try:
        while True:
            res = minimize(fg, vflat, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=inner, maxfun=int(1.5 * inner), maxcor=20, ftol=1e-15, gtol=1e-9))
            vflat = res.x
            Vm = vflat.reshape(s, k)
            x = np.einsum("pk,pk->p", Vm[S.I], Vm[S.J]); r = S.resid(x)
            z += beta * r
            inf = float(np.abs(r).max())
            if inf > drop * prev_inf:
                beta = min(beta * growth, beta_max)
            prev_inf = inf
            outer += 1
    except _Stop:
        pass
    Vm = vflat.reshape(s, k)
    hist.add(time.perf_counter() - t0, state["val"], state["inf"], force=True)
    return dict(val=state["val"], infeas_inf=state["inf"], iters=cnt[0], outer=outer, beta_final=beta,
                time=time.perf_counter() - t0, hist=hist.rows, status=status[0])
