"""CGAL (conditional-gradient augmented Lagrangian, Yurtsever-Fercoq-Cevher / Yurtsever et al. 2021
'Scalable SDP') WITHOUT the Nystrom sketch: X is kept implicitly as its slot vector x (dense s x s
matrix is only formed for the LMO).

    min  <-C, X>   s.t.  A(X) = 0 ([dev; diag-1] residual, see common.py),  X psd,  tr X <= s.

Iteration t (0-based):
    grad = -c + A^T(z + beta_t A(X_t))            (slot vector -> dense symmetric D)
    LMO  : (lam, v) = lowest eigenpair of D (scipy eigsh, warm-started); H = s v v^T if lam < 0 else 0
    X   <- (1-eta) X + eta H,  eta = 2/(t+2),  beta_t = beta0 sqrt(t+2)
    z   <- z + sigma A(X),  sigma = min(beta0, 4 beta_t s^2 ||A||^2 eta^2 / ||A(X)||^2)
The dual step is the CGAL rule up to constants (||A||^2 is bounded by 3 here).  Nothing here is a
rigorous upper bound: the reported value is the primal objective of an infeasible iterate.
"""
import time
import numpy as np
from scipy.sparse.linalg import eigsh
from common import History


def run(S, beta0=1.0, budget=1200.0, ref=None, tol_stop=1e-3, normA2=3.0, eig_tol=1e-6, **_):
    s, ns = S.s, S.nslots
    x = np.zeros(ns); x[S.diag] = 1.0              # X_0 = I (trace s, feasible diag)
    z = np.zeros(ns + s)
    r = S.resid(x)
    D = np.zeros((s, s))
    v0 = np.ones(s) / np.sqrt(s)
    hist = History(); t0 = time.perf_counter()
    t = 0; status = "budget"
    first = {}
    while True:
        el = time.perf_counter() - t0
        val = S.value(x); inf = float(np.abs(r).max())
        hist.add(el, val, inf)
        if ref is not None and abs(val - ref) <= tol_stop and inf <= tol_stop:
            status = "reached"; break
        if el > budget:
            break
        beta = beta0 * np.sqrt(t + 2)
        gvec = -S.c + S.resid_T(z + beta * r)
        S.to_dense(gvec, D)
        lam, V = eigsh(D, k=1, which="SA", v0=v0, tol=eig_tol)
        v = V[:, 0]; v0 = v
        eta = 2.0 / (t + 2)
        if lam[0] < 0:
            hx = s * v[S.I] * v[S.J]
        else:
            hx = np.zeros(ns)
        x = (1 - eta) * x + eta * hx
        r = S.resid(x)
        rn2 = float(r @ r)
        sigma = beta0 if rn2 == 0 else min(beta0, 4 * beta * s * s * normA2 * eta ** 2 / rn2)
        z += sigma * r
        t += 1
    hist.add(time.perf_counter() - t0, S.value(x), float(np.abs(r).max()), force=True)
    return dict(val=S.value(x), infeas_inf=float(np.abs(r).max()), iters=t, time=time.perf_counter() - t0,
                hist=hist.rows, status=status)
