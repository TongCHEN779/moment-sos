"""Heuristic primal recovery (feasible points => lower bounds for a maximisation)."""
from __future__ import annotations
import numpy as np
import torch


def _moment_vectors(res):
    """Smallest-eigenvalue eigenvector of every clique Gram matrix, normalised by its constant entry."""
    R, O = res.relaxation, res.oracle
    with torch.no_grad():
        Qs = O.gram_blocks(res.theta)
    n = R.prob.n
    acc, cnt = np.zeros(n), np.zeros(n)
    for k, B in enumerate(R.cl_basis):
        Q = Qs[k].cpu().numpy()
        w, V = np.linalg.eigh(Q)
        v = V[:, 0]
        z = np.where(B.sum(1) == 0)[0]
        if not len(z) or abs(v[int(z[0])]) < 1e-10:
            continue
        zero = int(z[0])
        for i in range(n):
            idx = np.where((B[:, i] == 1) & (B.sum(1) == 1))[0]
            if len(idx):
                acc[i] += v[idx[0]] / v[zero]; cnt[i] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def round_solution(res, local_search=True, restarts=50, seed=0):
    """Return (x, value) with x feasible for the ORIGINAL problem (best found), or (None, None)."""
    P = res.problem
    t_hat = np.clip(_moment_vectors(res), -1, 1)
    rng = np.random.default_rng(seed)
    cands = [t_hat]
    if P.boolean:
        cands = [np.where(t_hat >= 0, 1.0, -1.0)] + [np.where(t_hat + 0.3 * rng.standard_normal(P.n) >= 0, 1.0, -1.0) for _ in range(restarts)]
    best = (None, None)
    sgn = 1.0 if P.sense == "max" else -1.0
    for t in cands:
        if P.boolean and local_search:
            t = _flip_search(P, t)
        elif local_search and not P.boolean:
            t = _polish(P, t)
        x = P.to_original(t)
        if P.violation(x) <= 1e-6:
            v = P.value(x)
            if best[0] is None or sgn * v > sgn * best[1]:
                best = (x, v)
    return best


def _flip_search(P, t):
    t = t.copy()
    f = lambda tt: P.value(tt)
    sgn = 1.0 if P.sense == "max" else -1.0
    cur = sgn * f(t)
    improved = True
    while improved:
        improved = False
        for i in range(P.n):
            t[i] = -t[i]
            val = sgn * f(t)
            if val > cur + 1e-12 and P.violation(t) <= 1e-9:
                cur, improved = val, True
            else:
                t[i] = -t[i]
    return t


def _polish(P, t):
    from scipy.optimize import minimize
    sgn = 1.0 if P.sense == "max" else -1.0
    x0 = P.to_original(t)
    cons = [{"type": "ineq" if c.kind == "ineq" else "eq", "fun": (lambda x, c=c: c.poly(x))} for c in P.constraints]
    try:
        r = minimize(lambda x: -sgn * P.value(x), x0, method="SLSQP", bounds=list(zip(P.lo, P.hi)), constraints=cons,
                     options=dict(maxiter=100, ftol=1e-10))
        if P.violation(r.x) <= 1e-7:
            return P.to_internal(r.x)
    except Exception:
        pass
    return t
