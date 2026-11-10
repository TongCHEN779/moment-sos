"""Rigorous a-posteriori certificate (Lemma 'verified lower bound on lambda_min' and Theorem 'rigorous certificate').

The floating-point output theta_hat of a solver defines Gram matrices Q_b (the float values returned by
`Oracle.gram_blocks`).  Two ingredients give a bound that holds in exact arithmetic for the *stored* (normalised)
problem data:
  (a) a verified lower bound  ell_b <= lambda_min(Q_b)  from a floating-point Cholesky factorisation of fl(Q_b + alpha I)
      and the Cholesky backward-error theorem (Higham, Accuracy and Stability of Numerical Algorithms, Thm 10.3);
  (b) the exact residual E = sum_gamma |e_gamma| of the Gram identity (every double is a dyadic rational, so the
      residual is computed with Python Fractions).
Then  U_rig = sigma_hat + sum_k s_k Phi_k(-ell_k) + sum_i s_i B_i [-ell_i]_+ + E  >= max f  on K (in exact arithmetic).
NOTE the guarantee is for the normalised floating-point problem stored in `Problem` (rescaling to [-1,1]^n and
constraint/objective normalisation round coefficients); the affine map back to the user's objective is done exactly
but the rounding made when the data were normalised is not accounted for.  For data that are exactly representable
after normalisation (e.g. unit-weight MAX-CUT) the statement is exact.
"""
from __future__ import annotations
import math
import time
from fractions import Fraction
import numpy as np
import torch
from .relaxation import box_sup_bound_exact

U_ROUND = Fraction(1, 2 ** 53)


def _gamma(m):
    mu = m * U_ROUND
    if mu >= 1:
        raise ValueError("gamma_m undefined: m u >= 1")
    return mu / (1 - mu)


def _float_up(fr):
    x = float(fr)
    while Fraction(x) < fr:
        x = float(np.nextafter(x, np.inf))
    return x


def _float_down(fr):
    x = float(fr)
    while Fraction(x) > fr:
        x = float(np.nextafter(x, -np.inf))
    return x


def cholesky_columns(A):
    """Column-oriented (left-looking) Cholesky A = R^T R written explicitly so that every inner product is a
    floating-point inner product (the hypothesis of the backward-error theorem). Returns (R, ok)."""
    A = np.asarray(A, dtype=np.float64)
    s = A.shape[0]
    R = np.zeros((s, s))
    for j in range(s):
        row = A[j, j:] - R[:j, j] @ R[:j, j:]
        piv = row[0]
        if not (piv > 0.0) or not np.isfinite(piv):
            return R, False
        d = math.sqrt(piv)
        R[j, j] = d
        R[j, j + 1:] = row[1:] / d
    return R, True


def verified_lmin_lower_bound(Q, margin=None, max_doublings=40):
    """ell <= lambda_min(Q) for a symmetric float64 matrix Q, or -inf if the factorisation never succeeds."""
    Q = np.asarray(Q, dtype=np.float64)
    s = Q.shape[0]
    if not np.array_equal(Q, Q.T):
        raise ValueError("Q must be symmetric")
    lmin_hat = float(np.linalg.eigvalsh(Q)[0])
    u = float(U_ROUND)
    eps = float(margin) if margin is not None else max(1e-12, 1e-10 * abs(lmin_hat), 64.0 * u * float(np.linalg.norm(Q, "fro")))
    diag = np.diag(Q).copy()
    for k in range(max_doublings + 1):
        alpha = float(-lmin_hat + eps)
        At = Q.copy()
        At[np.arange(s), np.arange(s)] = diag + alpha
        R, ok = cholesky_columns(At)
        if ok:
            N = s * (s + 1) // 2
            nu = Fraction(float(np.sum(R * R))) * (1 + _gamma(N + 1))
            md = Fraction(float(np.max(np.abs(np.diag(At))))) * (1 + U_ROUND)
            ell = _float_down(Fraction(-alpha) - _gamma(s + 1) * nu - U_ROUND * md)
            return ell, dict(ok=True, lmin_hat=lmin_hat, slack=lmin_hat - ell, doublings=k)
        eps *= 2.0
    return -math.inf, dict(ok=False, lmin_hat=lmin_hat, slack=math.inf, doublings=max_doublings)


def exact_residual(relax, oracle, theta):
    """Exact E = sum_gamma |e_gamma| of the Gram identity (as a Fraction), for the float Gram matrices of theta."""
    th = theta.detach().cpu().numpy().astype(np.float64)
    ns, nu = relax.nslots, relax.nu
    acc = [Fraction(0)] * relax.nmom
    F = lambda x: Fraction(float(x))
    acc[relax.empty_id] += F(th[0])
    for g, v in enumerate(relax.fvec):
        if v != 0.0:
            acc[g] -= F(v)
    if nu:
        rows, cols, vals = relax.Rloc
        u = th[1 + ns:1 + ns + nu]
        for r_, c_, v_ in zip(rows, cols, vals):
            acc[int(r_)] -= F(v_) * F(u[c_])
    if relax.np_:
        rows, cols, vals = relax.Req
        p = th[1 + ns + nu:]
        for r_, c_, v_ in zip(rows, cols, vals):
            acc[int(r_)] -= F(v_) * F(p[c_])
    with torch.no_grad():
        Qs = [Q.cpu().numpy() for Q in oracle.gram_blocks(theta.detach())]
    for k, (I, J) in enumerate(zip(relax.cl_I, relax.cl_J)):
        sl = slice(relax.cl_off[k], relax.cl_off[k + 1])
        vals = Qs[k][I, J]
        w = relax.slot_w[sl]
        for g_, v_, w_ in zip(relax.slot_g[sl], vals, w):
            acc[int(g_)] -= int(w_) * F(v_)
    return sum((abs(a) for a in acc), Fraction(0))


def rigorous_bound(result):
    """Rigorous (exact-arithmetic) upper bound U_rig for the normalised problem and the corresponding bound for the
    user's problem (upper bound for max, lower bound for min).  Returns a dict."""
    t0 = time.perf_counter()
    R, O, th = result.relaxation, result.oracle, result.theta
    with torch.no_grad():
        Qs = [Q.cpu().numpy().astype(np.float64) for Q in O.gram_blocks(th.detach())]
    nl = len(R.cl_basis)
    ells, oks, total = [], [], Fraction(float(th[0]))
    for b, (blk, Q) in enumerate(zip(R.blocks, Qs)):
        if blk["weight"] == 0.0:
            continue
        ell, info = verified_lmin_lower_bound(Q)
        ells.append(ell); oks.append(info["ok"])
        if not info["ok"]:
            total = None
            break
        lam = Fraction(-ell)
        if blk["plus"]:
            lam = max(lam, Fraction(0))
        if blk["kind"] == "moment":
            wt = Fraction(blk["size"])
        else:
            wt = Fraction(blk["size"]) * box_sup_bound_exact(R.ineq_terms[b - nl], R.prob.boolean, R.tight_weights)
        total += wt * lam
    E = exact_residual(R, O, th)
    if total is None:
        return dict(U_rig=math.inf, bound=math.inf if result.problem.sense == "max" else -math.inf, ok=False, time=time.perf_counter() - t0)
    U_int = total + E
    P = result.problem
    u_orig = Fraction(float(P.f_off)) + Fraction(float(P.f_scale)) * U_int          # exact affine map back
    if P.sense == "max":
        bound = _float_up(u_orig)
    else:
        bound = _float_down(-u_orig)
    return dict(U_rig=_float_up(U_int), U=result.U, bound=bound, E=float(E), ells=ells, ok=all(oks), time=time.perf_counter() - t0)
