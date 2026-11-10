"""Rigorous a posteriori certificate (paper: Lemma lem:chol, Theorem thm:rigor).

Two ingredients turn the floating-point output theta_hat of the solver into a bound that is
valid in exact arithmetic:

  (a) verified_lmin_lower_bound : a lower bound ell <= lmin(Q) for a symmetric floating-point
      matrix Q, from a floating-point Cholesky factorisation of  A~ = fl(Q + alpha I)  and the
      backward-error theorem for Cholesky (Higham 2002, Thm. 10.3):
          lmin(Q) >= -alpha - gamma_{s+1} * ||R_hat||_F^2 - u * max_a |Q_aa + alpha|,
      u = 2^-53, gamma_m = m u / (1 - m u).  The Cholesky factorisation is written out
      column by column in NumPy (every inner product is a floating-point inner product, in
      some order, which is what the lemma assumes); LAPACK is used only for a pre-check.
  (b) exact_residual : the residual polynomial
          e = sigma_hat - f - sum_k v_k^T Qhat_k v_k - sum_i (z_i^T Qhat_i z_i) g_i - sum_j phat_j h_j
      (mod the ideal) in EXACT rational arithmetic.  Every double is a dyadic rational, so after
      scaling by a common power of two all coefficients are Python integers; E = sum_gamma |e_gamma|.
  (c) rigorous_bound : U_rig = sigma_hat + sum_k s_k Phi_k(-ell_k) + sum_i s_i B_i [-ell_i]_+ + E,
      final sum in Fractions, rounded up to a double.
"""
import math
import time
from fractions import Fraction

import numpy as np

from .core import box_sup_bound_exact

U_ROUND = Fraction(1, 2 ** 53)          # unit round-off of IEEE double


def gamma(m):
    """gamma_m = m u / (1 - m u) as an exact Fraction."""
    mu = m * U_ROUND
    if mu >= 1:
        raise ValueError("gamma_m undefined: m u >= 1")
    return mu / (1 - mu)


def _float_up(fr):
    """Smallest double >= the Fraction fr."""
    x = float(fr)
    while Fraction(x) < fr:
        x = np.nextafter(x, np.inf)
    return float(x)


def _float_down(fr):
    """Largest double <= the Fraction fr."""
    x = float(fr)
    while Fraction(x) > fr:
        x = np.nextafter(x, -np.inf)
    return float(x)


# --------------------------------------------------------------------------- #
#  (a) verified lower bound on lambda_min                                      #
# --------------------------------------------------------------------------- #
def cholesky_columns(A):
    """Column-oriented (left-looking) Cholesky  A = R^T R  with R upper triangular, written
    explicitly in NumPy.  Returns (R, ok); ok=False (and the partial R) if a pivot is <= 0 or
    not finite.  Only the upper triangle of A is read."""
    A = np.asarray(A, dtype=np.float64)
    s = A.shape[0]
    R = np.zeros((s, s))
    for j in range(s):
        row = A[j, j:] - R[:j, j] @ R[:j, j:]          # inner products in floating point
        piv = row[0]
        if not (piv > 0.0) or not np.isfinite(piv):
            return R, False
        d = math.sqrt(piv)
        R[j, j] = d
        R[j, j + 1:] = row[1:] / d
    return R, True


def verified_lmin_lower_bound(Q, lmin_hat=None, margin=None, max_doublings=40, precheck=True):
    """Verified lower bound ell <= lmin(Q) for a symmetric double matrix Q (Lemma lem:chol).

    Q        : symmetric (s, s) float64 array (its entries are taken as exact).
    lmin_hat : approximate smallest eigenvalue (eigvalsh if None).
    margin   : initial eps > 0; default max(1e-12, 1e-10*|lmin_hat|, 64 u ||Q||_F).  Doubled
               until the floating-point Cholesky of fl(Q + alpha I), alpha = -lmin_hat + eps,
               runs to completion (at most `max_doublings` doublings).

    Returns (ell, info) with ell a double such that lmin(Q) >= ell (ell = -inf on failure) and
    info = dict(alpha, eps, doublings, nu, gamma, s, slack=lmin_hat-ell, time, ok)."""
    t0 = time.perf_counter()
    Q = np.asarray(Q, dtype=np.float64)
    s = Q.shape[0]
    if Q.shape != (s, s) or not np.array_equal(Q, Q.T):
        raise ValueError("Q must be a square symmetric float64 array")
    if lmin_hat is None:
        lmin_hat = float(np.linalg.eigvalsh(Q)[0])
    lmin_hat = float(lmin_hat)
    u = float(U_ROUND)
    if margin is None:
        margin = max(1e-12, 1e-10 * abs(lmin_hat), 64.0 * u * float(np.linalg.norm(Q, "fro")))
    eps = float(margin)
    info = dict(s=s, lmin_hat=lmin_hat, ok=False, doublings=0)
    diag = np.diag(Q).copy()
    for k in range(max_doublings + 1):
        alpha = float(-lmin_hat + eps)                   # alpha in F
        At = Q.copy()
        At[np.arange(s), np.arange(s)] = diag + alpha    # fl(Q + alpha I): only the diagonal is rounded
        ok = True
        if precheck:
            try:
                np.linalg.cholesky(At)                   # cheap LAPACK pre-check (NOT the certificate)
            except np.linalg.LinAlgError:
                ok = False
        if ok:
            R, ok = cholesky_columns(At)
        if ok:
            N = s * (s + 1) // 2
            nu_hat = float(np.sum(R * R))                # fl(||R||_F^2)
            nu = Fraction(nu_hat) * (1 + gamma(N + 1))   # >= ||R_hat||_F^2  (any summation order)
            g = gamma(s + 1)
            # exact: |Delta_d| <= u max_a |fl(Q_aa + alpha)| ... we bound max|Q_aa + alpha| by the
            # computed max inflated by (1+u) (fl(x) differs from x by at most u|x|).
            md = Fraction(float(np.max(np.abs(At[np.arange(s), np.arange(s)])))) * (1 + U_ROUND)
            ell_fr = Fraction(-alpha) - g * nu - U_ROUND * md
            ell = _float_down(ell_fr)
            info.update(alpha=alpha, eps=eps, doublings=k, nu=float(nu), gamma=float(g), ok=True,
                        slack=lmin_hat - ell, time=time.perf_counter() - t0)
            return ell, info
        eps *= 2.0
    info.update(alpha=None, eps=eps, time=time.perf_counter() - t0, slack=math.inf)
    return -math.inf, info


# --------------------------------------------------------------------------- #
#  (b) exact residual of the Gram identity                                     #
# --------------------------------------------------------------------------- #
def _mant_exp(x):
    """x = M * 2**e with integer M (|M| < 2^53) and integer e, elementwise, exact.
    Returns (M as object array of Python ints, e as int array)."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite coefficient")
    m, e = np.frexp(x)                                   # x = m 2^e, 0.5 <= |m| < 1
    M = np.ldexp(m, 53).astype(np.int64)                 # exact integer
    return M.astype(object), (e.astype(np.int64) - 53)


def exact_residual(dual, theta, return_poly=True):
    """Exact residual polynomial e of the Gram identity for the floating-point Gram matrices
    Qhat_b = dual.gram_blocks(theta), sigma_hat = theta[0], phat = theta[dual.sl_p].

    Returns (E_float, E_frac, e_dict, info):
      E_float : sum_gamma |e_gamma| as a double rounded UP,
      E_frac  : the same as an exact Fraction,
      e_dict  : {monomial key (int): Fraction} of the non-zero residual coefficients
                (None if return_poly=False),
      info    : dict(scale_exp=S, time=...).
    """
    t0 = time.perf_counter()
    theta = np.asarray(theta, dtype=np.float64)
    Qs = dual.gram_blocks(theta)
    K = len(dual.cliques)
    ring = dual.ring

    def idx(keys):
        return np.searchsorted(dual.reg, np.asarray(keys, dtype=ring.dtype))

    # collect (moment index array, integer mantissa array, exponent array) contributions
    contribs = []

    # sigma_hat at the constant monomial
    M, e = _mant_exp(np.array([theta[0]]))
    contribs.append((np.array([dual.empty_id]), M, e))
    # - f
    if dual.pop.f:
        fk = np.array([ring.key(ex) for ex in dual.pop.f], dtype=ring.dtype)
        fc = np.array([c for c in dual.pop.f.values()], dtype=np.float64)
        M, e = _mant_exp(fc)
        contribs.append((idx(fk), -M, e))
    # - sum_k v_k^T Qhat_k v_k :  slot (I<=J) contributes w_slot * Q[I,J] to moment slot_g
    for k in range(K):
        I, J = dual.cl_I[k], dual.cl_J[k]
        sl = slice(dual.cl_off[k], dual.cl_off[k + 1])
        vals = Qs[k][I, J]
        w = dual.slot_w[sl].astype(np.int64).astype(object)   # 1 or 2 (exact integers)
        M, e = _mant_exp(vals)
        contribs.append((dual.slot_g[sl], -(M * w), e))
    # - sum_i (z_i^T Qhat_i z_i) g_i :  for each term (gk, c) of g_i and each a<=b:
    #       wq * c * Q_i[a,b]  at key  Z[a] Z[b] gk      (mirrors core.py R_loc)
    for i, info in enumerate(dual.ineq_info):
        Qi = Qs[K + i]
        a, b = info["a"], info["b"]
        vals = Qi[a, b]
        wq = np.where(a == b, 1, 2).astype(object)
        Mq, eq = _mant_exp(vals)
        for kp, (_, c) in zip(info["kpt"], info["terms"]):
            Mc, ec = _mant_exp(np.array([c]))
            contribs.append((idx(kp), -(Mq * wq * Mc[0]), eq + ec[0]))
    # - sum_j phat_j h_j :  for each term (hk, c) of h_j and each m:  c * p_j[m] at key M[m] hk
    p = theta[dual.sl_p]
    for j, info in enumerate(dual.eq_info):
        pj = p[dual.p_off[j]:dual.p_off[j + 1]]
        Mp, ep = _mant_exp(pj)
        for kp, (_, c) in zip(info["kpt"], info["terms"]):
            Mc, ec = _mant_exp(np.array([c]))
            contribs.append((idx(kp), -(Mp * Mc[0]), ep + ec[0]))

    # common power-of-two denominator 2^S
    S = 0
    for _, _, e in contribs:
        if len(e):
            S = max(S, int(-e.min()))
    acc = np.zeros(dual.nmom, dtype=object)
    acc[:] = 0
    for g, M, e in contribs:
        if len(e) == 0:
            continue
        sh = (e + S).astype(np.int64)                    # >= 0
        scaled = np.array([int(m) << int(s_) for m, s_ in zip(M, sh)], dtype=object)
        np.add.at(acc, g, scaled)
    den = 1 << S
    E_int = 0
    e_dict = {} if return_poly else None
    for gidx in range(dual.nmom):
        v = acc[gidx]
        if v:
            E_int += abs(v)
            if return_poly:
                e_dict[int(dual.reg[gidx])] = Fraction(v, den)
    E_frac = Fraction(E_int, den)
    E_float = _float_up(E_frac)
    return E_float, E_frac, e_dict, dict(scale_exp=S, time=time.perf_counter() - t0, nnz=len(e_dict) if return_poly else None)


def eval_residual_exact(dual, e_dict, x):
    """Evaluate the residual polynomial e (from exact_residual) at a point x exactly (Fractions)."""
    xf = [Fraction(float(t)) for t in x]
    tot = Fraction(0)
    for key, c in e_dict.items():
        ex = dual.ring.decode(key)
        m = Fraction(1)
        for xi, ei in zip(xf, ex):
            if ei:
                m *= xi ** int(ei)
        tot += c * m
    return tot


def residual_float_at_point(dual, theta, x):
    """Float evaluation of sigma - f(x) - sum v^T Q v - sum (z^T Q z) g - sum p h at x (for checks)."""
    Qs = dual.gram_blocks(theta)
    K = len(dual.cliques)
    x = np.asarray(x, float)
    ring = dual.ring

    def mono(keys):
        return np.array([np.prod(x ** np.array(ring.decode(k), dtype=float)) for k in keys])

    def ev(poly):
        return sum(c * np.prod(x ** np.array(e, dtype=float)) for e, c in poly.items())

    val = theta[0] - ev(dual.pop.f)
    for k, B in enumerate(dual.cl_basis):
        v = mono(B)
        val -= v @ Qs[k] @ v
    for i, (g, info) in enumerate(zip(dual.pop.ineq, dual.ineq_info)):
        z = mono(info["Z"])
        val -= (z @ Qs[K + i] @ z) * ev(g)
    p = theta[dual.sl_p]
    for j, (h, info) in enumerate(zip(dual.pop.eq, dual.eq_info)):
        pj = p[dual.p_off[j]:dual.p_off[j + 1]]
        val -= (pj @ mono(info["M"])) * ev(h)
    return float(val)


def sanity_check_residual(dual, theta=None, trials=5, seed=0):
    """For a (tiny) problem: compare the exact residual polynomial evaluated at random points of
    the box (or of {+-1}^n) with the float evaluation of the identity.  Returns the max |difference|
    (should be ~1e-13 or less) and E."""
    rng = np.random.default_rng(seed)
    if theta is None:
        theta = rng.standard_normal(dual.dim) * 0.5
    E, E_fr, e_dict, _ = exact_residual(dual, theta)
    worst = 0.0
    for _ in range(trials):
        x = rng.choice([-1.0, 1.0], dual.pop.n) if dual.pop.boolean else rng.uniform(-1, 1, dual.pop.n)
        ex = float(eval_residual_exact(dual, e_dict, x))
        fl = residual_float_at_point(dual, theta, x)
        worst = max(worst, abs(ex - fl))
    return worst, E


# --------------------------------------------------------------------------- #
#  (c) rigorous certificate                                                    #
# --------------------------------------------------------------------------- #
def rigorous_bound(dual, theta, verbose=False):
    """Rigorous upper bound U_rig (Theorem thm:rigor) for the floating-point dual point theta.

    Returns dict(U, U_rig, U_rig_minus_U, E, E_frac, ell (per block), lmin_hat (per block),
    slack (per block, lmin_hat - ell), max_slack, t_chol, t_res, t_eig, chol_ok (per block),
    block_sizes, s_max)."""
    theta = np.asarray(theta, dtype=np.float64)
    U = float(dual.certificate(theta))
    Qs = dual.gram_blocks(theta)
    # eigenvalue estimates
    t0 = time.perf_counter()
    lmin_hat = [float(np.linalg.eigvalsh(Q)[0]) for Q in Qs]
    t_eig = time.perf_counter() - t0
    # verified lower bounds
    ells, slacks, oks, infos = [], [], [], []
    t_chol = 0.0
    for b, (Q, lh) in enumerate(zip(Qs, lmin_hat)):
        ell, info = verified_lmin_lower_bound(Q, lh)
        t_chol += info["time"]
        ells.append(ell); slacks.append(lh - ell); oks.append(info["ok"]); infos.append(info)
        if verbose:
            print(f"  block {b}: s={Q.shape[0]} lmin_hat={lh:.3e} ell={ell:.3e} slack={lh-ell:.2e} doublings={info['doublings']}")
    # exact residual
    E, E_frac, _, rinfo = exact_residual(dual, theta, return_poly=False)
    t_res = rinfo["time"]
    # final sum in exact arithmetic, rounded up
    total = Fraction(float(theta[0])) + E_frac
    K = len(dual.cliques)
    for b, blk in enumerate(dual.blocks):
        ell = ells[b]
        if ell == -math.inf:
            total = None
            break
        lam = Fraction(-ell)
        if blk["plus"]:
            lam = max(lam, Fraction(0))
        if blk["kind"] == "moment":
            wt = Fraction(blk["size"])
        else:
            info = dual.ineq_info[blk["idx"]]
            wt = Fraction(blk["size"]) * box_sup_bound_exact(dual.pop.ineq[blk["idx"]], dual.pop.boolean)   # exact B_i (a float sum may round below sup g)
        total += wt * lam
    U_rig = _float_up(total) if total is not None else math.inf
    return dict(U=U, U_rig=U_rig, U_rig_minus_U=U_rig - U, E=E, E_frac=E_frac, ell=ells, lmin_hat=lmin_hat,
                slack=slacks, max_slack=max(slacks) if slacks else 0.0, chol_ok=oks, all_ok=all(oks),
                t_chol=t_chol, t_res=t_res, t_eig=t_eig, block_sizes=[Q.shape[0] for Q in Qs],
                s_max=max(Q.shape[0] for Q in Qs), doublings=[i["doublings"] for i in infos])
