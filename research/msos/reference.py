"""Independent exact moment relaxation (MOSEK via cvxpy), built directly from moment matrices --
shares no code with the Gram-matrix dual in core.py except the monomial key arithmetic."""
import time
import numpy as np
import cvxpy as cp
from .core import Ring, degree


def default_solver():
    """MOSEK when its cvxpy interface is available, otherwise the open-source interior-point solver Clarabel.
    Override with the environment variable MSOS_SOLVER."""
    import os
    env = os.environ.get("MSOS_SOLVER")
    if env:
        return env
    return "MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL"


def _basis_exps(variables, d, n, boolean):
    out = []
    for k in range(d + 1):
        it = itertools.combinations(variables, k) if boolean else itertools.combinations_with_replacement(variables, k)
        for c in it:
            e = [0] * n
            for i in c:
                e[i] += 1
            out.append(tuple(e))
    return out


import itertools


UNBOUNDED_VALUE = 1e6   # a relaxation value above this is treated as "unbounded" (see moment_relaxation)


def moment_relaxation(pop, r, cliques=None, trace_cuts=True, solver=None, verbose=False, cuts=None):
    """Value of the order-r (sparse) Lasserre relaxation.

    cuts : "none"  -- plain Lasserre relaxation rho_r;
           "trace" -- adds tr(M_k) <= s_k and tr(M_i^loc) <= s_i*B_i (the box-implied trace cuts,
                      Eq. (cuts) of the paper) -- the value is identical to the spectral dual, rho_r^tr;
           "diag"  -- adds the diagonal cuts M_k(y)_aa <= 1 and M_i^loc(y)_aa <= B_i for every diagonal
                      entry (Proposition prop:diag), value rho_r^d <= rho_r^tr <= rho_r.
           cuts=None (default) falls back to the legacy flag trace_cuts: True -> "trace", False -> "none".
    Returns (value, solve_time).  If the relaxation is unbounded (status 'unbounded'/'unbounded_inaccurate',
    or an optimal value above UNBOUNDED_VALUE = 1e6 -- e.g. the box-free example of the paper, where the
    moment vectors of Dirac masses at (t,0) are feasible for every t) the value is float('inf').
    If the primary solver (Clarabel by default; MOSEK if installed) raises or ends with an inaccurate status, SCS
    is tried as a fallback; if both are inaccurate, a big-M probe (diagonal cuts M_aa <= 1e4 and <= 1e8) decides:
    a value that still grows with the big-M bound means the relaxation is unbounded -> float('inf'); otherwise
    the probe value (with the larger bound, which is then inactive) is returned."""
    from .core import box_sup_bound
    solver = solver or default_solver()
    if cuts is None:
        cuts = "trace" if trace_cuts else "none"
    if cuts not in ("none", "trace", "diag"):
        raise ValueError("cuts must be 'none', 'trace' or 'diag'")
    trace_cuts = cuts != "none"
    n = pop.n
    cliques = [list(range(n))] if cliques is None else cliques
    bl = pop.boolean
    D = 2 * r
    ring = Ring(n, D, bl)
    mom = {}

    def var(key):
        if key not in mom:
            mom[key] = len(mom)
        return mom[key]

    def addk(a, b):
        return (a ^ b) if bl else (a + b)

    # first pass: collect keys, then create variable
    def mexp(e):
        return ring.key(e)

    blocks = []           # (list of list of {key:coef}, trace_bound or None)
    for c in cliques:
        Bk = ring.basis(c, r)
        Bk = [int(b) for b in Bk]
        M = [[{addk(a, b): 1.0} for b in Bk] for a in Bk]
        blocks.append((M, len(Bk) if trace_cuts else None, "moment"))
    for g in pop.ineq:
        supp = {i for e in g for i, x in enumerate(e) if x}
        host = next(c for c in cliques if supp <= set(c))
        om = (degree(g) + 1) // 2
        Z = [int(z) for z in ring.basis(host, r - om)]
        gt = [(ring.key(e), cf) for e, cf in g.items()]
        M = [[{addk(addk(a, b), gk): cf for gk, cf in gt} for b in Z] for a in Z]
        blocks.append((M, len(Z) * box_sup_bound(g, bl) if trace_cuts else None, "loc"))
    eqs = []
    for h in pop.eq:
        supp = {i for e in h for i, x in enumerate(e) if x}
        host = next(c for c in cliques if supp <= set(c))
        Mb = [int(m) for m in ring.basis(host, D - degree(h))]
        ht = [(ring.key(e), cf) for e, cf in h.items()]
        for m in Mb:
            eqs.append({addk(m, hk): cf for hk, cf in ht})
    for M, _, _ in blocks:
        for row in M:
            for d in row:
                for k in d:
                    var(k)
    for d in eqs:
        for k in d:
            var(k)
    objd = {ring.key(e): c for e, c in pop.f.items()}
    for k in objd:
        var(k)
    zero = ring.key(tuple([0] * n)); var(zero)
    y = cp.Variable(len(mom))
    cons = [y[mom[zero]] == 1]

    def lin(d):
        idx = [mom[k] for k in d]
        return sum(c * y[i] for i, c in zip(idx, d.values()))

    Ss = []
    for M, tb, kind in blocks:
        s = len(M)
        if kind == "moment" and not any(len(d) > 1 for row in M for d in row):
            idxm = np.array([[mom[next(iter(d))] for d in row] for row in M])
            S = cp.reshape(y[idxm.flatten()], (s, s), order="C")
        else:
            S = cp.bmat([[lin(d) for d in row] for row in M])
        cons.append((S + S.T) / 2 >> 0)
        Ss.append((S, tb, s))
        if tb is not None:
            if cuts == "trace":
                cons.append(cp.trace(S) <= tb)
            else:                                   # "diag": every diagonal entry  <= 1  (moment) / <= B_i (loc)
                cons.append(cp.diag(S) <= tb / s)
    for d in eqs:
        cons.append(lin(d) == 0)
    obj = lin(objd)
    prob = cp.Problem(cp.Maximize(obj), cons)
    t = time.perf_counter()
    val = _solve_with_fallback(prob, solver, verbose)
    if val is None:
        # Both solvers ended with an *inaccurate* status (typical for relaxations that are unbounded without an
        # improving ray, e.g. the box-free example at r >= 2, where the value grows only like |y|^(1/2r)).
        # Probe: add the big-M diagonal cuts  M_aa <= Mbig  for two values of Mbig.  The value is nondecreasing in
        # Mbig and constant once Mbig exceeds the largest moment of an optimal solution of a bounded relaxation;
        # if it still grows between the two probes we declare the relaxation unbounded (float('inf')).
        vals = []
        for Mbig in (1e4, 1e8):
            pc = cp.Problem(cp.Maximize(obj), cons + [cp.diag(S) <= Mbig for S, _, _ in Ss])
            v = _solve_with_fallback(pc, solver, verbose, allow_inaccurate=True)
            vals.append(v)
        if verbose:
            print(f"[moment_relaxation] big-M probe values: {vals}")
        if any(v is None for v in vals):
            raise RuntimeError(f"moment_relaxation: solver status {prob.status!r} (value {prob.value!r}) and probe failed")
        val = float("inf") if (not np.isfinite(vals[1]) or vals[1] > vals[0] * (1 + 1e-3) + 1e-6) else vals[1]
    return val, time.perf_counter() - t


def _status_value(prob, allow_inaccurate=False):
    """Interpret a solved cvxpy problem: float value, float('inf') for unbounded, None if uninformative
    (solver failure, or an 'inaccurate' status unless allow_inaccurate)."""
    st = prob.status
    if st in ("unbounded", "unbounded_inaccurate"):
        return float("inf")
    ok = ("optimal", "optimal_inaccurate") if allow_inaccurate else ("optimal",)
    if st in ok and prob.value is not None and np.isfinite(prob.value):
        v = float(prob.value)
        return float("inf") if v > UNBOUNDED_VALUE else v
    return None


def _solve_with_fallback(prob, solver, verbose=False, allow_inaccurate=False):
    """Solve with `solver` (Clarabel by default); if it raises or returns an uninformative status, retry with the
    first-order solver SCS.  Returns float('inf') if either solver reports 'unbounded' or a value > UNBOUNDED_VALUE,
    the optimal value if either solver reports 'optimal', and None if both end inaccurate/failed."""
    try:
        prob.solve(solver=solver, verbose=verbose)
        val = _status_value(prob, allow_inaccurate)
    except Exception as exc:                        # SolverError etc.
        if verbose:
            print(f"[moment_relaxation] {solver} failed: {exc!r}; falling back to SCS")
        val = None
    if val is None and solver != "SCS":
        try:
            prob.solve(solver="SCS", verbose=verbose, max_iters=100000, eps=1e-8)
            val = _status_value(prob, allow_inaccurate)
        except Exception as exc:
            if verbose:
                print(f"[moment_relaxation] SCS failed as well: {exc!r}")
            val = None
    return val
