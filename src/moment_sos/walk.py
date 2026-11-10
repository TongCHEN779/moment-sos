"""Adaptive relaxations between order 1 and order 2 for quadratic spin problems (the "triangle walk").

For MAX-CUT-type problems (spins ``x_i = +-1``, objective of degree 2) the order-2 relaxation is large, but most of its
strength over order 1 comes from a few *triples* ``{i, j, k}``: adding a triple means adding the three pair monomials
``x_i x_j, x_i x_k, x_j x_k`` to the moment basis, and the resulting relaxation contains the four triangle inequalities

    1 + s_1 Y_ij + s_2 Y_ik + s_3 Y_jk >= 0,   (s_1, s_2, s_3) in {(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}

of the moment matrix ``Y_ij = E[x_i x_j]``. A *round* picks a batch of ``K`` triples among the triples violated by the
current solution and re-solves; a *walk* is a sequence of rounds, every one of which gives a valid, monotone, anytime
bound (certified by the spectral dual like every other bound of this package).

Selection rules (``rule=``):

``"topk"``    the ``K`` most violated triples (the standard cutting-plane rule);
``"nms"``     the most violated triples subject to no two sharing a pair ("non-maximum suppression"): diversity avoids
              adding several triples that cut off the same part of the moment matrix;
``"random"``  ``K`` random triples (baseline).

Two ways to impose the chosen triples (``mode=``):

``"cuts"``    the four triangle inequalities are added as polynomial constraints to an order-1 relaxation (scalar
              localizing blocks): a polyhedral relaxation, cheap, which is what one uses at scale;
``"basis"``   the pair monomials are added to the moment block of the order-1 relaxation (partial order-2 lifting): the
              block grows by 3 rows per triple.

The selection rules and the idea of walking between the orders come from the ai4opt study of learned lifting paths, in
which hand-designed rules (``nms`` in particular) were found to beat the learned policies at a matched budget.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
import torch
from .poly import Constraint, Poly
from .problem import Problem

SIGNS = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
RULES = ("nms", "topk", "random")
MODES = ("cuts", "basis")


# --------------------------------------------------------------------------- separation and selection
def triangle_violations(Y, max_candidates=4000, chunk=8, tol=1e-6):
    """Exact separation of the triangle inequalities over *all* triples ``i < j < k``.

    Parameters
    ----------
    Y : (n, n) array
        Moment matrix entries ``Y_ij = E[x_i x_j]`` (diagonal ignored).
    max_candidates : int
        Keep this many most violated triples.
    chunk : int
        Number of ``i`` rows processed at once (memory/speed trade-off).
    tol : float
        Only triples violated by more than ``tol`` are returned.

    Returns
    -------
    tris : (m, 3) int array, viol : (m,) float array
        The ``m <= max_candidates`` most violated triples, sorted by decreasing violation
        ``max_s [-1 - s . (Y_ij, Y_ik, Y_jk)]`` over the four valid sign patterns.
    """
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    if n < 3:
        return np.zeros((0, 3), int), np.zeros(0)
    Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
    jj = torch.arange(n)
    upper = jj[:, None] < jj[None, :]
    vals, idxs = [], []
    for i0 in range(0, n - 2, chunk):
        i1 = min(n, i0 + chunk)
        A, B, C = Yt[i0:i1, :, None], Yt[i0:i1, None, :], Yt[None, :, :]
        v = torch.maximum(torch.maximum(-1 - (A + B + C), -1 - (A - B - C)),
                          torch.maximum(-1 - (-A + B - C), -1 - (-A - B + C)))
        ii = torch.arange(i0, i1)[:, None, None]
        mask = upper[None] & (jj[None, :, None] > ii) & (jj[None, None, :] > ii)
        v = torch.where(mask, v, torch.full_like(v, -1e9))
        flat = v.reshape(-1)
        tv, ti = torch.topk(flat, min(max_candidates, flat.numel()))
        keep = tv > tol
        tv, ti = tv[keep], ti[keep]
        if len(tv):
            c = ti // (n * n)
            rem = ti % (n * n)
            vals.append(tv)
            idxs.append(torch.stack([c + i0, rem // n, rem % n], 1))
    if not vals:
        return np.zeros((0, 3), int), np.zeros(0)
    V, I = torch.cat(vals), torch.cat(idxs)
    tv, order = torch.topk(V, min(max_candidates, len(V)))
    return I[order].numpy().astype(int), tv.numpy().astype(float)


def select_triples(Y, K, rule="nms", max_candidates=4000, rng=None):
    """Pick up to ``K`` triples of vertices for the next round (see the module docstring for the rules).

    Returns an ``(m, 3)`` int array, ``m <= K``; empty when no triangle inequality is violated (``rule`` ``"random"``
    ignores violations and always returns ``K`` distinct triples)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    rng = np.random.default_rng(0) if rng is None else rng
    if rule == "random":
        K = min(K, n * (n - 1) * (n - 2) // 6)
        out = set()
        while len(out) < K:
            out.add(tuple(sorted(rng.choice(n, 3, replace=False).tolist())))
        return np.array(sorted(out), int).reshape(-1, 3)
    tris, viol = triangle_violations(Y, max_candidates)
    if len(tris) == 0:
        return tris
    if rule == "topk":
        return tris[:K]
    used, sel = set(), []
    for q, (i, j, k) in enumerate(tris):
        pairs = ((i, j), (i, k), (j, k))
        if any(p in used for p in pairs):
            continue
        sel.append(q)
        used.update(pairs)
        if len(sel) >= K:
            break
    if len(sel) < K:                                             # fewer than K pair-disjoint candidates: fill by violation
        chosen = set(sel)
        sel += [q for q in range(len(tris)) if q not in chosen][:K - len(sel)]
    return tris[sel]


def triangle_constraints(n, tris):
    """The four triangle inequalities of every triple as polynomial constraints ``1 + s . (x_i x_j, x_i x_k, x_j x_k) >= 0``."""
    out = []
    for i, j, k in tris:
        for s in SIGNS:
            terms = {(0,) * n: 1.0}
            for sign, (a, b) in zip(s, ((i, j), (i, k), (j, k))):
                e = [0] * n
                e[a] = e[b] = 1
                terms[tuple(e)] = float(sign)
            out.append(Constraint(Poly(n, terms), "ineq"))
    return out


def pair_monomials(n, tris):
    """Exponent rows of the pair monomials of a set of triples (no duplicates)."""
    pairs = sorted({p for i, j, k in tris for p in ((i, j), (i, k), (j, k))})
    E = np.zeros((len(pairs), n), np.uint8)
    for r, (a, b) in enumerate(pairs):
        E[r, a] = E[r, b] = 1
    return E


def moment_matrix_Y(res):
    """``Y_ij = E[x_i x_j]`` (n x n, unit diagonal) read from the primal moment matrix of a ``solve_alm`` result."""
    X = res.extra["X"][0]
    B = res.relaxation.cl_basis[0]
    n = res.problem.n
    pos = {}
    for r, row in enumerate(B):
        if row.sum() == 1:
            pos[int(np.argmax(row))] = r
    idx = np.array([pos[i] for i in range(n)])
    Y = np.array(X)[np.ix_(idx, idx)] / X[0, 0] if False else np.array(X)[np.ix_(idx, idx)]
    z = int(np.where(B.sum(1) == 0)[0][0])
    Y = Y / float(X[z, z])
    np.fill_diagonal(Y, 1.0)
    return 0.5 * (Y + Y.T)


# --------------------------------------------------------------------------- the walk
@dataclass
class WalkResult:
    """Outcome of :func:`walk`. ``bounds[r]`` is the certified bound after round ``r`` (``r = 0`` is order 1)."""
    bounds: list = field(default_factory=list)
    times: list = field(default_factory=list)            # seconds per round (solve + selection)
    sizes: list = field(default_factory=list)            # active triples (cuts) or pair monomials (basis) after each round
    triples: np.ndarray = None                           # all triples chosen (in order)
    result: object = None                                # Result of the last round
    rule: str = ""
    mode: str = ""
    stopped: str = ""                                    # "rounds", "converged" (no violated triple) or "time"

    @property
    def bound(self):
        """Certified bound after the last round."""
        return self.bounds[-1]

    def __repr__(self):
        return (f"WalkResult({self.rule}, {self.mode}: {len(self.bounds) - 1} rounds, bound {self.bounds[0]:.6g} -> "
                f"{self.bounds[-1]:.6g}, {self.stopped})")


def walk(problem, rounds=10, per_round=None, rule="nms", mode="cuts", max_candidates=4000, drop_inactive=True,
         seed=0, time_limit=None, verbose=False, **solver_kw):
    """Walk from order 1 towards order 2 by adding triples chosen by ``rule``; every round gives a certified bound.

    Parameters
    ----------
    problem : Problem
        Fully spin (``boolean=True``) problem with an objective of degree at most 2 (MAX-CUT, Ising, QUBO in spins).
    rounds : int
        Number of rounds after the initial order-1 solve.
    per_round : int, optional
        Triples added per round ``K`` (default: the number of variables ``n``).
    rule : {"nms", "topk", "random"}
    mode : {"cuts", "basis"}
        Impose the triples as triangle inequalities (order-1 relaxation with scalar localizing blocks) or as pair
        monomials in the moment basis (partial order-2 lifting).
    drop_inactive : bool
        (``mode="cuts"``) discard triples whose four inequalities have zero multiplier after a round.
    time_limit : float, optional
        Total seconds; the walk stops after the round that exceeds it.
    **solver_kw
        Passed to :func:`moment_sos.solve_alm` (the primal moment matrix needed for the separation is read from its
        multipliers).

    Returns
    -------
    WalkResult
    """
    from .solve import solve_alm
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not problem.boolean:
        raise ValueError("walk needs a fully spin (boolean=True) problem")
    if problem.objective.degree() > 2:
        raise ValueError("walk needs an objective of degree at most 2")
    n = problem.n
    K = int(per_round or n)
    rng = np.random.default_rng(seed)
    user_cons = list(problem.constraints)
    base = lambda cons: Problem(problem.objective, cons, sense=problem.sense, boolean=True, name=problem.name)
    out = WalkResult(rule=rule, mode=mode)
    t_start = time.perf_counter()
    tris_all = np.zeros((0, 3), int)
    active = np.zeros((0, 3), int)
    t0 = time.perf_counter()
    res = solve_alm(base(user_cons), order=1, **solver_kw)
    out.bounds.append(res.bound); out.times.append(time.perf_counter() - t0); out.sizes.append(0)
    Y = moment_matrix_Y(res)
    out.stopped = "rounds"
    for r in range(rounds):
        t0 = time.perf_counter()
        new = select_triples(Y, K, rule, max_candidates, rng)
        if len(new) == 0:
            out.stopped = "converged"
            break
        tris_all = np.concatenate([tris_all, new]) if len(tris_all) else new
        active = np.concatenate([active, new]) if len(active) else new
        if mode == "cuts":
            cons = user_cons + triangle_constraints(n, active)
            res = solve_alm(base(cons), order=1, **solver_kw)
            if drop_inactive:
                nl = len(res.relaxation.cl_basis)
                d = np.array([float(res.extra["X"][nl + len(user_cons) + 4 * t + s][0, 0]) for t in range(len(active)) for s in range(4)]).reshape(-1, 4)
                keep = d.max(1) > 1e-7 * max(1.0, d.max())
                active = active[keep]
            size = len(active)
        else:
            extra = pair_monomials(n, active)
            res = solve_alm(base(user_cons), order=1, extra_basis=extra, **solver_kw)
            size = len(extra)
        Y = moment_matrix_Y(res)
        out.bounds.append(res.bound); out.times.append(time.perf_counter() - t0); out.sizes.append(size)
        if verbose:
            print(f"round {r + 1:2d}: bound {res.bound:.6f}  {mode} size {size}  {out.times[-1]:.1f}s")
        if time_limit and time.perf_counter() - t_start > time_limit:
            out.stopped = "time"
            break
    out.triples, out.result = tris_all, res
    return out
