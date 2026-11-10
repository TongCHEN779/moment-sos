"""Interior-point reference (cvxpy + MOSEK/Clarabel): solves the *same* dual problem inf_theta U(theta) exactly,
i.e.   min sigma + sum_b w_b t_b   s.t.  Q_b(theta) + t_b I >= 0,  t_b >= 0   (t_b free for constant-trace blocks).
By Theorem 2 of the paper this equals the moment relaxation with trace cuts."""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from .relaxation import Relaxation


def _sym_index(I, J, s, off=0):
    idx = np.zeros((s, s), dtype=int)
    idx[I, J] = np.arange(len(I)) + off
    idx[J, I] = np.arange(len(I)) + off
    return idx


def solve_dual_ipm(R: Relaxation, solver=None, verbose=False):
    """Solve the exact dual  min sigma + sum_b w_b t_b  s.t.  Q_b(theta) + t_b I >= 0  with an interior-point solver (cvxpy). Returns (value in the internal normalisation, seconds)."""
    import cvxpy as cp
    if solver is None:                      # MOSEK if available, else Clarabel (open-source interior point)
        import os
        solver = os.environ.get("MSOS_SOLVER") or ("MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL")
    ns, nu, npp = R.nslots, R.nu, R.np_
    sigma = cp.Variable()
    q = cp.Variable(ns)
    u = cp.Variable(max(nu, 1)); p = cp.Variable(max(npp, 1))
    cons = []
    S = sp.csr_matrix((R.slot_w, (R.slot_g, np.arange(ns))), shape=(R.nmom, ns))
    e0 = np.zeros(R.nmom); e0[R.empty_id] = 1.0
    rexpr = -R.fvec + e0 * sigma
    if nu:
        rexpr = rexpr - sp.csr_matrix((R.Rloc[2], (R.Rloc[0], R.Rloc[1])), shape=(R.nmom, nu)) @ u
    if npp:
        rexpr = rexpr - sp.csr_matrix((R.Req[2], (R.Req[0], R.Req[1])), shape=(R.nmom, npp)) @ p
    cons.append(S @ q == rexpr)
    obj = sigma
    for k, b in enumerate(R.blocks):
        t = cp.Variable(nonneg=b["plus"])
        obj = obj + b["weight"] * t if b["weight"] != 0 else obj
        s = b["size"]
        if b["kind"] == "moment":
            idx = _sym_index(R.cl_I[k], R.cl_J[k], s, R.cl_off[k])
            Q = q[idx.reshape(-1)].reshape((s, s), order="C")
        else:
            info = R.ineq_info[k - len(R.cl_basis)]
            off = R.u_off[k - len(R.cl_basis)]
            idx = _sym_index(info["a"], info["b"], s, off)
            Q = u[idx.reshape(-1)].reshape((s, s), order="C")
        cons.append(Q + t * np.eye(s) >> 0)
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose)
    return float(prob.value), time.perf_counter() - t0
