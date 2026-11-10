"""Polynomial-optimization encodings of ReLU-network questions.

* ``relu_verification``  -- max of a linear function of the logits over an l_inf ball around x0.
  Exact ReLU constraints z>=0, z>=pre, z*(z-pre)=0 (+ triangle) for *unstable* neurons only;
  stable neurons are substituted away.  Order 1 of the relaxation is the SDP of Raghunathan et al.
  (a Shor relaxation); orders >= 2 are strictly stronger.
* ``lipschitz_problem`` -- squared l2 Lipschitz bound  ``max over |u|<=1, s in {0,1}^w of |W1 diag(s) W0 u|^2``
  of a one-hidden-layer ReLU network (the activation-pattern bound that LipSDP relaxes).
"""
from __future__ import annotations
import numpy as np
from .poly import Poly, variables
from .problem import Problem


# --------------------------------------------------------------------------- interval bounds
def ibp(Ws, bs, lo, hi):
    """Interval bound propagation: list of pre-activation (lo, hi) per layer (last layer = logits)."""
    out = []
    a_lo, a_hi = np.asarray(lo, float), np.asarray(hi, float)
    for l, (W, b) in enumerate(zip(Ws, bs)):
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        pl = Wp @ a_lo + Wn @ a_hi + b
        ph = Wp @ a_hi + Wn @ a_lo + b
        out.append((pl, ph))
        a_lo, a_hi = np.maximum(pl, 0), np.maximum(ph, 0)
    return out


def forward(Ws, bs, x):
    """Forward pass of a ReLU network given weight matrices `Ws` and biases `bs` (last layer linear)."""
    a = np.asarray(x, float)
    for W, b in zip(Ws[:-1], bs[:-1]):
        a = np.maximum(W @ a + b, 0)
    return Ws[-1] @ a + bs[-1]


# --------------------------------------------------------------------------- verification POP
def relu_verification(Ws, bs, x0, eps, c, input_lo=None, input_hi=None, triangle=True, name="verify"):
    """Return (problem, info). `c` is the vector whose product with the logits is maximised."""
    x0 = np.asarray(x0, float)
    lo0 = x0 - eps if input_lo is None else np.maximum(x0 - eps, input_lo)
    hi0 = x0 + eps if input_hi is None else np.minimum(x0 + eps, input_hi)
    bounds = ibp(Ws, bs, lo0, hi0)
    d = len(x0)
    unstable = []                                              # (layer, neuron)
    for l in range(len(Ws) - 1):
        pl, ph = bounds[l]
        unstable += [(l, j) for j in range(len(pl)) if pl[j] < 0 < ph[j]]
    n = d + len(unstable)
    var = variables(n)
    lo = np.concatenate([lo0, np.zeros(len(unstable))])
    hi = np.concatenate([hi0, [max(bounds[l][1][j], 1e-9) for l, j in unstable]])
    cons = []
    a = list(var[:d])
    nxt = d
    for l in range(len(Ws) - 1):
        pl, ph = bounds[l]
        W, b = Ws[l], bs[l]
        new = []
        for j in range(W.shape[0]):
            pre = Poly.constant(n, b[j])
            for i in range(W.shape[1]):
                if W[j, i] != 0:
                    pre = pre + W[j, i] * a[i]
            if ph[j] <= 0:
                new.append(Poly.constant(n, 0.0))
            elif pl[j] >= 0:
                new.append(pre)
            else:
                z = var[nxt]; nxt += 1
                cons += [z >= 0, z >= pre, z * (z - pre) == 0]
                if triangle:
                    cons.append(ph[j] * (pre - pl[j]) - (ph[j] - pl[j]) * z >= 0)
                new.append(z)
        a = new
    out = Poly.constant(n, float(np.dot(c, bs[-1])))
    cW = np.asarray(c, float) @ Ws[-1]
    for i, ai in enumerate(a):
        if cW[i] != 0:
            out = out + cW[i] * ai
    P = Problem(out, cons, sense="max", box=(lo, hi), name=name)
    info = dict(n=n, d=d, n_unstable=len(unstable), n_hidden=sum(len(w) for w in bs[:-1]),
                ibp_upper=float(_ibp_objective(Ws, bs, bounds, c)), bounds=bounds)
    return P, info


def _ibp_objective(Ws, bs, bounds, c):
    pl, ph = bounds[-1]
    c = np.asarray(c, float)
    return float(np.sum(np.where(c > 0, c * ph, c * pl)))


def verification_exact_milp(Ws, bs, x0, eps, c, input_lo=None, input_hi=None, time_limit=120):
    """Exact max of c^T logits over the l_inf ball by big-M MILP (scipy/HiGHS). Returns (value, status)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    x0 = np.asarray(x0, float)
    lo0 = x0 - eps if input_lo is None else np.maximum(x0 - eps, input_lo)
    hi0 = x0 + eps if input_hi is None else np.minimum(x0 + eps, input_hi)
    bounds = ibp(Ws, bs, lo0, hi0)
    d = len(x0)
    # variable layout: x, then per hidden layer: z (post), y (binary, only unstable)
    cols = [("x", d)]
    idx = {"x": slice(0, d)}
    pos = d
    layers = []
    for l in range(len(Ws) - 1):
        w = Ws[l].shape[0]
        pl, ph = bounds[l]
        un = [j for j in range(w) if pl[j] < 0 < ph[j]]
        zs = slice(pos, pos + w); pos += w
        ys = {j: pos + k for k, j in enumerate(un)}; pos += len(un)
        layers.append((zs, ys))
    nvar = pos
    A, lb_, ub_ = [], [], []

    def row(coefs, lo, hi):
        r = np.zeros(nvar)
        for k, v in coefs:
            r[k] += v
        A.append(r); lb_.append(lo); ub_.append(hi)
    prev = list(range(d))
    for l in range(len(Ws) - 1):
        W, b = Ws[l], bs[l]
        pl, ph = bounds[l]
        zs, ys = layers[l]
        for j in range(W.shape[0]):
            z = zs.start + j
            pre = [(prev[i], W[j, i]) for i in range(W.shape[1]) if W[j, i] != 0]
            if ph[j] <= 0:
                row([(z, 1)], 0, 0)
            elif pl[j] >= 0:
                row([(z, 1)] + [(k, -v) for k, v in pre], b[j], b[j])
            else:
                y = ys[j]
                row([(z, 1)] + [(k, -v) for k, v in pre], b[j], np.inf)                     # z >= pre
                row([(z, 1)] + [(k, -v) for k, v in pre] + [(y, -pl[j])], -np.inf, b[j] - pl[j])  # z <= pre - lb(1-y)
                row([(z, 1), (y, -ph[j])], -np.inf, 0)                                       # z <= ub y
        prev = list(range(zs.start, zs.start + W.shape[0]))
    obj = np.zeros(nvar)
    cW = np.asarray(c, float) @ Ws[-1]
    for i, k in enumerate(prev):
        obj[k] = -cW[i]
    lo = np.full(nvar, 0.0); hi = np.full(nvar, np.inf)
    lo[:d], hi[:d] = lo0, hi0
    integ = np.zeros(nvar)
    for zs, ys in layers:
        for k in ys.values():
            integ[k] = 1; hi[k] = 1
        for j in range(zs.stop - zs.start):
            hi[zs.start + j] = max(hi[zs.start + j], 1e6)
    res = milp(obj, constraints=LinearConstraint(np.array(A), lb_, ub_), integrality=integ, bounds=Bounds(lo, hi),
               options=dict(time_limit=time_limit))
    val = -res.fun + float(np.dot(c, bs[-1])) if res.x is not None else None
    return val, res.status, getattr(res, "mip_dual_bound", None)


def verification_pgd(Ws, bs, x0, eps, c, input_lo=None, input_hi=None, steps=300, restarts=20, seed=0):
    """Projected-gradient lower bound on the same maximisation (feasible points)."""
    import torch
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    lo = x0 - eps if input_lo is None else np.maximum(x0 - eps, input_lo)
    hi = x0 + eps if input_hi is None else np.minimum(x0 + eps, input_hi)
    Wt = [torch.as_tensor(W) for W in Ws]; bt = [torch.as_tensor(b) for b in bs]
    ct = torch.as_tensor(np.asarray(c, float))
    best = -np.inf
    for r in range(restarts):
        x = torch.as_tensor(rng.uniform(lo, hi)).requires_grad_(True)
        for s in range(steps):
            a = x
            for W, b in zip(Wt[:-1], bt[:-1]):
                a = torch.relu(W @ a + b)
            out = ct @ (Wt[-1] @ a + bt[-1])
            out.backward()
            with torch.no_grad():
                x += (eps * 0.05 * (1 - s / steps) + 1e-4) * x.grad.sign()
                x.copy_(torch.clamp(x, torch.as_tensor(lo), torch.as_tensor(hi)))
            x.grad = None
        best = max(best, float(ct.numpy() @ forward(Ws, bs, x.detach().numpy())))
    return best


# --------------------------------------------------------------------------- Lipschitz POP
def lipschitz_problem(W0, W1, name="lipschitz", mixed=True):
    """Squared l2 Lipschitz bound as a POP: ``max |W1 diag(s) W0 u|^2`` s.t. ``|u|<=1``, ``s in {0,1}^w`` (``s = (1+t)/2``, ``t`` spin). Variables ``[u, t]``."""
    W0, W1 = np.asarray(W0, float), np.asarray(W1, float)
    w, d = W0.shape
    m = W1.shape[0]
    n = d + w
    v = variables(n)
    u, t = v[:d], v[d:]
    wu = [sum((W0[j, i] * u[i] for i in range(d) if W0[j, i] != 0), Poly.constant(n, 0.0)) for j in range(w)]
    f = Poly.constant(n, 0.0)
    for k in range(m):
        vk = Poly.constant(n, 0.0)
        for j in range(w):
            if W1[k, j] != 0:
                vk = vk + W1[k, j] * 0.5 * (1 + t[j]) * wu[j]
        f = f + vk * vk
    cons = [sum((ui * ui for ui in u), Poly.constant(n, 0.0)) <= 1]
    if mixed:                                    # t_j^2 = 1 used as a rewriting rule
        mask = np.array([False] * d + [True] * w)
        return Problem(f, cons, sense="max", box=(-1, 1), boolean=mask, name=name)
    cons += [tj * tj == 1 for tj in t]           # same problem with explicit equality constraints
    return Problem(f, cons, sense="max", box=(-1, 1), name=name + "-eq")


def lipschitz_exact_pattern(W0, W1):
    """max over activation patterns of ||W1 diag(s) W0||_2^2 by enumeration (w <= ~20)."""
    w = W0.shape[0]
    best = 0.0
    for mask in range(2 ** w):
        s = np.array([(mask >> j) & 1 for j in range(w)], float)
        best = max(best, np.linalg.norm(W1 @ (s[:, None] * W0), 2) ** 2)
    return best


def lipsdp_neuron(W0, W1):
    """LipSDP-Neuron bound (Fazlyab et al. 2019) on the squared Lipschitz constant, via cvxpy/MOSEK."""
    import cvxpy as cp
    w, d = W0.shape
    rho = cp.Variable(); T = cp.Variable(w, nonneg=True)
    Tm = cp.diag(T)
    M = cp.bmat([[-rho * np.eye(d), W0.T @ Tm], [Tm @ W0, W1.T @ W1 - 2 * Tm]])
    prob = cp.Problem(cp.Minimize(rho), [(M + M.T) / 2 << 0])
    prob.solve(solver=cp.MOSEK)
    return float(rho.value)
