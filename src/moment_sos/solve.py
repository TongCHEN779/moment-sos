"""First-order solver: entropic smoothing + L-BFGS with eta-continuation (all in PyTorch)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
import torch
from .problem import Problem
from .relaxation import Relaxation
from .oracle import Oracle, _safe_eigh

def min_order(problem):
    """Smallest relaxation order that can represent every polynomial of the problem."""
    degs = [problem.f.degree()] + [p.degree() for p in problem.ineq + problem.eq]
    return max(1, max((d + 1) // 2 for d in degs))


DEFAULT_ETAS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
PRECISE_ETAS = DEFAULT_ETAS + (3e-6, 1e-6, 3e-7, 1e-7)    # four more stages: the smoothing bias is ~eta, so this lowers the error ~100x
SHORT_ETAS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)      # coarse start: ~20x faster certification of integer optima (Boolean MAX-CUT)


@dataclass
class Result:
    problem: Problem
    relaxation: Relaxation
    oracle: Oracle
    theta: torch.Tensor
    U: float                               # certified bound in the internal (maximisation) sense
    history: list = field(default_factory=list)    # (time, n_eval, best U)
    time: float = 0.0
    n_eval: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def bound(self):
        """Certified bound on the optimum of the ORIGINAL problem: an upper bound if sense='max',
        a lower bound if sense='min'."""
        u = self.problem.f_off + self.problem.f_scale * self.U        # undo the internal normalisation
        return u if self.problem.sense == "max" else -u

    @property
    def bound_kind(self):
        return "upper" if self.problem.sense == "max" else "lower"

    def round(self, **kw):
        """Heuristic primal recovery: best feasible point found from the dual iterate (see `moment_sos.rounding`)."""
        from .rounding import round_solution
        return round_solution(self, **kw)

    def rigorous(self):
        """Bound that holds in exact arithmetic for the stored normalised problem (see `moment_sos.verify`)."""
        from .verify import rigorous_bound
        return rigorous_bound(self)

    def summary(self):
        """One-line description of the bound and of the relaxation."""
        R = self.relaxation
        return (f"{self.bound_kind} bound = {self.bound:.8g}  (order {R.r}, {len(R.blocks)} PSD blocks, "
                f"largest {max(R.sizes())}, {R.dim:,} multipliers; {self.n_eval} evaluations, {self.time:.1f}s)")

    __repr__ = summary


def solve(problem: Problem, order: int = 2, cliques="dense", etas=DEFAULT_ETAS, iters_per_stage: int = 500,
          history_size: int = 30, time_limit: float | None = None, device: str = "cpu",
          theta0: torch.Tensor | None = None, verbose: bool = False, warm_start: bool = True,
          warm_iters: int | None = None, stop_at: float | None = None, tight_weights: bool = False,
          term_sparsity=None, extra_basis=None) -> Result:
    """Certified bound through the order-``order`` moment-SOS relaxation (entropic smoothing + L-BFGS).

    Parameters
    ----------
    problem : Problem
    order : int
        Relaxation order; must satisfy ``2 * order >= degree`` of every polynomial (see :func:`min_order`).
    cliques : "dense", "auto" or list of index lists
        One clique, the correlative-sparsity cliques, or explicit cliques.
    etas : sequence of float
        Smoothing parameters of the continuation (``SHORT_ETAS`` starts coarser).
    iters_per_stage : int
        L-BFGS iterations per smoothing value.
    time_limit : float, optional
        Seconds; the best certified bound so far is returned.
    warm_start : bool
        Solve order 1, 2, ... first and embed; the start is already a valid certificate equal to the lower-order bound.
    stop_at : float, optional
        Stop as soon as the certified bound is at least this good (<= ``stop_at`` for a max problem, >= for a min
        problem), e.g. ``best_cut + 1 - 1e-9`` to certify an integer optimum.
    term_sparsity : int or "max", optional
        Split each clique's moment block into overlapping sub-blocks by term-sparsity closure (``steps`` iterations, or
        until stable). The bound stays valid for any block structure but the relaxation is weaker than the dense one;
        warm start is disabled. See the sparsity guide for when it helps.
    device : str
        ``"cpu"``; ``"cuda"`` is accepted but untested.

    Returns
    -------
    Result
        ``Result.bound`` is a valid bound (up to floating-point eigensolver error) at the best iterate seen; every
        iterate is a valid certificate, so a ``time_limit`` simply stops early.
    """
    if term_sparsity:
        warm_start = False                      # block structures of different orders are not nested: no embedding
    if extra_basis is not None and len(extra_basis):
        warm_start = False                      # the caller supplies theta0 (see moment_sos.walk)
    R = Relaxation(problem, order, cliques, tight_weights, term_sparsity, extra_basis)
    O = Oracle(R, device=device)
    if warm_start and theta0 is None and order > min_order(problem):
        # solve order-1 first (cheap), then embed: the starting certificate is already the order-1 bound
        lo = solve(problem, order - 1, cliques, etas=etas, iters_per_stage=warm_iters or iters_per_stage,
                   history_size=history_size, time_limit=time_limit, device=device, verbose=verbose,
                   warm_start=True, tight_weights=tight_weights)
        theta0 = lift_theta(lo, R, O)
        if verbose:
            print(f"  warm start from order {order-1}: U = {lo.U:.8g}; lifted U = {O.certificate(theta0):.8g}")
    th = O.theta0() if theta0 is None else theta0.clone().to(O.device)
    theta = torch.nn.Parameter(th)
    best = dict(U=float("inf"), theta=th.clone())
    stop_int = None
    if stop_at is not None:                                 # convert to the internal normalised maximisation scale
        v = stop_at if problem.sense == "max" else -stop_at
        stop_int = (v - problem.f_off) / problem.f_scale
    hist, n_eval, t0 = [], [0], time.perf_counter()

    for eta in etas:
        if time_limit and time.perf_counter() - t0 > time_limit:
            break
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters_per_stage, max_eval=int(1.25 * iters_per_stage),
                                history_size=history_size, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Us, Uh = O.evaluate(theta, eta)
            if not torch.isfinite(Us):                       # let the line search back off
                Us = torch.zeros((), dtype=Us.dtype, device=Us.device) + 1e30 + 0 * theta.sum()
            Us.backward()
            n_eval[0] += 1
            if Uh < best["U"]:
                best["U"], best["theta"] = Uh, theta.detach().clone()
            hist.append((time.perf_counter() - t0, n_eval[0], best["U"]))
            if time_limit and time.perf_counter() - t0 > time_limit:
                raise TimeoutError
            if stop_int is not None and best["U"] <= stop_int:
                raise TimeoutError                         # certified bound reached the requested level
            return Us
        try:
            opt.step(closure)
        except TimeoutError:
            break
        if verbose:
            print(f"  eta={eta:.0e}: best certified U = {best['U']:.8g}   evals = {n_eval[0]}   t = {time.perf_counter()-t0:.1f}s")
    return Result(problem, R, O, best["theta"], best["U"], hist, time.perf_counter() - t0, n_eval[0])


# --------------------------------------------------------------------------- warm start across orders
def _row_index(B_small, B_big):
    """index in B_big of every row of B_small (rows are exponent vectors)."""
    key = lambda B: [bytes(r) for r in B]
    pos = {k: i for i, k in enumerate(key(B_big))}
    return np.array([pos[k] for k in key(B_small)])


def lift_theta(res: "Result", relax_high: Relaxation, oracle_high: Oracle) -> torch.Tensor:
    """Embed a dual point of order r into order r+1 (zero-padded Gram matrices).  The embedded point is a dual
    point of the larger relaxation; its certified bound equals the old one when no diagonal shift was active
    and is slightly larger otherwise (the shift weight grows with the block size)."""
    Rl, Rh = res.relaxation, relax_high
    with torch.no_grad():
        Qs = [Q.cpu().numpy() for Q in res.oracle.gram_blocks(res.theta)]
    th = np.zeros(Rh.dim)
    th[0] = float(res.theta[0])
    nl = len(Rl.cl_basis)
    if Rl.prob.boolean:
        # constant-trace blocks: absorb the diagonal shift (Q <- Q - lmin I, sigma <- sigma - s*lmin), so the
        # embedded point needs no shift and keeps the bound U despite the larger block weight.
        for k in range(nl):
            lam = np.linalg.eigvalsh(Qs[k])[0]
            Qs[k] = Qs[k] - lam * np.eye(len(Qs[k]))
            th[0] -= len(Qs[k]) * lam
    # localising Gram entries and equality multipliers carry over entrywise
    for k, (il, ih) in enumerate(zip(Rl.ineq_info, Rh.ineq_info)):
        m = _row_index(il["Z"], ih["Z"])
        Q = Qs[nl + k]
        big = np.zeros((len(ih["Z"]),) * 2)
        big[np.ix_(m, m)] = Q
        th[1 + Rh.nslots + Rh.u_off[k]: 1 + Rh.nslots + Rh.u_off[k + 1]] = big[ih["a"], ih["b"]]
    pl = res.theta[1 + Rl.nslots + Rl.nu:].cpu().numpy()
    for k, (el, eh) in enumerate(zip(Rl.eq_info, Rh.eq_info)):
        m = _row_index(el["M"], eh["M"])
        seg = np.zeros(eh["nm"]); seg[m] = pl[Rl.p_off[k]:Rl.p_off[k + 1]]
        th[1 + Rh.nslots + Rh.nu + Rh.p_off[k]: 1 + Rh.nslots + Rh.nu + Rh.p_off[k + 1]] = seg
    # moment Gram blocks: Theta~ := Q_embedded - spread (zero-sum automatically)
    Oh = oracle_high
    thj = torch.as_tensor(th)
    with torch.no_grad():
        # r at moment level for the new multipliers (Theta=0)
        Qsp = Oh.gram_blocks(thj)
    q_target = np.zeros(Rh.nslots)
    for k in range(nl):
        m = _row_index(Rl.cl_basis[k], Rh.cl_basis[k])
        big = np.zeros((len(Rh.cl_basis[k]),) * 2)
        big[np.ix_(m, m)] = Qs[k]
        q_target[Rh.cl_off[k]:Rh.cl_off[k + 1]] = big[Rh.cl_I[k], Rh.cl_J[k]]
    q_spread = np.concatenate([Qsp[k].numpy()[Rh.cl_I[k], Rh.cl_J[k]] for k in range(nl)])
    th[1:1 + Rh.nslots] = q_target - q_spread
    return torch.as_tensor(th, dtype=oracle_high.dtype, device=oracle_high.device)


# --------------------------------------------------------------------------- augmented Lagrangian (dual ALM)
class _AlmPenalty(torch.autograd.Function):
    """Batched  (|Pi_+(X - rho G)|_F^2 - |X|_F^2) / (2 rho)  with gradient  -Pi_+(X - rho G)  w.r.t. G."""

    @staticmethod
    def forward(ctx, G, X, rho):
        M = X - rho * G
        w, V = _safe_eigh(M)
        wp = torch.clamp(w, min=0.0)
        Pi = (V * wp[:, None, :]) @ V.transpose(1, 2)
        val = ((wp ** 2).sum(dim=1) - (X ** 2).sum(dim=(1, 2))) / (2 * rho)
        ctx.save_for_backward(Pi)
        return val

    @staticmethod
    def backward(ctx, g):
        (Pi,) = ctx.saved_tensors
        return -g[:, None, None] * Pi, None, None


def _embed(M, idx_map, size):
    big = np.zeros((size, size))
    big[np.ix_(idx_map, idx_map)] = M
    return big


def lift_alm_state(lo: "Result", Rh: Relaxation, Oh: Oracle, eps_new: float = 1e-2):
    """Embed the full ALM state (theta, t, multipliers X) of order r into order r+1.  The embedded point is the
    exact optimum of order r viewed inside order r+1 (zero-padded Gram / moment matrices)."""
    Rl = lo.relaxation
    theta = lift_theta(lo, Rh, Oh)
    nb = len(Rh.blocks)
    t = np.zeros(nb)
    nl = len(Rl.cl_basis)
    Xh = {}
    for b in range(nb):
        if b < nl:
            m = _row_index(Rl.cl_basis[b], Rh.cl_basis[b]); size = len(Rh.cl_basis[b])
        else:
            il, ih = Rl.ineq_info[b - nl], Rh.ineq_info[b - nl]
            m = _row_index(il["Z"], ih["Z"]); size = len(ih["Z"])
        Xl = lo.extra["X"].get(b)
        if Xl is not None:
            Xe = _embed(Xl, m, size)
            new = np.ones(size, bool); new[m] = False
            scale = max(float(np.trace(Xl)) / max(len(Xl), 1), 1e-3)
            Xe[new, new] += eps_new * scale                         # break the zero-gradient boundary point
            Xh[b] = Xe
        if Rh.blocks[b]["plus"]:
            t[b] = lo.extra["t"][b]
    return theta, t, Xh



def _newton_inner(O, groups, Xg, wt_t, lbm, x0, rho, max_newton=30, cg_max=int(__import__('os').environ.get('POLYOPT_CG', 60)), gtol=1e-9, deadline=None):
    """Semismooth Newton-CG for the smooth inner problem of the dual ALM,
        phi(x) = sigma + w.t + sum_b (|Pi_+(X_b - rho G_b(x))|^2 - |X_b|^2) / (2 rho),   x = (theta, t),  t_b >= 0 on `lbm`.
    Hessian-vector products use the spectral operator of Pi_+ at the current eigendecomposition (no extra eigh).
    Returns (x, number of eigendecompositions, projected-gradient norm)."""
    dim = O.dim
    nb = sum(len(idx) for _, idx, _ in groups)
    eyes = {}
    lbm_t = torch.as_tensor(lbm, device=O.device)

    def G_of(xv):
        th, t = xv[:dim], xv[dim:]
        Qs = O.gram_blocks(th)
        out = []
        for _, idx, _ in groups:
            s = Qs[idx[0]].shape[0]
            if s not in eyes: eyes[s] = torch.eye(s, dtype=O.dtype, device=O.device)
            out.append(torch.stack([Qs[i] + t[i] * eyes[s] for i in idx]))
        return out

    def phi_only(xv):
        with torch.no_grad():
            Gs = G_of(xv); val = xv[0] + (wt_t * xv[dim:]).sum()
            for G, X in zip(Gs, Xg):
                w = _safe_eigh(X - rho * G)[0]
                val = val + ((torch.clamp(w, min=0.0) ** 2).sum() - (X ** 2).sum()) / (2 * rho)
            return float(val)

    def state(xnp):
        xv = torch.as_tensor(xnp, dtype=O.dtype, device=O.device).clone().requires_grad_(True)
        Gs = G_of(xv); cache, val, cot = [], xv[0].detach() + (wt_t * xv[dim:]).sum().detach(), []
        with torch.no_grad():
            for G, X in zip(Gs, Xg):
                w, V = _safe_eigh(X - rho * G); wp = torch.clamp(w, min=0.0)
                Pi = (V * wp[:, None, :]) @ V.transpose(1, 2)
                val = val + ((wp ** 2).sum() - (X ** 2).sum()) / (2 * rho)
                d = w[:, :, None] - w[:, None, :]; a = wp[:, :, None] - wp[:, None, :]
                Om = torch.where(d.abs() > 1e-12 * (1 + w.abs().max()), a / torch.where(d == 0, torch.ones_like(d), d),
                                 (w[:, :, None] > 0).to(w.dtype) * torch.ones_like(d))
                cache.append((V, Om)); cot.append(-Pi)
        lin = xv[0] + (wt_t * xv[dim:]).sum()
        g = torch.autograd.grad(lin, xv, retain_graph=True)[0] + torch.autograd.grad(Gs, xv, grad_outputs=cot, retain_graph=True)[0]
        return xv, Gs, cache, float(val), g.detach().cpu().numpy()

    x, n_eig = x0.copy(), 0
    for it in range(max_newton):
        xv, Gs, cache, f, g = state(x); n_eig += 1
        free = ~(lbm & (x <= 0.0) & (g > 0.0))
        pg = np.where(free, g, 0.0); gn = float(np.linalg.norm(pg))
        if gn <= gtol * (1.0 + abs(x[0])) or (deadline and time.perf_counter() > deadline):
            break

        # linear map v -> dG (affine part removed)
        with torch.no_grad():
            G0 = G_of(torch.zeros_like(xv))

        def Hmul(vnp):
            v = torch.as_tensor(np.where(free, vnp, 0.0), dtype=O.dtype, device=O.device)
            with torch.no_grad():
                dG = [a - b for a, b in zip(G_of(v), G0)]
                cot = []
                for d, (V, Om) in zip(dG, cache):
                    cot.append(rho * (V @ (Om * (V.transpose(1, 2) @ d @ V)) @ V.transpose(1, 2)))
            h = torch.autograd.grad(Gs, xv, grad_outputs=cot, retain_graph=True)[0].cpu().numpy()
            return np.where(free, h, 0.0) + 1e-6 * rho * np.where(free, vnp, 0.0)

        # truncated CG on  H d = -g  (free variables)
        d = np.zeros_like(x); r = -pg.copy(); pdir = r.copy(); rs = float(r @ r)
        tol_cg = min(0.1, gn ** 0.5) * gn
        for _ in range(cg_max):
            Hp = Hmul(pdir); pHp = float(pdir @ Hp)
            if pHp <= 1e-10 * rho * float(pdir @ pdir): break
            al = rs / pHp; d = d + al * pdir; r = r - al * Hp; rs_new = float(r @ r)
            if rs_new ** 0.5 <= tol_cg: break
            pdir = r + (rs_new / rs) * pdir; rs = rs_new
        if not np.any(d): d = -pg / rho
        cap = 10.0 * (1.0 + float(np.linalg.norm(x)))
        if np.linalg.norm(d) > cap: d = d * (cap / np.linalg.norm(d))
        # projected Armijo line search
        step, f0, ok = 1.0, f, False
        for _ in range(25):
            xn = x + step * d; xn[lbm] = np.maximum(xn[lbm], 0.0)
            fn = phi_only(torch.as_tensor(xn, dtype=O.dtype, device=O.device)); n_eig += 1
            if fn <= f0 + 1e-4 * float(g @ (xn - x)):
                ok = True; break
            step *= 0.5
        if not ok: break
        x = xn
    return x, n_eig, gn


def solve_alm(problem: Problem, order: int = 2, cliques="dense", rho0: float = 1000.0, outer: int = 300, inner: int = 600,
              time_limit: float | None = None, device: str = "cpu", theta0=None, warm_start: bool = False,
              verbose: bool = False, tol: float = 1e-7, adapt_rho: bool = False, shift: str = "uniform",
              tight_weights: bool = False, inner_solver: str = "lbfgs", term_sparsity=None, extra_basis=None) -> Result:
    """Dual augmented-Lagrangian (boundary-point type) solver for  min sigma + sum_b w_b t_b  s.t.  Q_b(theta) + t_b I >= 0.

    * inner problems: smooth (C^1), bound-constrained, solved by L-BFGS-B (inner_solver="lbfgs") or by a semismooth
      Newton-CG method with exact spectral Hessian-vector products (inner_solver="newton");
    * penalty parameter doubled whenever the primal infeasibility does not drop by 30% (residual balancing, `adapt_rho`,
      is available but misbehaves with inexact inner solves);
    * warm start: order r is started from the full ALM state (theta, t, X) of order r-1, embedded by zero padding;
    * every outer iterate gives a certified bound U(theta); the best one is returned.

    shift : "uniform" (default) -- one scalar shift t_b per block, G_b = Q_b + t_b I, value inf U = rho_r^tr;
            "diag"              -- one vector d_b in R^{s_b} per block (Proposition prop:diag), G_b = Q_b + diag(d_b),
                                   objective sigma + sum_b (w_b/s_b) 1^T d_b with w_b/s_b = 1 (moment) or B_i (loc),
                                   d_b >= 0 for 'plus' blocks and free for constant-trace (Boolean) blocks;
                                   value inf U^d = rho_r^d (diagonal cuts).  The certified bound at an iterate is
                                   U^d(theta, d) = sigma + sum_b (w_b/s_b) 1^T d_b + sum_b w_b Phi_b(Q_b + diag d_b)
                                   (the uniform-shift certificate applied to the shifted blocks; valid by
                                   Theorem thm:certificate), and Result.U is the best such bound over the outer
                                   iterates.  Result.extra["d"] holds the final shift vector (concatenated over
                                   blocks, offsets Result.extra["d_off"]) and Result.extra["U_uniform"] the best
                                   uniform certificate U(theta) seen along the way."""
    from scipy.optimize import minimize
    if shift not in ("uniform", "diag"):
        raise ValueError("shift must be 'uniform' or 'diag'")
    if term_sparsity and shift == "diag":
        raise ValueError("term_sparsity is not supported with shift='diag'")
    if term_sparsity or (extra_basis is not None and len(extra_basis)):
        warm_start = False                      # block structures are not nested across orders: no embedding
    if shift == "diag":
        return _solve_alm_diag(problem, order, cliques, rho0, outer, inner, time_limit, device, theta0, warm_start,
                               verbose, tol, adapt_rho, tight_weights)
    R = Relaxation(problem, order, cliques, tight_weights, term_sparsity, extra_basis)
    O = Oracle(R, device=device)
    groups, nb, dim = O.groups, len(R.blocks), O.dim
    plus = np.array([b["plus"] for b in R.blocks]); wts = np.array(O.weights); active = wts != 0.0
    t_init, X_init = np.zeros(nb), {}
    t_start = time.perf_counter()
    if warm_start and theta0 is None and order > min_order(problem):
        lo = solve_alm(problem, order - 1, cliques, rho0, outer, inner, time_limit, device, None, True, verbose, tol, adapt_rho, tight_weights=tight_weights)
        theta0, t_init, X_init = lift_alm_state(lo, R, O)
        if verbose:
            print(f"  warm start from order {order-1}: U = {lo.U:.8g}; lifted U = {O.certificate(theta0):.8g}")
    th0 = (O.theta0() if theta0 is None else theta0.clone().to(O.device)).detach()
    x0 = np.concatenate([th0.cpu().numpy(), t_init])
    lo_b = np.concatenate([np.full(dim, -np.inf), np.where(plus & active, 0.0, -np.inf)])
    hi_b = np.full(dim + nb, np.inf)
    bounds = list(zip(lo_b, hi_b))
    Xg = []
    for plus_g, idx, wt in groups:
        s = R.blocks[idx[0]]["size"]
        X = torch.zeros(len(idx), s, s, dtype=O.dtype, device=O.device)
        for pos, b in enumerate(idx):
            if b in X_init:
                X[pos] = torch.as_tensor(X_init[b], dtype=O.dtype, device=O.device)
        Xg.append(X)
    wt_t = torch.as_tensor(wts * active, dtype=O.dtype, device=O.device)
    eyes = {}

    def build(xnp, requires_grad=True):
        x = torch.as_tensor(xnp, dtype=O.dtype, device=O.device).clone().requires_grad_(requires_grad)
        return x, x[:dim], x[dim:], O.gram_blocks(x[:dim])

    def stackG(Qs, t, idx):
        s = Qs[idx[0]].shape[0]
        if s not in eyes:
            eyes[s] = torch.eye(s, dtype=O.dtype, device=O.device)
        return torch.stack([Qs[i] + t[i] * eyes[s] for i in idx])

    def fun(xnp, rho):
        x, th, t, Qs = build(xnp)
        val = th[0] + (wt_t * t).sum()
        for (plus_g, idx, wt), X in zip(groups, Xg):
            val = val + _AlmPenalty.apply(stackG(Qs, t, idx), X, rho).sum()
        val.backward()
        return float(val.detach()), x.grad.cpu().numpy().astype(np.float64)

    def update(xnp, rho):
        with torch.no_grad():
            x, th, t, Qs = build(xnp, False)
            pinf = 0.0
            for g, ((plus_g, idx, wt), X) in enumerate(zip(groups, Xg)):
                G = stackG(Qs, t, idx)
                w, V = _safe_eigh(X - rho * G)
                Xg[g] = (V * torch.clamp(w, min=0.0)[:, None, :]) @ V.transpose(1, 2)
                pinf += float((torch.clamp(-_safe_eigh(G)[0][:, 0], min=0.0) ** 2).sum())
            return pinf ** 0.5

    best = dict(U=float("inf"), theta=th0.clone())
    best["U"] = O.certificate(th0)
    hist, n_eval, rho, prev, x = [], 0, rho0, np.inf, x0
    for k in range(outer):
        if inner_solver == "newton":
            dl = (t_start + time_limit) if time_limit else None
            x, ne, _ = _newton_inner(O, groups, Xg, wt_t, np.concatenate([np.zeros(dim, bool), plus & active]), x, rho,
                                     max_newton=inner, deadline=dl)
            n_eval += ne
            g = fun(x, rho)[1]
        else:
            res = minimize(fun, x, args=(rho,), jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=inner, maxcor=30, ftol=1e-15, gtol=1e-10, maxfun=int(1.3 * inner)))
            x = res.x
            n_eval += res.nfev
            g = res.jac.copy()
        tpart = x[dim:]
        g[dim:] = np.where((tpart > 0) | ~(plus & active), g[dim:], np.minimum(g[dim:], 0.0))
        dinf = float(np.linalg.norm(g)) / (1.0 + abs(x[0]))
        pinf = update(x, rho)
        th = torch.as_tensor(x[:dim], dtype=O.dtype, device=O.device)
        U = O.certificate(th)
        if U < best["U"]:
            best["U"], best["theta"] = U, th.clone()
        hist.append((time.perf_counter() - t_start, n_eval, best["U"]))
        if verbose and (k % 5 == 0 or k == outer - 1):
            print(f"  outer {k:3d}  rho={rho:.3g}  pinf={pinf:.2e}  dinf={dinf:.2e}  certified U={best['U']:.8g}  evals={n_eval}  t={time.perf_counter()-t_start:.0f}s")
        if adapt_rho:
            ratio = pinf / (dinf + 1e-300)
            if ratio > 10.0:
                rho = min(rho * 2.0, 1e5)
            elif ratio < 0.1:
                rho = max(rho / 2.0, 1e-2)
        elif pinf > 0.7 * prev:
            rho = min(rho * 2.0, 1e5)
        prev = pinf
        if pinf < tol and dinf < tol:
            break
        if pinf < 1e-7 and len(hist) > 4 and hist[-5][2] - hist[-1][2] < 1e-9 * (1 + abs(hist[-1][2])):
            break                                          # converged: no progress in the last 5 outer iterations
        if time_limit and time.perf_counter() - t_start > time_limit:
            break
    Xb = {}
    for (plus_g, idx, wt), X in zip(groups, Xg):
        for pos, b in enumerate(idx):
            Xb[b] = X[pos].cpu().numpy()
    out = Result(problem, R, O, best["theta"], best["U"], hist, time.perf_counter() - t_start, n_eval)
    out.extra = dict(X=Xb, t=x[dim:].copy(), rho=rho)
    return out


# --------------------------------------------------------------------------- diagonal-shift ALM (Proposition prop:diag)
def _block_index_maps(Rl: Relaxation, Rh: Relaxation):
    """For every block b of Rh: (positions in Rh's basis of the rows of Rl's basis, size in Rh)."""
    nl = len(Rl.cl_basis)
    out = []
    for b in range(len(Rh.blocks)):
        if b < nl:
            out.append((_row_index(Rl.cl_basis[b], Rh.cl_basis[b]), len(Rh.cl_basis[b])))
        else:
            il, ih = Rl.ineq_info[b - nl], Rh.ineq_info[b - nl]
            out.append((_row_index(il["Z"], ih["Z"]), len(ih["Z"])))
    return out


def _lift_diag_shift(lo: "Result", Rh: Relaxation):
    """Embed the per-block diagonal shift vectors of order r into order r+1 (new basis entries get shift 0)."""
    d_lo, off_lo = lo.extra["d"], lo.extra["d_off"]
    sizes = [b["size"] for b in Rh.blocks]
    off = np.concatenate([[0], np.cumsum(sizes)])
    d = np.zeros(off[-1])
    for b, (m, size) in enumerate(_block_index_maps(lo.relaxation, Rh)):
        seg = np.zeros(size); seg[m] = d_lo[off_lo[b]:off_lo[b + 1]]
        d[off[b]:off[b + 1]] = seg
    return d


def _solve_alm_diag(problem, order, cliques, rho0, outer, inner, time_limit, device, theta0, warm_start, verbose, tol,
                    adapt_rho, tight_weights=False) -> Result:
    """solve_alm with shift='diag': variables (theta, d) with d = concatenation of the per-block vectors d_b,
         min  sigma + sum_b (w_b/s_b) 1^T d_b   s.t.   Q_b(theta) + diag(d_b) >= 0,   d_b >= 0 ('plus' blocks).
    Same augmented Lagrangian, same inner solver and same penalty update as the uniform version."""
    from scipy.optimize import minimize
    R = Relaxation(problem, order, cliques, tight_weights)
    O = Oracle(R, device=device)
    groups, nb, dim = O.groups, len(R.blocks), O.dim
    plus = np.array([b["plus"] for b in R.blocks]); wts = np.array(O.weights); active = wts != 0.0
    sizes = np.array([b["size"] for b in R.blocks])
    d_off = np.concatenate([[0], np.cumsum(sizes)]); nd = int(d_off[-1])
    per_entry = wts / sizes                                    # w_b / s_b  (= 1 for moment blocks, B_i for loc blocks)
    d_init, X_init = np.zeros(nd), {}
    t_start = time.perf_counter()
    if warm_start and theta0 is None and order > min_order(problem):
        lo = _solve_alm_diag(problem, order - 1, cliques, rho0, outer, inner, time_limit, device, None, True, verbose,
                             tol, adapt_rho)
        theta0, _, X_init = lift_alm_state(lo, R, O)
        d_init = _lift_diag_shift(lo, R)
        if verbose:
            print(f"  warm start from order {order-1}: U^d = {lo.U:.8g}; lifted uniform U = {O.certificate(theta0):.8g}")
    th0 = (O.theta0() if theta0 is None else theta0.clone().to(O.device)).detach()
    x0 = np.concatenate([th0.cpu().numpy(), d_init])
    d_lo = np.concatenate([np.full(sizes[b], 0.0 if (plus[b] and active[b]) else -np.inf) for b in range(nb)]) if nb else np.zeros(0)
    lo_b = np.concatenate([np.full(dim, -np.inf), d_lo])
    hi_b = np.full(dim + nd, np.inf)
    bounds = list(zip(lo_b, hi_b))
    d_nonneg = np.concatenate([np.full(sizes[b], bool(plus[b] and active[b])) for b in range(nb)]) if nb else np.zeros(0, bool)
    Xg = []
    for plus_g, idx, wt in groups:
        s = R.blocks[idx[0]]["size"]
        X = torch.zeros(len(idx), s, s, dtype=O.dtype, device=O.device)
        for pos, b in enumerate(idx):
            if b in X_init:
                X[pos] = torch.as_tensor(X_init[b], dtype=O.dtype, device=O.device)
        Xg.append(X)
    cost_d = torch.as_tensor(np.repeat(per_entry * active, sizes), dtype=O.dtype, device=O.device)   # objective weights of d
    # index tensors to gather the d-segment of each group (len(idx) x s)
    gidx = [torch.as_tensor(np.stack([np.arange(d_off[b], d_off[b + 1]) for b in idx]), dtype=torch.long, device=O.device)
            for plus_g, idx, wt in groups]

    def build(xnp, requires_grad=True):
        x = torch.as_tensor(xnp, dtype=O.dtype, device=O.device).clone().requires_grad_(requires_grad)
        return x, x[:dim], x[dim:], O.gram_blocks(x[:dim])

    def stackG(Qs, d, g, idx):
        return torch.stack([Qs[i] for i in idx]) + torch.diag_embed(d[gidx[g]])

    def fun(xnp, rho):
        x, th, d, Qs = build(xnp)
        val = th[0] + (cost_d * d).sum()
        for g, ((plus_g, idx, wt), X) in enumerate(zip(groups, Xg)):
            val = val + _AlmPenalty.apply(stackG(Qs, d, g, idx), X, rho).sum()
        val.backward()
        return float(val.detach()), x.grad.cpu().numpy().astype(np.float64)

    def update(xnp, rho):
        """Multiplier update X_b <- Pi_+(X_b - rho G_b).  Returns (pinf, dX): primal infeasibility of the G_b and the
        relative change of the multipliers.  dX is needed in the stopping test: dinf (the gradient of the augmented
        Lagrangian at the inner solution) only measures the accuracy of the inner solve, and pinf alone can be tiny
        long before the multipliers have converged (observed on the sparse chain-quartic instance of E9)."""
        with torch.no_grad():
            x, th, d, Qs = build(xnp, False)
            pinf, dx2, nx2 = 0.0, 0.0, 0.0
            for g, ((plus_g, idx, wt), X) in enumerate(zip(groups, Xg)):
                G = stackG(Qs, d, g, idx)
                w, V = _safe_eigh(X - rho * G)
                Xn = (V * torch.clamp(w, min=0.0)[:, None, :]) @ V.transpose(1, 2)
                dx2 += float(((Xn - X) ** 2).sum()); nx2 += float((Xn ** 2).sum())
                Xg[g] = Xn
                pinf += float((torch.clamp(-_safe_eigh(G)[0][:, 0], min=0.0) ** 2).sum())
            return pinf ** 0.5, dx2 ** 0.5 / (1.0 + nx2 ** 0.5)

    def certificate_d(xnp):
        """U^d(theta, d) = sigma + sum_b (w_b/s_b) 1^T d_b + sum_b w_b Phi_b(Q_b + diag d_b)  (hard, valid bound)."""
        with torch.no_grad():
            x, th, d, Qs = build(xnp, False)
            U = float(th[0]) + float((cost_d * d).sum())
            for g, (plus_g, idx, wt) in enumerate(groups):
                lam = _safe_eigh(stackG(Qs, d, g, idx))[0][:, 0]
                pen = torch.clamp(-lam, min=0.0) if plus_g else -lam
                U += float((wt * pen).sum())
            return U

    best = dict(U=certificate_d(x0), theta=th0.clone(), d=x0[dim:].copy())
    best_unif = O.certificate(th0)
    hist, n_eval, rho, prev, x = [], 0, rho0, np.inf, x0
    for k in range(outer):
        res = minimize(fun, x, args=(rho,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=inner, maxcor=30, ftol=1e-15, gtol=1e-10, maxfun=int(1.3 * inner)))
        x = res.x
        n_eval += res.nfev
        g = res.jac.copy()
        dpart = x[dim:]
        g[dim:] = np.where((dpart > 0) | ~d_nonneg, g[dim:], np.minimum(g[dim:], 0.0))
        dinf = float(np.linalg.norm(g)) / (1.0 + abs(x[0]))
        pinf, dX = update(x, rho)
        th = torch.as_tensor(x[:dim], dtype=O.dtype, device=O.device)
        U = certificate_d(x)
        best_unif = min(best_unif, O.certificate(th))
        if U < best["U"]:
            best["U"], best["theta"], best["d"] = U, th.clone(), x[dim:].copy()
        hist.append((time.perf_counter() - t_start, n_eval, best["U"]))
        if verbose and (k % 5 == 0 or k == outer - 1):
            print(f"  outer {k:3d}  rho={rho:.3g}  pinf={pinf:.2e}  dinf={dinf:.2e}  dX={dX:.2e}  certified U^d={best['U']:.8g}  evals={n_eval}  t={time.perf_counter()-t_start:.0f}s")
        if adapt_rho:
            ratio = pinf / (dinf + 1e-300)
            if ratio > 10.0:
                rho = min(rho * 2.0, 1e5)
            elif ratio < 0.1:
                rho = max(rho / 2.0, 1e-2)
        elif pinf > 0.7 * prev:
            rho = min(rho * 2.0, 1e5)
        prev = pinf
        if pinf < tol and dinf < tol and dX < tol:
            break
        if pinf < 1e-7 and len(hist) > 4 and hist[-5][2] - hist[-1][2] < 1e-9 * (1 + abs(hist[-1][2])):
            break                                          # converged: no progress in the last 5 outer iterations
        if time_limit and time.perf_counter() - t_start > time_limit:
            break
    Xb = {}
    for (plus_g, idx, wt), X in zip(groups, Xg):
        for pos, b in enumerate(idx):
            Xb[b] = X[pos].cpu().numpy()
    out = Result(problem, R, O, best["theta"], best["U"], hist, time.perf_counter() - t_start, n_eval)
    # extra["t"] (per-block max of d_b) keeps lift_alm_state usable; "d"/"d_off" carry the full diagonal shift
    t_eq = np.array([x[dim + d_off[b]:dim + d_off[b + 1]].max() if sizes[b] else 0.0 for b in range(nb)])
    out.extra = dict(X=Xb, t=t_eq, rho=rho, d=x[dim:].copy(), d_best=best["d"], d_off=d_off, U_uniform=best_unif,
                     shift="diag", outer_iters=k + 1, pinf=pinf, dinf=dinf, dX=dX)
    return out
