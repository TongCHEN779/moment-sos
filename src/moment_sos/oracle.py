"""Torch oracle: dual point -> certified bound U(theta) and its smoothed version.

theta = [ sigma | Theta~ (one entry per Gram slot) | u (localising Gram entries) | p (equality multipliers) ].
All maps theta -> Gram matrices are affine and built from gather/scatter ops, so autograd returns the
exact (sub)gradient; the spectral part has an analytic backward pass (no eigenvector derivatives).
"""
from __future__ import annotations
import torch
from .relaxation import Relaxation


def _safe_eigh(Q):
    """Batched symmetric eigendecomposition with fallbacks (non-finite input, LAPACK failure)."""
    if not bool(torch.isfinite(Q).all()):
        B, s, _ = Q.shape
        w = torch.full((B, s), -1e12, dtype=Q.dtype, device=Q.device)      # huge penalty -> line search backs off
        V = torch.eye(s, dtype=Q.dtype, device=Q.device).expand(B, s, s).clone()
        return w, V
    try:
        return torch.linalg.eigh(Q)
    except Exception:
        import scipy.linalg as sla
        ws, Vs = [], []
        for q in Q.detach().cpu().numpy():
            w_, V_ = sla.eigh(q, driver="evr")
            ws.append(w_); Vs.append(V_)
        import numpy as np
        return (torch.as_tensor(np.stack(ws), dtype=Q.dtype, device=Q.device),
                torch.as_tensor(np.stack(Vs), dtype=Q.dtype, device=Q.device))


class _SpecShift(torch.autograd.Function):
    """(smoothed, hard) shifts of a batch of symmetric matrices Q of shape (B, s, s).

    hard   = [-lmin(Q)]_+   (or -lmin(Q) for constant-trace blocks)
    smooth = eta*log(plus + tr exp(-Q/eta)) >= hard, backward = -P with P the softmax density."""

    @staticmethod
    def forward(ctx, Q, eta, plus):
        w, V = _safe_eigh(Q)                                      # (B,s), (B,s,s)
        lam = -w[:, 0]
        hard = torch.clamp(lam, min=0.0) if plus else lam
        if eta is None:                                           # exact subgradient
            v0 = V[:, :, 0]
            P = v0[:, :, None] * v0[:, None, :]
            if plus:
                P = P * (lam > 0).to(P.dtype)[:, None, None]
            val = hard.clone()
        else:
            z = -w / eta
            m = z.max(dim=1).values
            if plus:
                m = torch.clamp(m, min=0.0)
            S = torch.exp(z - m[:, None]).sum(dim=1) + (torch.exp(-m) if plus else 0.0)
            L = m + torch.log(S)
            val = eta * L
            P = (V * torch.exp(z - L[:, None])[:, None, :]) @ V.transpose(1, 2)
        ctx.save_for_backward(P)
        ctx.mark_non_differentiable(hard)
        return val, hard

    @staticmethod
    def backward(ctx, g_val, g_hard):
        (P,) = ctx.saved_tensors
        return -g_val[:, None, None] * P, None, None


class Oracle:
    def __init__(self, relax: Relaxation, device="cpu", dtype=torch.float64):
        self.relax, self.device, self.dtype = relax, torch.device(device), dtype
        R = relax
        L = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)
        F = lambda a: torch.as_tensor(a, dtype=dtype, device=self.device)
        self.slot_g, self.slot_w = L(R.slot_g), F(R.slot_w)
        self.W, self.W2, self.fvec = F(R.W), F(R.W2), F(R.fvec)
        self.Wslot = self.W[self.slot_g]
        self.cl = [(L(R.cl_I[k]), L(R.cl_J[k]), R.cl_off[k], R.cl_off[k + 1], len(R.cl_basis[k]))
                   for k in range(len(R.cl_basis))]
        self.Rloc = (L(R.Rloc[0]), L(R.Rloc[1]), F(R.Rloc[2]))
        self.Req = (L(R.Req[0]), L(R.Req[1]), F(R.Req[2]))
        self.loc = [(L(i["a"]), L(i["b"]), R.u_off[k], R.u_off[k + 1], len(i["Z"])) for k, i in enumerate(R.ineq_info)]
        self.ns, self.nu, self.np_ = R.nslots, R.nu, R.np_
        self.dim = R.dim
        self.weights = [b["weight"] for b in R.blocks]
        self.plus = [b["plus"] for b in R.blocks]
        groups = {}                                               # batch blocks of equal size / shift type
        for i, b in enumerate(R.blocks):
            if b["weight"] != 0.0:
                groups.setdefault((b["size"], b["plus"]), []).append(i)
        self.groups = [(k[1], idx, torch.as_tensor([self.weights[i] for i in idx], dtype=dtype, device=self.device))
                       for k, idx in groups.items()]

    # ------------------------------------------------------------------ parameters
    def theta0(self, sigma=0.0):
        """Zero dual point (with sigma set to `sigma`)."""
        th = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        th[0] = sigma
        return th

    # ------------------------------------------------------------------ preconditioning
    def coordinate_scales(self):
        """Per-coordinate scale s_i = 1/sqrt(L_i) with L_i = sum_b w_b |dQ_b/dtheta_i|_F^2 (diagonal
        Lipschitz preconditioner); optimising x with theta = s * x equilibrates the coordinates."""
        R = self.relax
        ns, nu = self.ns, self.nu
        mp = torch.where(self.slot_w == 1.0, 1.0, 2.0).to(self.dtype)             # matrix entries per slot (diag:1, off:2)
        wslot = torch.zeros(ns, dtype=self.dtype, device=self.device)
        for k, (I, J, o0, o1, s) in enumerate(self.cl):
            wslot[o0:o1] = self.weights[k]
        S = torch.zeros(R.nmom, dtype=self.dtype, device=self.device).index_add(0, self.slot_g, wslot * mp)
        Lg = (S / self.W ** 2)                                                   # per moment: sum_p w_p m_p / W^2
        L = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        # sigma: moment-level coordinate at the empty moment
        L[0] = Lg[R.empty_id]
        # Theta~: own slot entries (projection only reduces it)
        L[1:1 + ns] = wslot * mp
        # localising Gram entries: own block + moment level
        if nu:
            rows, cols, vals = self.Rloc
            Lu = torch.zeros(nu, dtype=self.dtype, device=self.device).index_add(0, cols, vals ** 2 * Lg[rows])
            own = torch.zeros(nu, dtype=self.dtype, device=self.device)
            for k, (a, b, o0, o1, s) in enumerate(self.loc):
                m = torch.where(a == b, 1.0, 2.0).to(self.dtype)
                own[o0:o1] = self.weights[len(self.cl) + k] * m
            L[1 + ns:1 + ns + nu] = Lu + own
        if self.np_:
            rows, cols, vals = self.Req
            L[1 + ns + nu:] = torch.zeros(self.np_, dtype=self.dtype, device=self.device).index_add(0, cols, vals ** 2 * Lg[rows])
        L = torch.clamp(L, min=1e-12)
        return 1.0 / torch.sqrt(L)

    # ------------------------------------------------------------------ Gram matrices
    def gram_blocks(self, th):
        """Dense symmetric Gram matrices Q_b(theta): clique moment blocks, then localising blocks."""
        ns, nu = self.ns, self.nu
        sigma, Th = th[0], th[1:1 + ns]
        u, p = th[1 + ns:1 + ns + nu], th[1 + ns + nu:]
        r = -self.fvec
        r = r + torch.nn.functional.one_hot(torch.tensor(self.relax.empty_id, device=self.device), len(r)).to(r.dtype) * sigma
        if nu:
            r = r.index_add(0, self.Rloc[0], -self.Rloc[2] * u[self.Rloc[1]])
        if self.np_:
            r = r.index_add(0, self.Req[0], -self.Req[2] * p[self.Req[1]])
        gsum = torch.zeros_like(r).index_add(0, self.slot_g, self.slot_w * Th)
        Theta = Th - self.slot_w * (gsum / self.W2)[self.slot_g]          # zero-sum projection
        q = r[self.slot_g] / self.Wslot + Theta
        Qs = []
        for I, J, o0, o1, s in self.cl:
            A = torch.zeros(s * s, dtype=q.dtype, device=q.device).index_put((I * s + J,), q[o0:o1]).view(s, s)
            Qs.append(A + A.T - torch.diag(torch.diagonal(A)))
        for a, b, o0, o1, s in self.loc:
            vals = u[o0:o1]
            A = torch.zeros(s * s, dtype=q.dtype, device=q.device).index_put((a * s + b,), vals).view(s, s)
            Qs.append(A + A.T - torch.diag(torch.diagonal(A)))
        return Qs

    # ------------------------------------------------------------------ value
    def evaluate(self, th, eta=None):
        """Return (U_smooth tensor with grad, U_hard float). eta=None gives the exact subgradient oracle."""
        Qs = self.gram_blocks(th)
        Us, Uh = th[0], float(th[0].detach())
        for plus, idx, wts in self.groups:
            val, hard = _SpecShift.apply(torch.stack([Qs[i] for i in idx]), eta, plus)
            Us = Us + (wts * val).sum()
            Uh += float((wts * hard).sum())
        return Us, Uh

    def certificate(self, theta):
        """Hard certified upper bound (internal maximisation sense)."""
        with torch.no_grad():
            return self.evaluate(theta, None)[1]
