"""Validate msos.solvers.alm (NumPy port of moment_sos.solve_alm) on Petersen r=2 and disk r=2:
   (i) against moment_sos.solve_alm on the identically normalised problem (same value to 1e-6),
   (ii) against the interior-point reference (reference.moment_relaxation: MOSEK if present, else Clarabel)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "3"); os.environ.setdefault("MKL_NUM_THREADS", "3")
sys.path.insert(0, ".."); sys.path.insert(0, "../../src")
import time
import numpy as np
import torch; torch.set_num_threads(3)
import moment_sos as po
from msos import POP, SpectralDual, problems as pb, solvers, reference


def scaled_copy(pop):
    """msos POP with moment_sos's internal normalisation  f -> (f - f_0) / sum|f_gamma|  (constraints unchanged:
    for the two test problems moment_sos's constraint scale is 1)."""
    zero = tuple([0] * pop.n)
    f_off = pop.f.get(zero, 0.0)
    f_scale = max(sum(abs(c) for e, c in pop.f.items() if e != zero), 1e-12)
    f = {e: (c - (f_off if e == zero else 0.0)) / f_scale for e, c in pop.f.items()}
    f = {e: c for e, c in f.items() if c != 0.0}
    return POP(pop.n, f, ineq=pop.ineq, eq=pop.eq, boolean=pop.boolean, name=pop.name + "_scaled"), f_off, f_scale


def check(name, pop, P, r, rho0):
    d = SpectralDual(pop, r)
    t = time.perf_counter(); U, th, h, info = solvers.alm(d, rho0=rho0, tol=1e-7); ta = time.perf_counter() - t
    # same normalisation as moment_sos -> values comparable to 1e-6
    pops, f_off, f_scale = scaled_copy(pop)
    ds = SpectralDual(pops, r)
    Us, _, hs, infos = solvers.alm(ds, rho0=rho0, tol=1e-7)
    Us_orig = f_off + f_scale * Us
    t = time.perf_counter(); res = po.solve_alm(P, order=r, rho0=rho0, tol=1e-7, warm_start=False); tp = time.perf_counter() - t
    ref, tr = reference.moment_relaxation(pop, r)
    print(f"{name:12s} r={r}: msos alm U={U:.8f} ({ta:.1f}s, {info['outer_iters']} outer, {info['n_eval']} evals, {info['status']})"
          f"  msos alm [moment_sos scaling] U={Us_orig:.8f}  moment_sos alm bound={res.bound:.8f} ({tp:.1f}s, {res.n_eval} evals)"
          f"  reference={ref:.8f} ({tr:.1f}s)")
    assert abs(Us_orig - res.bound) < 1e-6, (Us_orig, res.bound)
    assert abs(U - ref) < 1e-5, (U, ref)
    assert abs(U - res.bound) < 1e-5, (U, res.bound)
    assert U >= ref - 1e-7                       # certified upper bound
    assert d.certificate(th) == U                # returned theta reproduces the bound
    return U, Us_orig, res.bound, ref


if __name__ == "__main__":
    e = pb.graph("petersen", 10)
    s = po.variables(10)
    Ppet = po.Problem(sum((1 - s[i] * s[j]) / 2 for i, j in e), boolean=True)
    x, y = po.variables(2)
    Pdisk = po.Problem(x + y, [x ** 2 + y ** 2 <= 1], box=(-1, 1), add_box_constraints=False)
    out = {}
    out["petersen"] = check("Petersen", pb.maxcut(10, e), Ppet, 2, 1000.0)
    out["disk"] = check("disk", pb.disk(), Pdisk, 2, 1000.0)
    # finite-difference check of the augmented-Lagrangian gradient (theta and t) on the disk, random multipliers
    d = SpectralDual(pb.disk(), 2); rng = np.random.default_rng(0)
    X0 = []
    for b in d.blocks:
        A = rng.standard_normal((b["size"], b["size"])); X0.append(A @ A.T)
    rho = 37.0
    def L(xv):                                   # the same function alm minimises in its inner loop
        th, t = xv[:d.dim], xv[d.dim:]
        Qs = d.gram_blocks(th); val = th[0] + sum(b["weight"] * t[i] for i, b in enumerate(d.blocks))
        for i, (b, Q) in enumerate(zip(d.blocks, Qs)):
            w = np.linalg.eigvalsh(X0[i] - rho * (Q + t[i] * np.eye(b["size"]))); wp = np.maximum(w, 0)
            val += (wp @ wp - np.sum(X0[i] ** 2)) / (2 * rho)
        return val
    xv = np.concatenate([rng.standard_normal(d.dim) * 0.3, np.abs(rng.standard_normal(len(d.blocks)))])
    th = xv[:d.dim]; pr = np.bincount(d.slot_g, weights=d.slot_w * th[d.sl_Theta], minlength=d.nmom)
    th[d.sl_Theta] -= d.slot_w * (pr / d.W2)[d.slot_g]                   # zero-sum Theta (the space alm moves in)
    # analytic gradient, computed with the same formula alm uses in its inner function
    Qs = d.gram_blocks(th); Gq = []; gt = np.zeros(len(d.blocks))
    for i, (b, Q) in enumerate(zip(d.blocks, Qs)):
        w, V = np.linalg.eigh(X0[i] - rho * (Q + xv[d.dim + i] * np.eye(b["size"]))); wp = np.maximum(w, 0)
        Pi = (V * wp) @ V.T; Gq.append(-Pi); gt[i] = -wp.sum() + b["weight"]
    g = np.concatenate([d._backprop(Gq, th), gt])
    # directional derivatives along zero-sum-compatible random directions
    worst = 0.0
    for _ in range(5):
        dz = rng.standard_normal(len(xv)); dth = dz[:d.dim]
        pr = np.bincount(d.slot_g, weights=d.slot_w * dth[d.sl_Theta], minlength=d.nmom)
        dth[d.sl_Theta] -= d.slot_w * (pr / d.W2)[d.slot_g]
        h = 1e-6; fd = (L(xv + h * dz) - L(xv - h * dz)) / (2 * h)
        worst = max(worst, abs(fd - g @ dz) / (1 + abs(fd)))
    print(f"ALM gradient finite-difference check (disk r=2): max rel. error {worst:.2e}")
    assert worst < 1e-6
    print("all assertions passed")
