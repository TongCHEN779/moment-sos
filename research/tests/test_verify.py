"""Tests for msos/verify.py (rigorous a posteriori certificate).
(i)   verified_lmin_lower_bound on random symmetric matrices: ell <= lmin and lmin - ell small;
(ii)  exact_residual: E tiny for solver output, and the exact residual polynomial evaluated at random
      points agrees with the float evaluation of the identity;
(iii) U_rig >= U and U_rig - U < 1e-8 on small instances.
Run with:  python test_verify.py   (from tests/)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
import sys; sys.path.insert(0, "..")
import math
from fractions import Fraction
import numpy as np
from msos import POP, SpectralDual, problems as pb, solvers
from msos import verify as V


def test_lmin_bound():
    rng = np.random.default_rng(1)
    worst = 0.0
    for s in (5, 8, 13, 20, 40, 75, 120, 200, 300):
        for scale in (1.0, 1e-3, 1e3):
            A = rng.standard_normal((s, s)) * scale
            Q = (A + A.T) / 2
            Q = (Q + Q.T) / 2                               # exactly symmetric
            lm = float(np.linalg.eigvalsh(Q)[0])
            ell, info = V.verified_lmin_lower_bound(Q)
            assert info["ok"], (s, scale)
            assert ell <= lm, (s, scale, ell, lm)
            rel = (lm - ell) / max(1.0, abs(lm))
            worst = max(worst, rel)
            assert rel < 1e-8, (s, scale, lm - ell)
        # a PSD matrix and a matrix with a tiny negative eigenvalue
        B = rng.standard_normal((s, s)); P = B @ B.T / s
        P = (P + P.T) / 2
        lmP = float(np.linalg.eigvalsh(P)[0])
        ell, info = V.verified_lmin_lower_bound(P)
        assert info["ok"] and ell <= lmP and lmP - ell < 1e-8, (s, lmP, ell)
    # the lemma also holds when lmin_hat is wrong: pass a bad estimate and check validity
    A = rng.standard_normal((30, 30)); Q = (A + A.T) / 2; Q = (Q + Q.T) / 2
    lm = float(np.linalg.eigvalsh(Q)[0])
    ell, info = V.verified_lmin_lower_bound(Q, lmin_hat=lm + 1e-3)   # too optimistic -> doublings
    assert info["ok"] and ell <= lm and info["doublings"] > 0, info
    ell, info = V.verified_lmin_lower_bound(Q, lmin_hat=lm - 1.0)    # too pessimistic -> still valid
    assert info["ok"] and ell <= lm
    # the explicit Cholesky agrees with LAPACK on a PD matrix
    R, ok = V.cholesky_columns(P)
    assert ok and np.allclose(R, np.linalg.cholesky(P).T, atol=1e-12)
    R, ok = V.cholesky_columns(Q - (abs(lm) + 1) * np.eye(30))        # indefinite -> must fail
    assert not ok
    print(f"(i)   lmin bound: ok; worst relative slack {worst:.2e}")


def instances():
    sph = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")
    pc, cl = pb.chain_quartic(5, 1)
    return [("ball r2", pb.ball(), 2, None), ("disk r2", pb.disk(), 2, None), ("sphere r2", sph, 2, None),
            ("K3 r2", pb.maxcut(3, pb.graph("K", 3)), 2, None), ("boxqp4 r2", pb.box_qp(4, 1), 2, None),
            ("chain5 sparse r2", pc, 2, cl)]


def test_exact_residual():
    rng = np.random.default_rng(0)
    for name, pop, r, cl in instances():
        d = SpectralDual(pop, r, cl)
        # (a) random theta (not zero-sum: large residual) -- exact polynomial vs float evaluation
        th = rng.standard_normal(d.dim) * 0.5
        E, E_fr, e_dict, info = V.exact_residual(d, th)
        assert Fraction(E) >= E_fr and E_fr == sum(abs(c) for c in e_dict.values())
        worst = 0.0
        for _ in range(5):
            x = rng.choice([-1.0, 1.0], pop.n) if pop.boolean else rng.uniform(-1, 1, pop.n)
            ex = float(V.eval_residual_exact(d, e_dict, x))
            fl = V.residual_float_at_point(d, th, x)
            worst = max(worst, abs(ex - fl))
        assert worst < 1e-12 * max(1.0, E), (name, worst)
        # (b) zero-sum theta: the identity holds up to rounding, so E is tiny
        t = th[d.sl_Theta]; pr = np.bincount(d.slot_g, weights=d.slot_w * t, minlength=d.nmom)
        th[d.sl_Theta] = t - d.slot_w * (pr / d.W2)[d.slot_g]
        E0, _, _, _ = V.exact_residual(d, th)
        assert E0 < 1e-12, (name, E0)
        # (c) solver output
        U, ths, h = solvers.lbfgs_continuation(d, iters_per_stage=300)
        Es, Es_fr, e_s, _ = V.exact_residual(d, ths)
        assert Es < 1e-12 * max(1.0, abs(U)), (name, Es)
        worst_s = 0.0
        for _ in range(5):
            x = rng.choice([-1.0, 1.0], pop.n) if pop.boolean else rng.uniform(-1, 1, pop.n)
            ex = float(V.eval_residual_exact(d, e_s, x))
            fl = V.residual_float_at_point(d, ths, x)
            worst_s = max(worst_s, abs(ex - fl))
        assert worst_s < 1e-12, (name, worst_s)
        print(f"(ii)  {name:18s} E(random)={E:.2e} |exact-float|={worst:.1e}  E(zero-sum)={E0:.1e}  E(solver)={Es:.1e} |exact-float|={worst_s:.1e}")


def test_rigorous_bound():
    for name, pop, r, cl in instances():
        d = SpectralDual(pop, r, cl)
        U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=300)
        out = V.rigorous_bound(d, th)
        assert out["all_ok"], name
        assert out["U_rig"] >= out["U"], (name, out["U_rig"], out["U"])
        assert out["U_rig"] - out["U"] < 1e-8, (name, out["U_rig"] - out["U"])
        assert abs(out["U"] - U) < 1e-12
        for ell, lh in zip(out["ell"], out["lmin_hat"]):
            assert ell <= lh
        print(f"(iii) {name:18s} U={out['U']:.10f} U_rig-U={out['U_rig_minus_U']:.2e} E={out['E']:.1e} max slack={out['max_slack']:.1e}")
    # sanity check helper
    w, E = V.sanity_check_residual(SpectralDual(pb.ball(), 2))
    assert w < 1e-13


if __name__ == "__main__":
    test_lmin_bound()
    test_exact_residual()
    test_rigorous_bound()
    print("all verify tests passed")


def test_exact_B_is_not_below_float_B():
    """B_i of the rigorous certificate is summed exactly (a float sum could round below sup g)."""
    from fractions import Fraction
    from msos.core import box_sup_bound, box_sup_bound_exact
    g = {(0,): 0.1, (1,): 0.2, (2,): 0.3, (3,): -0.7, (1, 1): -0.1, (2, 1): -0.1}      # inexact decimals
    assert box_sup_bound_exact(g, False) >= Fraction(box_sup_bound(g, False)) - Fraction(1, 2 ** 50)
    assert box_sup_bound_exact({(0, 0): 1.0, (2, 0): -1.0}, False) == 1                # 1 - x^2: B = 1
    # positive-coefficient sum is exact even where the float accumulation rounds down
    c = [0.1] * 10
    assert box_sup_bound_exact({(i + 1,): ci for i, ci in enumerate(c)}, False) == sum(Fraction(x) for x in c)
