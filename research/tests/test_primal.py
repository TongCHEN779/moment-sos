"""Primal recovery (Proposition `primal recovery'): (i) two-sided certificate and Lagrangian identity on
Boolean MAX-CUT, (ii) componentwise gradient identities on constrained non-Boolean problems.
Plain asserting script:  python test_primal.py   (from tests/)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.optimize import minimize
from msos import POP, SpectralDual, problems as pb
from msos.reference import moment_relaxation


def tight_lbfgs(d, eta_final=1e-3, warm=(1e-1, 1e-2), gtol=1e-9, maxiter=3000):
    th = d.theta0()
    fg = lambda x, eta: (lambda o: (o["Us"], o["g"]))(d.evaluate(x, eta=eta))
    for eta in warm:
        th = minimize(fg, th, args=(eta,), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=300, maxcor=30, ftol=1e-12, gtol=1e-10)).x
    res = minimize(fg, th, args=(eta_final,), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxcor=30, ftol=1e-15, gtol=gtol, maxfun=int(1.5 * maxiter)))
    return res.x, res


def part_i(name, pop, r=2, eta=1e-3):
    d = SpectralDual(pop, r)
    th, res = tight_lbfgs(d, eta)
    pm = d.primal_moments(th, eta)
    o = d.evaluate(th, eta=eta)
    rho, _ = moment_relaxation(pop, r, trace_cuts=False)
    lb = d.boolean_lower_bound(th, eta)
    LB, U = d.two_sided(th, eta)
    ent = eta * pm["entropy"]
    ident_g0 = pm["Us"] - ent - pm["Lf"]                     # paper (b): valid when g = 0
    ident_full = pm["Us"] - ent - float(th @ pm["g"]) - pm["Lf"]   # paper (a): valid for every theta
    print(f"[{name}] r={r} eta={eta:g}  L-BFGS iters={res.nit} |g|={pm['grad_norm']:.2e} |g_Theta|={pm['g_Theta_norm']:.2e}")
    print(f"   y0-1={pm['y0']-1:.2e}  hankel={pm['infeas']['hankel']:.2e}  lmin M={pm['lmin_M'][0]:.2e}  delta={lb['delta']:.2e} tau={lb['tau']:.2e}")
    print(f"   rho_r(Clarabel)={rho:.8f}  LB={LB:.8f}  Lf={pm['Lf']:.8f}  U={U:.8f}   U-LB={U-LB:.2e}")
    print(f"   identity (a) Us - eta*H - <theta,g> - Lf = {ident_full:.2e};   (b) Us - eta*H - Lf = {ident_g0:.2e}")
    assert abs(pm["y0"] - 1) < 1e-12, "ybar_0 != 1"
    assert np.abs(o["g"] - pm["g"]).max() < 1e-12 and abs(o["Us"] - pm["Us"]) < 1e-12
    assert pm["infeas"]["hankel"] <= pm["g_Theta_norm"] + 1e-14, "Hankel infeasibility > |g_Theta|"
    assert pm["g_Theta_norm"] <= pm["grad_norm"] * (1 + 1e-12) + 1e-14, "|g_Theta| > |g|"   # rounding-safe (was: same eps on both sides)
    assert lb["delta"] <= pm["g_Theta_norm"] + 1e-14
    assert LB <= rho + 1e-6, f"LB {LB} > rho {rho}"
    assert rho <= U + 1e-6, f"rho {rho} > U {U}"
    assert pm["Lf"] <= rho + 1e-5, "L_ybar(f) exceeds rho (ybar should be nearly feasible)"
    assert abs(ident_full) < 1e-10, "Lagrangian identity (a) fails"
    # (b) holds exactly only at g = 0; L-BFGS in double precision stalls at |g| ~ 1e-6 (the relative decrease
    # of U_eta falls below machine epsilon), so (b) is checked to the Cauchy-Schwarz slack |<theta,g>|.
    slack = np.linalg.norm(th) * pm["grad_norm"]
    print(f"   L-BFGS stop: {res.message};  |theta||g| = {slack:.2e}")
    assert abs(ident_g0) <= slack + 1e-10, "identity (b) at the (approximate) minimiser fails beyond |theta||g|"
    assert abs(ident_g0) < 1e-6
    return dict(name=name, rho=rho, LB=LB, Lf=pm["Lf"], U=U, g=pm["grad_norm"], hankel=pm["infeas"]["hankel"])


def zero_sum(d, th):
    t = th[d.sl_Theta]
    pr = np.bincount(d.slot_g, weights=d.slot_w * t, minlength=d.nmom)
    th[d.sl_Theta] = t - d.slot_w * (pr / d.W2)[d.slot_g]
    return th


def part_ii(name, pop, r=2, eta=1e-2, seed=0):
    """Componentwise gradient identities at a random theta."""
    d = SpectralDual(pop, r)
    rng = np.random.default_rng(seed)
    th = rng.standard_normal(d.dim) * 0.3
    th[0] = 2.0
    zero_sum(d, th)                                 # Theta is a dual point only on the zero-sum subspace
    pm = d.primal_moments(th, eta)
    g = d.evaluate(th, eta=eta)["g"]
    K = len(d.cliques)
    e_sigma = abs(g[0] - (1 - pm["y0"]))
    e_u_paper = e_u_code = 0.0
    for i, info in enumerate(d.ineq_info):
        wq = np.where(info["a"] == info["b"], 1.0, 2.0)
        diff = wq * (pm["X"][K + i][info["a"], info["b"]] - pm["Mloc"][i][info["a"], info["b"]])
        gu = g[d.sl_u][d.u_off[i]:d.u_off[i + 1]]
        e_u_paper = max(e_u_paper, np.abs(gu - diff).max())      # paper statement:  g_u = +omega (X - L(g x^a x^b))
        e_u_code = max(e_u_code, np.abs(gu + diff).max())        # derivation:       g_u = -omega (X - L(g x^a x^b))
    e_p_paper = e_p_code = 0.0
    if d.np_:
        gp = g[d.sl_p]
        e_p_paper = np.abs(gp + pm["eq_res"]).max()               # paper: g_p = -L(h x^mu)
        e_p_code = np.abs(gp - pm["eq_res"]).max()                # derivation: g_p = +L(h x^mu)
    # g_Theta = zero-sum projection of -(omega_ab X_k[a,b]); also Hankel bound (c)
    xs = np.concatenate([pm["X"][k][d.cl_I[k], d.cl_J[k]] for k in range(K)])
    v = -d.slot_w * xs
    pr = np.bincount(d.slot_g, weights=d.slot_w * v, minlength=d.nmom)
    gT = v - d.slot_w * (pr / d.W2)[d.slot_g]
    e_T = np.abs(g[d.sl_Theta] - gT).max()
    print(f"[{name}] r={r}: |g_sigma-(1-y0)|={e_sigma:.1e}  g_u: paper-sign err={e_u_paper:.1e} / minus-sign err={e_u_code:.1e}"
          f"  g_p: paper-sign err={e_p_paper:.1e} / plus-sign err={e_p_code:.1e}  g_Theta err={e_T:.1e}"
          f"  hankel={pm['infeas']['hankel']:.3e} <= |g_Theta|={pm['g_Theta_norm']:.3e}  lagr.res={pm['lagr_residual']:.1e}")
    assert e_sigma < 1e-10
    assert e_T < 1e-10
    assert e_u_code < 1e-10, "g_u identity (with the sign g_u = -omega(X_i - L_ybar(g_i x^a x^b))) fails"
    assert e_p_code < 1e-10, "g_p identity (with the sign g_p = +L_ybar(h_j x^mu)) fails"
    assert pm["infeas"]["hankel"] <= pm["g_Theta_norm"] + 1e-12
    assert abs(pm["lagr_residual"]) < 1e-10
    return dict(e_u_paper=e_u_paper, e_p_paper=e_p_paper)


if __name__ == "__main__":
    print("== (i) Boolean two-sided certificate ==")
    part_i("Petersen", pb.maxcut(10, pb.graph("petersen", 10)))
    part_i("reg3 n=12", pb.maxcut(12, pb.graph("reg3", 12, seed=1)))
    print("== (ii) componentwise gradient identities (random theta) ==")
    sph = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")
    r1 = part_ii("disk", pb.disk())
    r2 = part_ii("boxqp4", pb.box_qp(4, 1))
    r3 = part_ii("sphere (eq.)", sph)
    pc, cl = pb.chain_quartic(5, 1)
    d = SpectralDual(pc, 2, cl); th = zero_sum(d, np.random.default_rng(1).standard_normal(d.dim) * 0.3)
    pm = d.primal_moments(th, 1e-2)
    assert pm["infeas"]["hankel_rss"] <= pm["g_Theta_norm"] and abs(pm["lagr_residual"]) < 1e-10
    print(f"[chain5 sparse] hankel (root-sum-square over 4 cliques)={pm['infeas']['hankel_rss']:.3e} <= |g_Theta|={pm['g_Theta_norm']:.3e}"
          f"  (sum of norms {pm['infeas']['hankel']:.3e} may exceed it)  lagr.res={pm['lagr_residual']:.1e}")
    if max(r1["e_u_paper"], r2["e_u_paper"], r3["e_p_paper"]) > 1e-8:
        print("NOTE: g_u = omega_ab (L_ybar(g_i x^a x^b) - X_i[a,b]) and g_p = +L_ybar(h_j x^mu) (the derivative);"
              " the first draft of Proposition (b) had the opposite signs, corrected in the revision.")
    print("ALL PRIMAL TESTS PASSED")
