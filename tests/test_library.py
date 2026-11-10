import itertools
import numpy as np
import pytest
import torch
import moment_sos as po
from moment_sos.relaxation import Relaxation
from moment_sos.oracle import Oracle


def maxcut(n, edges, w=None):
    s = po.variables(n)
    w = [1.0] * len(edges) if w is None else w
    return po.Problem(sum(wi * (1 - s[i] * s[j]) / 2 for (i, j), wi in zip(edges, w)), boolean=True)


def test_poly_arithmetic_and_constraints():
    x, y = po.variables(2)
    p = (x + 2 * y) ** 2 - 1
    assert p.degree() == 2 and abs(p([1.0, 1.0]) - 8) < 1e-12
    c = (x ** 2 + y ** 2 <= 1)
    assert c.kind == "ineq" and abs(c.poly([0.0, 0.0]) - 1) < 1e-12
    e = (x == y)
    assert e.kind == "eq" and abs(e.poly([2.0, 2.0])) < 1e-12
    q = p.substitute_affine(np.array([1.0, 0.0]), np.array([2.0, 1.0]))      # x=1+2t0, y=t1
    t = np.array([0.3, -0.7])
    assert abs(q(t) - p(np.array([1 + 2 * t[0], t[1]]))) < 1e-12


@pytest.mark.parametrize("make,order", [
    (lambda: po.Problem(po.variables(2)[0] * po.variables(2)[1], [po.variables(2)[0] ** 2 + po.variables(2)[1] ** 2 <= 1], box=(-1, 1)), 2),
    (lambda: maxcut(4, list(itertools.combinations(range(4), 2))), 2),
])
def test_gram_identity(make, order):
    """sigma - f - sum s_i g_i - sum p_j h_j == sum_k v_k^T Q_k v_k at random points."""
    P = make()
    R = Relaxation(P, order); O = Oracle(R)
    rng = np.random.default_rng(0)
    th = torch.as_tensor(rng.standard_normal(O.dim) * 0.4)
    Qs = [Q.numpy() for Q in O.gram_blocks(th)]
    K = len(R.cliques)

    def mono(E, x):
        return np.array([np.prod(x ** e.astype(float)) for e in E])
    for _ in range(5):
        x = rng.choice([-1.0, 1.0], P.n) if P.boolean else rng.uniform(-1, 1, P.n)
        ev = lambda poly: sum(c * np.prod(x ** np.array(e)) for e, c in poly.terms.items())
        lhs = float(th[0]) - ev(P.f)
        for i, (g, info) in enumerate(zip(P.ineq, R.ineq_info)):
            z = mono(info["Z"], x)
            lhs -= z @ Qs[K + i] @ z * ev(g)
        rhs = sum(mono(B, x) @ Qs[k] @ mono(B, x) for k, B in enumerate(R.cl_basis))
        assert abs(lhs - rhs) < 1e-10


def test_gradient_matches_finite_differences():
    P = maxcut(5, list(itertools.combinations(range(5), 2)))
    R = Relaxation(P, 2); O = Oracle(R)
    th = torch.as_tensor(np.random.default_rng(1).standard_normal(O.dim) * 0.3).requires_grad_(True)
    Us, _ = O.evaluate(th, 0.2); Us.backward()
    g = th.grad.clone()
    d = torch.as_tensor(np.random.default_rng(2).standard_normal(O.dim))
    e = 1e-6
    fp = O.evaluate(th.detach() + e * d, 0.2)[0].item(); fm = O.evaluate(th.detach() - e * d, 0.2)[0].item()
    # gradient is projected onto zero-sum directions only through the parametrisation, so compare directly
    assert abs((fp - fm) / (2 * e) - float(g @ d)) < 1e-5 * max(1, abs(float(g @ d)))


def test_certified_bounds_known_values():
    K5 = maxcut(5, list(itertools.combinations(range(5), 2)))
    assert abs(po.solve(K5, order=1).bound - 6.25) < 1e-3          # Goemans-Williamson / Shor value
    r3 = po.solve(K5, order=3)
    assert 6.0 - 1e-6 <= r3.bound < 6.0 + 1e-3                       # closes at order 3
    tri = maxcut(3, [(0, 1), (0, 2), (1, 2)])
    assert abs(po.solve(tri, order=1).bound - 2.25) < 1e-3
    assert abs(po.solve(tri, order=2).bound - 2.0) < 1e-3


def test_bound_is_valid_at_every_iterate_ball_and_min_sense():
    x = po.variables(2)
    P = po.Problem(x[0] + x[1], [x[0] ** 2 + x[1] ** 2 <= 1], box=(-1, 1))
    r = po.solve(P, order=2)
    assert r.bound >= 2 ** 0.5 - 1e-9 and r.bound < 2 ** 0.5 + 1e-3
    Pm = po.Problem(x[0] + x[1], [x[0] ** 2 + x[1] ** 2 <= 1], sense="min", box=(-1, 1))
    rm = po.solve(Pm, order=2)
    assert rm.bound_kind == "lower" and rm.bound <= -(2 ** 0.5) + 1e-9 and rm.bound > -(2 ** 0.5) - 1e-3


def test_equality_constraint_and_general_box():
    x, y = po.variables(2)
    P = po.Problem(x + y, [x ** 2 + y ** 2 == 2], box=([0, 0], [3, 3]))     # max at (1,1): value 2
    r = po.solve(P, order=2)
    assert abs(r.bound - 2.0) < 2e-3
    xr, v = r.round()
    assert xr is not None and abs(v - 2.0) < 1e-3


def test_sparse_cliques_agree_with_dense_on_chain():
    n = 6
    x = po.variables(n)
    f = sum(x[i] * x[i + 1] - 0.3 * x[i] ** 4 + 0.5 * x[i] ** 2 * x[i + 1] ** 2 for i in range(n - 1))
    P = po.Problem(f, box=(-1, 1))
    rd = po.solve(P, order=2, cliques="dense")
    rs = po.solve(P, order=2, cliques="auto")
    assert max(rs.relaxation.sizes()) < max(rd.relaxation.sizes())
    assert abs(rd.bound - rs.bound) < 5e-3 and rs.bound >= rd.bound - 5e-3


def test_boolean_rounding_finds_maxcut():
    C5 = maxcut(5, [(i, (i + 1) % 5) for i in range(5)])
    r = po.solve(C5, order=2)
    x, v = r.round()
    assert abs(v - 4.0) < 1e-9 and 4.0 - 1e-6 <= r.bound < 4.0 + 1e-3


def test_warm_start_lift_is_valid_dual_point():
    K5 = maxcut(5, list(itertools.combinations(range(5), 2)))
    r1 = po.solve(K5, order=1, warm_start=False)
    R2 = Relaxation(K5, 2); O2 = Oracle(R2)
    th = po.lift_theta(r1, R2, O2)
    # lifted point certifies a valid bound (>= max cut = 6) and, as no shift is active for Boolean K5, equals the old bound
    U = O2.certificate(th)
    assert K5.f_off + K5.f_scale * U >= 6.0 - 1e-9 and abs(U - r1.U) < 1e-3     # internal (normalised) scale


def test_version_matches_pyproject():
    import os, re
    here = os.path.dirname(os.path.abspath(__file__))
    txt = open(os.path.join(here, "..", "pyproject.toml")).read()
    assert po.__version__ == re.search(r'^version = "([^"]+)"', txt, re.M).group(1)


def test_clear_errors():
    x = po.variables(2)
    with pytest.raises(ValueError, match="box"):
        po.Problem(x[0] + x[1], [x[0] ** 2 <= 1])
    with pytest.raises(ValueError, match="use order >= 2"):
        po.solve(po.Problem(x[0] ** 3, box=(-1, 1)), order=1)


def test_stop_at_certifies_integer_optimum_early():
    import itertools
    n = 8
    s = po.variables(n)
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 4), (2, 6)]
    P = po.Problem(sum((1 - s[i] * s[j]) / 2 for i, j in edges), boolean=True)
    best_cut = max(sum(1 for i, j in edges if x[i] != x[j]) for x in itertools.product([0, 1], repeat=n))
    full = po.solve(P, order=2, etas=po.SHORT_ETAS)
    fast = po.solve(P, order=2, etas=po.SHORT_ETAS, stop_at=best_cut + 1 - 1e-9)
    assert fast.bound < best_cut + 1 and fast.bound >= best_cut - 1e-6
    assert fast.n_eval <= full.n_eval


def test_rigorous_bound_is_a_valid_bound_above_the_float_bound():
    K5 = maxcut(5, list(itertools.combinations(range(5), 2)))
    r = po.solve(K5, order=2, etas=po.SHORT_ETAS)
    v = r.rigorous()
    assert v["ok"] and v["bound"] >= r.bound - 1e-12 and v["bound"] >= 6.0          # true max cut is 6
    assert v["bound"] - r.bound < 1e-6 and v["E"] < 1e-12
    x, y = po.variables(2)
    Q = po.Problem(x * y + x ** 2 - 0.5 * y ** 2, [x ** 2 + y ** 2 <= 1], box=(-1, 1))
    rq = po.solve(Q, order=2); vq = rq.rigorous()
    ref = po.solve_dual_ipm(po.Relaxation(Q, 2))[0]
    ref = Q.f_off + Q.f_scale * ref
    assert vq["bound"] >= rq.bound - 1e-12 and vq["bound"] >= ref - 1e-7
    Qm = po.Problem(x * y + x ** 2 - 0.5 * y ** 2, [x ** 2 + y ** 2 <= 1], sense="min", box=(-1, 1))
    rm = po.solve(Qm, order=2); vm = rm.rigorous()
    assert vm["bound"] <= rm.bound + 1e-12                                          # lower bound for a minimisation


def test_verified_lmin_lower_bound():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30)); A = (A + A.T) / 2
    ell, info = po.verified_lmin_lower_bound(A)
    assert info["ok"] and ell <= np.linalg.eigvalsh(A)[0] and np.linalg.eigvalsh(A)[0] - ell < 1e-8


def test_newton_inner_solver_matches_lbfgs_alm():
    # Boolean (constant-trace blocks) and a constrained problem (inequality blocks with t >= 0)
    prob_b = maxcut(5, list(itertools.combinations(range(5), 2)))
    x, y = po.variables(2)
    prob_c = po.Problem(x**2 * y + x * y**2 - x - y, [x**2 + y**2 <= 1], sense="max", box=(-1, 1))
    for prob in (prob_b, prob_c):
        a = po.solve_alm(prob, order=2, outer=60, inner=300)
        b = po.solve_alm(prob, order=2, outer=60, inner=40, inner_solver="newton")
        assert abs(a.bound - b.bound) < 1e-5 * (1 + abs(a.bound))
        assert b.bound >= po.solve(prob, order=2).bound - 1e-2        # both are certified upper bounds of the relaxation


def test_json_roundtrip_and_cli(tmp_path):
    import json
    from moment_sos.cli import main
    x, y = po.variables(2)
    prob = po.Problem(x**2*y + x*y**2 - x - y, [x**2 + y**2 <= 1, x - y == 0.1], sense="max", box=(-1, 1), name="demo")
    d = po.problem_to_dict(prob)
    json.dumps(d)                                             # serialisable
    prob2 = po.problem_from_dict(d)
    pt = np.array([0.3, -0.4])
    assert abs(prob2.objective(pt) - prob.objective(pt)) < 1e-14 and len(prob2.constraints) == 2 and prob2.sense == "max"
    ref = po.solve(prob, order=2).bound
    path, out = tmp_path / "p.json", tmp_path / "r.json"
    po.save_problem(prob, path)
    assert main(["solve", str(path), "--order", "2", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["bound_kind"] == "upper" and abs(r["bound"] - ref) < 1e-3 and r["order"] == 2
    # Boolean problem with a mask and a rigorous certificate through the CLI
    s = po.variables(4)
    pb = po.Problem(sum((1 - s[i] * s[(i + 1) % 4]) / 2 for i in range(4)), boolean=True)
    po.save_problem(pb, path)
    assert main(["solve", str(path), "--order", "2", "--rigorous", "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert abs(r["bound"] - 4.0) < 1e-2 and "rigorous" in r and r["value"] == 4.0


def test_json_errors():
    with pytest.raises(ValueError):
        po.problem_from_dict({"n": 2})
    with pytest.raises(ValueError):
        po.problem_from_dict({"n": 1, "objective": [{"c": 1.0, "m": [[3, 1]]}], "box": {"lo": -1, "hi": 1}})


def test_term_sparsity_is_valid_and_weaker_than_dense():
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 4)]
    w = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    prob = maxcut(n, edges, w)
    best = max(sum(wi * (1 - (1 - 2 * ((m >> i) & 1)) * (1 - 2 * ((m >> j) & 1))) / 2 for (i, j), wi in zip(edges, w)) for m in range(1 << n))
    dense = Relaxation(prob, 2)
    bounds = {}
    for steps in (1, "max"):
        R = Relaxation(prob, 2, term_sparsity=steps)
        assert max(R.sizes()) < max(dense.sizes())                 # blocks really are smaller
        res = po.solve(prob, order=2, term_sparsity=steps, iters_per_stage=600)
        assert res.bound >= best - 1e-6                            # certificate: valid for every block structure
        x, val = res.round()
        assert val <= res.bound + 1e-6
        bounds[steps] = res.bound
    d = po.solve(prob, order=2, iters_per_stage=600).bound
    assert d <= bounds[1] + 1e-2 and d <= bounds["max"] + 1e-2     # sparse relaxations are weaker (up to solver accuracy)
    assert bounds["max"] <= bounds[1] + 1e-2                       # more closure steps tighten
    # self-consistency with the exact interior-point value of the same dual
    R2 = Relaxation(prob, 2, term_sparsity=1)
    v, _ = po.solve_dual_ipm(R2)
    ipm = prob.f_off + prob.f_scale * v
    assert abs(po.solve_alm(prob, order=2, term_sparsity=1, inner=300).bound - ipm) < 1e-4


def test_precise_etas_lower_the_smoothing_error():
    rng = np.random.default_rng(3)
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 4), (2, 6)]
    w = [float(round(1e4 * rng.standard_normal())) for _ in edges]
    prob = maxcut(n, edges, w)
    exact, _ = po.solve_dual_ipm(Relaxation(prob, 2))
    exact = prob.f_off + prob.f_scale * exact
    e_def = po.solve(prob, order=2).bound - exact
    e_pre = po.solve(prob, order=2, etas=po.PRECISE_ETAS).bound - exact
    assert e_def >= -1e-6 * abs(exact) and e_pre >= -1e-6 * abs(exact)      # both are valid bounds (>= relaxation value)
    assert e_pre <= e_def + 1e-9 * abs(exact)                                # the longer schedule is at least as accurate
    assert e_pre <= 1e-4 * abs(exact)
