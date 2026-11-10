import itertools
import numpy as np
import pytest
import moment_sos as po
from moment_sos.walk import select_triples, triangle_violations, triangle_constraints, pair_monomials


def k_graph(n):
    s = po.variables(n)
    return po.Problem(sum((1 - s[i] * s[j]) / 2 for i, j in itertools.combinations(range(n), 2)), boolean=True)


def test_triangle_violations_known_values():
    Y = np.eye(4)
    Y[0, 1] = Y[0, 2] = Y[1, 2] = -1.0          # 1 + Y01 + Y02 + Y12 = -2 < 0: violation 2
    Y = np.triu(Y) + np.triu(Y, 1).T
    tris, viol = triangle_violations(Y)
    assert tris.tolist() == [[0, 1, 2]] and abs(viol[0] - 2.0) < 1e-6
    assert len(triangle_violations(np.eye(5))[0]) == 0           # feasible matrix: nothing violated
    assert len(triangle_violations(np.eye(2))[0]) == 0


def test_selection_rules():
    rng = np.random.default_rng(0)
    n = 12
    A = rng.standard_normal((n, n))
    Y = np.tanh(2 * (A + A.T))
    np.fill_diagonal(Y, 1.0)
    tris, viol = triangle_violations(Y)
    assert np.all(np.diff(viol) <= 1e-6)                          # sorted by decreasing violation
    top = select_triples(Y, 6, "topk")
    assert np.array_equal(top, tris[:6])
    nms = select_triples(Y, 6, "nms")
    pairs = [p for i, j, k in nms for p in ((i, j), (i, k), (j, k))]
    assert len(pairs) == len(set(pairs)) and len(nms) == 6        # no two selected triples share a pair
    assert tuple(nms[0]) == tuple(tris[0])                        # nms starts from the most violated triple
    rnd = select_triples(Y, 6, "random")
    assert len({tuple(t) for t in rnd}) == 6
    with pytest.raises(ValueError):
        select_triples(Y, 3, "learned")


def test_cuts_and_pair_monomials():
    cons = triangle_constraints(4, [(0, 1, 2)])
    assert len(cons) == 4 and all(c.kind == "ineq" for c in cons)
    for s in itertools.product([-1, 1], repeat=3):                # valid on every spin assignment
        x = np.array([s[0], s[1], s[2], 1.0])
        assert all(c.poly(x) >= -1e-12 for c in cons)
    E = pair_monomials(5, [(0, 1, 2), (0, 1, 3)])
    assert E.shape == (5, 5) and sorted(map(tuple, E.tolist())) == sorted({tuple(int(i in p) for i in range(5)) for p in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]})


def test_extra_basis_grows_block_and_tightens():
    prob = k_graph(5)
    base = po.Relaxation(prob, 1)
    ex = pair_monomials(5, [(0, 1, 2)])
    R = po.Relaxation(prob, 1, extra_basis=ex)
    assert max(R.sizes()) == max(base.sizes()) + 3
    b1 = po.solve_alm(prob, order=1).bound
    b = po.solve_alm(prob, order=1, extra_basis=ex).bound
    b2 = po.solve_alm(prob, order=2).bound
    assert b2 - 1e-6 <= b <= b1 + 1e-6                            # between order 1 and order 2


def cycle(m):
    s = po.variables(m)
    return po.Problem(sum((1 - s[i] * s[(i + 1) % m]) / 2 for i in range(m)), boolean=True)


@pytest.mark.parametrize("mode", ["cuts", "basis"])
@pytest.mark.parametrize("rule", ["nms", "topk"])
def test_walk_closes_the_gap_on_the_5_cycle(mode, rule):
    # C5: max cut 4, order 1 (Goemans-Williamson) 4.5225, order 2 exactly 4
    prob = cycle(5)
    w = po.walk(prob, rounds=4, rule=rule, mode=mode)
    assert abs(w.bounds[0] - 4.5225) < 1e-3
    assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(w.bounds, w.bounds[1:]))     # anytime: bounds never increase
    assert w.bounds[-1] >= 4.0 - 1e-6                                          # always valid
    assert abs(w.bounds[-1] - 4.0) < 1e-3 and w.stopped == "converged"         # the walk reaches order 2 and stops
    x, val = w.result.round()
    assert val == 4.0 and val <= w.bound + 1e-6


def test_walk_on_the_7_cycle_between_order_1_and_2():
    prob = cycle(7)                                   # max cut 6; order 1: 6.6534; order 2: 6
    w = po.walk(prob, rounds=1, rule="nms", mode="cuts")
    assert 6.0 - 1e-6 <= w.bounds[1] < w.bounds[0] - 0.1       # one round of K=n triples already gives a large part of the gap
    assert w.bounds[1] > 6.0 + 0.1                             # but not all of it: a genuine intermediate relaxation


def test_walk_stops_when_nothing_is_violated():
    w = po.walk(k_graph(5), rounds=3)                 # K5: the order-1 optimum satisfies all triangle inequalities
    assert w.stopped == "converged" and len(w.bounds) == 1


def test_walk_errors():
    x = po.variables(3)
    with pytest.raises(ValueError):
        po.walk(po.Problem(x[0] * x[1], sense="max", box=(-1, 1)))                # not a spin problem
    with pytest.raises(ValueError):
        po.walk(po.Problem(x[0] * x[1] * x[2], boolean=True))                      # degree 3
    with pytest.raises(ValueError):
        po.walk(k_graph(4), rule="bogus")
    with pytest.raises(ValueError):
        po.walk(k_graph(4), mode="bogus")
