"""Cross-validation of the two code bases: moment_sos (PyTorch, public package) against msos (NumPy reference with the
independent primal moment model).  moment_sos must reproduce the exact moment-relaxation value (msos.reference, solved by
an interior-point method on the PRIMAL moment problem) within first-order accuracy on every instance below.
Run from tests/:  python -m pytest -q test_crossval.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import itertools
import numpy as np
import pytest
import moment_sos as po
from msos import POP, reference, problems as pb


def to_poly(pop_terms, n):
    return po.Poly(n, {tuple(e): c for e, c in pop_terms.items()})


def from_msos(pop, box=True):
    n = pop.n
    f = to_poly(pop.f, n)
    cons = [po.Constraint(to_poly(g, n), "ineq") for g in pop.ineq] + [po.Constraint(to_poly(h, n), "eq") for h in pop.eq]
    if pop.boolean:
        return po.Problem(f, cons, boolean=True)
    return po.Problem(f, cons, box=(-1, 1), add_box_constraints=False)


CASES = [
    ("K5 r1", pb.maxcut(5, pb.graph("K", 5)), 1),
    ("K5 r3", pb.maxcut(5, pb.graph("K", 5)), 3),
    ("C5 r2", pb.maxcut(5, pb.graph("C", 5)), 2),
    ("Petersen r2", pb.maxcut(10, pb.graph("petersen", 10)), 2),
    ("ball r2", pb.ball(), 2),
    ("disk r2", pb.disk(), 2),
    ("boxqp4 r2", pb.box_qp(4, 1), 2),
]


@pytest.mark.parametrize("name,pop,r", CASES, ids=[c[0] for c in CASES])
def test_polyopt_matches_primal_reference(name, pop, r):
    ref = reference.moment_relaxation(pop, r)[0]
    P = from_msos(pop)
    res = po.solve(P, order=r)
    assert res.bound >= ref - 1e-6, f"{name}: certified bound {res.bound} below the relaxation value {ref}"
    assert res.bound <= ref + 2e-3, f"{name}: first-order bound {res.bound} too far above {ref}"
    resa = po.solve_alm(P, order=r)
    assert abs(resa.bound - ref) < 1e-4, f"{name}: ALM {resa.bound} vs {ref}"
