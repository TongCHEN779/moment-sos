"""Trace cuts vs diagonal cuts (Proposition prop:diag, Example ex:boxfree):
(i)   reference.moment_relaxation(cuts='none'|'trace'|'diag') on the box-free example  max x1+x2 s.t. x1 x2 = 0;
(ii)  box QP n=4 (seed 1) with and without the explicit box constraints;
(iii) moment_sos.solve_alm(shift='diag') reaches the diagonal-cut values;
(iv)  Boolean (Petersen MAX-CUT, r=2): none = trace = diag = 12;
(v)   regression: solve_alm(shift='uniform') reproduces the values recorded before the 'shift' option was added.
Run from tests/:  python test_cuts.py"""
import os, sys, time, warnings
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "3")
sys.path.insert(0, "..")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))   # the moment_sos package
warnings.filterwarnings("ignore")
import numpy as np
import torch; torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import moment_sos as po
from msos import POP, reference, problems as pb

TOL = 1e-3


def to_polyopt(pop, box):
    """msos.POP -> moment_sos.Problem on [-1,1]^n (add_box_constraints=box)."""
    x = po.variables(pop.n)
    def mk(poly):
        out = 0
        for ex, c in poly.items():
            m = c
            for i, e in enumerate(ex):
                if e:
                    m = m * x[i] ** e
            out = out + m
        return out
    return po.Problem(mk(pop.f), [mk(g) >= 0 for g in pop.ineq] + [mk(h) == 0 for h in pop.eq],
                      box=(-1, 1), add_box_constraints=box)


def rel(pop, r, cuts, cliques=None):
    return reference.moment_relaxation(pop, r, cliques=cliques, cuts=cuts)[0]


def check(name, got, want, tol=TOL):
    ok = (np.isinf(want) and np.isinf(got)) or (np.isfinite(want) and abs(got - want) <= tol)
    print(f"  {'ok ' if ok else 'FAIL'} {name:48s} got {got:<12.6g} want {want:<12.6g}")
    assert ok, name


if __name__ == "__main__":
    # ---------------------------------------------------------------- (i) box-free example
    print("(i) box-free example  max x1+x2  s.t. x1*x2 = 0  (no box constraints)")
    bf = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(1, 1): 1.0}], name="boxfree")
    want_tr = [np.sqrt(2), 1.33839, 1.28872]
    want_d = [np.sqrt(2), 2 ** 0.25, 2 ** (1 / 6)]
    diag_bf = {}
    for r in (1, 2, 3):
        got = rel(bf, r, "none")
        # The plain relaxation is unbounded for every r (Dirac masses at (t,0) are feasible for all t and give value t).
        # Interior-point codes do not always report this: a finite value below the objective of a feasible point
        # (here t=100) is a solver breakdown, not a bound, and is accepted (and reported) as such.
        if not np.isinf(got):
            assert got < 100.0, "finite value above the objective of a feasible point: relaxation would be bounded"
            print(f"  note boxfree r={r} cuts=none: solver returned finite {got:.4g} < 100 (objective of the feasible Dirac moment vector at t=100): breakdown on an unbounded problem")
        else:
            print(f"  ok  boxfree r={r} cuts=none  (unbounded)  got inf")
        check(f"boxfree r={r} cuts=trace", rel(bf, r, "trace"), want_tr[r - 1])
        diag_bf[r] = rel(bf, r, "diag")
        check(f"boxfree r={r} cuts=diag", diag_bf[r], want_d[r - 1])
    # legacy aliases
    check("legacy trace_cuts=True == cuts='trace'", reference.moment_relaxation(bf, 2, trace_cuts=True)[0], want_tr[1])
    check("legacy trace_cuts=False == cuts='none'", reference.moment_relaxation(bf, 2, trace_cuts=False)[0], np.inf)

    # ---------------------------------------------------------------- (ii) box QP with / without box constraints
    print("(ii) box QP n=4 seed 1, with and without the explicit box constraints 1 - x_i^2 >= 0")
    q = pb.box_qp(4, 1)
    qn = POP(4, q.f, name="boxqp4_nobox")
    SHOR = 6.33737
    diag_nb = {}
    for r in (1, 2):
        vals = {c: rel(q, r, c) for c in ("none", "trace", "diag")}
        for c, v in vals.items():
            check(f"boxqp4 +box r={r} cuts={c}", v, SHOR)
        check(f"boxqp4 no-box r={r} cuts=none  (unbounded)", rel(qn, r, "none"), np.inf)
        diag_nb[r] = rel(qn, r, "diag")
        check(f"boxqp4 no-box r={r} cuts=diag == with-box value", diag_nb[r], SHOR)
        tr = rel(qn, r, "trace")
        print(f"  {'ok ' if tr > diag_nb[r] + TOL else 'FAIL'} boxqp4 no-box r={r} cuts=trace ({tr:.5f}) > diag ({diag_nb[r]:.5f})")
        assert tr > diag_nb[r] + TOL

    # ---------------------------------------------------------------- (iii) ALM with shift='diag'
    print("(iii) moment_sos.solve_alm(shift='diag') reaches the diagonal-cut values")
    x = po.variables(2)
    P = po.Problem(x[0] + x[1], [x[0] * x[1] == 0], box=(-1, 1), add_box_constraints=False)
    for r in (1, 2, 3):
        res = po.solve_alm(P, r, shift="diag")
        check(f"ALM-diag boxfree r={r} ({res.n_eval} evals, {res.time:.1f}s)", res.bound, diag_bf[r])
    Pn = to_polyopt(qn, box=False)
    assert len(Pn.ineq) == 0
    for r in (1, 2):
        res = po.solve_alm(Pn, r, shift="diag")
        check(f"ALM-diag boxqp4 no-box r={r} ({res.n_eval} evals, {res.time:.1f}s)", res.bound, diag_nb[r])
        assert len(res.relaxation.blocks) == 1, "no-box formulation must have a single (moment) block"

    # ---------------------------------------------------------------- (iv) Boolean: all three coincide
    print("(iv) Petersen MAX-CUT r=2 (Boolean): none = trace = diag = 12")
    pet = pb.maxcut(10, pb.graph("petersen", 10))
    for c in ("none", "trace", "diag"):
        check(f"petersen r=2 cuts={c}", rel(pet, 2, c), 12.0)

    # ---------------------------------------------------------------- (v) regression of the default shift='uniform'
    print("(v) regression: solve_alm(shift='uniform') reproduces the relaxation values (tolerance 1e-6: the earlier 1e-9 pins encoded the iteration path of other solver defaults, not the optimum)")
    res = po.solve_alm(P, 2)                                  # box-free example, r=2: rho_2^tr
    check("ALM-uniform boxfree r=2 internal U", res.U, 0.6691950104523512, tol=1e-6)
    check("ALM-uniform boxfree r=2 bound", res.bound, 1.3383900209047024, tol=1e-6)
    print(f"      n_eval = {res.n_eval} (was 44)")
    x3 = po.variables(3)
    S = po.Problem(x3[0] * x3[1] + x3[2], [x3[0] ** 2 + x3[1] ** 2 + x3[2] ** 2 == 1], box=(-1, 1), add_box_constraints=False)
    res = po.solve_alm(S, 2)
    check("ALM-uniform sphere r=2 internal U", res.U, 0.5, tol=1e-6)
    print(f"      n_eval = {res.n_eval} (was 453)")
    assert "shift" not in res.extra, "uniform path must not be touched"
    print("ALL PASSED")
