# Tutorial: reading convergence, choosing options

## The anytime history

`Result.history` stores `(seconds, evaluations, best certified value)` along the run. The value is in the *internal*
normalized units; `Result.bound` is in your units. To see the bound over time in your units:

```python
import itertools
import numpy as np
import moment_sos as po

n = 8
s = po.variables(n)
rng = np.random.default_rng(0)
W = rng.choice([-1.0, 0.0, 1.0], size=(n, n))
f = sum(W[i, j] * (1 - s[i]*s[j]) / 2 for i, j in itertools.combinations(range(n), 2))
problem = po.Problem(f, boolean=True)

res = po.solve(problem, order=2)
P = problem
bound_over_time = [(t, P.f_off + P.f_scale * u) for t, _, u in res.history]       # max problem: internal -> original
assert abs(bound_over_time[-1][1] - res.bound) < 1e-9
assert all(b >= a - 1e-9 for a, b in zip([b for _, b in bound_over_time][1:], [b for _, b in bound_over_time]))   # non-increasing
print(len(res.history), "points; first", round(bound_over_time[0][1], 3), "last", round(bound_over_time[-1][1], 5))
```

The best certified bound never increases, so you can stop whenever the bound is good enough for your purpose.

## Picking the solver and the schedule

| situation | recommendation |
|---|---|
| quick certified bound, MAX-CUT-like Boolean problem | `solve` (default) |
| integer optimum expected | `solve(..., etas=po.SHORT_ETAS, stop_at=best+1-1e-9)` |
| need the relaxation value to $10^{-6}$ or better | `solve_alm` |
| many inequality or equality constraints | `solve_alm` |
| big problem with weak coupling | `cliques="auto"` |
| huge weights spanning orders of magnitude | expect lower accuracy from `solve`; try `solve_alm` |

## Checking a result against an interior-point solver

For problems small enough, `po.solve_dual_ipm` solves the same dual exactly with MOSEK (or Clarabel when MOSEK is not
installed). It needs `cvxpy` and a solver (`pip install cvxpy clarabel`):

```python
import itertools
import moment_sos as po

s = po.variables(6)
problem = po.Problem(sum((1 - s[i]*s[j]) / 2 for i, j in itertools.combinations(range(6), 2)), boolean=True)

res = po.solve_alm(problem, order=2)
relax = po.Relaxation(problem, 2)
value, seconds = po.solve_dual_ipm(relax)
value = problem.f_off + problem.f_scale * value               # back to original units
print(res.bound, value)
assert abs(res.bound - value) < 1e-4
```

## Common problems

- **The bound barely moves from the order-1 value.** At order 2 with the smoothing solver, the first stages may sit at
  the order-1 value until $\eta$ is small; raise `time_limit` or use `solve_alm`.
- **The bound is far from the feasible value.** Increase the order, tighten the box, or check `res.history` for
  convergence.
- **`ValueError` mentioning cliques.** A constraint or objective monomial spans two cliques; merge them or use `"dense"`.
- **Memory.** The moment block takes $8s^2$ bytes per copy and the solver keeps several; budget about $50s^2$ bytes.
