# Tutorial: equality constraints (graph bisection)

Constraints on spin variables are polynomials too. **Graph bisection** is MAX-CUT with the two sides of equal size,
$\sum_ix_i=0$, an equality constraint:

```python
import networkx as nx
import numpy as np
import moment_sos as po

G = nx.random_regular_graph(3, 10, seed=1)
s = po.variables(10)
cut = sum((1 - s[i]*s[j]) / 2 for i, j in G.edges())

problem = po.Problem(cut, [sum(s) == 0], boolean=True)
res = po.solve(problem, order=2, etas=po.SHORT_ETAS)
x, value = res.round()
print(res.bound, value, "balanced:", int(sum(x)) == 0)
assert int(sum(x)) == 0                       # rounding returns a feasible (balanced) point
assert np.floor(res.bound + 1e-6) >= value
```

`res.round()` only returns points that satisfy the constraints (here a balanced partition), so `value` is a genuine lower
bound for the constrained problem and `bound` an upper bound.

## What the constraint does to the relaxation

The equality $\sum x_i=0$ is imposed as a moment constraint $L_y\big((\sum x_i)\,x^\mu\big)=0$ for all monomials $x^\mu$
that fit in the block. At order 1 this is weaker than it looks: order 1 contains only the products $x_ix_j$, so the
constraint is enforced only through $L_y\big((\sum x_i)x_j\big)=0$, i.e. $\sum_{i\ne j}y_{ij}=-1$. Raising the order enforces it
against more monomials; compare:

```python
import moment_sos as po

for order in (1, 2):
    r = po.solve(problem, order=order, etas=po.SHORT_ETAS)
    print(order, round(r.bound, 4))
```

## Inequality constraints

Inequalities are written with `<=` and `>=`. Their moment relaxation uses *localizing* matrices, an extra PSD block per
inequality of size about $\binom{n+r-1}{r-1}$. Equalities add multipliers but no PSD blocks, so they are cheaper:

```python
import moment_sos as po

x = po.variables(3)
f = x[0]*x[1] + x[1]*x[2] - x[0]*x[2]
problem = po.Problem(f, [x[0] + x[1] + x[2] == 1, x[0]**2 + x[1]**2 + x[2]**2 <= 1], sense="max", box=(-1, 1))
res = po.solve_alm(problem, order=2)
print(res)
```

For problems with many constraints (neural-network verification) prefer `solve_alm`; see {doc}`../guide/solvers`.
