# Tutorial: MAX-CUT and optimality proofs

MAX-CUT asks for a partition of the vertices of a graph that maximizes the number (or weight) of cut edges. With
spins $x_i\in\{-1,+1\}$ the cut value is $\sum_{ij\in E}w_{ij}(1-x_ix_j)/2$, a quadratic polynomial, so it is a POP with
`boolean=True`. This tutorial shows how a relaxation bound becomes a **proof of optimality**.

## Order 1 sometimes suffices

Order 1 is the Goemans–Williamson / Shor SDP. On the Petersen graph (maximum cut 12) it already proves optimality:

```python
import networkx as nx
import moment_sos as po

G = nx.petersen_graph()
s = po.variables(G.number_of_nodes())
problem = po.Problem(sum((1 - s[i]*s[j]) / 2 for i, j in G.edges()), boolean=True)

res = po.solve(problem, order=1)
x, value = res.round()                  # best cut found: a lower bound
print(res.bound, value)                 # 12.5 and 12
assert int(res.bound) == value == 12    # the cut is at most floor(12.5) = 12 and a cut of 12 exists: optimal
```

The logic: for integer weights the optimum is an integer, so `floor(bound)` is also an upper bound. If a feasible point
attains it, that point is **optimal, and the pair (bound, point) is the proof**.

## When order 1 is not enough

On a random graph the order-1 bound is often too weak, and the order-2 relaxation (a much bigger block, size
$1+n+\binom n2$) is needed:

```python
import networkx as nx
import numpy as np
import moment_sos as po

G = nx.gnp_random_graph(20, 0.3, seed=3)
s = po.variables(20)
problem = po.Problem(sum((1 - s[i]*s[j]) / 2 for i, j in G.edges()), boolean=True)

r1 = po.solve(problem, order=1)
x, value = r1.round()
print("order 1:", round(r1.bound, 3), " best cut:", value)          # 39.1 vs 38: floor(39.1) = 39 != 38
assert np.floor(r1.bound + 1e-9) > value                            # order 1 cannot prove optimality here

r2 = po.solve(problem, order=2, etas=po.SHORT_ETAS, stop_at=value + 1 - 1e-9)
print("order 2:", round(r2.bound, 4))                                # 38.876
assert np.floor(r2.bound + 1e-9) == value                           # proved optimal
```

Two options make this fast:

- `etas=po.SHORT_ETAS` starts the smoothing continuation coarser; for 0/1 and ±1-weighted MAX-CUT this certifies integer
  optima about 20× faster than the default schedule (not tested on other problem classes).
- `stop_at=value + 1 - 1e-9` stops the moment the certified bound drops below `value + 1`, i.e. as soon as optimality is
  proved. You do not wait for convergence.

## Making the proof airtight

`res.bound` is exact up to the eigensolver's floating-point error. For a proof that holds in exact arithmetic use the
rigorous certificate:

```python
rig = r2.rigorous()
print(rig["ok"], rig["bound"])
assert rig["ok"] and np.floor(rig["bound"]) == value
```

See {doc}`../guide/certificates` for what it certifies (the *normalized* floating-point problem; for unit weights that
is exact).

## Weighted graphs

Weights are ordinary coefficients: `sum(w[i, j] * (1 - s[i]*s[j]) / 2 for ...)`. The optimum is then an integer only for
integer weights; with non-integer weights compare `res.bound` with `value` and read the gap, there is no rounding down.
On instances with weights spanning many orders of magnitude (the Biq Mac torus instances have weights up to $10^5$) the
smoothing solver is noticeably less accurate; see {doc}`../benchmarks`.
