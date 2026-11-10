# Tutorial: correlative sparsity on a chain

When each term of the objective involves only a few neighbouring variables, a single dense moment block is wasteful. A
chain $f=\sum_i\big(x_ix_{i+1}-\tfrac12x_i^2x_{i+1}^2+0.3x_i^3\big)$ couples only neighbours.

```python
import time
import moment_sos as po

n = 12
x = po.variables(n)
f = sum(x[i]*x[i+1] - 0.5*x[i]**2*x[i+1]**2 + 0.3*x[i]**3 for i in range(n - 1))
problem = po.Problem(f, sense="max", box=(-1, 1))

cliques = po.correlative_sparsity_cliques(problem)
print(len(cliques), "cliques of size", {len(c) for c in cliques})       # 11 cliques of size 2

out = {}
for name in ("dense", "auto"):
    t = time.time()
    res = po.solve(problem, order=2, cliques=name, iters_per_stage=500)
    out[name] = (res.bound, time.time() - t, max(res.relaxation.sizes()))
    print(name, "bound", round(res.bound, 5), "largest block", out[name][2], f"{out[name][1]:.1f}s")

assert out["auto"][2] < out["dense"][2]                          # much smaller blocks
assert abs(out["auto"][0] - out["dense"][0]) < 1e-2              # same bound to within solver accuracy
```

The sparse relaxation is at least as weak as the dense one in theory (it relaxes fewer constraints), but here the bounds
agree to solver accuracy while the largest block is far smaller. The saving grows quickly with $n$: the dense block is
$\binom{n+2}{2}$, the sparse blocks stay at $\binom{4}{2}=6$.

## Requirements

- Each constraint must involve variables from **one** clique, otherwise `moment_sos` raises an error that names the
  offending constraint.
- Each objective monomial must be a product of two monomials from a single clique. For quadratic MAX-CUT on a sparse
  graph that holds for any chordal-extension cliques.

You can also pass your own cliques, e.g. for a layered structure:
`po.solve(problem, order=2, cliques=[[0, 1, 2], [2, 3, 4]])`.

:::{note}
Term sparsity (TSSOS) exploits sparsity *within* a clique; see the end of {doc}`../guide/sparsity` for the (mixed) results.
:::
