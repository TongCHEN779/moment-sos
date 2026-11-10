# Correlative sparsity

If the variables couple only in small groups (a chain, a grid, a network with layers), the single big moment block can
be replaced by one smaller block per **clique** of the correlative-sparsity graph. The relaxation is weaker or equal to
the dense one and far cheaper.

```python
import moment_sos as po

n = 8
x = po.variables(n)
f = sum(x[i]*x[i+1] - 0.5*x[i]**2*x[i+1]**2 for i in range(n - 1))      # chain
problem = po.Problem(f, [], sense="max", box=(-1, 1))
print(po.correlative_sparsity_cliques(problem))

dense = po.solve(problem, order=2, cliques="dense")
sparse = po.solve(problem, order=2, cliques="auto")
print(dense.bound, sparse.bound)
```

`cliques` accepts `"dense"` (one clique), `"auto"` (maximal cliques of a min-fill-in chordal extension of the
correlative-sparsity graph) or an explicit list of index lists. Two requirements, checked with clear error messages:

- every constraint must involve variables from a single clique;
- every objective monomial must be a product of two monomials of one clique.

## Term sparsity

`term_sparsity=k` additionally splits each clique's moment block into overlapping **sub-blocks** (TSSOS-style): a graph
on the monomial basis connects two monomials when their product occurs in the support (objective monomials and the
products arising from the constraints); the blocks are the maximal cliques of a chordal extension, and `k` iterations of
the support closure refine them (`"max"` iterates until the blocks stop changing).

```python
import moment_sos as po

n = 8
s = po.variables(n)
edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 4)]
problem = po.Problem(sum((1 - s[i]*s[j]) / 2 for i, j in edges), boolean=True)

dense = po.Relaxation(problem, 2)
sparse = po.Relaxation(problem, 2, term_sparsity=1)
print(max(dense.sizes()), "->", max(sparse.sizes()), "largest block;", len(sparse.blocks), "blocks")
assert max(sparse.sizes()) < max(dense.sizes())

res = po.solve(problem, order=2, term_sparsity=1)
print(res.bound)
```

What is guaranteed: the bound is valid for **any** block structure (the certificate only needs the polynomial identity and
the norm bound of each sub-basis). What is *not* guaranteed: the relaxation is weaker than the dense (or the
correlatively-sparse) one, and more closure steps tighten it. Warm start is switched off, and only moment blocks are split
(localizing blocks keep their clique size).

:::{warning}
**Term sparsity did not pay off on the problems we tried.** On MAX-CUT with random $\pm1$ weights on a $6\times6$ torus
($n=36$, best cut 30; raw output in `research/experiments/results/term_sparsity_torus6.txt`), order 2:

| relaxation | blocks (largest) | bound | time |
|---|---|---|---|
| dense | 1 (667) | 30.339 | 156 s |
| correlative sparsity | 24 (92) | 30.204 | 28 s |
| term sparsity, 1 step | 422 (212) | 33.853 | 302 s (time limit) |
| correlative + term, 1 step | 566 (5) | 34.715 | 67 s |
| correlative + term, 2 steps | 440 (9) | 32.036 | 77 s |
| correlative + term, until stable | 404 (11) | 32.019 | 53 s |

Only the first two certify the optimum ($\lfloor U\rfloor=30$). Term sparsity gives tiny blocks but a much weaker bound, and with
hundreds of overlapping blocks the smoothing solver is not faster. (The dense value 30.339 is above the correlative one,
30.204, although the dense relaxation is the tighter of the two: the dense run had not converged to solver accuracy.)
Use correlative sparsity first; try term sparsity on problems whose objective and constraints have very few monomials
relative to the basis, and compare the bound before trusting the speed-up.
:::
