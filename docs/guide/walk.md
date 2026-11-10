# Adaptive lifting: the triangle walk

For quadratic spin problems (MAX-CUT, Ising, QUBO written in $x_i=\pm1$) the order-2 relaxation is large, but most of
its strength over order 1 comes from a few **triples** of variables. `moment_sos.walk` grows the relaxation one batch of
triples at a time, so you can stop between order 1 and order 2 with a certified bound at every round.

A triple $\{i,j,k\}$ stands for the three pair monomials $x_ix_j,\ x_ix_k,\ x_jx_k$. Adding it makes the relaxation
contain the four **triangle inequalities** of the moment matrix $Y_{ij}=\mathbb E[x_ix_j]$,

$$1+s_1Y_{ij}+s_2Y_{ik}+s_3Y_{jk}\ \ge\ 0,\qquad (s_1,s_2,s_3)\in\{(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)\}.$$

A **round** picks $K$ triples among those violated by the current solution and re-solves. The bound after every round is a
valid, certified bound (it is the bound of a relaxation of the same problem), and bounds never increase.

```python
import moment_sos as po

m = 7
s = po.variables(m)
problem = po.Problem(sum((1 - s[i]*s[(i + 1) % m]) / 2 for i in range(m)), boolean=True)   # 7-cycle: max cut 6

for rule in ("nms", "topk"):
    w = po.walk(problem, rounds=3, rule=rule, mode="basis")
    print(rule, [round(b, 4) for b in w.bounds], "monomials added:", w.sizes[-1], w.stopped)

assert all(b2 <= b1 + 1e-6 for b1, b2 in zip(w.bounds, w.bounds[1:]))
assert abs(w.bounds[0] - 6.6534) < 1e-3 and abs(w.bounds[-1] - 6.0) < 1e-3       # order 1 -> order 2 = the optimum
```

## Selection rules

| `rule=` | which triples |
|---|---|
| `"topk"` | the $K$ most violated (the standard cutting-plane rule) |
| `"nms"` | the most violated, **no two sharing a pair** ("non-maximum suppression") |
| `"random"` | $K$ random triples (baseline) |

`nms` adds diversity: several triples that share a pair tend to cut off the same part of the moment matrix, so
`topk` can spend a whole batch on one defect. The rules and the walk come from the ai4opt study of learned lifting paths,
in which `nms` was consistently better than `topk` and both beat the learned policies at a matched budget.

## Two ways to impose the triples

| `mode=` | what is solved | cost |
|---|---|---|
| `"cuts"` | order-1 relaxation plus the four triangle inequalities of every selected triple, as scalar localizing blocks | cheap blocks, but many of them |
| `"basis"` | order-1 moment block plus the pair monomials of the selected triples (partial order-2 lifting) | the block grows by 3 rows per triple |

`"cuts"` drops triples whose inequalities become inactive after a round (`drop_inactive=True`). The two modes give different
relaxations: the example below on a 7-cycle shows that `"basis"` reaches the optimum in fewer rounds.

```python
import moment_sos as po

m = 7
s = po.variables(m)
problem = po.Problem(sum((1 - s[i]*s[(i + 1) % m]) / 2 for i in range(m)), boolean=True)
cuts = po.walk(problem, rounds=1, rule="nms", mode="cuts")
basis = po.walk(problem, rounds=1, rule="nms", mode="basis")
print("one round:", round(cuts.bounds[1], 4), "(cuts)", round(basis.bounds[1], 4), "(basis)")
assert basis.bounds[1] <= cuts.bounds[1] + 1e-6
```

## Options

```python
w = po.walk(problem, rounds=10, per_round=None, rule="nms", mode="cuts", max_candidates=4000, time_limit=600)
```

- `per_round` is $K$, default the number of variables $n$.
- `max_candidates` is the size of the pool of most violated triples the rules choose from.
- Extra keyword arguments go to `solve_alm` (the primal moment matrix needed to find violations is read from its
  multipliers).
- `WalkResult` has `bounds` (index 0 is order 1), `times`, `sizes` (active triples or pair monomials), `triples`, `result`
  (the last `Result`, for `round()`), `stopped` (`"rounds"`, `"converged"` or `"time"`).

You can also use the pieces directly: `po.triangle_violations(Y)` returns the most violated triples of any matrix `Y`,
`po.select_triples(Y, K, rule)` applies a rule, and `po.solve_alm(..., extra_basis=rows)` adds arbitrary monomials (exponent
rows) to the moment block of a relaxation.

## Limits

- **Only problems whose objective has degree at most 2 and whose variables are all spins.** Other problems raise a clear
  error.
- **Violation-based rules stop when nothing is violated.** On the complete graph $K_5$ the order-1 optimum satisfies every
  triangle inequality, so the walk stops at the order-1 bound even though order 2 is stronger: its extra strength does not
  come from triangle inequalities.
- **It is not fast.** Every round re-solves the relaxation with `solve_alm`; for $n=20$ one round with $K=20$ triples takes
  tens to a few hundred seconds here, and the per-round cost grows with the number of active triples. The ai4opt study
  reports that a first-order ADMM for the same polyhedral relaxation is faster than every cutting-plane rule (their
  implementation; not reproduced here).
- The `"cuts"` relaxation is a polyhedral strengthening of order 1 and is weaker than order 2 (its extra strength is only the triangle inequalities).

## Benchmark (one graph, small)

`G(20, 0.3)`, seed 1 (58 edges), $K=20$ triples per round, 3 rounds, `solve_alm`. Order 1: 45.058; order 2: 43.0000 (= the maximum cut 43). Raw data: `research/experiments/results/walk_gnp20.json`; one machine, shared, single instance.

| mode / rule | round 1 | round 2 | round 3 | total time | size after round 3 |
|---|---|---|---|---|---|
| cuts / nms | 43.803 | 43.462 | 43.211 | 785 s | 60 triples |
| cuts / topk | 44.271 | 43.680 | 43.283 | 1660 s | 60 triples |
| cuts / random | 44.594 | 44.402 | 44.223 | 377 s | 60 triples |
| basis / nms | 43.598 | 43.005 | 43.000 | 2975 s | 149 pair monomials |
| basis / topk | 44.263 | 43.431 | 43.000 | 190 s | 94 pair monomials |

What this does and does not show: `nms` has the tighter bound after every round in both modes, by a large margin in the first
round, and both beat `random`; in `basis` mode both reach the order-2 value after 3 rounds, but `topk` gets there with fewer
monomials (94 vs 149) and much less time (190 s vs 2975 s) on this instance. It is a single graph; no conclusion about
which rule is better overall should be drawn from it.
