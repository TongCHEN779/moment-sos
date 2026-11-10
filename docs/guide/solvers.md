# Solvers

Two solvers share the same eigenvalue oracle (one batched dense `eigh` per block and iteration).

Examples on this page use the order-2 MAX-CUT relaxation of the complete graph on 6 spins:

```python
import itertools
import moment_sos as po

s = po.variables(6)
problem = po.Problem(sum((1 - s[i]*s[j]) / 2 for i, j in itertools.combinations(range(6), 2)), boolean=True)
```

## `solve`: entropic smoothing + L-BFGS (anytime)

```python
res = po.solve(problem, order=2, cliques="dense", time_limit=60)
```

Minimizes the entropically smoothed bound $U_\eta$ with L-BFGS and $\eta$-continuation
($\eta=0.1\to10^{-5}$ by default). It is the best choice for a **quick certified bound** and for Boolean problems such
as MAX-CUT. The value you get is always a hard certified bound $U(\theta)$ of the best iterate, not the smoothed value.

Useful options:

| option | effect |
|---|---|
| `time_limit` | seconds; returns the best certified bound so far |
| `etas=po.PRECISE_ETAS` | four more smoothing stages (down to $10^{-7}$): lowers the smoothing error by about two orders of magnitude for roughly 40% more time. On a $5\times5$ torus with integer weights $\sim10^5$ the error drops from $2.2\times10^{-3}$ to $1.5\times10^{-5}$ (relative) |
| `etas=po.SHORT_ETAS` | coarse start of the schedule; certifies integer optima of 0/1 and ±1-weighted MAX-CUT about 20× faster (not tested on other problem classes) |
| `stop_at=value` | stop as soon as the certified bound is at least that good, e.g. `best_cut + 1 - 1e-9` to certify an integer optimum |
| `iters_per_stage` | L-BFGS iterations per smoothing value (default 500) |
| `warm_start=True` | solve order $r-1$ first and embed it (default) |
| `device` | `"cpu"` (default); `"cuda"` is untested |

**Accuracy.** The smoothing bias is about $\eta\cdot\sum_b w_b\log s_b$, so the relative accuracy is typically
$10^{-3}$–$10^{-4}$, and it degrades when the block weights span orders of magnitude (see {doc}`../benchmarks`).

## `solve_alm`: dual augmented Lagrangian

```python
res = po.solve_alm(problem, order=2, cliques="auto", rho0=1000, time_limit=60)
```

A boundary-point (SDPNAL-style) method: smooth inner problems solved by L-BFGS-B, projection of the multiplier
matrices onto the PSD cone in the outer loop. There is no smoothing bias, so it reaches much higher accuracy
($10^{-8}$ on order-2 MAX-CUT up to 30 spins), and it is the better choice **when the relaxation value itself is
wanted** and for problems with many inequality or equality constraints (ReLU networks). It is slower to a first useful
bound and depends on the penalty parameter `rho0`.

| option | effect |
|---|---|
| `rho0` | initial penalty (default 1000; doubled when infeasibility stalls) |
| `shift="diag"` | one diagonal shift per block: computes the diagonal-cut relaxation (tighter for non-Boolean problems) |
| `warm_start` | embed the full state of order $r-1$; helps small problems, did not help at $n=40$ |
| `inner_solver="newton"` | experimental semismooth Newton–CG for the inner problems; **not competitive** in our tests (see {doc}`../benchmarks`) |

:::{warning}
`inner_solver="newton"` is experimental: on a 40-variable verification problem it did not converge in 25 minutes,
where the default L-BFGS-B inner solver reached 78% of the gap. Prefer the default.
:::

## Reading the result

```python
res.bound          # certified bound in the original sense (upper for max, lower for min)
res.bound_kind     # "upper" or "lower"
res.history        # [(time, evaluations, best certified value in internal units), ...]
res.time, res.n_eval
res.round()        # (x, value): best feasible point found, or (None, None)
res.rigorous()     # exact-arithmetic verified bound (see Certificates)
```

## Cost of an order

The expensive object is the moment block of size $s$ (a dense symmetric $s\times s$ matrix): memory $O(s^2)$ and time
$O(s^3)$ per iteration for the eigendecomposition, which takes 73–89% of the time. In our experiments blocks up to
$s\approx1800$ (60 spins at order 2) were practical on the machine we used (a Mac with 68 GiB of free memory); $s\gtrsim2500$ is beyond the dense eigensolver. Use
correlative sparsity (next page) to split large problems into many small blocks.
