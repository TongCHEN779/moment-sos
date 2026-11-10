# Certificates and primal recovery

## What is certified

For a maximization problem, `res.bound` is an **upper bound** on the optimum of your problem: for every feasible $x$,
$f(x)\le U(\theta)$, where $\theta$ is the dual point at the best iterate. This holds for *every* $\theta$, not only at
convergence, because the dual constraint (a polynomial identity) is satisfied by construction and the positive
semidefiniteness of the Gram matrices is restored by a spectral shift whose cost is added to the bound
(see {doc}`../theory`). For a minimization problem `res.bound` is a **lower bound**.

Two caveats, both important:

1. **The box must contain the feasible set** (see {doc}`modeling`).
2. **Floating point.** `res.bound` is exact up to the error of the dense symmetric eigensolver (about
   $10^{-12}$ relative in our tests). If you need a bound that is valid in exact arithmetic, use `res.rigorous()`.

## The rigorous certificate

```python
import itertools
import moment_sos as po

s = po.variables(6)
cut = sum((1 - s[i]*s[j]) / 2 for i, j in itertools.combinations(range(6), 2))
res = po.solve(po.Problem(cut, boolean=True), order=2)

out = res.rigorous()
print(out["ok"], out["bound"], out["time"])
```

`rigorous()` replaces the floating-point $\lambda_{\min}$ by a *verified* lower bound (a Cholesky factorization of
the shifted matrix with a rigorous rounding-error allowance, following Higham) and evaluates the polynomial-identity
residual in exact rational arithmetic. The returned dictionary has `bound` (the verified bound for your problem),
`U_rig`, `ok` (whether every block was verified) and `time`. It certifies the **normalized** floating-point problem
that `Problem` stores: rescaling the box to $[-1,1]^n$ rounds the coefficients, and this rounding is not accounted for.
For data that stay exact, such as unit-weight MAX-CUT, the statement is exact. It was run up to blocks of size 821 in
our experiments.

## Primal recovery and optimality proofs

```python
x, value = res.round()
```

`round()` recovers a feasible point from the dual iterate (for Boolean problems: random rounding of the recovered
moment vectors plus 1-flip local search; for continuous problems, the recovered first moments, checked for feasibility).
It returns `(None, None)` if no feasible point is found. Its value is a *lower* bound for a max problem, so together
with `res.bound` it brackets the optimum:

$$\text{value}\ \le\ f^\star\ \le\ \text{res.bound}.$$

If the optimum is an integer (e.g. unit-weight MAX-CUT) and `floor(res.bound) == value`, `value` is **proved
optimal**. This is how all 60 generated and all 5 real Biq Mac `g05_60` instances in our benchmarks were certified.

:::{note}
The bracket can be loose for relaxations that are not tight. On the Biq Mac torus instance `t2g10` the relative gap
between the best cut and the order-2 bound is 3.0%, so no optimality
proof is obtained there.
:::

## Bound versus relaxation value

`res.bound` is a bound on the POP optimum $f^\star$ and is at least the order-$r$ relaxation value $\rho_r$ (up to the
solver accuracy). If you want to *reproduce $\rho_r$ to high accuracy* use `solve_alm` and compare with the
interior-point reference `po.solve_dual_ipm(res.relaxation)` on problems small enough for it.
