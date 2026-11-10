# Tutorial: polynomial minimization on a box

For `sense="min"`, `res.bound` is a **lower bound** on the minimum. As an example take the Styblinski–Tang function in two
variables,

$$f(x)=\tfrac12\sum_{i=1}^2\big(x_i^4-16x_i^2+5x_i\big),\qquad x\in[-5,5]^2 ,$$

a classic non-convex test function whose global minimum is $-78.3323$ at $x_i\approx-2.9035$.

```python
import moment_sos as po

x = po.variables(2)
f = sum((xi**4 - 16*xi**2 + 5*xi) / 2 for xi in x)
problem = po.Problem(f, sense="min", box=(-5, 5))

res = po.solve(problem, order=2, iters_per_stage=2000)
pt, value = res.round()
print("lower bound:", res.bound, " best point:", pt, value)
assert res.bound <= value + 1e-6                  # lower bound below the feasible value
assert abs(value - (-78.33233)) < 1e-3            # the recovered point is the global minimizer
assert value - res.bound < 0.05                   # and the bound is tight to about 0.02
```

Three things to note.

1. **The degree sets the minimum order.** The objective has degree 4, so `order=2` is the smallest admissible order
   (`po.min_order(problem)` returns 2). A degree-$d$ polynomial needs $2r\ge d$.
2. **The box matters.** The certificate uses the box to bound $\|v(x)\|^2$, and the bound degrades as the box grows
   relative to the region of interest. Use the tightest box you can justify.
3. **A tight bound plus a feasible point is a certificate of near-optimality**: here the minimum lies in
   $[\text{bound},\,\text{value}]$, an interval of width about $0.02$.

## A problem where order 2 is not enough

Raising the order is the only way to tighten the relaxation; when it does not help the problem may need a higher order
than is affordable. Polynomial problems with no structure grow quickly: the moment block has $\binom{n+r}{r}$ rows.

```python
import math

for n in (2, 5, 10):
    print(n, [math.comb(n + r, r) for r in (1, 2, 3)])
```

Exploit structure when you can ({doc}`sparse_chain`).
