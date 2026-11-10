# Modeling problems

A problem is $\max$ (or $\min$) $f(x)$ subject to $g_i(x)\ge0$, $h_j(x)=0$ and $x$ in a box.

## Polynomials and constraints

`po.variables(n)` returns $n$ coordinate polynomials. Ordinary arithmetic builds polynomials; comparison operators
build constraints:

```python
import moment_sos as po

x = po.variables(3)
f = x[0]*x[1] - 2*x[2]**2 + 1          # objective
g = x[0]**2 + x[1]**2 <= 1             # inequality  1 - x0^2 - x1^2 >= 0
h = x[0] + x[1] + x[2] == 0.5          # equality
print(f.degree(), g.kind, h.kind)
```

Polynomials are sparse dictionaries `{exponent tuple: coefficient}`; `f(point)` evaluates one.

## The `Problem`

```python
problem = po.Problem(f, [g, h], sense="max", box=(-1, 1))
```

| argument | meaning |
|---|---|
| `objective` | a polynomial |
| `constraints` | list of constraints built with `>=`, `<=`, `==` |
| `sense` | `"max"` (the certified bound is an **upper** bound) or `"min"` (a **lower** bound) |
| `box` | `(lo, hi)`, scalars or length-$n$ arrays. **It must contain the feasible set.** |
| `boolean` | `True` for all variables $x_i\in\{-1,+1\}$, or a length-$n$ list of booleans for a mixed problem |
| `add_box_constraints` | add $(hi-x)(x-lo)\ge0$ for every continuous variable (default `True`) |

:::{important}
The certificate relies on the box: it bounds the polynomial $v(x)^\top v(x)$ on the feasible set. If the true
feasible set is *not* inside `box`, the reported bound is not guaranteed. The package rescales the box to $[-1,1]^n$ and
normalizes the scale of the constraints and of the objective internally (for conditioning); results are reported in
your original units.
:::

## Spin (Boolean) variables

`boolean=True` treats every variable as a spin $x_i\in\{-1,+1\}$ and works modulo $x_i^2=1$, which keeps monomials
square-free and makes the moment-block trace constant. A **mixed** problem passes a mask:

```python
import numpy as np
import moment_sos as po

v = po.variables(4)
mask = np.array([True, True, False, False])          # first two are spins, last two continuous in [-1, 1]
problem = po.Problem(v[0]*v[2] + v[1]*v[3] - v[2]**2*v[3], [v[2]**2 + v[3]**2 <= 1],
                     sense="max", box=(-1, 1), boolean=mask)
print(problem)
```

## Choosing the order

`po.min_order(problem)` returns the smallest order that can represent every polynomial of the problem
($r\ge\lceil\deg/2\rceil$). Higher orders give tighter bounds at much higher cost: the moment block has
$s=\binom{n+r}{r}$ rows (square-free: $\sum_{k\le r}\binom nk$ for spins). See {doc}`solvers` for the cost.

```python
print(po.min_order(problem))
```
