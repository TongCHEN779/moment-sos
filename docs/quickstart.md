# Quickstart

## A constrained polynomial problem

Maximize $f(x,y)=x^2y+xy^2-x-y$ over the unit disk:

```python
import moment_sos as po

x, y = po.variables(2)
f = x**2*y + x*y**2 - x - y
problem = po.Problem(f, [x**2 + y**2 <= 1], sense="max", box=(-1, 1))

res = po.solve(problem, order=2)
print(res.bound)             # certified upper bound on max f
pt, val = res.round()        # a feasible point and its value: a lower bound
assert val <= res.bound + 1e-6
```

`order` is the relaxation order $r$; it must satisfy $2r\ge\deg f$ (here $r\ge2$ because $\deg f=3$).
`box` must contain the feasible set: the certificate uses it to size a spectral shift (see {doc}`theory`).

## MAX-CUT on a graph (spins)

Variables with $x_i^2=1$ are declared with `boolean=True`:

```python
import itertools
import moment_sos as po

n = 5
s = po.variables(n)
edges = list(itertools.combinations(range(n), 2))          # K5: the maximum cut is 6
cut = sum((1 - s[i]*s[j]) / 2 for i, j in edges)
problem = po.Problem(cut, boolean=True)

for order in (1, 2):
    res = po.solve(problem, order=order)
    print(order, res.bound)          # 6.25 at order 1 (Goemans-Williamson), 6 at order 2
```

When the optimum is an integer and a feasible point attains $\lfloor\text{bound}\rfloor$, that is a **proof of
optimality**: `floor(res.bound) == value` where `value` comes from `res.round()`.

## Stopping early

Every iterate is a certified bound, so a time limit simply returns the best one so far:

```python
res = po.solve(problem, order=2, time_limit=5.0)
print(res.history[-1])       # (time, evaluations, best certified bound in internal units)
```

Next: {doc}`guide/modeling`, {doc}`guide/solvers`, {doc}`guide/certificates`.
