# moment-sos

**Certified first-order bounds for polynomial optimization.** Write a polynomial optimization problem (POP) in a few
lines; `moment-sos` builds its order-$r$ moment–SOS (Lasserre) relaxation and solves the *dual* as an eigenvalue
optimization problem with first-order methods in PyTorch. There is no SDP solver and no Schur complement: memory grows
like $O(s^2)$ in the size $s$ of the moment blocks, and **every iterate is a valid bound**, so you can stop at any
time.

```python
import moment_sos as po

x, y = po.variables(2)
problem = po.Problem(x**2*y + x*y**2 - x - y, [x**2 + y**2 <= 1], sense="max", box=(-1, 1))
res = po.solve(problem, order=2)
print(res)                     # certified upper bound for a max problem
x_hat, value = res.round()     # best feasible point found: the other side of the gap
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: install
:link-type: doc
Install the package and run your first problem.
:::
:::{grid-item-card} User guide
:link: guide/modeling
:link-type: doc
Modeling problems, choosing solvers, sparsity, certificates, neural-network tools.
:::
:::{grid-item-card} Tutorials
:link: tutorials/maxcut
:link-type: doc
MAX-CUT optimality proofs, constraints, minimization, sparsity, neural-network certification.
:::
:::{grid-item-card} Julia and command line
:link: julia
:link-type: doc
`MomentSOS.jl` and the JSON / CLI interface for any language.
:::
:::{grid-item-card} Benchmarks and limits
:link: benchmarks
:link-type: doc
What it is good at, what it is not, measured against MOSEK, TSSOS, SCS and others.
:::
::::

## When to use it

Use `moment-sos` when you want a bound you can trust at any moment and the relaxation is too big for an interior-point
solver (the dense order-2 MAX-CUT relaxation on 60 spins has a $1831\times1831$ moment block; MOSEK is killed at 40
spins on a 68 GiB machine, `moment-sos` needs 2.5 GiB). Use an interior-point solver (MOSEK through TSSOS or
GloptiPoly) when the relaxation fits and you need high accuracy; they are faster on small problems.
See {doc}`benchmarks` for the numbers and for the cases where this package does **not** win.

```{toctree}
:hidden:
:maxdepth: 2

install
quickstart
tutorials/maxcut
tutorials/constrained
tutorials/minimization
tutorials/sparse_chain
tutorials/neural_network
tutorials/convergence
guide/modeling
guide/solvers
guide/certificates
guide/sparsity
guide/walk
guide/neural_networks
guide/cli_json
julia
theory
benchmarks
faq
api
contributing
changelog
```
