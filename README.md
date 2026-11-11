# moment-sos

[![PyPI](https://img.shields.io/pypi/v/moment-sos)](https://pypi.org/project/moment-sos/)
[![CI](https://github.com/TongCHEN779/moment-sos/actions/workflows/ci.yml/badge.svg)](https://github.com/TongCHEN779/moment-sos/actions/workflows/ci.yml)
[![Docs](https://github.com/TongCHEN779/moment-sos/actions/workflows/docs.yml/badge.svg)](https://tongchen779.github.io/moment-sos/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Certified first-order bounds for polynomial optimization, in PyTorch (with a Julia interface and a command line tool).

Write a polynomial optimization problem in a few lines. `moment-sos` builds its order-`r` moment–SOS (Lasserre)
relaxation — with correlative sparsity, equality and inequality constraints, spin (Boolean) and mixed variables — and
solves its **dual as an eigenvalue-optimization problem with first-order methods**. No SDP solver, no Schur complement:
memory is `O(s²)` in the moment-block size `s`, and **every iterate is a valid bound**, so you can stop at any time.

```python
import moment_sos as po

x, y = po.variables(2)
problem = po.Problem(x**2*y + x*y**2 - x - y, [x**2 + y**2 <= 1], sense="max", box=(-1, 1))
res = po.solve(problem, order=2)       # certified upper bound for a max problem (lower bound for min)
print(res)
x_hat, value = res.round()             # a feasible point: the other side of the gap
```

**Documentation:** <https://tongchen779.github.io/moment-sos/> — installation, user guide, tutorials, theory summary,
API reference, and a benchmarks page that states where the package does *not* win. (Source in `docs/`; build with
`sphinx-build -W docs docs/_build/html`.)

## Install

```bash
pip install moment-sos
pip install -e ".[test]"        # from a checkout, to run the tests
```

## Interfaces

| | |
|---|---|
| Python | `import moment_sos as po` |
| Command line | `moment-sos solve problem.json --order 2` (JSON problem format in the docs) |
| Julia | `julia/MomentSOS` — DynamicPolynomials front end that calls the command line tool (Python required) |

## When to use it — and when not

Use it when the relaxation is too large for an interior-point solver (e.g. a dense order-2 MAX-CUT relaxation on 60 spins,
a 1831×1831 moment block: 2.5 GiB here, MOSEK is killed at 40 spins) or when you need a valid bound at any moment, or an
optimality proof for integer problems (all 60 generated and 5 real Biq Mac instances we tried were certified optimal).

Use an interior-point solver (TSSOS, GloptiPoly, SumOfSquares.jl) when the relaxation fits and you need high accuracy:
they are faster below roughly 26 spins, TSSOS has term sparsity and more variable types, and non-certified first-order
solvers (Burer–Monteiro, SCS, CGAL) are faster than this package at every size we tried. Accuracy of the smoothing solver is
moderate (1e-3–1e-4); neural-network relaxations of about 40 unstable neurons were not solved to convergence.

## Status

Version 0.1.0 (alpha), released on PyPI. The test suite (library, triangle walk, and every Python block of the
documentation) runs in CI on Linux and macOS with Python 3.9, 3.11 and 3.12 (CPU only). Not tested: GPU, Windows.
The Julia package is not yet in the General registry; install it from this repository with
`Pkg.add(url="https://github.com/TongCHEN779/moment-sos", subdir="julia/MomentSOS")` (Python and `moment-sos` must be
on the path).

## Citation and license

MIT license. See `CITATION.cff`.

## Repository layout

| path | content |
|---|---|
| `src/moment_sos/` | the Python package |
| `tests/`, `docs/`, `examples/` | tests, Sphinx documentation, example scripts |
| `julia/MomentSOS/` | Julia interface |
| `research/` | manuscript, NumPy reference implementation (`msos`), experiments and raw results behind the benchmark numbers |
