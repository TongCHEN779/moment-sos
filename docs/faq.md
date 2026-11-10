# FAQ

**Is the bound really a proof?** It is a valid bound for every iterate under two conditions: the box contains the
feasible set, and floating-point eigenvalue error is negligible. `res.rigorous()` removes the second condition for the
normalized problem. See {doc}`guide/certificates`. Interior-point solvers also return bounds (their dual objective); the
difference is that their dual feasibility holds only up to a tolerance, whereas here every parameter value is feasible
by construction.

**Why not just use TSSOS?** If your relaxation fits an interior-point solver, TSSOS is mature, accurate and has term
sparsity, complex and noncommutative variables; use it. `moment-sos` targets relaxations that do not fit (memory) and
use cases where you want a valid bound at any time. See {doc}`benchmarks`.

**Which solver should I pick?** `solve` for a quick certified bound, Boolean problems and integer optimality proofs;
`solve_alm` for accurate relaxation values and constrained problems (e.g. networks).

**The bound is far above the best feasible value.** Either the relaxation is not tight at this order (try a higher
order), or the solver has not converged (look at `res.history`), or your box is much larger than the feasible set (a
tight box tightens the bound).

**`ValueError: order too low`.** The relaxation needs $2r\ge\deg$ of every polynomial; use `po.min_order(problem)`.

**Can it run on a GPU?** `device="cuda"` is accepted but untested.

**Can I use it from MATLAB / R / C++?** Use the JSON format and the `moment-sos` command ({doc}`guide/cli_json`).
Only Julia has a wrapper.

**How do I cite it?** See `CITATION.cff` in the repository.
