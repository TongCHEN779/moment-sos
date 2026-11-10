# Changelog

## 0.1.0 (unreleased)
- Package renamed `moment-sos` (import `moment_sos`).
- Polynomial front end (`variables`, `Problem`), automatic box normalisation, mixed Boolean/continuous variables.
- Sparse moment-SOS relaxation builder (`Relaxation`), correlative-sparsity cliques (`cliques="auto"`).
- Torch oracle with analytic spectral backward pass; solvers `solve` (smoothing + L-BFGS, optional warm start,
  `stop_at`) and `solve_alm` (dual augmented Lagrangian; `shift="uniform"|"diag"`; experimental
  `inner_solver="newton"`).
- Rigorous a-posteriori certificate `Result.rigorous()` (verified Cholesky, exact rational residual).
- JSON problem/result format (`problem_to_dict`, `problem_from_dict`, `load_problem`, `save_problem`, `result_to_dict`)
  and the `moment-sos` command line tool.
- Julia package `MomentSOS.jl` (wrapper around the command line tool).
- `moment_sos.nn`: ReLU robustness verification and Lipschitz encodings with MILP / PGD / LipSDP references.
- `moment_sos.reference.solve_dual_ipm`: exact interior-point reference of the same dual.
- Term sparsity (`term_sparsity=` in `solve`, `solve_alm`, `Relaxation`; CLI `--term-sparsity`): valid for any block structure, weaker than the dense relaxation; not recommended for MAX-CUT (see the sparsity guide).
- Adaptive lifting `moment_sos.walk` (`walk`, `select_triples`, `triangle_violations`): triangle walk between order 1 and order 2 for quadratic spin problems with the `nms`, `topk` and `random` selection rules, in cut mode and basis mode; `extra_basis=` option of `Relaxation`, `solve`, `solve_alm`. Rules from the ai4opt study.
- Sphinx documentation (`docs/`), executed documentation examples, 18 tests.
