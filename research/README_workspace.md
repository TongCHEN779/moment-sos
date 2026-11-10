# Anytime-certified first-order bounds for the moment–SOS hierarchy

Paper: `moment-sos.tex` → `moment-sos.pdf` (run `pdflatex` three times; `moment-sos_prev.pdf` is the previous version).

* `msos/` – package: `core.py` (Gram-matrix dual, cliques, equalities, Boolean mode, exact + smoothed eigen-oracle,
  primal recovery `primal_moments` / `boolean_lower_bound` / `two_sided`), `solvers.py` (L-BFGS + eta-continuation,
  accelerated gradient, target-level subgradient, dual augmented Lagrangian `alm`), `verify.py` (rigorous a posteriori
  certificate: verified Cholesky eigenvalue bound + exact rational residual, `rigorous_bound`),
  `reference.py` (independent interior-point moment relaxation, `cuts="none"|"trace"|"diag"`; MOSEK if installed,
  otherwise Clarabel; override with `MSOS_SOLVER`), `problems.py` (instances).
* `tests/` – `test_identity.py` (polynomial identity), `test_smoothing.py` (sandwich bound), `test_verify.py`
  (rigorous certificate), `test_primal.py` (primal recovery, gradient identities, two-sided bound), `test_cuts.py`
  (trace vs diagonal cuts, diagonal-shift ALM), `test_alm.py` (NumPy ALM vs moment_sos ALM vs interior point),
  `quick*.py` (smoke / gradient checks). All are plain scripts: `cd tests && python test_x.py`.
* `experiments/` – `e1_correctness.py`, `e2_convergence.py`, `e2b_agd.py`, `e3_scaling.sh` (+`run_one.py`),
  `e4_higher_order.py`, `e4b_r2_tight.py`, `e5_structure.py`, `e7_rigor.py`, `e8_primal.py` (+`e8b_tight.py`),
  `e9_cuts.py`, `e10a_alm_maxcut.py`, `e10b_variability.py`; `make_figures.py`, `make_tables.py`, `make_tables_e7.py`,
  `make_tables_e8.py`, `make_tables_e8b.py`, `make_tables_e9.py`, `make_tables_e10.py`, `make_notes_e10.py`
  regenerate `figures/` and `tables_*.tex` from `experiments/results/`. Notes for E7–E10 are in
  `experiments/results/e*_notes.md`.
* the PyTorch library now lives at the repository root (`src/moment_sos`): write a polynomial optimization
  problem, get certified bounds; `moment_sos.nn` has the ReLU-network encodings; `solve_alm(shift="diag")` computes the
  diagonal-cut relaxation.
* `experiments/baselines/` – SCS, CGAL, Burer–Monteiro comparisons (`results/baselines.json`); `experiments/nn/` –
  neural-network experiments N1–N3 (`results/nn_*.json`), `make_tables_nn.py` builds `tables_n*.tex`.
* `legacy/` – first version of the note and its JAX/SCS prototype.

Notes: MOSEK at n=40, r=2 was killed by the OS (no result, see paper). Timings come from a shared machine.
Experiments E7–E10 (October 2026 revision) used Clarabel as the interior-point reference; see `CHANGES.md`.
