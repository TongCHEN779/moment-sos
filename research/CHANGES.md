# Changes — October 2026 revision

All numbers below are copied from `experiments/results/*.json` / `*_notes.md`. Nothing was deleted or renamed;
`moment-sos_prev.pdf` is the PDF before this revision. Interior-point reference values for the new experiments come
from Clarabel (open source) because MOSEK was not available on the machine used; all earlier MOSEK results are unchanged.

## Theory (moment-sos.tex)

1. **Theorem 2 (exactness), proof fixed.** The Boolean case claimed "the cut is an identity"; this is true for the
   moment-block cut `tr M_k = s_k` but not for the localizing cut `tr M_i^loc <= s_i B_i`. Added the argument that the
   localizing cut is implied by `|y_alpha| <= 1` (off-diagonal entries of a moment block with unit diagonal), so both
   cuts are redundant and `rho_r^tr = rho_r` in the Boolean case.
2. **Proposition "diagonal domination and diagonal cuts" (new, after Prop. "weighted domination").** Minimizing the
   weighted domination over the weights gives the diagonal domination `U^d <= U`, whose infimum is the diagonal-cut
   relaxation `rho_r^d`, with `f* <= rho_r^d <= rho_r^tr <= rho_r` (all equal in the Boolean case); `U^d` is convex but
   not an eigenvalue function; the ALM applies with `t_b I -> diag(d_b)`.
3. **Example "the box is not implied by the constraints" (new).** `max x1+x2 s.t. x1 x2 = 0` on the box: Lasserre
   unbounded at every order, trace cuts finite (1.414, 1.338, 1.289), diagonal cuts tighter (1.414, 2^{1/4}, 2^{1/6}),
   true optimum 1; for box QPs without explicit box constraints, `rho_1^d` is the Shor relaxation with box.
4. **Section "A rigorous a posteriori certificate" (new, replaces the remark on rigor).** Lemma (verified lower bound
   on lambda_min from a floating-point Cholesky factorization, Higham Thm. 10.3) and Theorem (rigorous certificate
   `U_rig = sigma + sum s_k Phi_k(-ell_k) + sum s_i B_i [-ell_i]_+ + E` with `E` the exact rational residual of the
   Gram identity).
5. **Section "Primal recovery and a two-sided certificate for the relaxation" (new).** Proposition: (a) variational
   form of `U_eta` and the Lagrangian identity `L_eta = <theta, g> + L_ybar(f) + eta sum w_b H_b`; (b) the gradient of
   `U_eta` is componentwise the infeasibility of the slot-averaged moment vector `ybar` (`g_sigma = 1 - ybar_0`,
   `g_Theta` = zero-sum projection of `-(omega X)`, `g_u = omega (L_ybar(g x^a x^b) - X_i)`, `g_p = L_ybar(h x^mu)`),
   so stationarity = feasibility and then `L_ybar(f) <= rho^tr <= U` with an explicit gap; (c) `sqrt(sum_k ||M_k(ybar)
   - X_k||_F^2) <= ||g_Theta||`; (d) for Boolean problems without constraints a two-sided certificate
   `(1-tau) L_ybar(f) + tau f_0 <= rho_r <= U(theta)` from one dual point. The signs in (b) were corrected after the
   unit test `tests/test_primal.py` exposed them.
6. Contributions list, abstract, Section 5.5 remark, Software, Discussion (limitations 2, 4, 8, new 9), Outlook and
   Reproducibility updated accordingly; counts "24 problem–order pairs" corrected to 23 (the number of rows of Table 1);
   `aliascnt` added so that cleveref prints "Proposition/Lemma/Corollary" instead of "Theorem" for the shared counter
   (a pre-existing rendering bug with current LaTeX); references Higham 2002, Jansson–Chaykin–Keil 2008,
   Peyrl–Parrilo 2008 added.

## Experiments (new sections 6.7–6.10, tables 6–12, figures 5–6)

* **E7 rigorous certificates** (`experiments/e7_rigor.py`, `tables_e7.tex`, `results/e7.json`, `e7_notes.md`):
  23 instances of Table 1 + order-2 MAX-CUT n=10..40. `U_rig >= U` always; `U_rig - U` between 4e-14 and 7.2e-9
  (dominated by the chosen Cholesky margin 1e-10|lmin|, not by the backward error); `E <= 1.8e-11`; verification
  0.07 s + 0.14 s at s=821 vs 559 s solve; all 22 MAX-CUT optimality certificates hold in exact arithmetic.
* **E8 primal recovery** (`e8_primal.py`, `e8b_tight.py`, `tables_e8.tex`, `tables_e8b.tex`, `figures/primal_gap.pdf`,
  `results/e8.json`, `e8_notes.md`): the bracket `LB <= rho_2 <= U` held at every evaluation; the recovered primal
  value reaches `|L_ybar(f) - rho_2| < 1e-3` before `U - rho_2 < 1e-3` on every instance and 1e-4 at n=24,30 while U
  is still 1e-2 away; the lower bound of the default run is limited by the gradient norm (rho - LB = 4e-3..8e-2);
  running L-BFGS to stationarity at eta = 3e-4 brings rho - LB to 3e-6..4e-5 (n <= 20) so that U - LB equals U - rho
  within 10%. References for n >= 20 are the integer max cuts (confirmed by the bracket itself to <= 2.7e-3).
* **E9 cuts** (`e9_cuts.py`, `tables_e9a.tex`, `tables_e9b.tex`, `results/e9.json`, `e9_notes.md`): with the box
  encoded or Boolean the three relaxations coincide (<= 2e-8); without the box, Lasserre is unbounded, trace cuts are
  loose, diagonal cuts reproduce the with-box value exactly with one block instead of n+1 (23–33% fewer multipliers,
  bound equal to <= 1.7e-6 for box QP n=4..10). Smoothing within 1.2e-4 of rho^tr, ALM-diag within 6.8e-7 of rho^d.
* **E10a ALM on MAX-CUT** (`e10a_alm_maxcut.py`, `tables_e10a.tex`, `figures/alm_maxcut.pdf`, `results/e10a.json`,
  `e10_notes.md`): the dual ALM reaches certified gaps between 4e-11 and 7e-8 (converged runs) on n=16..30 where smoothing stalls at 3e-4..3e-3;
  smoothing is faster to 1e-1 (and to 1e-3 at n=20); rho0=1000 is the best setting and the only one converging at
  n=30 within 900 s.
* **E10b variability** (`e10b_variability.py`, `tables_e10b.tex`, `results/e10b.json`): five random 3-regular graphs
  per size n=16,20,24: 15/15 certified optimal, margins 3e-4..2.7e-3, time spread about a factor 2 (larger at n=20
  because of machine load, as the CPU times show).

## Code

* `msos/verify.py` (new): `cholesky_columns`, `verified_lmin_lower_bound`, `exact_residual`, `rigorous_bound`.
* `msos/core.py`: appended `SpectralDual._densities`, `primal_moments`, `boolean_lower_bound`, `two_sided`
  (existing behaviour unchanged).
* `msos/solvers.py`: appended `alm` (NumPy port of `moment_sos.solve_alm`, validated to 1e-6 against it).
* `msos/reference.py`: `default_solver()` (MOSEK if installed else Clarabel; `MSOS_SOLVER` overrides);
  `moment_relaxation(..., cuts="none"|"trace"|"diag")` with `trace_cuts` kept as alias; Clarabel -> SCS fallback and
  unboundedness detection.
* `src/moment_sos/reference.py`: same MOSEK -> Clarabel fallback.
* `src/moment_sos/solve.py`: `solve_alm(..., shift="uniform"|"diag")`; the uniform path is byte-identical
  (regression check in `tests/test_cuts.py` reproduces the old value and evaluation count).
* New tests: `tests/test_verify.py`, `tests/test_primal.py`, `tests/test_cuts.py`, `tests/test_alm.py`; all old tests
  and the 10 `moment_sos` pytest tests still pass.
* New table/notes generators: `experiments/make_tables_e7.py`, `make_tables_e8.py`, `make_tables_e8b.py`,
  `make_tables_e9.py`, `make_tables_e10.py`, `make_notes_e10.py`; helper `experiments/e10_common.py`.

## Not done / caveats

* MOSEK runs were not repeated; the n=50,60 MAX-CUT instances were not re-solved, so the rigorous certificate (E7)
  covers n <= 40 only.
* All new timings come from a shared machine with four tracks running concurrently (3 BLAS threads each) and are
  indicative only; CPU times are reported where available.
* The two-sided certificate exists only for Boolean problems without constraints; for constrained problems the
  recovered `ybar` and its infeasibility are available but no lower bound on the relaxation value.
* The diagonal-shift relaxation needs the ALM (no eigenvalue-oracle solver), and the ALM was not run beyond n=30.
* Review notes: during the revision three low-severity reviewer remarks (a residual-range sentence in e7_notes.md, an
  unverified SCS attribution in e9_notes.md, a stale docstring in e8_primal.py) were fixed in the files; the n=16 E10a
  reference uses round() of the Clarabel value (21.99999977 -> 22), which equals the brute-force maximum cut.
