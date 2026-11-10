# E8 — primal recovery and two-sided certificates (track 'Primal')

## What was run

* `msos/core.py`: appended `SpectralDual._densities`, `primal_moments(theta, eta)`, `boolean_lower_bound(theta, eta)`, `two_sided(theta, eta)` (existing behaviour untouched; existing tests `tests/test_identity.py`, `tests/test_smoothing.py` still pass).
* `tests/test_primal.py` (plain asserting script, `cd tests && python test_primal.py`, ~5 s): passes.
* `experiments/e8_primal.py` (`cd experiments && python e8_primal.py all`): re-implements the eta-continuation of `solvers.lbfgs_continuation` (etas 1e-1 … 1e-5, 500 iterations per stage, maxcor 30, ftol 1e-9, gtol 1e-10, maxfun 650) but routes every oracle call through `primal_moments`, so U, L_ybar(f), LB, |g|, Hankel infeasibility, lmin M(ybar) and y0 are recorded at *every* function evaluation (`traj`) and at every stage end (`stages`). Results: `experiments/results/e8.json` (written after each instance), log `experiments/results/e8_maxcut.log`.
* `experiments/make_tables_e8.py` (run from the repo root) writes `tables_e8.tex` and `figures/primal_gap.pdf` (+ `experiments/results/primal_gap.png`).
* Timings were taken on a shared machine (load average 22–38 on 14 cores during the run, 3 BLAS threads): absolute times are inflated and only comparable within this experiment.

## MAX-CUT, order 2, random 3-regular graphs (networkx seed 1)

| n | s | rho_2 | source | final U-rho | final rho-LB | final |Lf-rho| | t(U-LB<1e-2) | t(U-LB<1e-3) | t(U-rho<1e-2) | t(U-rho<1e-3) | t(|Lf-rho|<1e-3) | t(|Lf-rho|<1e-4) | total time | evals |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 79 | 16.000000 | Clarabel | 1.02e-04 | 4.39e-03 | 8.76e-04 | 2.4 | -- | 1.3 | 2.4 | 1.3 | 2.3 | 2 s | 1221 |
| 16 | 137 | 22.000000 | Clarabel | 3.08e-04 | 8.01e-03 | 5.20e-04 | 10.6 | -- | 10.2 | 10.5 | 10.3 | 10.3 | 11 s | 1187 |
| 20 | 211 | 26.000000 | e3_bounds.json floor_ub (lb=26, ub=26.001) | 9.55e-04 | 3.51e-02 | 2.79e-04 | -- | -- | 30.4 | 32.0 | 29.4 | 29.7 | 32 s | 1610 |
| 24 | 301 | 32.000000 | exact max cut by enumeration (2s) | 2.67e-03 | 4.62e-02 | 2.52e-04 | -- | -- | 288.6 | -- | 172.0 | 219.9 | 296 s | 2013 |
| 30 | 466 | 40.000000 | e3_bounds.json floor_ub (lb=40, ub=40.0026) | 2.58e-03 | 7.70e-02 | 1.08e-03 | -- | -- | 575.8 | -- | 296.8 | 463.8 | 585 s | 2122 |

All times are 'first evaluation at which the criterion holds' (U and LB are best-so-far, hence monotone; |Lf-rho| is the raw per-evaluation value, which fluctuates along line searches, so t(|Lf-rho|<1e-4) is a first-hitting time, not a time after which it stays below).

* `bracket_ok` is True for every n: LB <= rho_2 <= U held at every one of the 1187–2122 evaluations per instance (rho_2 from Clarabel for n=12,16, exact max cut otherwise — see caveats). y0 = 1 to <= 1.8e-12 at every evaluation (Prop. (d)); the Hankel bound sqrt(sum_k ||M_k(ybar)-X_k||_F^2) <= ||g_Theta|| held at every evaluation (Prop. (c)).
* Primal vs dual: the recovered value L_ybar(f) reaches |Lf-rho| < 1e-3 before U-rho < 1e-3 on every instance (n=12: 1.3 s vs 2.4 s; n=16: 10.3 vs 10.5; n=20: 29.4 vs 32.0; n=24: 172 s vs never (final U-rho = 2.7e-3); n=30: 297 s vs never (2.6e-3)). At n=24/30, |Lf-rho| < 1e-4 is first hit at 220 s / 464 s while U-rho is still > 1e-2 (U-rho < 1e-2 only at 289 s / 576 s). The paper's phrase 'Lf reaches the relaxation value to 1e-5 while U is still 1e-2 away' is NOT reproduced at the 1e-5 level: stage-end |Lf-rho| is 1e-4–1e-3 (best stage-end values: 7.9e-5 (n=12), 2.3e-5 (n=16), 1.6e-4 (n=20), 2.5e-4 (n=24), 6.1e-4 (n=30)); 1e-4 is reached transiently. Suggested wording: 'to 1e-4 while U is still 1e-2 away'.
* The lower bound of Prop. (d) is the weak side: final rho-LB = 4.4e-3, 8.0e-3, 3.5e-2, 4.6e-2, 7.7e-2 for n=12..30. Reason: LB = Lf - tau (Lf - f_empty) with tau ~ delta = [-lmin M(ybar)]_+ <= ||g_Theta||, and Lf - f_empty = m/2 (11–22.5 here), so the LB loses ~ (m/2)·delta; L-BFGS leaves ||g|| ~ 1e-2–1e-1 at the end of the small-eta stages (the stages for eta <= 3e-4 stop after 3–143 iterations with the relative-decrease criterion, far from stationarity), and the best LB is typically obtained at the eta = 1e-3 or 3e-4 stage, not at the end. Consequently the certified bracket U-LB < 1e-2 is reached only for n=12 (2.4 s) and n=16 (10.6 s) and U-LB < 1e-3 never; for n >= 20 the final bracket is 3.6e-2 – 8.0e-2. A gradient-driven (rather than value-driven) stopping rule, or a few extra iterations at eta ~ 1e-3, would improve LB directly; not done here.

## Non-Boolean problems, r=2 (reference rho^tr with trace cuts, Clarabel)

| problem | rho^tr | final U-rho^tr | final Lf-rho^tr | best stage-end Lf-rho^tr (eta) | final y0 | final hankel_rss | final loc | final |g| | t(|Lf-rho|<1e-3) | t(U-rho<1e-3) |
|---|---|---|---|---|---|---|---|---|---|---|
| boxqp5_s1 | 8.610997 | 9.57e-05 | +9.58e-03 | -1.59e-06 (0.01) | 1.000930 | 4.8e-03 | 3.5e-02 | 2.4e-02 | 0.12 | 0.32 |
| boxqp5_s2 | 10.852041 | 1.09e-04 | +5.69e-03 | -1.49e-04 (0.003) | 0.999445 | 5.8e-03 | 1.5e-02 | 1.1e-02 | 0.53 | 0.66 |
| chain8_sparse | 2.014044 | 1.37e-04 | +1.45e-04 | +1.02e-04 (0.0001) | 1.000327 | 7.7e-03 | 1.5e-02 | 1.2e-02 | 0.42 | 0.99 |
| disk | 1.414214 | 3.73e-06 | -2.25e-04 | -1.70e-06 (0.01) | 1.000029 | 2.3e-04 | 2.0e-04 | 3.6e-04 | 0.01 | 0.01 |

* Here Lf is *not* a bound (ybar is infeasible by O(||g||)): it overshoots rho^tr at the final stages (box QPs: +9.6e-3, +5.7e-3) because the small-eta stages end with ||g|| ~ 1e-2 and y0 off by up to 1e-3. The best primal values occur at intermediate eta (1e-2 … 1e-4), where |Lf - rho^tr| is 1.6e-6 (box QP s1), 1.5e-4 (s2), 1.0e-4 (chain), 1.7e-6 (disk). |Lf-rho^tr| < 1e-3 is again reached before U-rho^tr < 1e-3 on all four problems.
* Clarabel reference values: boxqp5_s1 8.610997 (Clarabel printed 'solution may be inaccurate'; MOSEK value in tables_e1.tex: 8.61100), boxqp5_s2 10.852041 (MOSEK 10.85204), chain8 2.014044 (MOSEK 2.01404), disk 1.414214.

## Unit test (tests/test_primal.py) — key numbers

* Petersen r=2, eta=1e-3: L-BFGS stops at |g| = 4.2e-6 (message 'RELATIVE REDUCTION OF F <= FACTR*EPSMCH' — the requested gtol 1e-9 is not reachable in double precision; the decrease of U_eta per step is below machine epsilon relative to U ~ 12). y0-1 = -7e-15; Hankel 3.0e-6 <= |g_Theta| 4.2e-6; rho_2(Clarabel, no cuts) = 11.99999992, LB = 11.99999817, Lf = 11.99999994, U = 12.01635512; Lagrangian identity (a) U_eta - eta*sum w_b H_b - <theta,g> - Lf = 1.4e-13; identity (b) (without <theta,g>) = 5.8e-8 <= |theta||g| = 3.6e-6.
* reg3 n=12 r=2: |g| = 7.6e-6, y0-1 = -3e-15, Hankel 5.5e-6 <= 7.6e-6, rho_2 = 15.99999999, LB = 15.99999447, Lf = 16.00000016 (> rho by 1.6e-7: ybar is infeasible at the 1e-6 level, as allowed), U = 16.02063744; identity (a) -5.7e-14, (b) -1.8e-7.
* Componentwise identities at random zero-sum theta (disk, boxqp4, sphere r=2): g_sigma = 1 - y0 (0.0 error), g_Theta = zero-sum projection of -(omega_ab X_k[a,b]) (0.0), g_u = -omega_ab (X_i[a,b] - L_ybar(g_i x^a x^b)) (0.0), g_p = +L_ybar(h_j x^mu) (0.0); Lagrangian identity (a) to 4e-14.

## Discrepancies with Proposition prop:primal (to be fixed in the .tex by the lead)

1. **Signs in (b).** With U_eta = sigma + sum_b w_b phi_eta(Q_b(theta)) and dphi/dQ = -P_b, the derivative is g_u = -omega_ab (X_i[a,b] - L_ybar(g_i x^a x^b)) = omega_ab (L_ybar(g_i x^a x^b) - X_i[a,b]) and g_p = +(L_ybar(h_j x^mu))_{j,mu}; likewise g_Theta is the zero-sum projection of -(omega_ab X_k[a,b]). The paper states the opposite signs for g_u and g_p (and omits the sign for g_Theta); g_sigma = 1 - ybar_0 is correct. The test checks both signs: errors 1e1 / 3.5 with the paper's sign, 0.0 with the derived sign. The vanishing conditions and everything else in (b)-(d) are unaffected.
2. **(c), Hankel bound, several cliques.** The proof gives sum_k ||M_k(ybar) - X_k||_F^2 <= ||g_Theta||^2 (root-sum-square over cliques), which is what is tested (`infeas['hankel_rss']`); the per-clique statement in the paper follows. The *sum* of the Frobenius norms over cliques can exceed ||g_Theta|| (chain5 sparse: 16.5 > 10.7) — only relevant if someone reads 'summed over k'.
3. **Identity in (b)** 'U_eta - eta sum_b w_b H_b(P_b) = L_ybar(f)' needs g = 0 exactly; at the numerically stationary points reachable by L-BFGS (|g| ~ 4e-6) it holds to |<theta, g>| ~ 6e-8–2e-7, whereas the general identity (a) U_eta - eta sum w_b H_b = <theta, g> + L_ybar(f) holds to 1e-13 at every theta. Note also that theta must lie in the zero-sum Theta-subspace (the parametrization of Lemma gram); for a non-zero-sum Theta the identity (a) fails because the code's projected g_Theta differs from the unprojected derivative.
4. **Empirical claim in the text after the proposition** ('L_ybar(f) reaches the relaxation value to 1e-5 while U is still 1e-2 away'): in E8 the primal side is indeed faster, but to ~1e-4, not 1e-5 (see above).

## Caveats / deviations

* **Reference values for n >= 20.** Clarabel (3 threads, shared machine) did not finish the order-2 relaxation at n=20 within 27 min and was stopped; the project note 'n<=24 in seconds to a minute' did not hold under the current load. For n=20 and n=30 the stored `e3_bounds.json` floor values (26, 40; lb = floor(U) in E3) were used, and for n=24 the exact max cut 32 from exhaustive enumeration (vectorised, 2 s; the enumeration reproduces 16, 22, 26 for n=12, 16, 20). For these graphs rho_2 >= max cut always, and the bracket computed here gives rho_2 <= U: n=20: rho_2 in [26, 26.00096]; n=24: [32, 32.0027]; n=30: [40, 40.0026]. So 'rho_2 = max cut' is confirmed to within the final U-rho column, and the U-rho, rho-LB, |Lf-rho| numbers for n>=20 are exact up to that width (<= 2.7e-3) — they should be read as provisional at the 1e-3 level.
* Clarabel was used for n=12 (16.000000, 15.99999999) and n=16 (22.000000, 21.99999974) and for all non-Boolean references.
* Timings on a shared, heavily loaded machine; the n=30 continuation took 585 s (2122 evaluations, two eigendecompositions of size 466 per evaluation: one of Q, one of M(ybar) for the LB).
* Unit-test tolerance: gtol 1e-9 is not reachable (see above); the (b)-identity check was relaxed to the Cauchy–Schwarz slack |theta||g| (and < 1e-6 absolute) instead of 1e-8.
* Per-evaluation |Lf-rho| is noisy (line-search trial points); figure left panel therefore shows it as faint dots plus stage-end markers.

## Files

* modified: `msos/core.py` (appended methods only)
* created: `tests/test_primal.py`, `experiments/e8_primal.py`, `experiments/make_tables_e8.py`, `experiments/results/e8.json`, `experiments/results/e8_maxcut.log`, `experiments/results/e8_notes.md`, `experiments/results/primal_gap.png`, `tables_e8.tex`, `figures/primal_gap.pdf`

## Addendum: tight stationarity (experiments/e8b_tight.py, results/e8.json['tight'], tables_e8b.tex)

Hypothesis from the main run: the Boolean lower bound is limited by the gradient norm that the value-based L-BFGS stopping rule leaves at the small-eta stages. Remedy tested: after the standard continuation (identical options), from its final theta, L-BFGS-B on U_eta at fixed eta = 1e-3, 3e-4, 1e-4 (warm-started in sequence; maxiter 3000, maxfun 4000, ftol 1e-15, gtol 1e-8), total time limit 900 s per instance (callback raising StopIteration). U and LB are tracked at every evaluation from the start of the run (best-so-far), so t(U-LB < tol) counts from the start of the whole run. Machine still shared (timings indicative; the standard phase alone took 3 / 8 / 53 / 244 s here versus 2 / 11 / 32 / 296 s in the main run).

| n | stage | iter. | \|g\| | rho-LB | U-rho | U-LB | delta | Lf-rho | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 12 | standard (end, eta=1e-5) | 6 | 2.7e-02 | 4.4e-03 | 1.0e-04 | 4.5e-03 | 1.8e-03 | -8.8e-04 | 3 |
| 12 | tight eta=0.001 | 51 | 5.0e-06 | 4.3e-06 | 1.0e-04 | 1.1e-04 | 6.1e-07 | -4.8e-08 | 3 |
| 12 | tight eta=0.0003 | 33 | 6.0e-06 | 4.3e-06 | 1.0e-04 | 1.1e-04 | 6.8e-07 | +2.1e-08 | 3 |
| 12 | tight eta=0.0001 | 50 | 5.4e-06 | 3.2e-06 | 1.0e-04 | 1.1e-04 | 5.5e-07 | +7.6e-08 | 4 |
| 16 | standard (end, eta=1e-5) | 3 | 1.5e-02 | 8.0e-03 | 3.1e-04 | 8.3e-03 | 7.5e-04 | -5.2e-04 | 8 |
| 16 | tight eta=0.001 | 3000 | 4.1e-05 | 2.1e-05 | 3.1e-04 | 3.3e-04 | 4.4e-06 | -3.3e-06 | 44 |
| 16 | tight eta=0.0003 | 28 | 1.6e-05 | 1.7e-05 | 3.1e-04 | 3.2e-04 | 1.7e-06 | +2.5e-07 | 45 |
| 16 | tight eta=0.0001 | 38 | 2.5e-05 | 1.6e-05 | 3.1e-04 | 3.2e-04 | 2.9e-06 | +8.3e-07 | 45 |
| 20 | standard (end, eta=1e-5) | 15 | 8.8e-02 | 3.5e-02 | 9.6e-04 | 3.6e-02 | 3.4e-03 | -2.8e-04 | 53 |
| 20 | tight eta=0.001 | 3000 | 2.1e-03 | 2.2e-02 | 9.6e-04 | 2.3e-02 | 2.3e-04 | -2.1e-02 | 139 |
| 20 | tight eta=0.0003 | 3000 | 6.3e-05 | 4.1e-05 | 9.6e-04 | 1.0e-03 | 8.1e-06 | -2.7e-07 | 220 |
| 20 | tight eta=0.0001 | 102 | 1.4e-04 | 4.1e-05 | 9.6e-04 | 1.0e-03 | 1.2e-05 | -1.8e-06 | 223 |
| 24 | standard (end, eta=1e-5) | 50 | 2.0e-01 | 4.6e-02 | 2.7e-03 | 4.9e-02 | 1.4e-02 | -2.5e-04 | 244 |
| 24 | tight eta=0.001 | 3000 | 3.8e-03 | 3.7e-02 | 2.7e-03 | 4.0e-02 | 1.9e-04 | -3.5e-02 | 740 |
| 24 | tight eta=0.0003 | 2150 | 2.3e-03 | 8.8e-04 | 2.7e-03 | 3.6e-03 | 1.1e-04 | +1.0e-05 | 900 |

| n | t(U-LB<1e-2) standard / tight | t(U-LB<1e-3) standard / tight | final U-LB standard / tight | total time | note |
|---|---|---|---|---|---|
| 12 | 2.4 s / 3.1 s | -- / 3.3 s | 4.5e-03 / 1.1e-04 | 4 s | all three tight stages stop on the relative-decrease floor (|g| 5e-6) |
| 16 | 10.6 s / 8.5 s | -- / 8.7 s | 8.3e-03 / 3.2e-04 | 45 s | eta=1e-3 stage hits 3000 iterations (|g| 4e-5); the two others stop on the floor |
| 20 | -- / 139.7 s | -- / 210.9 s | 3.6e-02 / 1.0e-03 | 223 s | eta=1e-3 and 3e-4 stages hit 3000 iterations each |
| 24 | -- / 770.9 s | -- / -- | 4.9e-02 / 3.6e-03 | 900 s | 900 s limit hit during the eta=3e-4 stage (2150 iterations); eta=1e-4 stage not run |

Findings:
* The remedy works where the tight phase reaches |g| <~ 1e-4: rho - LB drops from 4.4e-3 / 8.0e-3 / 3.5e-2 (standard) to 3.2e-6 / 1.6e-5 / 4.1e-5 for n = 12 / 16 / 20, i.e. by three orders of magnitude, and LB <= rho_2 <= U held at every evaluation (bracket_ok). The bracket U - LB then equals U - rho_2 to within 5% (1.05e-4, 3.2e-4, 1.0e-3): after the tight phase the certificate is limited by the *dual* side (the best hard U, which the tight phase does not improve: the hard bound of the eta >= 1e-4 iterates stays above the best U found by the eta = 1e-5 stage of the standard continuation).
* U - LB < 1e-3 is now reached for n = 12 (3.3 s), n = 16 (8.7 s), n = 20 (211 s); U - LB < 1e-2 for n = 24 at 771 s (standard run: never for n >= 20). For n = 24 the 900 s budget ended in the eta = 3e-4 stage with |g| = 2.3e-3, rho - LB = 8.8e-4 and U - LB = 3.6e-3.
* Cost: the tight phase is where the iterations go (n=20: 3000 + 3000 + 102 iterations, 170 s on top of the 53 s continuation; n=24: 5150 iterations in 656 s). The tight stage at eta = 1e-3 is the least useful of the three for the bracket (its stationary point has L_ybar(f) = rho - 2e-2 (n=20) / rho - 3.5e-2 (n=24) because of the smoothing bias, so LB cannot exceed that); the eta = 3e-4 stage delivers the bound. A cheaper recipe would therefore be: standard continuation, then one tight stage at eta = 3e-4 (or 1e-4) only.
* The double-precision floor |g| ~ 5e-6 (relative-decrease stop) was reached at n = 12 and at the eta <= 3e-4 stages of n = 16; it corresponds to delta ~ 6e-7 and rho - LB ~ 3e-6 — the smallest bracket contribution of the LB side attainable with this solver.
* No bug was found in msos/core.py; it was not modified in the addendum.
