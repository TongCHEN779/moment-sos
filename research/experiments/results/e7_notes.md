# E7 — rigorous a posteriori certificates (track "Rigor")

## What was run
* `msos/verify.py` (new): 
  * `verified_lmin_lower_bound(Q, lmin_hat, margin)` — Lemma lem:chol. Column-oriented (left-looking)
    Cholesky written out in NumPy (`cholesky_columns`; every inner product is a floating-point
    `R[:j,j] @ R[:j,j:]`), applied to `A~ = fl(Q + alpha I)`, `alpha = -lmin_hat + eps`,
    `eps0 = max(1e-12, 1e-10|lmin_hat|, 64 u ||Q||_F)`, doubled until success (max 40 doublings).
    Returns `ell = -alpha - gamma_{s+1} nu - u max_a|Q_aa+alpha|` with
    `nu = fl(||R||_F^2)(1+gamma_{N+1})`, `N = s(s+1)/2`; the final scalar arithmetic is done in
    `fractions.Fraction` and rounded DOWN to a double. `np.linalg.cholesky` is used only as a cheap
    pre-check (a failed pre-check just skips the explicit factorisation for that eps).
  * `exact_residual(dual, theta)` — the residual polynomial `e` of the Gram identity for the
    floating-point `Qhat_b = dual.gram_blocks(theta)`, `sigma_hat = theta[0]`, `phat = theta[sl_p]`,
    with the float coefficients of `pop.f / ineq / eq`, computed exactly: every double is written
    `M 2^e` (`np.frexp`), everything is scaled to a common denominator `2^S`, products `c*Qhat[a,b]`
    are integer products, accumulation per moment key with Python ints (`np.add.at` on an object
    array). Mirrors `core.py`: slots `(cl_I, cl_J, slot_w, slot_g)` for the moment blocks,
    `ineq_info[i]['kpt'/'a'/'b'/'terms']` with weights 1/2 for the localising blocks,
    `eq_info[j]['kpt'/'terms']` for the equalities. Returns `E = sum|e_gamma|` as a Fraction and as
    a double rounded UP, plus the sparse residual `{key: Fraction}`.
  * `rigorous_bound(dual, theta)` — Theorem thm:rigor: `U_rig = sigma + sum_k s_k Phi_k(-ell_k)
    + sum_i s_i B_i [-ell_i]_+ + E`, final sum in Fractions (`s_i*B_i` as `Fraction(s)*Fraction(B)`),
    rounded up. `Phi_k(lambda) = lambda` for Boolean (constant-trace) moment blocks
    (`blocks[b]['plus'] == False`), `[lambda]_+` otherwise.
  * `sanity_check_residual`, `eval_residual_exact`, `residual_float_at_point` — the exact residual
    polynomial evaluated at random points vs. the float evaluation of the identity.
* `tests/test_verify.py` (new; `cd tests && python test_verify.py`, 2.3 s, passes):
  (i) random symmetric matrices s = 5..300, scales 1e-3/1/1e3, plus PSD matrices and deliberately
  wrong `lmin_hat`: `ell <= lmin(eigh)` always, worst relative slack 1.1e-10;
  (ii) exact residual on ball/disk/sphere/K3 r=2/boxqp(4)/chain(5, sparse): for random theta the exact
  residual polynomial and the float identity agree at random points to <= 1.1e-15; for zero-sum theta
  E <= 4.6e-15; for solver output E <= 8e-13 (< 1e-12 max(1,|U|));
  (iii) `U_rig >= U` and `U_rig - U <= 2.0e-10 < 1e-8` on those instances.
* `experiments/e7_rigor.py` (new): E1 case list (ball, disk, sphere, MAX-CUT K3/C5/K5/Petersen/K7/
  reg3-12 at r = 1,2(,3 for n<10), box QP n=5 seeds 1,2, chain n=8 sparse & dense) + order-2 MAX-CUT on
  `networkx.random_regular_graph(3, n, seed=1)`, n = 10, 14, 20, 26, 30, 40;
  `solvers.lbfgs_continuation(iters_per_stage=500)` (time_limit 900 s for n=40, not reached), then
  `rigorous_bound`. Results written incrementally to `results/e7.json` (29 rows); log in
  `results/e7.log`. Best cut: `brute_maxcut` for n <= 20; for n = 26, 30, 40 the max of a
  2000-restart single-flip local search and the `lb` of `results/e3_bounds.json` (both agreed:
  36, 40, 54).
* `experiments/make_tables_e7.py` (new) -> `tables_e7.tex` (repo root, booktabs tabular only; columns:
  problem, r, s, U, U_rig - U, E, max eig. slack, t_chol, t_res, certified?).

## Key numbers (from results/e7.json)
* `U_rig >= U` on all 29 rows; Cholesky verification succeeded on every block with the initial
  margin (0 doublings) — no failures.
* max over all instances of `U_rig - U` = 7.22e-09 (MAX-CUT reg3 n=40, s=821, U = 54.00642248);
  min = 4.0e-14 (sphere). Relative to |U| the maximum is 1.34e-10. For Boolean instances the
  correction is almost exactly `1e-10 * (U - sigma)`: it is dominated by the initial Cholesky
  margin `eps = 1e-10 |lmin_hat|` multiplied by the block weight s_k, not by the backward-error
  term (gamma_{s+1} nu <= 1e-11 even at s = 821).
* Exact residual E: median 1.1e-13; max 1.83e-11 (n=40); E is 1e-16..1e-14 for the tiny instances
  and 1e-13..1e-11 for s >= 50 (K7 r=3, s=64: 1.4e-13; all other s >= 50 rows: 1e-12..1e-11).
* Max over blocks of `lmin_hat - ell`: between 1.0e-12 and 1.5e-10 (largest for small r=1 blocks
  where |lmin_hat| ~ 1 and eps = 1e-10|lmin_hat| dominates); 8.8e-12 at n=40.
* All 22 Boolean (MAX-CUT) rows: floor(U_rig) == floor(U) == best cut (certified = True), including
  n = 26, 30, 40 (36, 40, 54).
* Largest instance (MAX-CUT reg3 n=40, r=2, s=821, 102091 moments, 2333 oracle calls):
  solve 558.6 s, t_chol = 0.068 s, t_res = 0.138 s, t_eig (eigvalsh for lmin_hat) = 0.068 s,
  U = 54.00642248150075, U_rig = 54.00642248871853, E = 1.83e-11.
  n=30 (s=466): solve 149.3 s, t_chol 0.12 s, t_res 0.10 s (t_chol noise: shared machine).
* Table: `tables_e7.tex`.

## Caveats
* Timings on a shared machine (3 other tracks running concurrently), 3 BLAS threads
  (`OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=3`).
* The inner products inside `cholesky_columns` are NumPy/BLAS matrix-vector products; the backward
  error bound gamma_{s+1} of Higham Thm. 10.3 holds for any summation order and also with FMA /
  extended-precision accumulation (errors only get smaller), so the hypothesis of Lemma lem:chol is
  met. `nu` uses the bound gamma_N/(1-gamma_N) <= gamma_{N+1} (valid while N(N+1)u < 1, i.e. s
  below ~13 000).
* The certificate is for the floating-point problem data as stored in `pop.f/ineq/eq`; for MAX-CUT
  (coefficients +-1/2) and the box constraints these are exact.
* `U_rig - U` is dominated by the deliberately chosen initial margin `eps = 1e-10|lmin_hat|`; a
  smaller margin (e.g. 1e-12 relative) would reduce the correction ~100x at the cost of occasional
  doublings. Not tuned, to stay with the task specification.
* The exact residual in Python integers costs ~0.14 s at s=821 (102k moments, 337k slots), i.e.
  about one oracle call (eigh of 821x821 ~ 0.07-0.2 s), consistent with the paper's remark.
* E1 instances were re-solved here (fresh L-BFGS runs), so the U column differs from
  `tables_e1.tex` in the last digits (same optima).

## Nothing failed.
