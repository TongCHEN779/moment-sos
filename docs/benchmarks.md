# Benchmarks and limitations

All numbers come from the raw data in `research/experiments/results/` of the repository (reproduction commands in
`research/REPRODUCE.md`). Timings were taken on one shared Apple-silicon machine (8 threads); load was recorded but not
controlled, so treat factors below 2 as noise. Instances are mostly single random draws.

## Dense order-2 MAX-CUT on random 3-regular graphs

| $n$ | block size $s$ | ours: value, time, memory | MOSEK: value, time, memory |
|---|---|---|---|
| 10 | 56 | 12.0002, 0.5 s, 0.13 GiB | 12.0000, 0.1 s, 0.14 GiB |
| 14 | 106 | 19.0000, 1.7 s, 0.17 GiB | 19.0000, 0.4 s, 0.21 GiB |
| 20 | 211 | 26.0010, 8.7 s, 0.21 GiB | 26.0000, 3.9 s, 0.99 GiB |
| 26 | 352 | 36.0000, 26.3 s, 0.32 GiB | 36.0000, 43.0 s, 5.41 GiB |
| 30 | 466 | 40.0026, 61.6 s, 0.44 GiB | 40.0000, 252.6 s, 13.27 GiB |
| 40 | 821 | 54.0065, 241 s, 1.04 GiB | killed (memory; Schur matrix ≳ 39 GiB) |
| 50 | 1276 | 68.0040, 583 s, 1.89 GiB | not attempted |
| 60 | 1831 | 81.0195, 2037 s, 2.49 GiB | not attempted |

**Reading it.** Below $n\approx26$ the interior-point solver is faster. Above it, ours is faster and uses a small
fraction of the memory. The cut found equals $\lfloor U\rfloor$ at every size, i.e. optimality is proved.

## Optimality proofs on Biq Mac-style and real instances

- 60 order-2 MAX-CUT instances drawn from the Biq Mac distributions (`g05`, `pm1s`, `pm1d`, torus Ising; 5 seeds,
  $n\le40$): **all 60 certified optimal** (best cut equals $\lfloor U\rfloor$).
- 5 real library instances `g05_60.0`–`g05_60.4` (downloaded from the Biq Mac library, $n=60$): **all 5 certified
  optimal**, after 794–1223 s (median 1102 s).
- Larger library instances ($n=80,100$) and the complete-graph `ising` instances do **not** fit the dense order-2 model
  and were not run.

## Comparison with other solvers

| solver (class) | speed vs. ours | memory | bound returned |
|---|---|---|---|
| MOSEK / TSSOS / CS-TSSOS (interior point) | faster for $n\lesssim26$; slower above | large (13 GiB at $n=30$) | dual objective, accurate to solver tolerance |
| Burer–Monteiro ALM (low rank) | **faster** (4.7 s vs 62 s at $n=30$) | small | none; multipliers are poor dual points |
| SCS (ADMM) | faster for $n\le30$ at $10^{-5}$ | up to 1.1 GiB at $n=40$ | none; at $10^{-3}$ the error is $2\times10^{-3}$–0.45 |
| CGAL (conditional gradient) | faster | small | none |
| **moment-sos** | — | 1.0 GiB at $n=40$ | **valid at every iterate** |

At every size we tried, at least one non-certified first-order method reached $10^{-3}$ faster than we did; the price of
having a bound at every iterate was a factor of 5–15 in time at $n\ge20$. Not compared: ManiSDP, SketchyCGAL with a
sketch, RiNNAL-POP, CDCS, COSMO, SDPNAL+.

**TSSOS** (Julia 1.13.3, MOSEK) returns the same values as the interior-point reference on the dense instances
$n=10$–30 and takes 12, 12, 16, 58 and 264 s (each including about 12 s of Julia start-up).

## Limitations — read these before using the package

1. **Not faster than interior point on small problems.** If the relaxation fits in memory and you need accuracy of
   $10^{-8}$, use an interior-point solver.
2. **Moderate accuracy** of the smoothing solver ($10^{-3}$–$10^{-4}$). `solve_alm` is more accurate but slower to a
   first useful bound.
3. **Dense block limit.** Blocks beyond $s\approx2500$ are out of reach (dense `eigh`); no partial eigensolver, no GPU
   path tested.
4. **Sparse library instances.** On the Biq Mac torus instance `t2g10` ($n=100$, weights up to $10^5$) the
   correlative-sparsity relaxation has 73 blocks of size ≤ 301; CS-TSSOS reaches the order-2 value $6.0495\times10^6$
   in 42 s using 11 GB, while our smoothing solver stops at $6.197\times10^6$ ($2.4\%$ higher) in 109 s using 0.7 GiB,
   and `solve_alm` is slow here: with warm start it did not leave the order-1 plateau in 400 s, and with default settings it reaches only $6.76\times10^6$ (12% above) in 980 s (about 70 s per outer iteration). The gap is smoothing bias, not a different relaxation: `etas=po.PRECISE_ETAS` reaches $6.04969\times10^6$ (relative difference $3.8\times10^{-5}$ from TSSOS) in 264 s. Term sparsity is implemented but weaker (see the sparsity guide).
5. **Neural networks.** Order 2 matches interior point to $10^{-5}$ up to about 14 variables. At 38–40 variables
   `solve_alm` had not converged after 25 min (78–94% of the order-1 → exact gap closed); at 60 it fails. On a small MNIST
   classifier order 1 already certified every robust point.
6. **Rigorous certificate** was run for blocks up to size 821; it assumes IEEE arithmetic without under/overflow and
   certifies the stored (normalized) floating-point problem.
7. **Engineering limits.** Monomials are 64-bit encoded: spin mode supports $n\le63$ and continuous mode $n\le26$ at
   order 2.
8. **Julia and the command line** run a Python process; there is no native Julia solver.
