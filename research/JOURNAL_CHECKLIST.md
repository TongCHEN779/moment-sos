# Journal-readiness checklist (target: computational / software journals, e.g. Math. Prog. Computation, ACM TOMS, Optim. Methods & Software)

Status codes: **DONE** · **IN PROGRESS** · **TODO** · **NEEDS USER** (needs a decision, permission, or resource only the author has).
Each item has an acceptance criterion; an item is not DONE until the criterion is met and verified.

## A. Story and scope
| id | item | acceptance criterion | status |
|---|---|---|---|
| A1 | One main claim, <= 5 contributions, software-journal framing | abstract <= 200 words, every number in it traceable to a table | PARTIAL (abstract 188 words and 5 contributions rewritten; numbers in it match tables; final wording after the benchmark) |
| A2 | Restructure: Intro / Method (short) / Algorithms / Software design / Computational study / Application / Limits | theory secondary to appendix; body <= ~25 pages | PARTIAL (Software moved before the experiments; three long proofs and E4/E5 moved to appendices; body still ~24 pages; further trimming of Sections 7-8 TODO) |
| A3 | No claim without data; neural-network section scaled to what was shown | each sentence of Sec. NN backed by a table/figure; failures stated | PARTIAL (NN section claims were checked against data; failures stated; abstract states the non-convergence) |
| A4 | Remove history/self-reference phrases; one terminology (certified / rigorous / anytime) | grep finds no "earlier version"; glossary consistent | DONE |

## B. Software quality (what the journals review)
| id | item | acceptance criterion | status |
|---|---|---|---|
| B1 | `moment_sos` installs in a fresh virtualenv, tests pass from the installed package | `pip install .` + `pytest` in a clean venv | DONE (venv with system site-packages, `--no-deps`: dependency resolution from the network was not tested) |
| B2 | One documented code base: either experiments use `moment_sos`, or `msos` is declared the NumPy reference and cross-validated | tests assert msos == moment_sos values on E1/E3 instances | DONE (moment_sos cross-validated against the independent primal reference; rigorous certificate now also in `moment_sos.verify`) |
| B3 | API docs, docstrings, error messages (order too low, box missing, clique coverage), type hints | doc page + every public function documented | DONE except type hints (`docs/api.md` generated; 12 tests incl. error messages) |
| B4 | LICENSE, CITATION.cff, CHANGELOG, version tag, minimum dependency versions | files present; `moment_sos.__version__` matches | DONE (CITATION.cff has a placeholder repository URL: NEEDS USER) |
| B5 | Reproduction package: environment file, one command per table/figure, seeds, expected runtimes, raw results with checksums | `reproduce/README` + script regenerates all tables from raw JSON | TODO |
| B6 | Profile where time goes (Gram assembly vs eigh vs backward), thread scaling, optional GPU path tested | profile table in paper; GPU stated as tested or untested | PARTIAL (profile done: eigh = 73-89% of an evaluation, no thread scaling on this machine; GPU and partial eigensolvers NOT tested) |

## C. Computational study
| id | item | acceptance criterion | status |
|---|---|---|---|
| C1 | Benchmark families from the standard distributions (Biq Mac / rudy: g05, pm1s, pm1d, torus Ising; box QP; sparse chain POPs). Generated with the same distributions, **not** the library instances (those need a download) | generators in repo, instances reproducible by seed | DONE for MAX-CUT families (60 instances), box QP and sparse/dense chain (45 instances, 5 seeds); real Biq Mac rudy g05_60.0-4 DONE: all 5 certified optimal, 794-1223 s (bench_real.jsonl); n=80/100 and ising instances too large for the dense order-2 model |
| C2 | >= 5 seeds per size, median and spread, performance profile (time to certified gap) | figure + table; no single-instance claims | DONE for MAX-CUT (5 seeds x 3 sizes x 4 families, median/min/max, certification-time profile); TODO for other problem classes |
| C3 | Timing protocol: idle machine, fixed threads, 3 repeats, load recorded, CPU and wall time | protocol stated; load average logged per run | PARTIAL (rerun on a quieter machine: median load 2.9, max 10.7, 8 threads, load and CPU time stored per run; no repeats of the same run; the machine is still shared) |
| C4 | Equal-accuracy comparison with relevant solvers | MOSEK, Clarabel, SCS, CGAL, Burer-Monteiro DONE; **TSSOS/CS-TSSOS (Julia), ManiSDP (MATLAB; Python port exists), SketchyCGAL with sketch, RiNNAL-POP not run** (no Julia/MATLAB on this machine) | PARTIAL (BM + dual certificate baseline DONE; TSSOS 1.13.3/MOSEK DONE on reg3 n=10-30 (same values as the interior-point reference, times follow MOSEK; CS-TSSOS reduces to dense on these graphs); ManiSDP (MATLAB), SketchyCGAL with sketch, RiNNAL-POP NOT run) |
| C5 | ALM at n >= 38: converge or explain | order-2 NN case n=40 reaches 1e-3 of the interior-point value, or a diagnosed obstruction | NOT SOLVED (semismooth Newton-CG inner solver implemented and unit-tested; 7x faster than L-BFGS for order 1 on t2g10 but on n=40 gives 293.8 vs L-BFGS -8.623 vs exact -9.618 after the divergence bug was fixed; cause not diagnosed; earlier runs with the buggy version are kept as PREFIX_BUGGY_* and not used) |
| C6 | One realistic network (e.g. small MNIST classifier, PCA inputs) certified at order 1 and 2 on several test points | certified-accuracy table | DONE for one small net (MNIST 3 vs 8, d=4, 8-8, 8 points x 3 eps; order 1 already certifies every robust point, order 2 exact to 2e-6 vs order 1 gap up to 0.49); larger nets NOT done |
| C7 | Sensitivity of the solvers to eta schedule, rho0, inner iterations, block size | table/figure | DONE for the smoothing solver on g05 n=30 (60 runs: eta0, iterations per stage, L-BFGS memory), rho0 of the ALM earlier; not done: other problem classes, block-size dependence |

## D. Rigor and verification
| id | item | acceptance criterion | status |
|---|---|---|---|
| D1 | Independent adversarial review of every theorem, lemma and proposition | reviewer report; each finding fixed or rebutted | DONE (adversarial review; no fatal error; 5 findings fixed: Boolean equality in the diagonal-domination proposition, K vs K-with-box convention and the box-QP hypothesis, monotonicity claim in the exactness theorem, exact rounding of B_i in `verify.py` (+test), phrasing in primal recovery; E7 numbers predate the B_i fix, effect < 1e-12 stated) |
| D2 | Automatic check that every number in the paper equals the raw data | `check_paper_numbers.py` passes | DONE for 25 prose claims (`check_paper_numbers.py`); extend as new numbers enter (bench) |
| D3 | Citation audit | every bibitem verified or flagged | PARTIAL (rigor refs verified; Higham Thm 10.3 only via a secondary source) |
| D4 | Related-work completeness and a feature table against GloptiPoly, SumOfSquares.jl, TSSOS, SOSTOOLS, SPOT, YALMIP | verified table | DONE for the table in Sec. Software (verified fields only; SPOT, YALMIP, SumOfSquares.jl, SketchyCGAL, CDCS, SDPNAL+, COSMO listed as not examined); related-work sentence added; closest GPU precedent (Kang et al. 2024) cited |

## E. Presentation
| id | item | acceptance criterion | status |
|---|---|---|---|
| E1 | Readable tables (no tiny resizebox), vector figures with consistent style | visual check at 100% | TODO |
| E2 | Language and notation pass; symbol list | read-through | TODO |
| E3 | Journal template and length | after A2 | TODO |

| C8 | Sparse MAX-CUT on library tori (t2g10) vs CS-TSSOS | converge to the CS-TSSOS order-2 value | NOT MET (ours 6.1972e6 vs 6.0495e6, 0.7 GiB vs 11 GB; t2g15/t2g20/t3g not run; stated as limitation 7b) |
