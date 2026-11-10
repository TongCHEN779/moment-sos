# Reproducing the paper

Environment: Python 3.13, see `requirements-lock.txt` (exact versions used). MOSEK needs a licence (E1-E6, Secs. NN);
E7-E10 and the benchmark use Clarabel (open source) where MOSEK is not available. All timings in the paper were taken
on a **shared** 14-core macOS machine with 36 GiB RAM; the per-run load average is stored in the raw results
(`load0` / `loadavg`). Expect different wall-clock times on other machines; memory figures and bound values should agree.

Check the prose numbers against the raw data at any time: `python check_paper_numbers.py` (exit code 1 if a claim fails).

| paper item | script(s) (run from the repo root unless noted) | raw data | rough runtime (estimate, not a measurement) |
|---|---|---|---|
| E1 correctness | `cd experiments && python e1_correctness.py` | `results/e1.json` | 10 min |
| E2 convergence | `python e2_convergence.py 16; python e2_convergence.py 24; python e2b_agd.py` | `results/e2_n*.json`, `e2b_n*.json` | 15 min |
| E3 scaling | `experiments/e3_scaling.sh` (n=10..60), n=60 rerun with `run_one.py fo 60 2 3000` | `results/e3.jsonl` | 1.5 h (n=60: 35 min) |
| E4, E5 | `python e4_higher_order.py 20 1 2 3; python e4_higher_order.py 24 2 3; python e5_structure.py` | `results/e4_n*.json`, `e5.json` | 1 h |
| E6 baselines | `experiments/baselines/driver.py`, `collect.py` | `results/baselines.json` | 3 h |
| E7 rigorous certificate | `experiments/e7_rigor.py` | `results/e7.json` | 40 min |
| E8 primal recovery | `experiments/e8_primal.py`, `e8b_tight.py` | `results/e8.json` | 1 h |
| E9 cuts | `experiments/e9_cuts.py` | `results/e9.json` | 30 min |
| E10 ALM, variability | `experiments/e10a_alm_maxcut.py`, `e10b_variability.py` | `results/e10a.json`, `e10b.json` | 2 h |
| Benchmark (Biq Mac-style families, E11) | `experiments/bench/run_all.sh`, then `python experiments/bench/aggregate.py` from the repo root | `results/bench.jsonl` (an earlier, partial run under heavy machine load is kept as `bench_loaded_partial.jsonl` and is not used) | 1.5 h measured (total wall time of the 60 runs) |
| E12 certification schedules and BM certificate | `experiments/bench/run_hybrid_all.sh`, then `python experiments/bench/aggregate_hybrid.py` (root); n=50/60 short schedule: `experiments/bench/cold_large.py 50 68`, `cold_large.py 60 81` | `results/bench_hybrid.jsonl`, `results/cold_large.txt` | 2.5 h (the n=50/60 runs took 4 and 10 min) |
| E5b structured benchmark (box QP, sparse/dense chain, 5 seeds) | `experiments/bench/run_struct_all.sh`, then `python experiments/bench/aggregate_struct.py` | `results/bench_struct.jsonl` | 40 min |
| E13 sensitivity of the smoothing solver | `experiments/bench/sensitivity.py`, then `python experiments/bench/aggregate_sens.py` | `results/sensitivity.jsonl` | 40 min |
| Profile of one oracle evaluation | `python experiments/bench/profile_oracle.py` | `results/profile_oracle.json` | 2 min |
| tight weights (negative result) | `cd experiments/nn && python tight_weights.py` | `results/nn_tight_weights.json` | 10 min |
| NN: N1 verification | `experiments/nn/n1_verification.py` (needs MOSEK), `n12_alm.py` | `results/nn_n1_new.json`, `nn_alm_n1.json` | 1.5 h |
| NN: N2 Lipschitz | `experiments/nn/n2_lipschitz.py`, `n12_alm.py` | `results/nn_n2_new.json`, `nn_alm_n2.json` | 1 h |
| NN: N3 large | `experiments/nn/run_n3.sh`, `run_n3_alm.sh` (use `rho0=1000`, no warm start) | `results/nn_n3_*.json`, `nn_alm_*.json` | 3 h |
| tables / figures | `experiments/make_tables*.py`, `experiments/make_figures.py`, `experiments/nn/make_tables_nn.py` | `tables_*.tex`, `figures/` | 1 min |

Tests: `cd moment_sos && pytest` (12 tests); `cd tests && pytest -q test_verify.py test_crossval.py` and
`python test_identity.py; python test_smoothing.py; python test_cuts.py; python test_primal.py; python test_alm.py`
(script-style tests; each prints `ALL ... PASSED` / `all assertions passed`).

Known provenance notes: the n=40 network result of Table (larger networks) is the run with `rho0=1000`, no warm start
(`nn_alm_5_40-40_11_0.1.json`); the rerun with `rho0=100` and warm start is kept separately
(`nn_alm_5_40-40_11_0.1_rho100_warm.json`) and is discussed in Sec. 5.3.

## Added: TSSOS, MNIST, real Biq Mac instances
| item | command | raw data | rough time |
|---|---|---|---|
| TSSOS (Julia 1.13.3, TSSOS from GitHub, Mosek.jl) | `julia experiments/tssos/maxcut_tssos.jl experiments/tssos/reg3_N_1.txt 2` | `experiments/results/tssos_reg3.jsonl` | 12-264 s (n=10-30) |
| MNIST 3 vs 8 verification | `python experiments/nn/n4_mnist.py MNIST_ROOT 4 8` (torchvision downloads MNIST) | `experiments/results/nn_mnist_d4_w8.json` | ~30 min |
| real Biq Mac g05_60 | `experiments/bench/run_real_all.sh` (needs `rudy_all.tar.gz` from biqmac.aau.at, path set in the script) | `experiments/results/bench_real.jsonl` | ~tens of min per instance |
