"""E10b: instance-to-instance variability of the smoothing solver (paper settings: lbfgs_continuation,
iters_per_stage=500) on order-2 MAX-CUT, five random 3-regular graphs per size (networkx seeds 1..5),
n = 16, 20, 24.  Per instance: solve time, evaluations, final U, floor(U), best cut (brute force for n=16 and,
if < 2 min, n=20; 2000-restart single-flip local search otherwise), certified-optimal flag floor(U) == cut,
margin U - floor(U); Clarabel interior-point reference (n=16, 20; n=24 skipped if one solve exceeds ~5 min).
Results appended to results/e10b.json after every instance."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
sys.path.insert(0, ".."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, time, resource
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
from e10_common import local_search_maxcut

NS = (16, 20, 24)
SEEDS = (1, 2, 3, 4, 5)
OUT = "results/e10b.json"


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else dict(rows=[], clarabel_skipped={})


def save(res):
    tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)


def main():
    res = load()
    done = {(r["n"], r["seed"]) for r in res["rows"]}
    brute20_ok = None; brute20_time = None
    for r in res['rows']:
        if r['n'] == 20 and 'brute_time' in r:
            brute20_ok, brute20_time = True, r['brute_time']
    for n in NS:
        for seed in SEEDS:
            if (n, seed) in done:
                continue
            edges = pb.graph("reg3", n, seed=seed)
            pop = pb.maxcut(n, edges)
            d = SpectralDual(pop, 2)
            t = time.perf_counter(); c0 = time.process_time(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500)
            dt = time.perf_counter() - t; cpu = time.process_time() - c0
            U = float(U); flU = int(np.floor(U + 1e-9))
            # best cut
            t = time.perf_counter()
            ls = local_search_maxcut(n, edges, 2000, seed=0); tls = time.perf_counter() - t
            cut, method = ls, "local_search_2000"
            # brute force: n=16 always; n=20 if it takes < 2 min; n=24 if the n=20 time extrapolates (x20) to < 3 min
            if n == 16 or (n == 20 and brute20_ok is not False) or (n == 24 and brute20_time is not None and brute20_time * 20 < 180):
                t = time.perf_counter(); bf = pb.brute_maxcut(n, edges); tbf = time.perf_counter() - t
                if n == 20:
                    brute20_ok = tbf < 120; brute20_time = tbf
                cut, method = bf, "brute"
                assert ls <= bf
                rec_bf = dict(brute_cut=bf, brute_time=tbf)
            else:
                rec_bf = {}
            row = dict(n=n, seed=seed, m=len(edges), block_size=d.sizes()[0], dim=d.dim, time=dt, cpu_time=cpu, loadavg=os.getloadavg(), evals=int(len(h)), U=U,
                       floor_U=flU, cut=cut, cut_method=method, local_search_cut=ls, local_search_time=tls,
                       certified=bool(flU == cut), margin=U - flU, **rec_bf)
            # interior-point reference (Clarabel unless MOSEK is installed)
            # skipped for every size >= the smallest size at which one solve exceeded 300 s (shared machine)
            skip_from = min([int(k) for k, v in res["clarabel_skipped"].items() if v] + [10 ** 9])
            if n < skip_from:
                t = time.perf_counter(); c0 = time.process_time(); val, _ = reference.moment_relaxation(pop, 2); tc = time.perf_counter() - t; cc = time.process_time() - c0
                row.update(ipm_value=float(val), ipm_time=tc, ipm_cpu_time=cc, ipm_solver=reference.default_solver(),
                           ipm_floor_equals_cut=bool(int(np.floor(val + 1e-6)) == cut), ipm_minus_cut=float(val) - cut)
                if tc > 300:
                    res["clarabel_skipped"][str(n)] = True
            else:
                row.update(ipm_skipped=f"interior point skipped: a size-{skip_from} solve exceeded 300 s")
            row["rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
            res["rows"].append(row); save(res)
            print(f"n={n:2d} seed={seed}: U={U:.6f} floor={flU} cut={cut} ({method}) certified={row['certified']} "
                  f"margin={row['margin']:.2e} t={dt:.1f}s evals={len(h)} ipm={row.get('ipm_value')} ipm_t={row.get('ipm_time')}", flush=True)


def ipm_one(n, seed):
    pop = pb.maxcut(n, pb.graph("reg3", n, seed=seed))
    t = time.perf_counter(); c0 = time.process_time(); val, _ = reference.moment_relaxation(pop, 2)
    return float(val), time.perf_counter() - t, time.process_time() - c0


def fill_ipm(n, cap=600.0):
    """Back-fill the interior-point reference for size-n rows that were skipped, one subprocess per instance,
    each capped at `cap` seconds (used after the machine load dropped)."""
    import subprocess
    res = load()
    for row in res["rows"]:
        if row["n"] != n or "ipm_value" in row:
            continue
        cmd = [sys.executable, __file__, "ipm", str(n), str(row["seed"])]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
            line = [l for l in p.stdout.splitlines() if l.startswith("IPM ")]
            val, tc, cc = json.loads(line[0][4:])
        except subprocess.TimeoutExpired:
            row["ipm_skipped"] = f"interior point did not finish within {cap:.0f} s (shared machine)"; save(res)
            print(f"n={n} seed={row['seed']}: Clarabel timed out ({cap:.0f} s)", flush=True); continue
        row.pop("ipm_skipped", None)
        row.update(ipm_value=val, ipm_time=tc, ipm_cpu_time=cc, ipm_solver=reference.default_solver(),
                   ipm_floor_equals_cut=bool(int(np.floor(val + 1e-6)) == row["cut"]), ipm_minus_cut=val - row["cut"],
                   ipm_loadavg=os.getloadavg())
        save(res)
        print(f"n={n} seed={row['seed']}: ipm={val:.9f} t={tc:.1f}s cpu={cc:.1f}s", flush=True)


def retime(n):
    """Re-run the smoothing solver for size-n rows that lack cpu_time (rows written before CPU timing was added);
    U and evaluations are deterministic, the first wall time is kept as time_first_run."""
    res = load()
    for row in res["rows"]:
        if row["n"] != n or "cpu_time" in row:
            continue
        d = SpectralDual(pb.maxcut(n, pb.graph("reg3", n, seed=row["seed"])), 2)
        t = time.perf_counter(); c0 = time.process_time(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500)
        dt = time.perf_counter() - t; cpu = time.process_time() - c0
        assert abs(float(U) - row["U"]) < 1e-9 and len(h) == row["evals"], (U, row["U"], len(h), row["evals"])
        row.update(time_first_run=row["time"], time=dt, cpu_time=cpu, loadavg=os.getloadavg())
        save(res); print(f"n={n} seed={row['seed']}: re-timed wall={dt:.1f}s cpu={cpu:.1f}s (first run {row['time_first_run']:.1f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "retime":
        retime(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "ipm":
        print("IPM " + json.dumps(ipm_one(int(sys.argv[2]), int(sys.argv[3]))), flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "fill":
        fill_ipm(int(sys.argv[2]))
    else:
        main()
