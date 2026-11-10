"""E10a: dual augmented Lagrangian (msos.solvers.alm) versus smoothing + L-BFGS on order-2 MAX-CUT,
random 3-regular graphs (networkx seed 1), n = 16, 20, 24, 30.
Each run is executed in its own subprocess (clean peak RSS); results are appended to results/e10a.json after
every run.  Usage:  python e10a_alm_maxcut.py            (driver)
                    python e10a_alm_maxcut.py run n solver rho0 time_limit   (one run; prints a JSON line)"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "3"
sys.path.insert(0, ".."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, time, resource, subprocess
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
from e10_common import local_search_maxcut, times_to_gap, thin

NS = (16, 20, 24, 30)
RHOS = (100.0, 300.0, 1000.0)
OUT = "results/e10a.json"
CERT_EVERY = 20              # ALM: evaluate the certificate every 20 inner evaluations (finer time resolution)


def instance(n):
    return pb.maxcut(n, pb.graph("reg3", n, seed=1))


def one_run(n, solver, rho0, time_limit):
    pop = instance(n)
    d = SpectralDual(pop, 2)
    t = time.perf_counter(); c0 = time.process_time()
    if solver == "smoothing":
        U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=time_limit)
        info = dict(status="schedule_done" if h[-1, 1] < time_limit else "time_limit", n_eval=len(h))
    else:
        U, th, h, info = solvers.alm(d, rho0=rho0, time_limit=time_limit, cert_every=CERT_EVERY)
        info = {k: v for k, v in info.items() if k not in ("X", "t")}
    dt = time.perf_counter() - t; cpu = time.process_time() - c0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30          # GiB (macOS: bytes)
    return dict(n=n, solver=solver, rho0=rho0, time_limit=time_limit, final_U=float(U), final_time=dt, cpu_time=cpu, loadavg=os.getloadavg(),
                evals=int(len(h) if solver == "smoothing" else info["n_eval"]), rss_gib=rss, status=info["status"],
                outer_iters=info.get("outer_iters"), rho_final=info.get("rho"), block_size=d.sizes()[0], dim=d.dim,
                hist=np.asarray(h).tolist())


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else dict(refs={}, runs=[])


def save(res):
    tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)


def clarabel_value(n):
    pop = instance(n)
    t = time.perf_counter(); val, _ = reference.moment_relaxation(pop, 2)
    return float(val), time.perf_counter() - t


def references(res, clarabel_cap=600.0, skip_clarabel=()):
    """Relaxation value = integer max cut on these graphs.  n=20,30 from results/e3_bounds.json (MOSEK floor_ub);
    n=16,24: interior-point value (Clarabel, in a subprocess capped at `clarabel_cap` s) rounded to the integer
    that equals the brute-force / local-search cut.  If Clarabel does not finish within the cap the reference is
    the cut itself, marked provisional until a solver run certifies U - cut < 1e-6 (see confirm_refs)."""
    e3 = json.load(open("results/e3_bounds.json"))
    for n in NS:
        key = str(n)
        if key in res["refs"]:
            continue
        edges = pb.graph("reg3", n, seed=1)
        cut = pb.brute_maxcut(n, edges) if n <= 16 else local_search_maxcut(n, edges, 2000, seed=0)
        rec = dict(cut=cut, cut_method="brute" if n <= 16 else "local_search_2000")
        if key in e3:
            rec.update(ref=float(e3[key]["floor_ub"]), source="e3_bounds.json floor_ub", e3=e3[key], provisional=False)
        else:
            val, tc = (None, None) if n in skip_clarabel else run_clarabel(n, clarabel_cap)
            if val is not None:
                fl = float(np.round(val))             # relaxation >= max cut; Clarabel may return value - 1e-7
                ok = (abs(val - fl) < 0.05) and (cut == fl)
                rec.update(clarabel_value=val, clarabel_time=tc, ref=fl if ok else val, provisional=not ok,
                           source="round(Clarabel value), certified by matching cut" if ok else "Clarabel value (integer test failed)")
            else:
                rec.update(clarabel_value=None, clarabel_time=None, ref=float(cut), provisional=True,
                           source=f"local-search cut (Clarabel exceeded {clarabel_cap:.0f} s cap); to be confirmed by certified U")
        res["refs"][key] = rec
        print(f"ref n={n}: {rec}", flush=True); save(res)


def run_clarabel(n, cap):
    cmd = [sys.executable, __file__, "clarabel", str(n)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
        line = [l for l in p.stdout.splitlines() if l.startswith("CLARABEL ")]
        return tuple(json.loads(line[0][9:])) if line else (None, None)
    except subprocess.TimeoutExpired:
        return None, None


def late_clarabel(res, cap=600.0):
    """Interior-point check of the references that were set from the cut alone (run after the solver runs)."""
    for key, rec in res["refs"].items():
        if rec.get("clarabel_value", 0) is None:
            val, tc = run_clarabel(int(key), cap)
            rec["clarabel_value"], rec["clarabel_time"] = val, tc
            if val is None:
                rec["clarabel_note"] = f"Clarabel did not finish within {cap:.0f} s (shared machine)"
            else:
                rec["clarabel_note"] = ("Clarabel value rounds to the cut" if abs(val - rec["cut"]) < 0.05
                                        else "Clarabel value does NOT round to the cut")
            print(f"late Clarabel n={key}: {val} ({tc})", flush=True); save(res)


def confirm_refs(res):
    """A provisional reference ref=cut is confirmed when some run certifies U - cut < 1e-6 (then rho_2 in [cut, cut+1e-6])."""
    for key, rec in res["refs"].items():
        if rec.get("provisional"):
            us = [r["final_U"] for r in res["runs"] if str(r["n"]) == key]
            if us and min(us) - rec["ref"] < 1e-6:
                rec["provisional"] = False; rec["confirmed_by"] = f"min certified U = {min(us):.9f}"
    save(res)


def main():
    res = load()
    references(res, skip_clarabel=(24,))             # n=24 Clarabel is deferred to the end (late_clarabel)
    done = {(r["n"], r["solver"], r["rho0"]) for r in res["runs"]}
    for n in NS:
        tl = 900 if n == 30 else 600
        runs = [("smoothing", 0.0)] + [("alm", rho) for rho in RHOS]
        for solver, rho0 in runs:
            if (n, solver, rho0) in done:
                continue
            cmd = [sys.executable, __file__, "run", str(n), solver, str(rho0), str(tl)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(f"FAILED n={n} {solver} rho0={rho0}:\n{p.stderr[-2000:]}", flush=True); continue
            r = json.loads(line[0][7:])
            ref = res["refs"][str(n)]["ref"]
            r["ref"] = ref; r["t_gap"] = times_to_gap(r["hist"], ref); r["final_gap"] = r["final_U"] - ref
            r["hist"] = thin(r["hist"])
            res["runs"].append(r); save(res)
            confirm_refs(res)
            print(f"n={n:2d} {solver:9s} rho0={rho0:6.0f}: U={r['final_U']:.6f} gap={r['final_gap']:.2e} t={r['final_time']:.1f}s "
                  f"evals={r['evals']} rss={r['rss_gib']:.2f}GiB t_gap={r['t_gap']} status={r['status']}", flush=True)
    late_clarabel(res)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clarabel":
        print("CLARABEL " + json.dumps(clarabel_value(int(sys.argv[2]))), flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        n, solver, rho0, tl = int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
        print("RESULT " + json.dumps(one_run(n, solver, rho0, tl)), flush=True)
    else:
        main()
