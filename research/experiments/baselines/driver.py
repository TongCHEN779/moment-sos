"""Sequential experiment driver.  Each (method, n, options) runs in its own subprocess (8 BLAS threads),
waits first until the 1-min load average is <= 6 (so timings are not contaminated; waited time is not counted).
Results are appended to experiments/results/baselines_raw.jsonl.

usage: python driver.py tune|main|ref [n ...]
"""
import sys, os, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "baselines_raw.jsonl")
ENV = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", VECLIB_MAXIMUM_THREADS="8")
NS = [16, 20, 26, 30, 40]


def refs():
    out = {}
    for l in open(RAW):
        d = json.loads(l)
        if d.get("method") == "mosek":
            out[d["n"]] = d["val"]
    out[40] = 54.006490432611415            # our own best certified bound (MOSEK is OOM-killed at n=40)
    return out


def run(method, n, tag, **opts):
    while os.getloadavg()[0] > 6:
        print("load", os.getloadavg()[0], "waiting", flush=True); time.sleep(20)
    time.sleep(2)
    args = [sys.executable, os.path.join(HERE, "run_baseline.py"), method, str(n)] + [f"{k}={v}" for k, v in opts.items()]
    p = subprocess.run(args, env=ENV, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        d = dict(method=method, n=n, opts=opts, error=p.stderr[-800:], returncode=p.returncode)
    else:
        d = json.loads(line[0][7:])
    d["tag"] = tag
    with open(RAW, "a") as f:
        f.write(json.dumps(d) + "\n")
    print(tag, method, n, opts, d.get("val"), d.get("time"), d.get("status"), flush=True)
    return d


def tuned(n, method, k=None):
    """Best tuning run for (method, n[, k]): earliest strict-1e-2 time, else smallest final strict error."""
    from metrics import time_to, strict_err
    R = refs()[n]; best = None
    for l in open(RAW):
        d = json.loads(l)
        if d.get("tag") != "tune" or d.get("method") != method or d["n"] != n or "hist" not in d:
            continue
        if k is not None and d["opts"].get("k") != k:
            continue
        t2 = time_to(d["hist"], R, 1e-2)
        key = (0, t2) if t2 is not None else (1, min(strict_err(r, R) for r in d["hist"]))
        if best is None or key < best[0]:
            best = (key, d["opts"])
    return best[1]


if __name__ == "__main__":
    mode = sys.argv[1]
    ns = [int(a) for a in sys.argv[2:]] or NS
    if mode == "ref":
        for n in ns:
            run("mosek", n, "ref")
    elif mode == "tune":
        R = refs()
        for n in ns:
            for b in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                run("cgal", n, "tune", beta0=b, budget=90 if n < 40 else 180, ref=R[n])
            for k in (8, 16):
                for b in (1.0, 10.0):
                    run("bm", n, "tune", k=k, beta0=b, budget=90 if n < 40 else 180, ref=R[n])
    elif mode == "tune2":      # extend the grids at the edges (best CGAL beta0 was at the lower edge; BM beta0=1 beat 10)
        R = refs()
        for n in ns:
            bud = 90 if n < 40 else 180
            for b in (1e-5, 3e-5):
                run("cgal", n, "tune", beta0=b, budget=bud, ref=R[n])
            for k in (8, 16):
                for b in (0.1, 0.3):
                    run("bm", n, "tune", k=k, beta0=b, budget=bud, ref=R[n])
    elif mode == "tune3":
        R = refs()
        for n in ns:
            for k in (8, 16):
                run("bm", n, "tune", k=k, beta0=0.03, budget=180 if n == 40 else 90, ref=R[n])
    elif mode == "ours16":     # fresh run of our method at n=16 (e3.jsonl starts at n=10/14/20) for memory + same-thread timing
        d = subprocess.run([sys.executable, os.path.join(HERE, "..", "run_one.py"), "fo", "16", "2"], cwd=os.path.join(HERE, "..", ".."),
                           env=ENV, capture_output=True, text=True)
        o = json.loads([l for l in d.stdout.splitlines() if l.startswith("RESULT ")][0][7:]); o["tag"] = "ours"
        open(RAW, "a").write(json.dumps(o) + "\n"); print(o["val"], o["time"], o["rss_gib"])
    elif mode == "main":
        R = refs(); R[40] = 54.0       # stopping reference only; collect.py scores against 54.0065 (and 54.0)
        for n in ns:
            run("scs", n, "main", eps=1e-3, budget=1200)
            run("scs", n, "main", eps=1e-5, budget=1200)
            o = tuned(n, "cgal"); run("cgal", n, "main", beta0=o["beta0"], budget=1200, ref=R[n])
            for k in (8, 16):
                o = tuned(n, "bm", k); run("bm", n, "main", k=k, beta0=o["beta0"], budget=1200, ref=R[n])
