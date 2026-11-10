"""Run ONE baseline on ONE instance in this process; print 'RESULT {json}'.

usage: python run_baseline.py METHOD n [key=value ...]
  METHOD in {scs, cgal, bm, mosek}.  Common options: budget=1200 (s), ref=<reference value>.
  scs : eps=1e-3
  cgal: beta0=...
  bm  : k=8 beta0=... inner=...
Peak memory = ru_maxrss of this process.  Run with OMP/OPENBLAS/VECLIB threads = 8.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *


def parse(argv):
    kw = {}
    for a in argv:
        k, v = a.split("=")
        try:
            kw[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            kw[k] = v
    return kw


if __name__ == "__main__":
    method, n = sys.argv[1], int(sys.argv[2])
    kw = parse(sys.argv[3:])
    load0 = os.getloadavg()[0]
    out = dict(method=method, n=n, opts=kw, load_start=load0)
    if method == "mosek":
        from msos import problems as pb, reference
        pop = pb.maxcut(n, pb.graph("reg3", n, seed=1))
        t = time.perf_counter(); val, ts = reference.moment_relaxation(pop, 2)
        out.update(val=val, time=time.perf_counter() - t, time_solve=ts)
    elif method == "scs":
        import scs_run; out.update(scs_run.run(n, **kw))
    else:
        t0 = time.perf_counter()
        pop, d = load_instance(n)
        S = Slots(d)
        tbuild = time.perf_counter() - t0
        if method == "cgal":
            import cgal; res = cgal.run(S, **kw)
        elif method == "bm":
            import bm; res = bm.run(S, **kw)
        else:
            raise SystemExit("unknown method")
        res["time_build"] = tbuild
        out.update(res)
    out["rss_gib"] = peak_rss_gib()
    out["load_end"] = os.getloadavg()[0]
    print("RESULT " + json.dumps(out))
