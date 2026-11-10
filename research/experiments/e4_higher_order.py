"""E4: order r=1,2,3 on random 3-regular MAX-CUT; r=3 is out of reach of the interior-point baseline."""
import sys, json, time, resource; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers

def brute(n, edges):
    m = np.arange(2 ** (n - 1), dtype=np.int64)
    cut = np.zeros(len(m), np.int16)
    for i, j in edges:
        cut += (((m >> i) ^ (m >> j)) & 1).astype(np.int16)
    return int(cut.max())

n = int(sys.argv[1]); rs = [int(a) for a in sys.argv[2:]] or [1, 2, 3]
edges = pb.graph("reg3", n, seed=1); opt = brute(n, edges)
pop = pb.maxcut(n, edges)
out = dict(n=n, opt=opt, runs=[])
for r in rs:
    d = SpectralDual(pop, r)
    t = time.perf_counter(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=500, time_limit=5400); dt = time.perf_counter() - t
    rec = dict(r=r, block=d.sizes()[0], dim=d.dim, U=U, time=dt, evals=len(h),
               rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, hist=h[:, [1, 3]].tolist())
    out["runs"].append(rec)
    print(f"n={n} r={r} s={rec['block']} U={U:.5f} opt={opt} t={dt:.0f}s rss={rec['rss_gib']:.2f}GiB", flush=True)
    json.dump(out, open(f"results/e4_n{n}.json", "w"))
