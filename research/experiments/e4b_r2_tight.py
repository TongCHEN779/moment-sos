"""E4b: is order 2 already tight on other MAX-CUT families?  (small sizes, MOSEK)"""
import sys, json, itertools; sys.path.insert(0, ".")
import numpy as np
from msos import problems as pb, reference
def brute(n, edges, w):
    m = np.arange(2 ** (n - 1), dtype=np.int64); c = np.zeros(len(m))
    for (i, j), x in zip(edges, w): c += x * (((m >> i) ^ (m >> j)) & 1)
    return float(c.max())
rng = np.random.default_rng(3); out = []
for name, n, edges, w in [("G(16,1/2)", 16, pb.graph("gnp", 16, seed=2), None),
                          ("spin glass K14", 14, list(itertools.combinations(range(14), 2)), rng.choice([-1., 1.], 91)),
                          ("spin glass K16", 16, list(itertools.combinations(range(16), 2)), rng.choice([-1., 1.], 120))]:
    w = np.ones(len(edges)) if w is None else w
    pop = pb.maxcut(n, edges, list(w)); opt = brute(n, edges, w)
    vals = [reference.moment_relaxation(pop, r)[0] for r in (1, 2, 3)]
    out.append(dict(name=name, n=n, opt=opt, r1=vals[0], r2=vals[1], r3=vals[2])); print(out[-1], flush=True)
json.dump(out, open("experiments/results/e4b.json", "w"), indent=1)
