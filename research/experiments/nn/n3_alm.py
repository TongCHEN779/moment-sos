"""N3 with the dual-ALM solver. usage: n3_alm.py d widths seed eps time_limit rho0 inner"""
import time, json, sys, resource
from common import *
d, widths, seed, eps = int(sys.argv[1]), [int(a) for a in sys.argv[2].split(",")], int(sys.argv[3]), float(sys.argv[4])
tl, rho0, inner = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
Ws, bs, acc = train_mlp(d, widths, seed=seed, steps=2500)
rng = np.random.default_rng(100 + seed)
for trial in range(500):
    x0 = rng.uniform(-0.8, 0.8, d); lg = pnn.forward(Ws, bs, x0)
    if abs(lg[0] - lg[1]) > 0.4: break
cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
cliq = po.correlative_sparsity_cliques(P)
out = dict(d=d, widths=widths, seed=seed, eps=eps, n=info["n"], n_unstable=info["n_unstable"])
for order, cl, name in [(1, "dense", "order1"), (2, cliq, "order2_sparse")]:
    t0 = time.time(); r = po.solve_alm(P, order=order, cliques=cl, rho0=rho0, inner=inner, outer=400, time_limit=tl, verbose=True)
    out[name] = r.bound; out["t_" + name] = time.time() - t0; out["hist_" + name] = [(h[0], r.problem.f_off + r.problem.f_scale * h[2]) for h in r.history]
    out["rss_gib_" + name] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    print(name, r.bound, f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(f"../results/nn_alm_{d}_{sys.argv[2].replace(',', '-')}_{seed}_{eps}.json", "w"))
