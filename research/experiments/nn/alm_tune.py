from common import *
import time, sys
d, widths, seed, eps = 4, [24, 24], 3, 0.0995
Ws, bs, acc = train_mlp(d, widths, seed=seed, steps=2500)
rng = np.random.default_rng(100 + seed)
for trial in range(500):
    x0 = rng.uniform(-0.8, 0.8, d); lg = pnn.forward(Ws, bs, x0)
    if abs(lg[0] - lg[1]) > 0.4: break
cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
cl = po.correlative_sparsity_cliques(P)
print("n", info["n"], "unstable", info["n_unstable"], "exact MILP", pnn.verification_exact_milp(Ws, bs, x0, eps, c)[0], flush=True)
for name, cls_ in (("dense", "dense"), ("sparse", cl)):
    v, t = exact_relaxation(P, 2, cliques=None if name == "dense" else cl); print("MOSEK order 2", name, v, f"{t:.0f}s", flush=True)
    R = po.Relaxation(P, 2, cls_); print("  blocks", len(R.blocks), "max", max(R.sizes()), "moments", R.nmom, flush=True)
