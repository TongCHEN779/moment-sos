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
print("target (MOSEK sparse order 2) = -0.4106693; order 1 = 2.0347", flush=True)
for rho0, ws in [(10, True), (100, True), (1000, True), (100, False)]:
    t = time.time(); r = po.solve_alm(P, order=2, cliques=cl, rho0=rho0, warm_start=ws, time_limit=200)
    print(f"rho0={rho0} warm={ws}: bound {r.bound:.7f}  err {r.bound+0.4106693:.2e}  evals {r.n_eval}  {time.time()-t:.0f}s", flush=True)
