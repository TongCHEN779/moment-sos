from common import *
import time
d, widths = 2, [8, 8]
Ws, bs, acc = train_mlp(d, widths, seed=1)
print("train acc", acc)
rng = np.random.default_rng(5)
# choose a correctly classified point and find the exact robust radius by bisection with the MILP
for trial in range(50):
    x0 = rng.uniform(-0.8, 0.8, d); lg = pnn.forward(Ws, bs, x0); cls = int(lg.argmax())
    c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
    if abs(lg[0] - lg[1]) > 0.5: break
lo, hi = 0.0, 1.0
for _ in range(14):
    mid = (lo + hi) / 2
    val = pnn.verification_exact_milp(Ws, bs, x0, mid, c)[0]
    lo, hi = (mid, hi) if val < 0 else (lo, mid)
eps_star = lo; print("point", x0, "margin", lg, "exact robust radius", eps_star)
for frac in (0.9, 1.0):
    eps = eps_star * frac
    P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
    ex = pnn.verification_exact_milp(Ws, bs, x0, eps, c)[0]
    r1, t1 = exact_relaxation(P, 1); r2, t2 = exact_relaxation(P, 2)
    print(f"eps={eps:.4f} n={info['n']} unstable={info['n_unstable']} IBP={info['ibp_upper']:.4f}  exact={ex:.4f}  MOSEK order1={r1:.4f} ({t1:.1f}s) order2={r2:.4f} ({t2:.1f}s)")
    for o in (1, 2):
        t = time.time(); res = po.solve(P, order=o, iters_per_stage=3000, etas=(1e-2, 1e-3, 1e-4, 1e-5)); print("   ours", o, res.bound, f"{time.time()-t:.0f}s")
