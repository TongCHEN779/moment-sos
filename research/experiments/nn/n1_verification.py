"""N1: robustness verification of small trained ReLU nets: IBP vs order 1 (Shor/SDP) vs order 2, exact = MILP."""
from common import *
import time, json
cases = [(2, [8, 8], 1), (2, [10, 10], 2), (3, [8, 8], 3), (3, [10, 10], 4), (4, [8, 8], 5), (2, [12, 12], 6)]
import os
old = {(r['d'], tuple(r['widths']), r['seed']): r for r in json.load(open('../results/nn_n1.json'))} if os.environ.get('REUSE_OURS') else {}
out = []
for d, widths, seed in cases:
    Ws, bs, acc = train_mlp(d, widths, seed=seed)
    rng = np.random.default_rng(100 + seed)
    for trial in range(200):
        x0 = rng.uniform(-0.8, 0.8, d); lg = pnn.forward(Ws, bs, x0)
        if abs(lg[0] - lg[1]) > 0.4: break
    cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1

    def bisect(fun, lo=0.0, hi=1.0, it=12):
        for _ in range(it):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fun(mid) < 0 else (lo, mid)
        return lo
    f_ibp = lambda e: pnn.relu_verification(Ws, bs, x0, e, c)[1]["ibp_upper"]
    f_ex = lambda e: pnn.verification_exact_milp(Ws, bs, x0, e, c)[0]
    f_r = lambda o: (lambda e: exact_relaxation(pnn.relu_verification(Ws, bs, x0, e, c)[0], o)[0])
    rad = dict(ibp=bisect(f_ibp), exact=bisect(f_ex), order1=bisect(f_r(1)), order2=bisect(f_r(2), it=9))
    eps = 0.97 * rad["exact"]
    P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
    rec = dict(d=d, widths=widths, seed=seed, acc=acc, x0=x0.tolist(), margin=float(abs(lg[0] - lg[1])), radius=rad, eps=eps, n=info["n"],
               n_unstable=info["n_unstable"], ibp=info["ibp_upper"], exact=f_ex(eps), pgd=pnn.verification_pgd(Ws, bs, x0, eps, c))
    for o in (1, 2):
        v, t = exact_relaxation(P, o); rec[f"mosek{o}"], rec[f"t_mosek{o}"] = v, t
        if (d, tuple(widths), seed) in old:
            rec[f"ours{o}"], rec[f"t_ours{o}"] = old[(d, tuple(widths), seed)][f"ours{o}"], old[(d, tuple(widths), seed)][f"t_ours{o}"]; continue
        t0 = time.time(); r = po.solve(P, order=o, iters_per_stage=4000, etas=(1e-3, 1e-4, 1e-5)); rec[f"ours{o}"], rec[f"t_ours{o}"] = r.bound, time.time() - t0
    out.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("x0",)}), flush=True)
    json.dump(out, open("../results/nn_n1_new.json", "w"), indent=1)
