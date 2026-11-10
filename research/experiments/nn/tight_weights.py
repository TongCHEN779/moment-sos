"""Effect of the tighter localizing weights B_i (term-wise interval bound including the constant) on the NN problems."""
from common import *
import json, time
out = []
# N1 cases
res1 = json.load(open("../results/nn_n1_new.json"))
for r in res1:
    d, widths, seed, eps = r["d"], r["widths"], r["seed"], r["eps"]
    Ws, bs, acc = train_mlp(d, widths, seed=seed)
    x0 = np.array(r["x0"]); lg = pnn.forward(Ws, bs, x0); cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
    P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
    rec = dict(case=f"d{d} {widths}", n=info["n"])
    for o in (2,):
        Wold = sum(b["weight"] for b in po.Relaxation(P, o).blocks); Wnew = sum(b["weight"] for b in po.Relaxation(P, o, tight_weights=True).blocks)
        rec[f"W_old"], rec["W_tight"] = Wold, Wnew
        v_old = po.solve_dual_ipm(po.Relaxation(P, o))[0]; v_new = po.solve_dual_ipm(po.Relaxation(P, o, tight_weights=True))[0]
        conv = lambda v: P.f_off + P.f_scale * v
        rec["ipm_old"], rec["ipm_tight"] = conv(v_old), conv(v_new)
        for tw in (False, True):
            t = time.time(); s = po.solve(P, order=o, iters_per_stage=2000, etas=(1e-3, 1e-4, 1e-5), tight_weights=tw, time_limit=300)
            rec[f"smooth_{'tight' if tw else 'old'}"] = s.bound; rec[f"t_{'tight' if tw else 'old'}"] = time.time() - t
    out.append(rec); print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
json.dump(out, open("../results/nn_tight_weights.json", "w"), indent=1)
