"""Re-solve the N1 (verification) and N2 (Lipschitz) instances with the dual-ALM solver (rho0=1000)."""
from common import *
import time, json
from n2_lipschitz import make

res1 = json.load(open("../results/nn_n1_new.json")); out1 = []
for r in res1:
    d, widths, seed, eps = r["d"], r["widths"], r["seed"], r["eps"]
    Ws, bs, acc = train_mlp(d, widths, seed=seed)
    x0 = np.array(r["x0"]); lg = pnn.forward(Ws, bs, x0); cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
    P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
    rec = dict(d=d, widths=widths, seed=seed, eps=eps)
    for o in (1, 2):
        t = time.time(); s = po.solve_alm(P, order=o, time_limit=600); rec[f"alm{o}"], rec[f"t_alm{o}"] = s.bound, time.time() - t
    out1.append(rec); print(json.dumps(rec), "| MOSEK", r["mosek1"], r["mosek2"], flush=True)
    json.dump(out1, open("../results/nn_alm_n1.json", "w"), indent=1)

res2 = json.load(open("../results/nn_n2_new.json")); out2 = []
for r in res2:
    W0, W1 = make(r["kind"], r["d"], r["w"], r["m"], r["seed"]); Pm = pnn.lipschitz_problem(W0, W1)
    rec = dict(kind=r["kind"], d=r["d"], w=r["w"], seed=r["seed"])
    for o in ((2, 3) if r["d"] + r["w"] <= 14 else (2,)):
        t = time.time(); s = po.solve_alm(Pm, order=o, time_limit=900); rec[f"alm{o}"], rec[f"t_alm{o}"] = s.bound, time.time() - t
    out2.append(rec); print(json.dumps(rec), "| exact", r["exact_pattern"], "lipsdp", r["lipsdp"], flush=True)
    json.dump(out2, open("../results/nn_alm_n2.json", "w"), indent=1)
