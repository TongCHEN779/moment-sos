"""N3: verification of larger ReLU nets: order 1 vs order 2 (sparse and dense cliques), MOSEK where it fits.
usage: n3_scale.py d widths seed eps time_limit  mosek_sparse(0/1)  dense(0/1)"""
import time, json, sys, resource
from common import *
d, widths, seed, eps = int(sys.argv[1]), [int(a) for a in sys.argv[2].split(",")], int(sys.argv[3]), float(sys.argv[4])
tl, do_mosek, do_dense = float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
Ws, bs, acc = train_mlp(d, widths, seed=seed, steps=2500)
rng = np.random.default_rng(100 + seed)
for trial in range(500):
    x0 = rng.uniform(-0.8, 0.8, d); lg = pnn.forward(Ws, bs, x0)
    if abs(lg[0] - lg[1]) > 0.4: break
cls = int(lg.argmax()); c = np.zeros(2); c[1 - cls] = 1; c[cls] = -1
P, info = pnn.relu_verification(Ws, bs, x0, eps, c)
rec = dict(d=d, widths=widths, seed=seed, eps=eps, n=info["n"], n_unstable=info["n_unstable"], ibp=info["ibp_upper"])
rec["exact"], rec["milp_status"], _ = pnn.verification_exact_milp(Ws, bs, x0, eps, c)
rec["pgd"] = pnn.verification_pgd(Ws, bs, x0, eps, c)
rec["mosek1"], rec["t_mosek1"] = exact_relaxation(P, 1)
for cl in ("dense", "auto"):
    R2 = po.Relaxation(P, 2, cl)
    rec[f"order2_{cl}"] = dict(max_block=max(R2.sizes()), blocks=len(R2.blocks), moments=R2.nmom, multipliers=R2.dim,
                              schur_gib_lower=R2.nmom ** 2 * 4 / 2 ** 30)
print(json.dumps(rec), flush=True)
cliq = po.correlative_sparsity_cliques(P)
if do_mosek:
    t0 = time.time(); rec["mosek2_sparse"], _ = exact_relaxation(P, 2, cliques=cliq); rec["t_mosek2_sparse"] = time.time() - t0
    rec["rss_after_mosek_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    print("MOSEK sparse order 2:", rec["mosek2_sparse"], rec["t_mosek2_sparse"], flush=True)
t0 = time.time(); r1 = po.solve(P, order=1, iters_per_stage=3000, etas=(1e-2, 1e-3, 1e-4, 1e-5), time_limit=900); rec["ours1"], rec["t_ours1"] = r1.bound, time.time() - t0
print("ours order 1:", rec["ours1"], flush=True)
for cl, name in ([("auto", "sparse")] + ([("dense", "dense")] if do_dense else [])):
    t0 = time.time()
    r = po.solve(P, order=2, cliques=cliq if cl == "auto" else "dense", iters_per_stage=3000, etas=(1e-3, 1e-4, 1e-5), time_limit=tl, verbose=True)
    rec[f"ours2_{name}"], rec[f"t_ours2_{name}"] = r.bound, time.time() - t0
    rec[f"hist_{name}"] = [(h[0], h[2]) for h in r.history[::25]]
    rec[f"rss_gib_{name}"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    print(f"ours order 2 ({name}):", r.bound, rec[f"t_ours2_{name}"], flush=True)
json.dump(rec, open(f"../results/nn_n3_{d}_{sys.argv[2].replace(',', '-')}_{seed}_{eps}.json", "w"))
