"""N2: squared l2 Lipschitz bound of one-hidden-layer ReLU nets: LipSDP-Neuron vs order-2 / order-3 relaxation."""
from common import *
import time, json

def make(kind, d, w, m, seed):
    rng = np.random.default_rng(seed)
    if kind == "rand":
        return rng.standard_normal((w, d)) / np.sqrt(d), rng.standard_normal((m, w)) / np.sqrt(w)
    B = rng.standard_normal((2, d))
    return rng.standard_normal((w, 2)) @ B / np.sqrt(d) + 0.3 * rng.standard_normal((w, d)) / np.sqrt(d), rng.standard_normal((m, w)) / np.sqrt(w)

if __name__ == '__main__':
    cases = [("rand", 3, 8, 3, 0), ("rand", 3, 8, 3, 2), ("rand", 3, 8, 3, 3), ("corr", 3, 10, 3, 2), ("corr", 4, 10, 4, 0),
             ("corr", 4, 12, 4, 0), ("corr", 4, 14, 4, 0), ("corr", 5, 14, 4, 1)]
    import os
    out = []
    old = {(r['kind'], r['d'], r['w'], r['m'], r['seed']): r for r in json.load(open('../results/nn_n2.json'))} if os.environ.get('REUSE_OURS') else {}
    done = {(r['kind'], r['d'], r['w'], r['m'], r['seed']) for r in out}
    for kind, d, w, m, seed in cases:
        if (kind, d, w, m, seed) in done:
            continue
        W0, W1 = make(kind, d, w, m, seed)
        rec = dict(kind=kind, d=d, w=w, m=m, seed=seed, n=d + w)
        rec["exact_pattern"] = pnn.lipschitz_exact_pattern(W0, W1)
        rec["lipsdp"] = pnn.lipsdp_neuron(W0, W1)
        rec["spectral_prod"] = (np.linalg.norm(W0, 2) * np.linalg.norm(W1, 2)) ** 2
        Pm = pnn.lipschitz_problem(W0, W1)
        if d + w <= 16:
            try:
                v, t = exact_relaxation(pnn.lipschitz_problem(W0, W1, mixed=False), 2); rec["mosek2"], rec["t_mosek2"] = v, t
            except Exception as e:
                rec["mosek2_error"] = str(e)[:80]
        for o, iters in ((2, 4000), (3, 3000)):
            if o == 3 and d + w > 14:
                continue
            if (kind, d, w, m, seed) in old and f"ours{o}" in old[(kind, d, w, m, seed)]:
                for k in (f"ours{o}", f"t_ours{o}", f"block{o}"):
                    rec[k] = old[(kind, d, w, m, seed)][k]
                continue
            t0 = time.time(); r = po.solve(Pm, order=o, iters_per_stage=iters, etas=(1e-3, 1e-4, 1e-5), time_limit=1500)
            rec[f"ours{o}"], rec[f"t_ours{o}"], rec[f"block{o}"] = r.bound, time.time() - t0, max(r.relaxation.sizes())
        out.append(rec); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        json.dump(out, open("../results/nn_n2_new.json", "w"), indent=1)
