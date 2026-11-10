"""Where does one oracle evaluation (value + gradient of the smoothed dual) spend its time?  (moment_sos, order-2 MAX-CUT)
usage (from the repo root):  python experiments/bench/profile_oracle.py"""
import os, sys, time, json
sys.path.insert(0, "../src"); sys.path.insert(0, ".")
import numpy as np, torch
import moment_sos as po
from moment_sos.oracle import Oracle
from msos import problems as pb

def maxcut_problem(n):
    e = pb.graph("reg3", n, seed=1); s = po.variables(n)
    return po.Problem(sum((1 - s[i] * s[j]) / 2 for i, j in e), boolean=True)

def timeit(fn, reps):
    fn(); t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps

rows = []
for n, reps in ((30, 20), (40, 8), (60, 3)):
    P = maxcut_problem(n); R = po.Relaxation(P, 2); O = Oracle(R)
    th = O.theta0().clone(); th += 0.01 * torch.randn_like(th)
    for thr in (1, 2, 4, 8):
        torch.set_num_threads(thr)
        t_gram = timeit(lambda: O.gram_blocks(th), reps)
        Q = O.gram_blocks(th)[0]
        t_eigh = timeit(lambda: torch.linalg.eigh(Q), reps)
        def full():
            x = th.clone().requires_grad_(True); Us, _ = O.evaluate(x, 1e-3); Us.backward()
        t_full = timeit(full, reps)
        rows.append(dict(n=n, s=int(Q.shape[0]), threads=thr, t_gram=t_gram, t_eigh=t_eigh, t_eval_grad=t_full, eigh_share=t_eigh / t_full))
        print(f"n={n:3d} s={Q.shape[0]:5d} threads={thr}: gram {t_gram*1e3:7.1f} ms | eigh {t_eigh*1e3:8.1f} ms | evaluation+gradient {t_full*1e3:8.1f} ms | eigh share {100*t_eigh/t_full:3.0f}%", flush=True)
json.dump(rows, open("experiments/results/profile_oracle.json", "w"), indent=1)
