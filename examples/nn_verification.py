"""Robustness verification of a small ReLU net: order-1 (Shor/SDP) vs order-2 relaxation."""
import numpy as np
import moment_sos as po
from moment_sos import nn

rng = np.random.default_rng(0)
d, w = 3, 6
Ws = [rng.standard_normal((w, d)), rng.standard_normal((w, w)) * 0.5, rng.standard_normal((2, w))]
bs = [rng.standard_normal(w) * 0.2, rng.standard_normal(w) * 0.2, np.zeros(2)]
x0, eps, c = rng.uniform(-0.5, 0.5, d), 0.3, np.array([1.0, -1.0])      # maximise logit_0 - logit_1

problem, info = nn.relu_verification(Ws, bs, x0, eps, c)
print(problem, "unstable neurons:", info["n_unstable"], " interval bound:", round(info["ibp_upper"], 4))
print("exact (MILP):", nn.verification_exact_milp(Ws, bs, x0, eps, c)[0])
for order in (1, 2):
    print(f"order {order}:", po.solve(problem, order=order, iters_per_stage=2000))
