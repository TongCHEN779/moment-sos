"""Check  U <= U_eta <= U + eta*sum_b w_b log(1+s_b)  (log s_b for constant-trace blocks)."""
import sys, math; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb
worst = 0
for pop, r in [(pb.box_qp(4, 1), 2), (pb.maxcut(6, pb.graph("K", 6)), 2), (pb.disk(), 2)]:
    d = SpectralDual(pop, r); rng = np.random.default_rng(1)
    for eta in (1.0, 0.1, 0.01):
        th = rng.standard_normal(d.dim) * 0.5
        o = d.evaluate(th, eta=eta, grad=False)
        slack = eta * sum(b["weight"] * math.log(b["size"] + (1 if b["plus"] else 0)) for b in d.blocks)
        assert o["U"] <= o["Us"] + 1e-12 and o["Us"] <= o["U"] + slack + 1e-12, (pop.name, eta, o, slack)
        worst = max(worst, (o["Us"] - o["U"]) / slack)
print("smoothing sandwich holds; max (Us-U)/bound =", round(worst, 3))
