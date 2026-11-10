"""Polynomial-identity test: sigma - f - sum s_i g_i - sum p_j h_j == sum_k v_k^T Q_k v_k at random points."""
import sys; sys.path.insert(0, "..")
import numpy as np
from msos import POP, SpectralDual, problems as pb

def mono(ring, keys, x):
    return np.array([np.prod(x ** np.array(ring.decode(k))) for k in keys])

def residual(pop, r, cliques=None, trials=5, seed=0):
    d = SpectralDual(pop, r, cliques); rng = np.random.default_rng(seed)
    th = rng.standard_normal(d.dim) * 0.5
    t = th[d.sl_Theta]; pr = np.bincount(d.slot_g, weights=d.slot_w * t, minlength=d.nmom)
    th[d.sl_Theta] = t - d.slot_w * (pr / d.W2)[d.slot_g]          # zero-sum part
    Qs = d.gram_blocks(th); K = len(d.cliques); worst = 0
    for _ in range(trials):
        x = rng.choice([-1.0, 1.0], pop.n) if pop.boolean else rng.uniform(-1, 1, pop.n)
        ev = lambda poly: sum(c * np.prod(x ** np.array(e)) for e, c in poly.items())
        lhs = th[0] - ev(pop.f)
        for i, (g, info) in enumerate(zip(pop.ineq, d.ineq_info)):
            z = mono(d.ring, info["Z"], x); lhs -= z @ Qs[K + i] @ z * ev(g)
        for j, (h, info) in enumerate(zip(pop.eq, d.eq_info)):
            pj = th[d.sl_p][d.p_off[j]:d.p_off[j + 1]]
            lhs -= (pj @ mono(d.ring, info["M"], x)) * ev(h)
        rhs = sum(mono(d.ring, B, x) @ Qs[k] @ mono(d.ring, B, x) for k, B in enumerate(d.cl_basis))
        worst = max(worst, abs(lhs - rhs))
    return worst

if __name__ == "__main__":
    cases = [("boxqp4 r2", pb.box_qp(4, 2), 2, None), ("maxcut K4 r2 (Boolean)", pb.maxcut(4, pb.graph("K", 4)), 2, None),
             ("disk r3", pb.disk(), 3, None)]
    pc, cl = pb.chain_quartic(5, 1); cases.append(("chain5 sparse r2", pc, 2, cl))
    # equality-constrained, non-Boolean: sphere  max x1 + x2 s.t. x1^2+x2^2-1 = 0
    sph = POP(2, {(1, 0): 1.0, (0, 1): 1.0}, eq=[{(2, 0): 1.0, (0, 2): 1.0, (0, 0): -1.0}], name="sphere")
    cases.append(("sphere (equality) r2", sph, 2, None))
    for name, pop, r, cl in cases:
        print(f"{name:26s} max |identity residual| = {residual(pop, r, cl):.2e}")
