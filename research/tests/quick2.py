import sys, time; sys.path.insert(0, "..")
import numpy as np
from msos import SpectralDual, problems as pb, solvers, reference
# gradient check
for pop, r in [(pb.box_qp(3, 1), 2), (pb.maxcut(4, pb.graph("K", 4)), 2)]:
    d = SpectralDual(pop, r); rng = np.random.default_rng(0)
    th = rng.standard_normal(d.dim) * 0.3
    # project Theta to zero-sum: not required for value; gradient is projected, so test along zero-sum dirs
    o = d.evaluate(th, eta=0.2)
    for _ in range(3):
        dirn = rng.standard_normal(d.dim)
        # make Theta part zero-sum
        t = dirn[d.sl_Theta]; pr = np.bincount(d.slot_g, weights=d.slot_w * t, minlength=d.nmom)
        dirn[d.sl_Theta] = t - d.slot_w * (pr / d.W2)[d.slot_g]
        e = 1e-6
        fd = (d.evaluate(th + e * dirn, 0.2, grad=False)["Us"] - d.evaluate(th - e * dirn, 0.2, grad=False)["Us"]) / (2 * e)
        print(pop.name, "grad check", fd, o["g"] @ dirn)
cases = [("K5 r2", pb.maxcut(5, pb.graph("K", 5)), 2), ("C5 r1", pb.maxcut(5, pb.graph("C", 5)), 1),
         ("petersen r2", pb.maxcut(10, pb.graph("petersen", 10)), 2), ("boxqp4 r2", pb.box_qp(4, 3), 2)]
pc, cl = pb.chain_quartic(5, 1); cases.append(("chain5 sparse r2", pc, 2, cl))
for c in cases:
    name, pop, r = c[:3]; cl = c[3] if len(c) > 3 else None
    d = SpectralDual(pop, r, cliques=cl)
    t = time.time(); U, th, h = solvers.lbfgs_continuation(d, iters_per_stage=400); t = time.time() - t
    ref = reference.moment_relaxation(pop, r, cliques=cl)[0]
    refn = reference.moment_relaxation(pop, r, cliques=cl, trace_cuts=False)[0]
    print(f"{name:18s} FO {U:.6f} ({t:.1f}s) | MOSEK+tracecuts {ref:.6f} | MOSEK plain Lasserre {refn:.6f}")
