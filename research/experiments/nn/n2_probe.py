from common import *
import time
rng = np.random.default_rng(0)
d, w, m = 3, 8, 3
W0 = rng.standard_normal((w, d)) / np.sqrt(d); W1 = rng.standard_normal((m, w)) / np.sqrt(w)
ex = pnn.lipschitz_exact_pattern(W0, W1); sd = pnn.lipsdp_neuron(W0, W1)
print("L^2: exact pattern", ex, "LipSDP-neuron", sd, "spectral product^2", (np.linalg.norm(W0, 2) * np.linalg.norm(W1, 2)) ** 2)
Pm = pnn.lipschitz_problem(W0, W1); Pe = pnn.lipschitz_problem(W0, W1, mixed=False)
print(Pm, Pe)
for name, P in [("eq", Pe)]:
    print(name, "MOSEK order2", exact_relaxation(P, 2)[0])
t = time.time(); r = po.solve(Pm, order=2, iters_per_stage=3000, etas=(1e-2, 1e-3, 1e-4, 1e-5)); print("ours mixed order2", r.bound, time.time() - t, r.relaxation)
