"""Squared l2 Lipschitz bound of a one-hidden-layer ReLU network: LipSDP vs order-2 relaxation."""
import numpy as np
import moment_sos as po
from moment_sos import nn

rng = np.random.default_rng(0)
d, w, m = 3, 8, 3
W0, W1 = rng.standard_normal((w, d)) / np.sqrt(d), rng.standard_normal((m, w)) / np.sqrt(w)
print("exact (activation patterns):", nn.lipschitz_exact_pattern(W0, W1))
print("LipSDP-Neuron             :", nn.lipsdp_neuron(W0, W1))
print("order 2 (ours)            :", po.solve(nn.lipschitz_problem(W0, W1), order=2, iters_per_stage=3000).bound)
