# Neural-network tools

`moment_sos.nn` contains encodings of two standard neural-network certification problems as POPs, with exact or
heuristic references.

## Robustness verification of a ReLU network

Maximize the margin $c^\top\text{logits}$ over an $\ell_\infty$ ball around $x_0$; a negative maximum proves
robustness. Stable neurons (by interval propagation) are substituted away and every unstable neuron becomes a variable
with $z\ge0$, $z\ge\text{pre}$, $z(z-\text{pre})=0$ and the triangle inequality.

```python
import numpy as np
import moment_sos as po
from moment_sos import nn

rng = np.random.default_rng(0)
d, w = 3, 4
Ws = [rng.standard_normal((w, d)), rng.standard_normal((w, w)) * 0.5, rng.standard_normal((2, w))]
bs = [rng.standard_normal(w) * 0.2, rng.standard_normal(w) * 0.2, np.zeros(2)]
x0, eps, c = rng.uniform(-0.5, 0.5, d), 0.3, np.array([1.0, -1.0])

problem, info = nn.relu_verification(Ws, bs, x0, eps, c)
print("unstable neurons:", info["n_unstable"], " interval bound:", round(info["ibp_upper"], 4))
print("exact (MILP):", nn.verification_exact_milp(Ws, bs, x0, eps, c)[0])
for order in (1, 2):
    print(order, po.solve_alm(problem, order=order, time_limit=15).bound)       # any iterate is a valid bound
```

Order 1 is the Shor-type SDP of the verification literature; order 2 is strictly stronger. Helpers:
`nn.verification_exact_milp` (big-M MILP through SciPy/HiGHS), `nn.verification_pgd` (projected-gradient lower bound).

## Lipschitz constants

```python
import numpy as np
import moment_sos as po
from moment_sos import nn

rng = np.random.default_rng(0)
d, w, m = 3, 6, 3
W0, W1 = rng.standard_normal((w, d)) / np.sqrt(d), rng.standard_normal((m, w)) / np.sqrt(w)
print("activation patterns:", nn.lipschitz_exact_pattern(W0, W1))
print("LipSDP-Neuron      :", nn.lipsdp_neuron(W0, W1))
print("order 2            :", po.solve_alm(nn.lipschitz_problem(W0, W1), order=2).bound)
```

## What to expect

On networks with up to about 14 unstable neurons the second-order relaxation is computed to $10^{-5}$ and matches the
interior-point value. At about 40 unstable neurons our ALM solver recovers 78–94% of the gap between order 1 and the
exact value in 25 minutes **without having converged**, and at 60 it fails; an interior-point solver on the
correlatively sparse relaxation solves the 40-variable case in the same time with 17× the memory. On a small MNIST
classifier (3 vs 8, PCA to 4 inputs) the order-1 relaxation already certified every robust test point and order 2 only
tightened the bounds. See {doc}`../benchmarks`.
