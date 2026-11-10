# Tutorial: certifying a ReLU network

We train a small classifier, then ask: *is the prediction at a point $x_0$ unchanged for every input within
$\ell_\infty$-distance $\varepsilon$?* The answer is a polynomial optimization problem; its relaxations give certificates.

## Train a network

```python
import numpy as np
import torch
from moment_sos import nn


def train(d, widths, seed, n_train=2000, steps=1500):
    """Small ReLU classifier on synthetic data labelled by a random smooth teacher; returns numpy (Ws, bs)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    teacher, phase = rng.standard_normal((d, 4)), rng.uniform(0, 6.28, 4)
    X = torch.as_tensor(rng.uniform(-1, 1, (n_train, d)))
    t = torch.sin(X @ torch.as_tensor(teacher) * 2.0 + torch.as_tensor(phase)).sum(1)
    y = (t > t.median()).long()
    dims, layers = [d] + list(widths) + [2], []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]).double())
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(X), y).backward()
        opt.step()
    lin = [l for l in layers if isinstance(l, torch.nn.Linear)]
    return [l.weight.detach().numpy() for l in lin], [l.bias.detach().numpy() for l in lin]


Ws, bs = train(4, [8, 8], seed=5)
rng = np.random.default_rng(105)
while True:                                        # a test point with a clear margin
    x0 = rng.uniform(-0.8, 0.8, 4)
    logits = nn.forward(Ws, bs, x0)
    if abs(logits[0] - logits[1]) > 0.4:
        break
cls = int(logits.argmax())
c = np.zeros(2)
c[1 - cls], c[cls] = 1.0, -1.0                    # maximise (wrong logit) - (true logit)
```

## The verification problem

For each $\varepsilon$ we maximize the margin violation $c^\top\text{logits}$ over the ball. **A negative maximum proves
robustness.** `nn.relu_verification` builds the POP (stable neurons are substituted away; each *unstable* neuron becomes a
variable with the exact ReLU constraints plus the triangle inequality):

```python
import moment_sos as po

rows = []
for eps in (0.2,):
    problem, info = nn.relu_verification(Ws, bs, x0, eps, c)
    row = dict(eps=eps, unstable=info["n_unstable"], ibp=info["ibp_upper"],
               exact=nn.verification_exact_milp(Ws, bs, x0, eps, c)[0],
               order1=po.solve_alm(problem, order=1).bound,
               order2=po.solve_alm(problem, order=2).bound)
    rows.append(row)
    print({k: round(v, 3) for k, v in row.items()})
```

The columns are: the interval-bound-propagation bound (`ibp`, the cheapest), the exact maximum from a MILP (`exact`), and
the order-1 (Shor SDP) and order-2 relaxations. At $\varepsilon=0.2$:

```python
r = rows[0]
assert r["ibp"] > 0 and r["order1"] > 0 > r["order2"]       # only the second-order relaxation proves robustness
assert abs(r["order2"] - r["exact"]) < 1e-3                  # and it is tight
```

(Other values of $\varepsilon$: at $0.22$ order 1 again fails and order 2 again certifies; at $0.15$ order 1 already suffices. The order-2 solve takes about a minute; since every iterate is a valid bound, `time_limit=` stops it early with a looser but still certified value.)

At $\varepsilon=0.2$ the interval bound ($\approx86$) and the order-1 SDP ($\approx4.2$) both fail to prove robustness; the
order-2 relaxation gives $-2.80$, equal to the exact maximum: **robustness is certified**, with a first-order solver and no
interior-point SDP.

## Practical advice

- Use `po.solve_alm` for these problems: they have many inequality and equality constraints, and `solve_alm` has no
  smoothing bias.
- The cost is governed by the number of **unstable** neurons $n$: the order-2 block has size $1+n+\binom n2$-ish. Up to
  about 14 unstable neurons the order-2 value is computed to $10^{-5}$. At about 40 it is not solved to convergence in 25
  minutes; use correlative sparsity (`cliques="auto"`) when layers decouple. See {doc}`../benchmarks`.
- On a small MNIST classifier (3 vs 8, PCA to 4 inputs) order 1 already certified every robust point, so check order 1
  first: order 2 helps only when order 1 leaves a gap.
