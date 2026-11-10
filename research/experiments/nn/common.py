import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "src")); sys.path.insert(0, ROOT)
import numpy as np, torch
import moment_sos as po
from moment_sos import nn as pnn
from msos.core import POP
from msos import reference


def exact_relaxation(prob, order, cliques=None, trace_cuts=True):
    """Exact value of the order-r (sparse) relaxation by an interior-point solve (MOSEK) of the same dual."""
    R = po.Relaxation(prob, order, "dense" if cliques is None else cliques)
    v, t = po.solve_dual_ipm(R)
    v = prob.f_off + prob.f_scale * v                      # undo the internal normalisation of Problem
    return (v if prob.sense == "max" else -v), t


def train_mlp(d, widths, seed=0, n_train=2000, steps=1500, k=2):
    """Small ReLU classifier on synthetic data labelled by a random smooth teacher; returns numpy (Ws, bs)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    teacher = rng.standard_normal((d, 4)); phase = rng.uniform(0, 6.28, 4)
    X = torch.as_tensor(rng.uniform(-1, 1, (n_train, d)), dtype=torch.float64)
    t = torch.sin(X @ torch.as_tensor(teacher) * 2.0 + torch.as_tensor(phase)).sum(1)
    y = (t > t.median()).long()
    dims = [d] + list(widths) + [k]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]).double())
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for s in range(steps):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(net(X), y); loss.backward(); opt.step()
    acc = (net(X).argmax(1) == y).double().mean().item()
    Ws = [l.weight.detach().numpy() for l in layers if isinstance(l, torch.nn.Linear)]
    bs = [l.bias.detach().numpy() for l in layers if isinstance(l, torch.nn.Linear)]
    return Ws, bs, acc
