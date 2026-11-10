"""Instance generators."""
import itertools
import numpy as np
import networkx as nx
from .core import POP


def _e(n, *idx):
    e = [0] * n
    for i in idx:
        e[i] += 1
    return tuple(e)


def maxcut(n, edges, weights=None, name="maxcut"):
    """max sum_{ij} w_ij (1 - x_i x_j)/2, x in {+-1}^n (Boolean mode)."""
    weights = [1.0] * len(edges) if weights is None else weights
    f = {}
    z = tuple([0] * n)
    for (i, j), w in zip(edges, weights):
        f[z] = f.get(z, 0.0) + w / 2
        f[_e(n, i, j)] = f.get(_e(n, i, j), 0.0) - w / 2
    return POP(n, f, boolean=True, name=name)


def graph(kind, n, seed=0):
    if kind == "K":
        return list(itertools.combinations(range(n), 2))
    if kind == "C":
        return [(i, (i + 1) % n) for i in range(n)]
    if kind == "petersen":
        return list(nx.petersen_graph().edges())
    if kind == "reg3":
        return list(nx.random_regular_graph(3, n, seed=seed).edges())
    if kind == "gnp":
        G = nx.gnp_random_graph(n, 0.5, seed=seed)
        return list(G.edges())
    if kind == "torus":
        k = int(round(n ** 0.5)); G = nx.grid_2d_graph(k, k, periodic=True)
        m = {v: i for i, v in enumerate(G.nodes())}
        return [(m[a], m[b]) for a, b in G.edges()]
    raise ValueError(kind)


def brute_maxcut(n, edges):
    best = -1
    for mask in range(2 ** (n - 1)):
        x = [(mask >> i) & 1 for i in range(n)]
        best = max(best, sum(1 for i, j in edges if x[i] != x[j]))
    return best


def ball():
    return POP(1, {(1,): 1.0}, ineq=[{(0,): 1.0, (2,): -1.0}], name="ball")


def disk():
    return POP(2, {(1, 0): 1.0, (0, 1): 1.0}, ineq=[{(0, 0): 1.0, (2, 0): -1.0, (0, 2): -1.0}], name="disk")


def box_qp(n, seed=0, density=1.0):
    """max x^T Q x over [-1,1]^n, Q random symmetric (non-convex)."""
    rng = np.random.default_rng(seed)
    f, cons = {}, []
    for i in range(n):
        for j in range(i, n):
            if rng.random() <= density:
                c = rng.standard_normal() * (1 if i == j else 2)
                f[_e(n, i, j)] = f.get(_e(n, i, j), 0.0) + c
    for i in range(n):
        cons.append({tuple([0] * n): 1.0, _e(n, i, i): -1.0})
    return POP(n, f, ineq=cons, name=f"boxqp{n}_s{seed}")


def chain_quartic(n, seed=0):
    """Correlatively sparse: max sum_i (a_i x_i x_{i+1} + b_i x_i^2 x_{i+1}^2 - c x_i^4) on box."""
    rng = np.random.default_rng(seed)
    f = {}
    for i in range(n - 1):
        f[_e(n, i, i + 1)] = f.get(_e(n, i, i + 1), 0) + rng.standard_normal()
        f[_e(n, i, i, i + 1, i + 1)] = f.get(_e(n, i, i, i + 1, i + 1), 0) + rng.standard_normal() * 0.5
        f[_e(n, i, i, i, i)] = f.get(_e(n, i, i, i, i), 0) - 0.3
    cons = [{tuple([0] * n): 1.0, _e(n, i, i): -1.0} for i in range(n)]
    cl = [[i, i + 1] for i in range(n - 1)]
    return POP(n, f, ineq=cons, name=f"chain{n}_s{seed}"), cl
