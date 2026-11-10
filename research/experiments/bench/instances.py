"""MAX-CUT instance generators following the distributions of the Biq Mac library / rudy
(g05: G(n,1/2) unweighted; pm1s: density 0.1, weights +-1; pm1d: density 0.99, weights +-1;
 torus: 2-D toroidal grid, Gaussian weights x100 rounded to integers).
These are NOT the library instances (those need a download); they are drawn from the same distributions by seed."""
import numpy as np


def instance(family, n, seed):
    if family == "reg3":                                    # random 3-regular graph (the family of E3), unit weights
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        from msos import problems as pb
        e = pb.graph("reg3", n, seed=seed)
        return n, [tuple(x) for x in e], [1.0] * len(e)
    rng = np.random.default_rng(1000 * seed + {"g05": 1, "pm1s": 2, "pm1d": 3, "torus": 4}[family])
    edges, w = [], []
    if family == "torus":
        k = int(round(n ** 0.5)); assert k * k == n
        idx = lambda i, j: (i % k) * k + (j % k)
        for i in range(k):
            for j in range(k):
                for (a, b) in ((idx(i, j), idx(i + 1, j)), (idx(i, j), idx(i, j + 1))):
                    edges.append((min(a, b), max(a, b))); w.append(float(np.rint(100 * rng.standard_normal())))
        # remove duplicate edges for k=2 (not used) and zero weights
        keep = [t for t in range(len(edges)) if w[t] != 0.0]
        return n, [edges[t] for t in keep], [w[t] for t in keep]
    p = {"g05": 0.5, "pm1s": 0.1, "pm1d": 0.99}[family]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j)); w.append(1.0 if family == "g05" else float(rng.choice([-1.0, 1.0])))
    return n, edges, w


def local_search_cut(n, edges, w, restarts=2000, seed=0):
    """Best cut value found by randomised 1-flip local search (a feasible point => lower bound)."""
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n)]
    for (i, j), x in zip(edges, w):
        adj[i].append((j, x)); adj[j].append((i, x))
    best = -np.inf
    for _ in range(restarts):
        s = rng.choice([-1, 1], n)
        improved = True
        while improved:
            improved = False
            for v in rng.permutation(n):
                gain = sum(x * s[u] for u, x in adj[v]) * s[v]      # > 0: flipping v increases the cut
                if gain > 1e-12:
                    s[v] = -s[v]; improved = True
        best = max(best, sum(x for (i, j), x in zip(edges, w) if s[i] != s[j]))
    return float(best)
