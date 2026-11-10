"""Shared helpers for E10 (ALM on MAX-CUT, instance variability).  Import AFTER setting the thread env vars."""
import numpy as np


def local_search_maxcut(n, edges, restarts=2000, seed=0):
    """Best cut from `restarts` single-flip local searches (best-improvement) from uniform random starts."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] += 1; A[j, i] += 1
    best = -1
    for _ in range(restarts):
        x = rng.choice([-1.0, 1.0], n)
        while True:
            s = A @ x                                 # s_i = sum_j A_ij x_j
            gain = x * s                              # cut change when flipping i:  +x_i s_i
            i = int(np.argmax(gain))
            if gain[i] <= 0:
                break
            x[i] = -x[i]
        cut = int(round(sum(1 for i, j in edges if x[i] != x[j])))
        best = max(best, cut)
    return best


def times_to_gap(hist, ref, gaps=(1e-1, 1e-2, 1e-3, 1e-4)):
    """First wall-clock time at which best_U - ref <= gap (hist rows: n_eval, time, U, best_U)."""
    h = np.asarray(hist)
    out = {}
    for g in gaps:
        idx = np.where(h[:, 3] - ref <= g)[0]
        out[f"{g:.0e}"] = float(h[idx[0], 1]) if len(idx) else None
    return out


def thin(hist, k=600):
    """Down-sample a history to at most k rows (always keeps the last row)."""
    h = np.asarray(hist)
    if len(h) <= k:
        return h.tolist()
    idx = np.unique(np.concatenate([np.linspace(0, len(h) - 1, k).astype(int), [len(h) - 1]]))
    return h[idx].tolist()
