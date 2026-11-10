"""Shared data for the first-order SDP baselines on the Boolean MAX-CUT order-2 moment relaxation.

The SDP (maximisation form) is
    max  sum_gamma f_gamma * ybar_gamma
    s.t. X psd (s x s),  X_p = ybar_gamma(p) for every upper-triangular slot p (slots with the same
         moment gamma are tied together),  diag(X) = 1,
where ybar_gamma = (sum_{p in gamma} w_p X_p) / W_gamma  is the weighted mean of the slots
(w_p = 1 on the diagonal, 2 off it, W_gamma = sum of the weights).  This is exactly the order-r
Lasserre moment relaxation of msos/reference.py (the trace cut tr X <= s is implied by diag = 1).

Slot-space notation: x = X[I, J] (length nslots, upper triangle incl. diagonal).
  value(x)     = sum_p c_p x_p,                 c_p = w_p f_g(p) / W_g(p)
  dev(x)       = x - ybar[g]                    (zero iff all slots of a moment agree)
  dg(x)        = x[diag slots] - 1
Matrix form of a slot coefficient vector c:  D[I,J] = D[J,I] = c_p / w_p, so <D, X> = sum_p c_p x_p.
"""
import os, sys, time, json, resource
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def load_instance(n, r=2, seed=1):
    from msos import SpectralDual, problems as pb
    pop = pb.maxcut(n, pb.graph("reg3", n, seed=seed))
    return pop, SpectralDual(pop, r)


class Slots:
    """Slot arrays of the single clique, taken from msos.SpectralDual (read only)."""

    def __init__(self, d):
        self.s = len(d.cl_basis[0])
        self.I = d.cl_I[0].astype(np.int64)
        self.J = d.cl_J[0].astype(np.int64)
        self.nslots = len(self.I)
        g_raw = d.slot_g
        self.W_full = d.W
        used, self.g = np.unique(g_raw, return_inverse=True)       # compress moment ids
        self.G = len(used)
        self.w = d.slot_w.astype(float)
        self.W = d.W[used]
        self.f = d.fvec[used]
        self.empty = int(np.searchsorted(used, d.empty_id))
        self.diag = np.where(self.I == self.J)[0]                    # slot indices of diagonal
        assert (self.I[self.diag] == np.arange(self.s)).all()
        self.c = self.w * self.f[self.g] / self.W[self.g]            # objective coefficients on slots
        self.cmat_val = self.f[self.g] / self.W[self.g]              # = c / w
        self.offd = self.I != self.J

    # ---- slot-space operators -------------------------------------------------
    def ybar(self, x):
        return np.bincount(self.g, weights=self.w * x, minlength=self.G) / self.W

    def value(self, x):
        return float(self.f @ self.ybar(x))

    def dev(self, x):
        return x - self.ybar(x)[self.g]

    def resid(self, x):
        """Stacked residual [dev(x); diag(x) - 1] (the equality constraints A x - b)."""
        return np.concatenate([self.dev(x), x[self.diag] - 1.0])

    def resid_T(self, z):
        """Adjoint of resid wrt the Euclidean slot inner product: returns slot coefficient vector."""
        zd, zg = z[:self.nslots], z[self.nslots:]
        out = zd - self.w * (np.bincount(self.g, weights=zd, minlength=self.G) / self.W)[self.g]
        out[self.diag] += zg
        return out

    def to_dense(self, cvec, out=None):
        """Symmetric matrix D with <D, X> = sum_p cvec_p x_p."""
        v = cvec / self.w
        D = np.zeros((self.s, self.s)) if out is None else out
        D[self.I, self.J] = v
        D[self.J, self.I] = v
        return D

    def slots_of(self, X):
        return X[self.I, self.J]


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # macOS reports bytes


class History:
    """Geometrically thinned (time, value, infeas) log."""

    def __init__(self, ratio=1.02):
        self.rows, self.last, self.ratio = [], 0.0, ratio

    def add(self, t, val, inf, force=False):
        if force or t > self.last * self.ratio + 1e-3:
            self.rows.append([float(t), float(val), float(inf)])
            self.last = t
