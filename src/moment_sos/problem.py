"""Problem definition and normalisation to the box [-1,1]^n."""
from __future__ import annotations
import numpy as np
from .poly import Poly, Constraint


class Problem:
    """max (or min) f(x)  s.t.  g_i(x) >= 0, h_j(x) = 0, x in a box (or x in {-1,+1}^n).

    Parameters
    ----------
    objective   : Poly
    constraints : iterable of Constraint (build them with ``>=``, ``<=``, ``==`` on Poly)
    sense       : "max" or "min"
    box         : (lo, hi) scalars or length-n sequences. Required unless ``boolean``.
                  The feasible set MUST lie in the box (the certified bound relies on it).
    boolean     : True (all variables) or a length-n boolean sequence marking the variables that are
                  spins x_i in {-1,+1}; x_i^2 = 1 is then used as a rewriting rule (mixed problems allowed).
    add_box_constraints : add (hi-x)(x-lo) >= 0 for every variable (recommended).
    """

    def __init__(self, objective, constraints=(), sense="max", box=None, boolean=False,
                 add_box_constraints=True, name=""):
        if sense not in ("max", "min"):
            raise ValueError("sense must be 'max' or 'min'")
        n = objective.n
        mask = np.ones(n, bool) if boolean is True else (np.zeros(n, bool) if boolean is False or boolean is None else np.asarray(boolean, bool))
        self.bool_mask = mask
        self.boolean = bool(mask.all())                       # fully Boolean (constant-trace) problem
        boolean = bool(mask.any())
        self.n, self.sense, self.name = n, sense, name
        self.objective = objective
        self.constraints = list(constraints)
        if box is None:
            if not self.boolean:
                raise ValueError("Problem needs `box=(lo, hi)` (scalars or length-n arrays) containing the feasible set: the certified bound uses the box to size the diagonal shift. For Boolean variables pass `boolean=True` instead.")
            lo, hi = -np.ones(n), np.ones(n)
        else:
            lo = np.broadcast_to(np.asarray(box[0], float), (n,)).copy()
            hi = np.broadcast_to(np.asarray(box[1], float), (n,)).copy()
            if np.any(hi <= lo):
                raise ValueError("empty box: need hi > lo in every coordinate")
        lo[mask], hi[mask] = -1.0, 1.0
        self.lo, self.hi = lo, hi
        self.c, self.d = (lo + hi) / 2, (hi - lo) / 2           # x = c + d * t,  t in [-1,1]^n

        sgn = 1.0 if sense == "max" else -1.0
        trivial = np.allclose(self.c, 0) and np.allclose(self.d, 1)
        self.f = (objective * sgn).substitute_affine(self.c, self.d) if not trivial else objective * sgn
        self.ineq, self.eq = [], []
        for con in self.constraints:
            p = con.poly.substitute_affine(self.c, self.d) if not trivial else con.poly
            (self.ineq if con.kind == "ineq" else self.eq).append(p)
        # ---- normalise scales (conditioning only; the feasible set and maximisers are unchanged)
        from .relaxation import box_sup_bound
        mk = lambda p: p
        self.ineq = [self._scale_poly(g, 1.0 / box_sup_bound(g.terms, False) if box_sup_bound(g.terms, False) > 0 else 1.0) for g in self.ineq]
        self.eq = [self._scale_poly(h, 1.0 / max(abs(c) for c in h.terms.values())) if h.terms else h for h in self.eq]
        zero = (0,) * n
        self.f_off = self.f.terms.get(zero, 0.0)
        self.f_scale = max(sum(abs(c) for e, c in self.f.terms.items() if e != zero), 1e-12)
        self.f = self._scale_poly(self.f - self.f_off, 1.0 / self.f_scale)
        if add_box_constraints:
            for i in range(n):
                if mask[i]:
                    continue
                e = [0] * n
                e[i] = 2
                self.ineq.append(Poly(n, {tuple([0] * n): 1.0, tuple(e): -1.0}))

    @staticmethod
    def _scale_poly(p, a):
        return Poly(p.n, {k: v * a for k, v in p.terms.items()})

    # ---- helpers ------------------------------------------------------------
    def to_original(self, t):
        """Map normalised coordinates t in [-1,1]^n back to the original variables."""
        return self.c + self.d * np.asarray(t, float)

    def to_internal(self, x):
        """Map original coordinates x to the normalised box [-1,1]^n used internally."""
        return (np.asarray(x, float) - self.c) / self.d

    def value(self, x):
        """Original objective at an original-coordinate point."""
        return self.objective(x)

    def violation(self, x):
        """Largest violation of the original constraints / box at x."""
        x = np.asarray(x, float)
        v = max(0.0, float(np.max(self.lo - x)), float(np.max(x - self.hi)))
        for con in self.constraints:
            val = con.poly(x)
            v = max(v, -val if con.kind == "ineq" else abs(val))
        return v

    def __repr__(self):
        return (f"Problem({self.sense}, n={self.n}, deg={self.objective.degree()}, "
                f"{len(self.ineq)} ineq, {len(self.eq)} eq, boolean={self.boolean})")
