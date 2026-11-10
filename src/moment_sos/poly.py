"""Sparse multivariate polynomials with operator overloading.

    x = moment_sos.variables(3)
    f = x[0]*x[1] - 2*x[2]**2 + 1
    c = (x[0]**2 + x[1]**2 <= 1)          # -> Constraint
"""
from __future__ import annotations
import numbers
import numpy as np


class Poly:
    """A polynomial in `n` variables: {exponent tuple: coefficient}."""
    __slots__ = ("n", "terms")
    __array_priority__ = 1000

    def __init__(self, n, terms=None):
        self.n = n
        self.terms = {} if terms is None else {k: float(v) for k, v in terms.items() if v != 0.0}

    # ---- constructors -------------------------------------------------------
    @staticmethod
    def constant(n, c):
        """Constant polynomial `c` in `n` variables."""
        return Poly(n, {(0,) * n: float(c)} if c != 0 else {})

    @staticmethod
    def variable(n, i):
        """Coordinate polynomial x_i in `n` variables."""
        e = [0] * n
        e[i] = 1
        return Poly(n, {tuple(e): 1.0})

    def _coerce(self, other):
        if isinstance(other, Poly):
            if other.n != self.n:
                raise ValueError("polynomials live in different numbers of variables")
            return other
        if isinstance(other, (numbers.Real, np.floating, np.integer)):
            return Poly.constant(self.n, float(other))
        return NotImplemented

    # ---- arithmetic ---------------------------------------------------------
    def __add__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        t = dict(self.terms)
        for k, v in o.terms.items():
            t[k] = t.get(k, 0.0) + v
        return Poly(self.n, t)
    __radd__ = __add__

    def __neg__(self):
        return Poly(self.n, {k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        o = self._coerce(other)
        return o if o is NotImplemented else self + (-o)

    def __rsub__(self, other):
        o = self._coerce(other)
        return o if o is NotImplemented else o + (-self)

    def __mul__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return o
        t = {}
        for k1, v1 in self.terms.items():
            for k2, v2 in o.terms.items():
                k = tuple(a + b for a, b in zip(k1, k2))
                t[k] = t.get(k, 0.0) + v1 * v2
        return Poly(self.n, t)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, (numbers.Real, np.floating, np.integer)):
            return NotImplemented
        return Poly(self.n, {k: v / float(other) for k, v in self.terms.items()})

    def __pow__(self, p):
        if not isinstance(p, int) or p < 0:
            raise ValueError("only non-negative integer powers")
        out = Poly.constant(self.n, 1.0)
        base = self
        while p:
            if p & 1:
                out = out * base
            base = base * base
            p >>= 1
        return out

    # ---- constraints --------------------------------------------------------
    def __ge__(self, other):
        o = self._coerce(other)
        return o if o is NotImplemented else Constraint(self - o, "ineq")

    def __le__(self, other):
        o = self._coerce(other)
        return o if o is NotImplemented else Constraint(o - self, "ineq")

    def __eq__(self, other):                       # NOTE: builds a Constraint, as in cvxpy
        o = self._coerce(other)
        return o if o is NotImplemented else Constraint(self - o, "eq")

    __hash__ = None

    # ---- inspection ---------------------------------------------------------
    def degree(self):
        """Total degree (0 for the zero polynomial)."""
        return max((sum(k) for k in self.terms), default=0)

    def support(self):
        """Set of variable indices that occur in some monomial."""
        return {i for k in self.terms for i, e in enumerate(k) if e}

    def __call__(self, x):
        """Evaluate at x (array-like of length n, or an array of shape (m, n))."""
        x = np.asarray(x, float)
        if x.ndim == 1:
            return float(sum(c * np.prod(x ** np.array(k)) for k, c in self.terms.items()))
        return np.array([self(r) for r in x])

    def substitute_affine(self, c, d):
        """Return p(c + d * t) as a polynomial in t."""
        n = self.n
        lin = [Poly(n, {tuple([0] * n): float(c[i]), **({tuple(1 if j == i else 0 for j in range(n)): float(d[i])} if d[i] != 0 else {})})
               for i in range(n)]
        cache = {}

        def power(i, e):
            if (i, e) not in cache:
                cache[(i, e)] = lin[i] ** e
            return cache[(i, e)]
        out = Poly(n)
        for k, v in self.terms.items():
            term = Poly.constant(n, v)
            for i, e in enumerate(k):
                if e:
                    term = term * power(i, e)
            out = out + term
        return out

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for k, v in sorted(self.terms.items()):
            mon = "*".join(f"x{i}" + (f"^{e}" if e > 1 else "") for i, e in enumerate(k) if e)
            parts.append(f"{v:+g}" + (f"*{mon}" if mon else ""))
        return " ".join(parts)


class Constraint:
    """poly >= 0 (kind='ineq') or poly == 0 (kind='eq')."""
    __slots__ = ("poly", "kind")

    def __init__(self, poly, kind):
        self.poly, self.kind = poly, kind

    def __repr__(self):
        return f"[{self.poly!r}] {'>= 0' if self.kind == 'ineq' else '== 0'}"


def variables(n):
    """Return a list of n coordinate polynomials x_0 ... x_{n-1}."""
    return [Poly.variable(n, i) for i in range(n)]
