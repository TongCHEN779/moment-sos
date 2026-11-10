"""Spectral (eigenvalue-optimisation) dual of the moment-SOS hierarchy.

Given   max f(x)  s.t.  g_i(x) >= 0,  h_j(x) = 0,   feasible set inside [-1,1]^n,
and a relaxation order r, every dual point theta defines Gram matrices
Q_0 (one per clique), Q_i (one per inequality) and a scalar sigma such that

    sigma - f  =  sum_k v_k^T Q_{0,k} v_k + sum_i (z_i^T Q_i z_i) g_i + sum_j p_j h_j     (mod ideal),

where the Q's are *free symmetric matrices* (not required to be PSD).  Dominating
the diagonal by the most negative eigenvalue gives the certified upper bound

    U(theta) = sigma + sum_k s_k [-lmin(Q_{0,k})]_+ + sum_i s_i B_i [-lmin(Q_i)]_+ ,

(the [.]_+ is dropped on blocks with constant trace, e.g. Boolean variables).
All index bookkeeping is vectorised; blocks are only ever touched through
matrix-vector products / extreme eigenpairs.
"""
import math
from itertools import combinations, combinations_with_replacement
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# --------------------------------------------------------------------------- #
#  Monomial arithmetic on integer keys                                         #
# --------------------------------------------------------------------------- #
class Ring:
    """Monomials as integer keys.  General mode: base-(D+1) digits, product = add.
    Boolean mode (x_i^2 = 1): bitmasks, product = xor."""

    def __init__(self, n, maxdeg, boolean=False):
        self.n, self.boolean = n, boolean
        if boolean:
            if n > 63:
                raise ValueError("Boolean mode supports n <= 63")
            self.dtype = np.uint64
        else:
            self.base = maxdeg + 1
            if float(self.base) ** n >= 2 ** 62:
                raise ValueError("monomial keys overflow int64; reduce n or r")
            self.dtype = np.int64

    def key(self, exp):
        if self.boolean:
            return int(sum(1 << i for i, e in enumerate(exp) if e % 2))
        return int(sum(int(e) * self.base ** i for i, e in enumerate(exp)))

    def decode(self, key):
        key = int(key)
        if self.boolean:
            return tuple((key >> i) & 1 for i in range(self.n))
        out = []
        for _ in range(self.n):
            out.append(key % self.base); key //= self.base
        return tuple(out)

    def mul(self, a, b):
        return np.bitwise_xor(a, b) if self.boolean else a + b

    def basis(self, variables, d):
        """keys of all monomials in `variables` of (squarefree) degree <= d."""
        out = []
        for k in range(d + 1):
            it = combinations(variables, k) if self.boolean else combinations_with_replacement(variables, k)
            for c in it:
                if self.boolean:
                    out.append(sum(1 << i for i in c))
                else:
                    out.append(sum(self.base ** i for i in c))
        return np.array(out, dtype=self.dtype)


def reduce_poly(poly, boolean):
    """Normalise a polynomial dict {exp: coef}; in Boolean mode reduce exps mod 2."""
    out = {}
    for e, c in poly.items():
        e = tuple((x % 2) for x in e) if boolean else tuple(int(x) for x in e)
        out[e] = out.get(e, 0.0) + float(c)
    return {e: c for e, c in out.items() if c != 0.0}


def degree(poly):
    return max((sum(e) for e in poly), default=0)


def box_sup_bound_exact(poly, boolean):
    """Same bound as `box_sup_bound`, but summed in exact rational arithmetic (every float coefficient is a
    dyadic rational); used by the rigorous certificate, where a rounded-down float sum would not be a valid bound."""
    from fractions import Fraction
    b = Fraction(0)
    for e, c in poly.items():
        const = sum(e) == 0
        c = Fraction(float(c))
        if c > 0:
            b += c
        elif (not const) and (boolean or any(x % 2 for x in e)):
            b += -c
    return max(b, Fraction(0))


def box_sup_bound(poly, boolean):
    """Upper bound on sup_{x in [-1,1]^n} poly(x) (>= 0):  positive coefficients count,
    negative ones count only if the monomial can be negative (odd exponent).  In Boolean
    mode every non-constant monomial is +-1."""
    b = 0.0
    for e, c in poly.items():
        const = sum(e) == 0
        if c > 0:
            b += c
        elif (not const) and (boolean or any(x % 2 for x in e)):
            b += -c
    return max(b, 0.0)


# --------------------------------------------------------------------------- #
#  Problem container                                                           #
# --------------------------------------------------------------------------- #
class POP:
    """max f(x) s.t. g_i >= 0, h_j = 0 .  Polynomials: {exponent tuple (len n): coef}.
    The caller guarantees the feasible set lies in [-1,1]^n (needed by the trace bounds)."""

    def __init__(self, n, f, ineq=(), eq=(), boolean=False, name=""):
        self.n, self.boolean, self.name = n, boolean, name
        self.f = reduce_poly(f, boolean)
        self.ineq = [reduce_poly(g, boolean) for g in ineq]
        self.eq = [reduce_poly(h, boolean) for h in eq]

    def evaluate(self, x):
        x = np.asarray(x, float)
        return sum(c * np.prod(x ** np.array(e)) for e, c in self.f.items())


# --------------------------------------------------------------------------- #
#  The spectral dual                                                           #
# --------------------------------------------------------------------------- #
class SpectralDual:
    """theta = [sigma | Theta (zero-sum Hankel/Gram freedom, one per slot) | u (localising Gram
    entries) | p (equality-multiplier coefficients)].

    Block 0..K-1  : clique moment blocks      (size s_k, weight s_k,    constant trace iff Boolean)
    Block K..     : inequality localising     (size s_i, weight s_i*B_i)
    """

    def __init__(self, pop, r, cliques=None, dense_limit=2500):
        self.pop, self.r = pop, r
        n = pop.n
        D = 2 * r
        for poly, nm in [(pop.f, "objective")] + [(g, "g") for g in pop.ineq] + [(h, "h") for h in pop.eq]:
            if degree(poly) > D:
                raise ValueError(f"{nm} has degree {degree(poly)} > 2r={D}")
        self.ring = ring = Ring(n, D, pop.boolean)
        self.cliques = [list(range(n))] if cliques is None else [sorted(c) for c in cliques]
        self.dense_limit = dense_limit
        ct = pop.boolean

        # ---- clique moment blocks and slot arrays --------------------------------
        self.cl_basis, self.cl_off, self.cl_I, self.cl_J, gk = [], [], [], [], []
        off = 0
        self.cl_off.append(0)
        for c in self.cliques:
            B = ring.basis(c, r)
            I, J = np.triu_indices(len(B))
            self.cl_basis.append(B)
            self.cl_I.append(I); self.cl_J.append(J)
            gk.append(ring.mul(B[I], B[J]))
            off += len(I)
            self.cl_off.append(off)
        slot_keys = np.concatenate(gk)
        self.nslots = off

        # ---- inequality localising blocks ----------------------------------------
        self.ineq_info = []
        loc_keys = []
        for g in pop.ineq:
            supp = {i for e in g for i, x in enumerate(e) if x}
            host = next((k for k, c in enumerate(self.cliques) if supp <= set(c)), None)
            if host is None:
                raise ValueError("an inequality constraint is not covered by any clique")
            om = (degree(g) + 1) // 2
            if r - om < 0:
                raise ValueError("relaxation order too low for a constraint")
            Z = ring.basis(self.cliques[host], r - om)
            a, b = np.triu_indices(len(Z))
            gterms = [(ring.key(e), c) for e, c in g.items()]
            keys_per_term = [ring.mul(ring.mul(Z[a], Z[b]), np.array(gk_, dtype=ring.dtype)) for gk_, _ in gterms]
            loc_keys.extend(keys_per_term)
            self.ineq_info.append(dict(Z=Z, a=a, b=b, terms=gterms, kpt=keys_per_term,
                                       B=box_sup_bound(g, pop.boolean), host=host))
        # ---- equality multiplier polynomials -------------------------------------
        self.eq_info = []
        eq_keys = []
        for h in pop.eq:
            supp = {i for e in h for i, x in enumerate(e) if x}
            host = next((k for k, c in enumerate(self.cliques) if supp <= set(c)), None)
            if host is None:
                raise ValueError("an equality constraint is not covered by any clique")
            dh = degree(h)
            M = ring.basis(self.cliques[host], 2 * r - dh)
            hterms = [(ring.key(e), c) for e, c in h.items()]
            kpt = [ring.mul(M, np.array(hk, dtype=ring.dtype)) for hk, _ in hterms]
            eq_keys.extend(kpt)
            self.eq_info.append(dict(M=M, terms=hterms, kpt=kpt))
        f_keys = np.array([ring.key(e) for e in pop.f] + [0], dtype=ring.dtype)

        # ---- moment registry ------------------------------------------------------
        allk = np.concatenate([slot_keys, f_keys] + loc_keys + eq_keys)
        self.reg, inv = np.unique(allk, return_inverse=True)
        self.nmom = len(self.reg)
        self.slot_g = inv[:self.nslots]
        self.slot_w = np.where(np.concatenate(self.cl_I) == np.concatenate(self.cl_J), 1.0, 2.0)
        covered = np.zeros(self.nmom, bool); covered[self.slot_g] = True
        self.W = np.bincount(self.slot_g, weights=self.slot_w, minlength=self.nmom)
        self.W2 = np.bincount(self.slot_g, weights=self.slot_w ** 2, minlength=self.nmom)

        def idx(keys):
            j = np.searchsorted(self.reg, keys)
            return j
        self.empty_id = int(idx(np.array([0], dtype=ring.dtype))[0])
        self.fvec = np.zeros(self.nmom)
        for e, c in pop.f.items():
            self.fvec[int(idx(np.array([ring.key(e)], dtype=ring.dtype))[0])] += c

        # localising R_i :  coefficient of x^gamma in z^T Q z g  (linear in the upper-tri entries of Q)
        self.u_off = [0]
        Rblocks = []
        for info in self.ineq_info:
            ne = len(info["a"])
            wq = np.where(info["a"] == info["b"], 1.0, 2.0)
            rows, cols, vals = [], [], []
            for kp, (_, c) in zip(info["kpt"], info["terms"]):
                rows.append(idx(kp)); cols.append(np.arange(ne)); vals.append(c * wq)
            info["R"] = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                      shape=(self.nmom, ne))
            info["ne"] = ne
            self.u_off.append(self.u_off[-1] + ne)
            Rblocks.append(info["R"])
        self.nu = self.u_off[-1]
        self.R_loc = sp.hstack(Rblocks, format="csr") if Rblocks else sp.csr_matrix((self.nmom, 0))
        # equality multipliers
        self.p_off = [0]
        Eblocks = []
        for info in self.eq_info:
            nm = len(info["M"])
            rows, cols, vals = [], [], []
            for kp, (_, c) in zip(info["kpt"], info["terms"]):
                rows.append(idx(kp)); cols.append(np.arange(nm)); vals.append(np.full(nm, c))
            E = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.nmom, nm))
            info["E"] = E
            self.p_off.append(self.p_off[-1] + nm)
            Eblocks.append(E)
        self.np_ = self.p_off[-1]
        self.R_eq = sp.hstack(Eblocks, format="csr") if Eblocks else sp.csr_matrix((self.nmom, 0))

        # block table: (size, weight, plus_part)
        self.blocks = []
        for k, B in enumerate(self.cl_basis):
            self.blocks.append(dict(kind="moment", size=len(B), weight=float(len(B)), plus=not ct, clique=k))
        for i, info in enumerate(self.ineq_info):
            s = len(info["Z"])
            self.blocks.append(dict(kind="loc", size=s, weight=s * info["B"], plus=True, idx=i))

        # parameter layout
        self.i_sigma = 0
        self.sl_Theta = slice(1, 1 + self.nslots)
        self.sl_u = slice(1 + self.nslots, 1 + self.nslots + self.nu)
        self.sl_p = slice(1 + self.nslots + self.nu, 1 + self.nslots + self.nu + self.np_)
        self.dim = 1 + self.nslots + self.nu + self.np_
        # dense scratch for moment blocks
        self._eig_cache = {}

    # ------------------------------------------------------------------ helpers
    def theta0(self, sigma=0.0):
        th = np.zeros(self.dim)
        th[0] = sigma
        return th

    def sizes(self):
        return [b["size"] for b in self.blocks]

    def _moment_level(self, th):
        u = th[self.sl_u]; p = th[self.sl_p]
        rvec = -self.fvec.copy()
        rvec[self.empty_id] += th[0]
        if self.nu:
            rvec -= self.R_loc @ u
        if self.np_:
            rvec -= self.R_eq @ p
        return rvec

    def gram_blocks(self, th):
        """Return the list of dense symmetric matrices Q_b(theta) (moment cliques, then localising)."""
        rvec = self._moment_level(th)
        q0 = rvec[self.slot_g] / self.W[self.slot_g] + th[self.sl_Theta]
        mats = []
        for k, B in enumerate(self.cl_basis):
            s = len(B)
            sl = slice(self.cl_off[k], self.cl_off[k + 1])
            Q = np.zeros((s, s))
            I, J = self.cl_I[k], self.cl_J[k]
            Q[I, J] = q0[sl]
            Q[J, I] = q0[sl]
            mats.append(Q)
        u = th[self.sl_u]
        for i, info in enumerate(self.ineq_info):
            s = len(info["Z"])
            Q = np.zeros((s, s))
            vals = u[self.u_off[i]:self.u_off[i + 1]]
            Q[info["a"], info["b"]] = vals
            Q[info["b"], info["a"]] = vals
            mats.append(Q)
        return mats

    # ----------------------------------------------------------------- eigen oracle
    def _spectrum(self, Q, eta, kprev):
        """Smallest eigenpairs of Q.  Returns (w ascending, V, complete).  complete=True when the
        full spectrum is returned.  For big blocks we return enough low eigenpairs that the discarded
        ones have softmax weight < 1e-12 (or at least a safe margin)."""
        s = Q.shape[0]
        if s <= self.dense_limit:
            w, V = np.linalg.eigh(Q)
            return w, V, True
        k = max(8, kprev)
        while True:
            k = min(k, s - 2)
            w, V = eigsh(Q, k=k, which="SA", tol=1e-10, ncv=min(s, max(2 * k + 1, 40)))
            o = np.argsort(w)
            w, V = w[o], V[:, o]
            if (w[-1] - w[0]) > 28 * (eta if eta else 0.0) or k >= s - 2 or eta in (None, 0):
                return w, V, False
            k *= 2

    def evaluate(self, th, eta=None, grad=True):
        """Return dict(U=hard certified bound, Us=smoothed bound (>=U), g=gradient of Us (or the
        subgradient of U when eta is None))."""
        Qs = self.gram_blocks(th)
        U = th[0]
        Us = th[0]
        Gq = []                      # gradient wrt Q matrices (dense)
        lmins = []
        for b, (blk, Q) in enumerate(zip(self.blocks, Qs)):
            w, V, comp = self._spectrum(Q, eta or 0.0, self._eig_cache.get(b, 8))
            lm = -w[0]                       # lambda_max(-Q)
            lmins.append(w[0])
            wt = blk["weight"]
            hard = max(lm, 0.0) if blk["plus"] else lm
            U += wt * hard
            if eta is None:
                Us += wt * hard
                if grad:
                    v = V[:, 0]
                    if (not blk["plus"]) or lm > 0:
                        Gq.append(-wt * np.outer(v, v))
                    else:
                        Gq.append(None)
            else:
                z = -w / eta
                if blk["plus"]:
                    m = max(z.max(), 0.0)
                    S = np.exp(z - m).sum() + math.exp(-m)
                else:
                    m = z.max()
                    S = np.exp(z - m).sum()
                L = m + math.log(S)
                Us += wt * eta * L
                if grad:
                    pw = np.exp(z - L)
                    keep = pw > 1e-16
                    Vk = V[:, keep]
                    P = (Vk * pw[keep]) @ Vk.T
                    Gq.append(-wt * P)
                    self._eig_cache[b] = max(8, int(keep.sum()) + 4) if not comp else 8
        out = dict(U=float(U), Us=float(Us), lmins=lmins)
        if grad:
            out["g"] = self._backprop(Gq, th)
        return out

    def _backprop(self, Gq, th):
        K = len(self.cliques)
        g = np.zeros(self.dim)
        G_slot = np.zeros(self.nslots)
        for k in range(K):
            if Gq[k] is None:
                continue
            I, J = self.cl_I[k], self.cl_J[k]
            G_slot[self.cl_off[k]:self.cl_off[k + 1]] = self.slot_w[self.cl_off[k]:self.cl_off[k + 1]] * Gq[k][I, J]
        # zero-sum projection for Theta
        proj = np.bincount(self.slot_g, weights=self.slot_w * G_slot, minlength=self.nmom)
        g[self.sl_Theta] = G_slot - self.slot_w * (proj / self.W2)[self.slot_g]
        rbar = np.bincount(self.slot_g, weights=G_slot, minlength=self.nmom) / self.W
        g[0] = 1.0 + rbar[self.empty_id]
        if self.nu:
            gu = np.zeros(self.nu)
            for i, info in enumerate(self.ineq_info):
                Gi = Gq[K + i]
                if Gi is not None:
                    wq = np.where(info["a"] == info["b"], 1.0, 2.0)
                    gu[self.u_off[i]:self.u_off[i + 1]] = wq * Gi[info["a"], info["b"]]
            g[self.sl_u] = gu - self.R_loc.T @ rbar
        if self.np_:
            g[self.sl_p] = -(self.R_eq.T @ rbar)
        return g

    # -------------------------------------------------------------- convenience
    def certificate(self, th):
        """Rigorous (up to floating-point eigensolver error) upper bound on max f."""
        return self.evaluate(th, eta=None, grad=False)["U"]

    def primal_density(self, th, eta):
        """Softmax density matrices (approximate primal moment/localising blocks, scaled to trace<=1)."""
        Qs = self.gram_blocks(th)
        Ps = []
        for blk, Q in zip(self.blocks, Qs):
            w, V = np.linalg.eigh(Q)
            z = -w / eta
            m = z.max(); pw = np.exp(z - m); pw /= pw.sum()
            Ps.append((V * pw) @ V.T)
        return Ps

    # ------------------------------------------------------------ primal recovery
    def _densities(self, th, eta):
        """Density matrices P_b(theta) of the smoothed objective (same normalisation as `evaluate`:
        for blocks with the positive part the pencil diag(-Q, 0) is used, so tr P_b <= 1; for
        constant-trace (Boolean) moment blocks tr P_b = 1).  Returns the Gram blocks Q_b, the P_b,
        the entropies H_b of diag(P_b, 1 - tr P_b) (of P_b for constant-trace blocks), the hard
        bound U, the smoothed bound U_eta and the smallest eigenvalue of every Q_b."""
        Qs = self.gram_blocks(th)
        U = Us = float(th[0])
        Ps, Hs, lmins = [], [], []
        for blk, Q in zip(self.blocks, Qs):
            w, V = np.linalg.eigh(Q)
            lm = -w[0]
            lmins.append(float(w[0]))
            wt = blk["weight"]
            U += wt * (max(lm, 0.0) if blk["plus"] else lm)
            z = -w / eta
            if blk["plus"]:
                m = max(z.max(), 0.0)
                S = np.exp(z - m).sum() + math.exp(-m)
            else:
                m = z.max()
                S = np.exp(z - m).sum()
            L = m + math.log(S)
            Us += wt * eta * L
            pw = np.exp(z - L)                                   # eigenvalues of P_b
            probs = np.append(pw, math.exp(-L)) if blk["plus"] else pw
            probs = probs[probs > 0]
            Hs.append(float(-(probs * np.log(probs)).sum()))
            Ps.append((V * pw) @ V.T)
        return dict(Q=Qs, P=Ps, H=Hs, U=U, Us=Us, lmins=lmins)

    def primal_moments(self, th, eta):
        """Primal recovery (Proposition `primal recovery' of the paper).

        X_b = w_b P_b(theta) are candidate moment / localising blocks; the slot average
            ybar_gamma = (1/W_gamma) sum_{(k;a,b) in slots(gamma)} omega_ab X_k[a,b]
        is a moment vector indexed like `self.reg`.  Returns a dict with
          ybar, Lf = L_ybar(f), y0 = ybar[empty], X (list, all blocks), M (moment matrices M_k(ybar)),
          Mloc (localising matrices M_i^loc(ybar)), eq_res = (L_ybar(h_j x^mu))_{j,mu} = R_eq^T ybar,
          infeas = dict(hankel = sum_k ||M_k(ybar) - X_k||_F, hankel_max, hankel_rss = sqrt(sum_k ||.||_F^2) <= ||g_Theta||,
                        loc = sum_i ||Mloc_i - X_i||_F,
                        loc_max, eq = ||eq_res||, y0 = |y0 - 1|),
          g (gradient of U_eta at theta, computed from the same P_b), grad_norm = ||g||,
          g_Theta_norm, lmin_M (smallest eigenvalue of each M_k(ybar)), U, Us, H (entropies),
          entropy = sum_b w_b H_b(P_b), and the Lagrangian identity check
          lagr_residual = Us - eta*entropy - <theta, g> - Lf  (should be ~1e-12).
        """
        d = self._densities(th, eta)
        K = len(self.cliques)
        X = [blk["weight"] * P for blk, P in zip(self.blocks, d["P"])]
        # slot average
        xs = np.concatenate([X[k][self.cl_I[k], self.cl_J[k]] for k in range(K)])
        num = np.bincount(self.slot_g, weights=self.slot_w * xs, minlength=self.nmom)
        ybar = np.divide(num, self.W, out=np.zeros(self.nmom), where=self.W > 0)
        Lf = float(self.fvec @ ybar)
        y0 = float(ybar[self.empty_id])
        # moment matrices from ybar
        Ms, lmin_M, hank = [], [], []
        for k in range(K):
            s = len(self.cl_basis[k])
            sl = slice(self.cl_off[k], self.cl_off[k + 1])
            M = np.zeros((s, s))
            I, J = self.cl_I[k], self.cl_J[k]
            M[I, J] = ybar[self.slot_g[sl]]
            M[J, I] = ybar[self.slot_g[sl]]
            Ms.append(M)
            lmin_M.append(float(np.linalg.eigvalsh(M)[0]))
            hank.append(float(np.linalg.norm(M - X[k])))
        # localising matrices from ybar
        Mloc, locres = [], []
        for i, info in enumerate(self.ineq_info):
            s = len(info["Z"])
            wq = np.where(info["a"] == info["b"], 1.0, 2.0)
            vals = (info["R"].T @ ybar) / wq                    # L_ybar(g_i x^a x^b)
            Ml = np.zeros((s, s))
            Ml[info["a"], info["b"]] = vals
            Ml[info["b"], info["a"]] = vals
            Mloc.append(Ml)
            locres.append(float(np.linalg.norm(Ml - X[K + i])))
        eq_res = np.asarray(self.R_eq.T @ ybar).ravel() if self.np_ else np.zeros(0)
        # gradient of U_eta from the same densities
        g = self._backprop([-x for x in X], th)
        entropy = float(sum(blk["weight"] * H for blk, H in zip(self.blocks, d["H"])))
        infeas = dict(hankel=float(sum(hank)), hankel_max=float(max(hank)),
                      hankel_rss=float(np.sqrt(sum(h * h for h in hank))),   # <= ||g_Theta||  (Prop. (c))
                      loc=float(sum(locres)) if locres else 0.0, loc_max=float(max(locres)) if locres else 0.0,
                      eq=float(np.linalg.norm(eq_res)), y0=abs(y0 - 1.0))
        return dict(ybar=ybar, Lf=Lf, y0=y0, X=X, M=Ms, Mloc=Mloc, eq_res=eq_res, infeas=infeas,
                    g=g, grad_norm=float(np.linalg.norm(g)), g_Theta_norm=float(np.linalg.norm(g[self.sl_Theta])),
                    lmin_M=lmin_M, U=d["U"], Us=d["Us"], H=d["H"], entropy=entropy, P=d["P"], Q=d["Q"],
                    lmins_Q=d["lmins"],
                    lagr_residual=float(d["Us"] - eta * entropy - float(th @ g) - Lf))

    def boolean_lower_bound(self, th, eta, pm=None):
        """Two-sided certificate, Proposition `primal recovery' (d): Boolean, one clique, no constraints.
        Returns dict(LB, tau, delta, Lf, U, y0).  LB <= rho_r <= U for every theta and eta."""
        if not (self.pop.boolean and len(self.cliques) == 1 and not self.pop.ineq and not self.pop.eq):
            raise ValueError("boolean_lower_bound requires a Boolean problem with a single clique and no constraints")
        pm = self.primal_moments(th, eta) if pm is None else pm
        ybar, y0 = pm["ybar"], pm["y0"]
        assert abs(y0 - 1.0) < 1e-8, f"ybar_0 = {y0} should be 1 for a Boolean clique block"
        if abs(y0 - 1.0) > 1e-12:
            ybar = ybar / y0
        Lf = float(self.fvec @ ybar)
        M = pm["M"][0] / y0 if abs(y0 - 1.0) > 1e-12 else pm["M"][0]
        lmin = float(np.linalg.eigvalsh(M)[0])
        delta = max(-lmin, 0.0)
        tau = delta / (1.0 + delta)
        f_empty = self.pop.f.get((0,) * self.pop.n, 0.0)
        LB = (1.0 - tau) * Lf + tau * f_empty
        return dict(LB=float(LB), tau=float(tau), delta=float(delta), Lf=Lf, U=pm["U"], y0=y0, lmin_M=lmin)

    def two_sided(self, th, eta):
        """(LB, U) with LB <= rho_r <= U (Boolean, single clique, unconstrained)."""
        b = self.boolean_lower_bound(th, eta)
        return b["LB"], b["U"]
