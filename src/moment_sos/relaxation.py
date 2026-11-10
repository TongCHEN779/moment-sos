"""Index bookkeeping of the sparse moment-SOS relaxation (numpy, no torch).

For a normalised problem (box [-1,1]^n) it builds the Gram-matrix dual of the paper:

    sigma - f = sum_k v_k^T Q0_k v_k + sum_i (z_i^T Q_i z_i) g_i + sum_j p_j h_j     (mod ideal),

with free symmetric Gram matrices Q.  Everything is expressed through integer index arrays so that
`moment_sos.oracle.Oracle` can evaluate it with a handful of torch gather / scatter operations.
"""
from __future__ import annotations
from itertools import combinations, combinations_with_replacement
import numpy as np
import networkx as nx
from .problem import Problem


# ----------------------------------------------------------------------------- monomial encoding
def _void(E):
    E = np.ascontiguousarray(E)
    return E.view(np.dtype((np.void, E.shape[1])))[:, 0]


def _enc(E, boolean):
    """Exponent rows -> product representation (bit-packed for fully Boolean, raw uint8 exponents otherwise)."""
    E = np.asarray(E, np.uint8)
    return np.packbits(E, axis=1) if boolean is True else E


def _mul(A, B, boolean, mask=None):
    if boolean is True:
        return A ^ B
    E = A + B
    if mask is not None and mask.any():
        E[:, mask] &= 1                                   # x_i^2 = 1 for the Boolean variables
    return E


def _basis(variables, d, n, boolean, mask=None):
    rows = []
    for k in range(d + 1):
        it = combinations(variables, k) if boolean is True else combinations_with_replacement(variables, k)
        for c in it:
            e = np.zeros(n, np.uint8)
            for i in c:
                e[i] += 1
            if mask is not None and np.any(e[mask] > 1):
                continue
            rows.append(e)
    return np.array(rows, np.uint8)


def _add_rows(B, extra, clique, boolean, mask, n):
    """Append the rows of `extra` (exponent vectors) supported inside `clique` that are not yet in the basis B."""
    cl = np.zeros(n, bool)
    cl[list(clique)] = True
    have = {bytes(r) for r in B}
    rows = []
    for e in extra:
        e = np.asarray(e, np.uint8).copy()
        if mask is not None and mask.any():
            e[mask] = np.minimum(e[mask], 1)                  # x_i^2 = 1 for spins
        if np.any(e[~cl]):
            continue
        k = bytes(e)
        if k not in have:
            have.add(k); rows.append(e)
    return np.vstack([B] + [np.array(rows, np.uint8)]) if rows else B


def box_sup_bound(terms, boolean, tight=False):
    """Upper bound (>= 0) on sup of the polynomial over [-1,1]^n, Eq. (B) of the paper.

    default : positive coefficients count, negative ones only if the monomial can be negative; a negative constant is
              ignored (valid, slightly loose).
    tight   : term-wise interval bound including the constant:  c_0 + sum_{even} [c_a]_+ + sum_{odd} |c_a|  (clipped at 0);
              never larger than the default and still >= sup over the box."""
    b = 0.0
    for e, c in terms.items():
        const = sum(e) == 0
        if const:
            b += c if tight else max(c, 0.0)
        elif c > 0:
            b += c
        elif boolean or any(x % 2 for x in e):
            b += -c
    return max(b, 0.0)


def box_sup_bound_exact(terms, boolean, tight=False):
    """`box_sup_bound` summed in exact rational arithmetic (a float sum can round below the true bound)."""
    from fractions import Fraction
    b = Fraction(0)
    for e, c in terms.items():
        c = Fraction(float(c)); const = sum(e) == 0
        if const:
            b += c if tight else max(c, Fraction(0))
        elif c > 0:
            b += c
        elif boolean or any(x % 2 for x in e):
            b += -c
    return max(b, Fraction(0))


def reduce_boolean(terms, mask=None):
    out = {}
    for e, c in terms.items():
        e = tuple(x % 2 if (mask is None or mask[i]) else x for i, x in enumerate(e))
        out[e] = out.get(e, 0.0) + c
    return {e: c for e, c in out.items() if c != 0.0}


def _degree(t):
    return max((sum(e) for e in t), default=0)


# ----------------------------------------------------------------------------- cliques
def correlative_sparsity_cliques(prob: Problem):
    """Maximal cliques of a chordal extension (min-fill-in) of the correlative sparsity graph."""
    n = prob.n
    G = nx.empty_graph(n)
    for p in [prob.f] + prob.ineq + prob.eq:
        for e in p.terms:
            G.add_edges_from(combinations([i for i, x in enumerate(e) if x], 2))
    bags = []
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            bags.append(tuple(comp))
            continue
        _, tree = nx.approximation.treewidth_min_fill_in(G.subgraph(comp))
        bags.extend(tuple(sorted(b)) for b in tree.nodes)
    bags = sorted(set(bags), key=len, reverse=True)
    kept = []
    for b in bags:
        if not any(set(b) <= set(k) for k in kept):
            kept.append(b)
    return [list(b) for b in kept]


# ----------------------------------------------------------------------------- term sparsity
def _chordal_max_cliques(G):
    """Maximal cliques of a min-fill-in chordal extension of G (all vertices are covered)."""
    bags = []
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            bags.append(tuple(comp))
            continue
        _, tree = nx.approximation.treewidth_min_fill_in(G.subgraph(comp))
        bags.extend(tuple(sorted(b)) for b in tree.nodes)
    bags = sorted(set(bags), key=len, reverse=True)
    kept = []
    for b in bags:
        sb = set(b)
        if not any(sb <= k for k in kept):
            kept.append(sb)
    return [np.array(sorted(k), np.int64) for k in kept]


def term_sparsity_blocks(Be, boolean, mask, support, steps=1, max_steps=25):
    """Sub-bases of a moment block from term-sparsity block closure (TSSOS-style).

    Be      : encoded basis monomials of the block (rows).
    support : encoded monomials (void keys) that the Gram matrices must be able to represent
              (objective support and the products arising from the constraints).
    steps   : number of iterations of the support extension, or "max" (until the blocks stop changing).

    Graph on the basis: {a, b} is an edge iff a*b lies in the current support (the diagonal is always present).
    The blocks are the maximal cliques of a chordal extension; the next support adds the products a*b of all
    pairs inside one block. Every choice of blocks gives a valid (weaker) relaxation; more steps = tighter."""
    s = len(Be)
    I, J = np.triu_indices(s, 1)
    P = _void(_mul(Be[I], Be[J], boolean, mask))
    S = support
    keys = np.concatenate([P, S])
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    pid, sid = inv[:len(P)], inv[len(P):]
    in_support = np.zeros(inv.max() + 1, bool)
    in_support[sid] = True
    cap = max_steps if steps == "max" else int(steps)
    blocks, prev = None, None
    for step in range(cap):
        edge = in_support[pid]
        G = nx.Graph()
        G.add_nodes_from(range(s))
        G.add_edges_from(zip(I[edge].tolist(), J[edge].tolist()))
        blocks = _chordal_max_cliques(G)
        sig = sorted(tuple(b.tolist()) for b in blocks)
        if sig == prev:
            break
        prev = sig
        if step == cap - 1:
            break
        # support extension: products of all pairs lying in a common block
        for b in blocks:
            if len(b) < 2:
                continue
            a_, c_ = np.triu_indices(len(b), 1)
            lo, hi = b[a_], b[c_]
            pos = lo * s - lo * (lo + 1) // 2 + (hi - lo - 1)
            in_support[pid[pos]] = True
    return blocks


# ----------------------------------------------------------------------------- relaxation
class Relaxation:
    """Pure-numpy index structure of the order-`order` relaxation of a `Problem`."""

    def __init__(self, prob: Problem, order: int, cliques="dense", tight_weights: bool = False, term_sparsity=None, extra_basis=None):
        self.prob, self.r = prob, int(order)
        self.term_sparsity = term_sparsity
        self.extra_basis = None if extra_basis is None else np.asarray(extra_basis, np.uint8).reshape(-1, prob.n)
        self.tight_weights = bool(tight_weights)
        n, r = prob.n, self.r
        bl = True if prob.boolean else False                 # fully Boolean fast path (bit-packed, constant trace)
        mask = prob.bool_mask
        self.mask = mask
        D = 2 * r
        anyb = bool(mask.any())
        f = reduce_boolean(prob.f.terms, mask) if anyb else dict(prob.f.terms)
        ineq = [reduce_boolean(g.terms, mask) if anyb else dict(g.terms) for g in prob.ineq]
        eq = [reduce_boolean(h.terms, mask) if anyb else dict(h.terms) for h in prob.eq]
        self.ineq_terms, self.eq_terms, self.f_terms = ineq, eq, f      # reduced polynomial data (kept for verification)
        for nm, t in [("objective", f)] + [("inequality", g) for g in ineq] + [("equality", h) for h in eq]:
            if _degree(t) > D:
                raise ValueError(f"the {nm} has degree {_degree(t)} > 2*order = {D}: use order >= {(_degree(t) + 1) // 2}")
        if cliques in ("dense", None):
            self.cliques = [list(range(n))]
        elif cliques == "auto":
            self.cliques = correlative_sparsity_cliques(prob)
        else:
            self.cliques = [sorted(c) for c in cliques]

        def host(t):
            sup = {i for e in t for i, x in enumerate(e) if x}
            return next((k for k, c in enumerate(self.cliques) if sup <= set(c)), None)

        chunks = []                                   # key arrays, registry is built from all of them
        def add_chunk(E):
            chunks.append(_void(E)); return len(chunks) - 1

        # ---- objective
        f_exps = np.array(list(f.keys()) + [(0,) * n], np.uint8)
        f_chunk = add_chunk(_enc(f_exps, bl))

        # ---- inequality localising blocks
        self.ineq_info = []
        for g in ineq:
            h_ = host(g)
            if h_ is None:
                raise ValueError("an inequality constraint involves variables that are not contained in a single clique: enlarge the cliques (or use cliques=\"dense\"/\"auto\")")
            om = (_degree(g) + 1) // 2
            if r - om < 0:
                raise ValueError("relaxation order too low for an inequality constraint: its localising matrix needs order >= ceil(deg/2)")
            Z = _basis(self.cliques[h_], r - om, n, bl, mask)
            Ze = _enc(Z, bl)
            a, b = np.triu_indices(len(Z))
            zz = _mul(Ze[a], Ze[b], bl, mask)
            terms = list(g.items())
            cidx = [add_chunk(_mul(zz, _enc(np.array([e], np.uint8), bl), bl, mask)) for e, _ in terms]
            self.ineq_info.append(dict(Z=Z, a=a, b=b, coefs=[c for _, c in terms], cidx=cidx,
                                       B=box_sup_bound(g, bl or False, tight_weights), host=h_, ne=len(a)))
        # ---- equality multiplier polynomials
        self.eq_info = []
        for h in eq:
            h_ = host(h)
            if h_ is None:
                raise ValueError("an equality constraint involves variables that are not contained in a single clique: enlarge the cliques (or use cliques=\"dense\"/\"auto\")")
            M = _basis(self.cliques[h_], D - _degree(h), n, bl, mask)
            Me = _enc(M, bl)
            terms = list(h.items())
            cidx = [add_chunk(_mul(Me, _enc(np.array([e], np.uint8), bl), bl, mask)) for e, _ in terms]
            self.eq_info.append(dict(M=M, coefs=[c for _, c in terms], cidx=cidx, nm=len(M)))

        # ---- clique moment blocks (upper-triangular slots); with term sparsity a clique block is split into sub-blocks
        support = np.concatenate([chunks[f_chunk]] + [chunks[ci] for info in self.ineq_info + self.eq_info for ci in info["cidx"]])
        self.cl_basis, self.cl_I, self.cl_J, self.cl_off, slot_chunks, self.block_clique = [], [], [], [0], [], []
        for kc, c in enumerate(self.cliques):
            B = _basis(c, r, n, bl, mask)
            if self.extra_basis is not None and len(self.extra_basis):
                B = _add_rows(B, self.extra_basis, c, bl, mask, n)
            if term_sparsity:
                parts = [B[idx] for idx in term_sparsity_blocks(_enc(B, bl), bl, mask, support, term_sparsity)]
            else:
                parts = [B]
            for Bk in parts:
                Be = _enc(Bk, bl)
                I, J = np.triu_indices(len(Bk))
                self.cl_basis.append(Bk); self.cl_I.append(I); self.cl_J.append(J); self.block_clique.append(kc)
                slot_chunks.append(add_chunk(_mul(Be[I], Be[J], bl, mask)))
                self.cl_off.append(self.cl_off[-1] + len(I))
        self.nslots = self.cl_off[-1]

        # ---- moment registry (one np.unique over everything)
        sizes = np.array([len(k) for k in chunks])
        offs = np.concatenate([[0], np.cumsum(sizes)])
        reg, inv = np.unique(np.concatenate(chunks), return_inverse=True)
        inv = inv.reshape(-1)
        ch = lambda i: inv[offs[i]:offs[i + 1]]
        self.nmom = len(reg)
        self.slot_g = np.concatenate([ch(i) for i in slot_chunks])
        self.slot_w = np.where(np.concatenate(self.cl_I) == np.concatenate(self.cl_J), 1.0, 2.0)
        self.W = np.bincount(self.slot_g, weights=self.slot_w, minlength=self.nmom)
        self.W2 = np.bincount(self.slot_g, weights=self.slot_w ** 2, minlength=self.nmom)
        fi = ch(f_chunk)
        self.empty_id = int(fi[-1])
        self.fvec = np.zeros(self.nmom)
        for (e, c), j in zip(f.items(), fi[:-1]):
            self.fvec[j] += c
        for t, nm in [(info["cidx"], "constraint") for info in self.ineq_info + self.eq_info]:
            for ci in t:
                if np.any(self.W[ch(ci)] == 0):
                    raise ValueError(f"a {nm} product is not representable by the clique Gram matrices")
        if np.any(self.W[fi] == 0):
            raise ValueError("an objective monomial is not a product of two basis monomials of one clique")

        # ---- linear maps u -> moment level, p -> moment level  (COO triplets)
        self.u_off, rows, cols, vals = [0], [], [], []
        for info in self.ineq_info:
            wq = np.where(info["a"] == info["b"], 1.0, 2.0)
            for c, ci in zip(info["coefs"], info["cidx"]):
                rows.append(ch(ci)); cols.append(np.arange(info["ne"]) + self.u_off[-1]); vals.append(c * wq)
            self.u_off.append(self.u_off[-1] + info["ne"])
        self.nu = self.u_off[-1]
        self.Rloc = tuple(np.concatenate(x) for x in (rows, cols, vals)) if rows else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        self.p_off, rows, cols, vals = [0], [], [], []
        for info in self.eq_info:
            for c, ci in zip(info["coefs"], info["cidx"]):
                rows.append(ch(ci)); cols.append(np.arange(info["nm"]) + self.p_off[-1]); vals.append(np.full(info["nm"], c))
            self.p_off.append(self.p_off[-1] + info["nm"])
        self.np_ = self.p_off[-1]
        self.Req = tuple(np.concatenate(x) for x in (rows, cols, vals)) if rows else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))

        # ---- block table: size, weight, whether the shift keeps its positive part
        ct = bl
        self.blocks = []
        for k, B in enumerate(self.cl_basis):
            self.blocks.append(dict(kind="moment", size=len(B), weight=float(len(B)), plus=not ct))
        for info in self.ineq_info:
            s = len(info["Z"])
            self.blocks.append(dict(kind="loc", size=s, weight=s * info["B"], plus=True))
        self.dim = 1 + self.nslots + self.nu + self.np_

    def sizes(self):
        """Sizes of all PSD blocks (clique moment blocks first, then localising blocks)."""
        return [b["size"] for b in self.blocks]

    def __repr__(self):
        return (f"Relaxation(order={self.r}, cliques={len(self.cliques)}, term sparsity={self.term_sparsity}, extra monomials={0 if self.extra_basis is None else len(self.extra_basis)}, blocks={len(self.blocks)}, "
                f"max block={max(self.sizes())}, multipliers={self.dim:,}, moments={self.nmom:,})")
