"""Language-independent problem and result format (JSON).

A problem is a dict (or a JSON file)::

    {"n": 2, "sense": "max", "name": "demo",
     "box": {"lo": -1, "hi": 1},                      # scalars or length-n lists; omit for Boolean problems
     "boolean": false,                                # true, false, or a length-n list of booleans
     "objective": [{"c": 1.0, "m": [[0, 2], [1, 1]]}, {"c": -1.0, "m": [[0, 1]]}],
     "constraints": [{"kind": "ineq", "terms": [{"c": 1.0, "m": []}, {"c": -1.0, "m": [[0, 2]]}]}]}

A term ``{"c": coefficient, "m": [[variable index, power], ...]}`` is ``c * prod x_i^p`` (0-based indices; the empty
list is the constant monomial). ``kind`` is ``"ineq"`` (poly >= 0) or ``"eq"`` (poly == 0).
This is the interface used by the command line tool and by the Julia package.
"""
from __future__ import annotations
import json
import numpy as np
from .poly import Poly, Constraint
from .problem import Problem

FORMAT_VERSION = 1


def poly_to_terms(p):
    return [{"c": float(c), "m": [[i, int(e)] for i, e in enumerate(exps) if e]} for exps, c in sorted(p.terms.items())]


def terms_to_poly(n, terms):
    out = {}
    for t in terms:
        exps = [0] * n
        for i, pw in t["m"]:
            if not 0 <= i < n:
                raise ValueError(f"variable index {i} out of range for n={n}")
            exps[i] += int(pw)
        key = tuple(exps)
        out[key] = out.get(key, 0.0) + float(t["c"])
    return Poly(n, out)


def problem_to_dict(problem):
    """Serialise a :class:`Problem` (the user's original, un-normalised data)."""
    d = {"format": FORMAT_VERSION, "n": problem.n, "sense": problem.sense, "name": problem.name,
         "boolean": [bool(b) for b in problem.bool_mask] if not problem.bool_mask.all() and problem.bool_mask.any() else bool(problem.bool_mask.all()),
         "objective": poly_to_terms(problem.objective),
         "constraints": [{"kind": c.kind, "terms": poly_to_terms(c.poly)} for c in problem.constraints]}
    if not problem.bool_mask.all():
        d["box"] = {"lo": problem.lo.tolist(), "hi": problem.hi.tolist()}
    return d


def problem_from_dict(d):
    """Build a :class:`Problem` from the dict format described in the module docstring."""
    if "n" not in d or "objective" not in d:
        raise ValueError("problem dict needs the keys 'n' and 'objective'")
    n = int(d["n"])
    obj = terms_to_poly(n, d["objective"])
    cons = []
    for c in d.get("constraints", []):
        if c["kind"] not in ("ineq", "eq"):
            raise ValueError("constraint kind must be 'ineq' or 'eq'")
        cons.append(Constraint(terms_to_poly(n, c["terms"]), c["kind"]))
    box = d.get("box")
    box = None if box is None else (box["lo"], box["hi"])
    return Problem(obj, cons, sense=d.get("sense", "max"), box=box, boolean=d.get("boolean", False), name=d.get("name", ""))


def load_problem(path):
    with open(path) as f:
        return problem_from_dict(json.load(f))


def save_problem(problem, path):
    with open(path, "w") as f:
        json.dump(problem_to_dict(problem), f)


def result_to_dict(res, rigorous=False, round_point=True):
    """Machine-readable summary of a :class:`Result` (bound, kind, point, history)."""
    P, R = res.problem, res.relaxation
    out = {"format": FORMAT_VERSION, "sense": P.sense, "bound": float(res.bound), "bound_kind": res.bound_kind,
           "order": int(R.r), "blocks": len(R.blocks), "largest_block": int(max(R.sizes())), "time": float(res.time),
           "evaluations": int(res.n_eval), "history": [[float(t), int(n), float(P.f_off + P.f_scale * u) * (1 if P.sense == "max" else -1)] for t, n, u in res.history]}
    if rigorous:
        out["rigorous"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.rigorous().items() if isinstance(v, (int, float, np.floating, bool, str))}
    if round_point:
        x, val = res.round()
        out["x"] = None if x is None else [float(v) for v in x]
        out["value"] = None if val is None else float(val)
    return out
