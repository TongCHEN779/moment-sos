"""Command line interface: ``moment-sos solve problem.json --order 2 --out result.json``."""
from __future__ import annotations
import argparse
import json
import sys
from .serialize import load_problem, result_to_dict


def _parse_cliques(v):
    if v in ("dense", "auto"):
        return v
    return json.loads(v)            # e.g. "[[0,1,2],[2,3]]"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moment-sos", description="Certified first-order bounds for polynomial optimization problems.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve a problem given as JSON (format: see moment_sos.serialize)")
    s.add_argument("problem", help="path to the problem JSON file")
    s.add_argument("--order", type=int, default=None, help="relaxation order (default: smallest admissible)")
    s.add_argument("--method", choices=["smooth", "alm"], default="smooth", help="smoothing + L-BFGS (anytime) or dual augmented Lagrangian")
    s.add_argument("--cliques", default="dense", type=_parse_cliques, help="dense | auto | JSON list of index lists")
    s.add_argument("--term-sparsity", default=None, help="term-sparsity steps: an integer or 'max' (default: off)")
    s.add_argument("--time-limit", type=float, default=None, help="seconds")
    s.add_argument("--device", default="cpu")
    s.add_argument("--rigorous", action="store_true", help="also compute the exact-arithmetic verified bound")
    s.add_argument("--no-round", action="store_true", help="skip primal recovery")
    s.add_argument("--out", default="-", help="result JSON path ('-' = stdout)")
    sub.add_parser("version", help="print the version")
    a = ap.parse_args(argv)
    if a.cmd == "version":
        from . import __version__
        print(__version__)
        return 0
    from . import solve, solve_alm, min_order
    P = load_problem(a.problem)
    order = a.order or min_order(P)
    fn = solve if a.method == "smooth" else solve_alm
    ts = None if a.term_sparsity is None else (a.term_sparsity if a.term_sparsity == "max" else int(a.term_sparsity))
    res = fn(P, order=order, cliques=a.cliques, time_limit=a.time_limit, device=a.device, term_sparsity=ts)
    out = result_to_dict(res, rigorous=a.rigorous, round_point=not a.no_round)
    text = json.dumps(out)
    if a.out == "-":
        print(text)
    else:
        open(a.out, "w").write(text)
        print(res.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
