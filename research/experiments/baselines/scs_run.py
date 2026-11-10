"""SCS (ADMM) on the moment model of msos/reference.py, via cvxpy.

msos.reference.moment_relaxation builds the model and calls prob.solve(solver=..., verbose=...) with no
extra options, so we (without editing msos/) wrap cvxpy.Problem.solve to inject the SCS tolerances and to
log the solver output (parsed for the trajectory of the primal objective).  Time 'total' includes the
cvxpy model build + canonicalisation, 'solve' is SCS' own setup+solve time.  SCS' value is approximate
(no certificate): we report it and its signed error vs the reference.
"""
import os, re, sys, time, tempfile
import numpy as np
import cvxpy as cp
from common import *


def run(n, eps=1e-3, budget=1200.0, r=2, max_iters=10 ** 7, **_):
    from msos import problems as pb, reference
    pop = pb.maxcut(n, pb.graph("reg3", n, seed=1))
    store = {}
    orig = cp.Problem.solve

    def patched(self, *a, **k):
        k.update(eps_abs=eps, eps_rel=eps, max_iters=max_iters, time_limit_secs=budget, verbose=True)
        store["prob"] = self
        store["t_call"] = time.perf_counter()
        return orig(self, *a, **k)

    cp.Problem.solve = patched
    log = tempfile.NamedTemporaryFile("w+", suffix=".scslog", delete=False)
    sys.stdout.flush(); fd = os.dup(1); os.dup2(log.fileno(), 1)
    t0 = time.perf_counter()
    try:
        val, _ = reference.moment_relaxation(pop, r, solver="SCS", verbose=True)
    finally:
        sys.stdout.flush(); os.dup2(fd, 1); os.close(fd)
        cp.Problem.solve = orig
    total = time.perf_counter() - t0
    st = store["prob"].solver_stats
    log.seek(0); txt = log.read(); log.close(); os.unlink(log.name)
    # rows: iter | pri res | dua res | gap | obj | scale | time(s)   (obj = primal objective of cvxpy's min problem)
    traj = []
    num = r"([-+0-9.e]+)"
    for m in re.finditer(r"^\s*(\d+)\|\s*" + r"\s+".join([num] * 6), txt, re.M):
        it = int(m.group(1)); pri_res, dua_res, gap, obj, _, tt = [float(m.group(i)) for i in range(2, 8)]
        traj.append([it, tt, -obj, pri_res, dua_res, gap])      # value = -obj (maximisation)
    return dict(val=float(val), time=total, time_solve=float(st.solve_time or 0) + float(st.setup_time or 0),
                time_build=total - float(st.solve_time or 0) - float(st.setup_time or 0),
                iters=int(st.num_iters), eps=eps, status=str(store["prob"].status), traj=traj,
                log_tail=txt[-600:])
