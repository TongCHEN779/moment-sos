"""First-order solvers for  min_theta U(theta)."""
import time
import numpy as np
from scipy.optimize import minimize


def lbfgs_continuation(dual, theta0=None, etas=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
                       iters_per_stage=300, maxcor=30, tol_stage=1e-9, log=None, time_limit=None, verbose=False):
    """Entropic smoothing + L-BFGS with eta-continuation.  Every function evaluation also yields the
    exact (hard) certified bound U(theta); the best one seen is returned (anytime)."""
    th = dual.theta0() if theta0 is None else theta0.copy()
    t0 = time.perf_counter()
    best = dict(U=np.inf, theta=None)
    hist = []
    nev = [0]

    def fg(x, eta):
        o = dual.evaluate(x, eta=eta)
        nev[0] += 1
        if o["U"] < best["U"]:
            best["U"] = o["U"]; best["theta"] = x.copy()
        hist.append((nev[0], time.perf_counter() - t0, o["U"], best["U"]))
        return o["Us"], o["g"]

    for eta in etas:
        if time_limit and time.perf_counter() - t0 > time_limit:
            break
        res = minimize(fg, th, args=(eta,), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=iters_per_stage, maxcor=maxcor, ftol=tol_stage, gtol=1e-10, maxfun=int(iters_per_stage * 1.3)))
        th = res.x
        if verbose:
            print(f"  eta={eta:.0e}  smoothed={res.fun:.6f}  best U={best['U']:.6f}  evals={nev[0]}  t={time.perf_counter()-t0:.1f}s")
    return best["U"], best["theta"], np.array(hist)


def subgradient_target(dual, theta0=None, iters=5000, delta0=None, log_every=0):
    """Original Polyak / target-level subgradient scheme (baseline)."""
    th = dual.theta0() if theta0 is None else theta0.copy()
    t0 = time.perf_counter()
    bestU, hist = np.inf, []
    delta, path = delta0, 0.0
    for k in range(iters):
        o = dual.evaluate(th, eta=None)
        U, g = o["U"], o["g"]
        improved = U < bestU - 1e-12
        bestU = min(bestU, U)
        hist.append((k + 1, time.perf_counter() - t0, U, bestU))
        if delta is None:
            delta = max(1.0, 0.25 * abs(U))
        gn2 = float(g @ g) + 1e-18
        step = max(min((U - (bestU - delta)) / gn2, 1.0), 1e-12)
        th = th - step * g
        if improved:
            path = 0.0
        else:
            path += step * np.sqrt(gn2)
            if path > 2.0:
                delta *= 0.5; path = 0.0
    return bestU, th, np.array(hist)


def agd_continuation(dual, theta0=None, etas=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
                     iters_per_stage=2000, L0=1.0, time_limit=None, verbose=False):
    """Nesterov accelerated gradient with backtracking on the Lipschitz constant and gradient-based
    adaptive restart (O'Donoghue-Candes), run on the entropically smoothed objective U_eta with
    eta-continuation.  This is the scheme covered by the O(1/eps) complexity theorem."""
    th = dual.theta0() if theta0 is None else theta0.copy()
    t0 = time.perf_counter()
    best = dict(U=np.inf, theta=None)
    hist = []
    nev = [0]

    def fg(x, eta):
        o = dual.evaluate(x, eta=eta)
        nev[0] += 1
        if o["U"] < best["U"]:
            best["U"] = o["U"]; best["theta"] = x.copy()
        hist.append((nev[0], time.perf_counter() - t0, o["U"], best["U"]))
        return o["Us"], o["g"]

    L = L0
    for eta in etas:
        if time_limit and time.perf_counter() - t0 > time_limit:
            break
        x = th.copy(); yk = th.copy(); tk = 1.0
        f_y, g_y = fg(yk, eta)
        for it in range(iters_per_stage):
            while True:                                  # backtracking
                xn = yk - g_y / L
                f_n, g_n = fg(xn, eta)
                d = xn - yk
                if f_n <= f_y + g_y @ d + 0.5 * L * (d @ d) + 1e-14:
                    break
                L *= 2.0
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            ynew = xn + ((tk - 1) / tn) * (xn - x)
            if g_n @ (xn - x) > 0:                       # gradient restart
                tn = 1.0; ynew = xn.copy()
            x, tk = xn, tn
            yk = ynew
            f_y, g_y = fg(yk, eta)
            L = max(L * 0.9, 1e-8)
            if np.linalg.norm(g_n) < 1e-9:
                break
            if time_limit and time.perf_counter() - t0 > time_limit:
                break
        th = x
        if verbose:
            print(f"  eta={eta:.0e} best U={best['U']:.6f} evals={nev[0]} t={time.perf_counter()-t0:.1f}s L={L:.2e}")
    return best["U"], best["theta"], np.array(hist)


def alm(dual, theta0=None, rho0=1000.0, outer=300, inner=600, time_limit=None, tol=1e-7, verbose=False,
        cert_every=0, X0=None, t0=None):
    """Dual augmented Lagrangian (boundary-point / SDPNAL type) for the exact dual
        min_{theta,t}  sigma + sum_b w_b t_b   s.t.  G_b = Q_b(theta) + t_b I >= 0,  t_b >= 0 ('plus' blocks),
    a NumPy port of moment_sos.solve_alm.  Inner problems in (theta, t) are solved by L-BFGS-B (bounds t_b >= 0 for
    'plus' blocks, free otherwise); penalty  (|Pi_+(X_b - rho G_b)|_F^2 - |X_b|_F^2) / (2 rho)  with gradient
    -Pi_+(X_b - rho G_b) w.r.t. G_b; multipliers  X_b <- Pi_+(X_b - rho G_b);  rho is doubled when the primal
    infeasibility does not drop by 30%.  The certified bound U(theta) is evaluated at every outer iterate (and,
    if cert_every > 0, every cert_every inner evaluations) and the best one is returned (anytime).
    Returns (bestU, theta, hist, info): hist rows (n_eval, time, U_current, best_U) as in lbfgs_continuation;
    info = dict(X=multipliers, t, rho, outer_iters, n_eval, time, pinf, dinf, status)."""
    blocks = dual.blocks
    nb, dim = len(blocks), dual.dim
    plus = np.array([b["plus"] for b in blocks]); wts = np.array([float(b["weight"]) for b in blocks])
    active = wts != 0.0
    wt_act = wts * active
    sizes = [b["size"] for b in blocks]
    th0 = dual.theta0() if theta0 is None else np.asarray(theta0, float).copy()
    t_init = np.zeros(nb) if t0 is None else np.asarray(t0, float).copy()
    x = np.concatenate([th0, t_init])
    lo_b = np.concatenate([np.full(dim, -np.inf), np.where(plus & active, 0.0, -np.inf)])
    bounds = list(zip(lo_b, np.full(dim + nb, np.inf)))
    X = [np.zeros((s, s)) for s in sizes] if X0 is None else [np.array(Xb, float, copy=True) for Xb in X0]
    eyes = {s: np.eye(s) for s in set(sizes)}
    t_start = time.perf_counter()
    best = dict(U=dual.certificate(th0), theta=th0.copy())
    hist = [(0, time.perf_counter() - t_start, best["U"], best["U"])]
    outer_best = []                                   # best U after each outer iteration (stall test)
    nev = [0]

    def record(th):
        U = dual.certificate(th)
        if U < best["U"]:
            best["U"], best["theta"] = U, th.copy()
        hist.append((nev[0], time.perf_counter() - t_start, U, best["U"]))
        return U

    def fun(xv, rho):
        th, t = xv[:dim], xv[dim:]
        Qs = dual.gram_blocks(th)
        val = th[0] + float(wt_act @ t)
        Gq, gt = [], np.zeros(nb)
        for b in range(nb):
            M = X[b] - rho * (Qs[b] + t[b] * eyes[sizes[b]])
            w, V = np.linalg.eigh(M)
            wp = np.maximum(w, 0.0)
            Pi = (V * wp) @ V.T
            val += (float(wp @ wp) - float(np.sum(X[b] * X[b]))) / (2 * rho)
            Gq.append(-Pi)
            gt[b] = -float(wp.sum()) + wt_act[b]
        g = np.concatenate([dual._backprop(Gq, th), gt])
        nev[0] += 1
        if cert_every and nev[0] % cert_every == 0:
            record(th)
        return val, g

    def update(xv, rho):
        th, t = xv[:dim], xv[dim:]
        Qs = dual.gram_blocks(th)
        pinf = 0.0
        for b in range(nb):
            G = Qs[b] + t[b] * eyes[sizes[b]]
            w, V = np.linalg.eigh(X[b] - rho * G)
            X[b] = (V * np.maximum(w, 0.0)) @ V.T
            pinf += max(-np.linalg.eigvalsh(G)[0], 0.0) ** 2
        return pinf ** 0.5

    rho, prev, status, k = rho0, np.inf, "max_outer", -1
    pinf = dinf = np.nan
    for k in range(outer):
        res = minimize(fun, x, args=(rho,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=inner, maxcor=30, ftol=1e-15, gtol=1e-10, maxfun=int(1.3 * inner)))
        x = res.x
        g = np.asarray(res.jac).copy()
        tpart = x[dim:]
        g[dim:] = np.where((tpart > 0) | ~(plus & active), g[dim:], np.minimum(g[dim:], 0.0))
        dinf = float(np.linalg.norm(g)) / (1.0 + abs(x[0]))
        pinf = update(x, rho)
        U = record(x[:dim])
        outer_best.append(best["U"])
        if verbose and (k % 5 == 0 or k == outer - 1):
            print(f"  outer {k:3d}  rho={rho:.3g}  pinf={pinf:.2e}  dinf={dinf:.2e}  U={U:.8g}  best U={best['U']:.8g}"
                  f"  evals={nev[0]}  t={time.perf_counter()-t_start:.0f}s", flush=True)
        if pinf > 0.7 * prev:
            rho = min(rho * 2.0, 1e5)
        prev = pinf
        if pinf < tol and dinf < tol:
            status = "converged"; break
        if pinf < 1e-7 and len(outer_best) > 4 and outer_best[-5] - outer_best[-1] < 1e-9 * (1 + abs(outer_best[-1])):
            status = "stalled"; break                  # converged: no progress in the last 5 outer iterations
        if time_limit and time.perf_counter() - t_start > time_limit:
            status = "time_limit"; break
    info = dict(X=X, t=x[dim:].copy(), rho=rho, outer_iters=k + 1, n_eval=nev[0], time=time.perf_counter() - t_start,
                pinf=float(pinf), dinf=float(dinf), status=status)
    return best["U"], best["theta"], np.array(hist), info
