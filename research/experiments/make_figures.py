import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
R = "experiments/results/"
C = {"lbfgs": "#1b6ca8", "agd": "#e08a00", "sub": "#8a8a8a", "mosek": "#c0392b"}

# ---- Fig 1: convergence ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
summ = {}
for a, n in zip(ax, (16, 24)):
    d = json.load(open(R + f"e2_n{n}.json")); b = json.load(open(R + f"e2b_n{n}.json")); ref = d["ref"]
    for key, lab, col, h in [("lbfgs", "smoothing + L-BFGS", C["lbfgs"], np.array(d["lbfgs"])),
                             ("agd", "smoothing + accel. gradient", C["agd"], np.array(b["agd"])),
                             ("sub", "target-level subgradient", C["sub"], np.array(d["subgrad"]))]:
        gap = np.maximum(h[:, 3] - ref, 1e-7)
        a.loglog(h[:, 1], gap, color=col, label=lab, lw=1.4)
        for tol in (1e-1, 1e-2, 1e-3):
            idx = np.where(h[:, 3] - ref <= tol)[0]
            summ[(n, key, tol)] = float(h[idx[0], 1]) if len(idx) else None
    a.axvline(d["t_mosek"], color=C["mosek"], ls="--", lw=1.2, label="MOSEK (interior point)")
    a.set_title(f"MAX-CUT, random 3-regular, $n={n}$, $r=2$"); a.set_xlabel("wall-clock time (s)")
ax[0].set_ylabel("certified bound $-$ relaxation value"); ax[0].legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig("figures/convergence.pdf"); plt.close()
json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summ.items()}, open(R + "e2_summary.json", "w"), indent=1)

# ---- Fig 2: scaling --------------------------------------------------------
rows = [json.loads(l) for l in open(R + "e3.jsonl")]
fo = sorted([r for r in rows if r["solver"] == "fo"], key=lambda r: r["n"]); mo = sorted([r for r in rows if r["solver"] == "MOSEK"], key=lambda r: r["n"])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].semilogy([r["n"] for r in fo], [r["time"] for r in fo], "o-", color=C["lbfgs"], label="first-order (ours)")
ax[0].semilogy([r["n"] for r in mo], [r["time"] for r in mo], "s-", color=C["mosek"], label="MOSEK")
ax[0].set_xlabel("$n$ (variables)"); ax[0].set_ylabel("time (s)"); ax[0].legend(fontsize=7)
ax[1].semilogy([r["n"] for r in fo], [r["rss_gib"] for r in fo], "o-", color=C["lbfgs"], label="first-order (ours)")
ax[1].semilogy([r["n"] for r in mo], [r["rss_gib"] for r in mo], "s-", color=C["mosek"], label="MOSEK")
ax[1].axhline(36, color="k", ls=":", lw=1); ax[1].text(10, 26, "RAM 36 GiB", fontsize=7)
ax[1].plot([40], [39], "x", color=C["mosek"], ms=8); ax[1].annotate("$n{=}40$: killed; Schur matrix\n$\\gtrsim$39 GiB (lower bound)", (40, 39), (10.5, 70), fontsize=7, arrowprops=dict(arrowstyle="-", lw=.6))
ax[1].set_ylim(0.08, 150); ax[1].set_xlabel("$n$ (variables)"); ax[1].set_ylabel("peak memory (GiB)")
plt.tight_layout(); plt.savefig("figures/scaling.pdf"); plt.close()

# ---- Fig 3: anytime certificate -------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for a, n in zip(ax, (40, 50, 60)):
    r = [x for x in fo if x["n"] == n][0]; h = np.array(r["hist"])
    a.plot(h[:, 0], h[:, 1], color=C["lbfgs"])
    import math; fl = math.floor(r["val"] + 1e-9)
    a.axhline(fl, color="k", ls=":", lw=1); a.set_ylim(fl - 0.5, fl + min(8, h[5, 1] - fl) if False else fl + 0.4 * (h[:, 1].max() - fl) + 1)
    a.set_title(f"$n={n}$; max cut $\\leq\\lfloor U\\rfloor={fl}$", fontsize=8); a.set_xlabel("time (s)")
ax[0].set_ylabel("best certified upper bound $U$")
plt.tight_layout(); plt.savefig("figures/anytime.pdf"); plt.close()
print(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summ.items()}, indent=0))
