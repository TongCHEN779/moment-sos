"""tables_e8.tex and figures/primal_gap.pdf from experiments/results/e8.json.  Run from the repo root."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
R = "experiments/results/"
res = json.load(open(R + "e8.json"))


def sci(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    s = f"{m:.1f}\\cdot10^{{{e}}}"
    if signed and x > 0:
        s = "+" + s
    return "$" + s + "$"


def tm(t):
    return "--" if t is None else (f"{t:.2f}" if t < 10 else f"{t:.1f}" if t < 100 else f"{t:.0f}")


lines = []
w = lines.append
mc = sorted(res["maxcut"].values(), key=lambda r: r["n"])
mc = [r for r in mc if r.get("done")]
w("\\begin{tabular}{r r r r r r r r r r}\\toprule")
w("$n$ & $s$ & $\\rho_2$ & $U-\\rho_2$ & $\\rho_2-{\\rm LB}$ & $|L_{\\bar y}(f)-\\rho_2|$ & $t_{10^{-2}}$ & $t_{10^{-3}}$ & $t^{U}_{10^{-3}}$ & $t^{L}_{10^{-3}}$ \\\\\\midrule")
for r in mc:
    f = r["final"]
    rho = f"{r['rho']:.5f}" + ("$^\\dagger$" if r["rho_source"] != "Clarabel" else "")
    w(f"{r['n']} & {r['s']} & {rho} & {sci(f['U_minus_rho'])} & {sci(f['rho_minus_LB'])} & {sci(abs(f['Lf_minus_rho']))} & "
      f"{tm(r['t_gap']['0.01'])} & {tm(r['t_gap']['0.001'])} & {tm(r['t_dual']['0.001'])} & {tm(r['t_primal']['0.001'])} \\\\")
w("\\midrule")
w("\\multicolumn{10}{l}{\\footnotesize non-Boolean, $r=2$ (no lower bound): problem, $\\rho_2^{\\rm tr}$, $U-\\rho^{\\rm tr}_2$, $L_{\\bar y}(f)-\\rho^{\\rm tr}_2$ (final / best stage), infeasibility, $\\|g\\|$} \\\\")
w("\\midrule")
names = {"boxqp5_s1": "box QP $n{=}5$ (seed 1)", "boxqp5_s2": "box QP $n{=}5$ (seed 2)", "chain8_sparse": "chain quartic $n{=}8$ (sparse)", "disk": "disk"}
for key in ("boxqp5_s1", "boxqp5_s2", "chain8_sparse", "disk"):
    r = res["nonbool"].get(key)
    if not r or not r.get("done"):
        continue
    f = r["final"]
    best = min(r["stages"], key=lambda s: abs(s["Lf"] - r["rho_tr"]))
    w(f"\\multicolumn{{2}}{{l}}{{{names[key]}}} & {r['rho_tr']:.5f} & {sci(f['U_minus_rho'])} & \\multicolumn{{2}}{{c}}{{{sci(f['Lf_minus_rho'], True)} / {sci(best['Lf'] - r['rho_tr'], True)} ($\\eta{{=}}10^{{{int(round(np.log10(best['eta'])))}}}$)}} "
      f"& \\multicolumn{{2}}{{c}}{{{sci(f['infeas_total'])}}} & \\multicolumn{{2}}{{c}}{{{sci(f['grad_norm'])}}} \\\\")
w("\\bottomrule\\end{tabular}")
open("tables_e8.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure ----------------------------------------------------------------
cols = {c: i for i, c in enumerate(res["meta"]["traj_columns"])}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
C = {"U": "#1b6ca8", "LB": "#c0392b", "Lf": "#2e8b57"}
ls = {20: "-", 30: "--"}
for n in (20, 30):
    r = res["maxcut"].get(str(n))
    if not r or not r.get("done"):
        continue
    T = np.array(r["traj"]); rho = r["rho"]; t = T[:, 0]
    floor = 1e-7
    ax[0].loglog(t, np.maximum(T[:, cols["bestU"]] - rho, floor), color=C["U"], ls=ls[n], lw=1.3, label=f"$U-\\rho_2$, $n={n}$")
    ax[0].loglog(t, np.maximum(rho - T[:, cols["bestLB"]], floor), color=C["LB"], ls=ls[n], lw=1.3, label=f"$\\rho_2-{{\\rm LB}}$, $n={n}$")
    ax[0].loglog(t, np.maximum(np.abs(T[:, cols["Lf"]] - rho), floor), ".", color=C["Lf"], ms=1.2, alpha=0.25)
    st = r["stages"]
    ax[0].loglog([q["time"] for q in st], [max(abs(q["Lf"] - rho), floor) for q in st], "o" if n == 20 else "s", color=C["Lf"], ms=3.5,
                 mfc="none" if n == 30 else C["Lf"], label=f"$|L_{{\\bar y}}(f)-\\rho_2|$, $n={n}$ (stage ends; dots: every evaluation)")
ax[0].set_xlabel("wall-clock time (s)"); ax[0].set_ylabel("error w.r.t. relaxation value")
ax[0].set_title("MAX-CUT $r=2$: $n=20$ (solid), $n=30$ (dashed)"); ax[0].legend(fontsize=5.5, ncol=1, loc="lower left")
cmap = plt.get_cmap("viridis")
ns = sorted(int(k) for k, v in res["maxcut"].items() if v.get("done"))
for i, n in enumerate(ns):
    T = np.array(res["maxcut"][str(n)]["traj"])
    gap = T[:, cols["bestU"]] - T[:, cols["bestLB"]]
    ax[1].loglog(T[:, 0], np.maximum(gap, 1e-7), color=cmap(i / max(len(ns) - 1, 1)), lw=1.3, label=f"$n={n}$")
for tol in (1e-2, 1e-3):
    ax[1].axhline(tol, color="k", ls=":", lw=0.7)
ax[1].set_xlabel("wall-clock time (s)"); ax[1].set_ylabel("certified bracket $U-{\\rm LB}$")
ax[1].set_title("two-sided certificate, Prop. (d)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/primal_gap.pdf"); plt.savefig("experiments/results/primal_gap.png"); plt.close()
print("wrote tables_e8.tex, figures/primal_gap.pdf")
