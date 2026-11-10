"""Tables and figure for E10 from results/e10a.json and results/e10b.json (run from experiments/):
   ../tables_e10a.tex, ../tables_e10b.tex, ../figures/alm_maxcut.pdf."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
R = "results/"
GAPS = ("1e-01", "1e-02", "1e-03", "1e-04")


def fmt_t(v):
    if v is None:
        return "--"
    return f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}"


def fmt_gap(g):
    if g <= 0:
        return "$0$"
    e = int(np.floor(np.log10(g))); m = g / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def label(r):
    return "smoothing + L-BFGS" if r["solver"] == "smoothing" else f"ALM, $\\rho_0={int(r['rho0'])}$"


def tables_e10a():
    if not os.path.exists(R + "e10a.json"):
        return
    d = json.load(open(R + "e10a.json"))
    runs = sorted(d["runs"], key=lambda r: (r["n"], r["solver"] != "smoothing", r["rho0"]))
    lines = ["\\begin{tabular}{r l c c c c c r r}\\toprule",
             "$n$ & solver & $t_{10^{-1}}$ & $t_{10^{-2}}$ & $t_{10^{-3}}$ & $t_{10^{-4}}$ & final gap & wall (s) & CPU (s) \\\\\\midrule"]
    last = None
    for r in runs:
        nlab = f"{r['n']}" if r["n"] != last else ""
        if r["n"] != last and last is not None:
            lines.append("\\addlinespace")
        last = r["n"]
        tg = [fmt_t(r["t_gap"][g]) for g in GAPS]
        lines.append(f"{nlab} & {label(r)} & " + " & ".join(tg) + f" & {fmt_gap(r['final_gap'])} & {r['final_time']:.0f} & {r.get('cpu_time', float('nan')):.0f} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    open("../tables_e10a.tex", "w").write("\n".join(lines) + "\n")
    # figure: certified gap vs time, one panel per n
    ns = sorted({r["n"] for r in runs})
    cols = {"smoothing": "#1b6ca8", 100.0: "#8fbc5a", 300.0: "#e08a00", 1000.0: "#c0392b"}
    fig, ax = plt.subplots(1, len(ns), figsize=(1.9 * len(ns) + 0.6, 2.8), sharey=True, squeeze=False)
    for a, n in zip(ax[0], ns):
        for r in [q for q in runs if q["n"] == n]:
            h = np.array(r["hist"]); gap = np.maximum(h[:, 3] - r["ref"], 1e-7)
            key = "smoothing" if r["solver"] == "smoothing" else r["rho0"]
            a.loglog(np.maximum(h[:, 1], 1e-2), gap, color=cols[key], lw=1.3, label=label(r))
        a.set_title(f"$n={n}$", fontsize=9); a.set_xlabel("wall-clock time (s)")
        for g in (1e-2, 1e-3, 1e-4):
            a.axhline(g, color="k", lw=.4, ls=":")
    ax[0][0].set_ylabel("certified bound $-$ relaxation value"); ax[0][0].set_ylim(5e-8, 30)
    ax[0][-1].legend(fontsize=6.5, loc="lower left")
    plt.tight_layout(); os.makedirs("../figures", exist_ok=True); plt.savefig("../figures/alm_maxcut.pdf"); plt.close()


def stat(v):
    v = np.asarray(v, float)
    return v.mean(), v.min(), v.max()


def tables_e10b():
    if not os.path.exists(R + "e10b.json"):
        return
    d = json.load(open(R + "e10b.json"))
    rows = d["rows"]
    ns = sorted({r["n"] for r in rows})
    lines = ["\\begin{tabular}{r r r l l l c c}\\toprule",
             "$n$ & $s$ & inst. & wall time (s) & CPU time (s) & margin $U-\\lfloor U\\rfloor$ & certified & IPM wall (s) \\\\\\midrule"]
    summ = {}
    for n in ns:
        rr = [r for r in rows if r["n"] == n]
        tm = stat([r["time"] for r in rr]); cpu = [r["cpu_time"] for r in rr if "cpu_time" in r]; ev = stat([r["evals"] for r in rr]); mg = stat([r["margin"] for r in rr])
        cert = sum(r["certified"] for r in rr)
        ipm = [r["ipm_time"] for r in rr if "ipm_time" in r]
        ipm_s = f"{np.mean(ipm):.1f}" + (f" ({len(ipm)}/{len(rr)})" if len(ipm) != len(rr) else "") if ipm else "--"
        cs = stat(cpu) if cpu else None
        cpu_s = (f"{cs[0]:.1f} [{cs[1]:.1f}, {cs[2]:.1f}]" + (f" ({len(cpu)}/{len(rr)})" if len(cpu) != len(rr) else "")) if cpu else "--"
        lines.append(f"{n} & {rr[0]['block_size']} & {len(rr)} & {tm[0]:.1f} [{tm[1]:.1f}, {tm[2]:.1f}] & {cpu_s} & "
                     f"{mg[0]:.1e} [{mg[1]:.1e}, {mg[2]:.1e}] & {cert}/{len(rr)} & {ipm_s} \\\\")
        summ[n] = dict(time=tm, cpu_time=cs, cpu_count=len(cpu), evals=ev, margin=mg, certified=cert, count=len(rr), ipm_time_mean=float(np.mean(ipm)) if ipm else None,
                       ipm_all_match=all(r.get("ipm_floor_equals_cut", True) for r in rr))
    lines.append("\\bottomrule\\end{tabular}")
    open("../tables_e10b.tex", "w").write("\n".join(lines) + "\n")
    json.dump({str(k): v for k, v in summ.items()}, open(R + "e10b_summary.json", "w"), indent=1, default=float)
    return summ


if __name__ == "__main__":
    tables_e10a(); s = tables_e10b()
    for f in ("../tables_e10a.tex", "../tables_e10b.tex"):
        if os.path.exists(f):
            print(f, "\n", open(f).read())
