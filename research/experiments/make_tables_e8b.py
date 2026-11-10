"""tables_e8b.tex (E8 addendum: tight stationarity) from experiments/results/e8.json['tight'].  Run from the repo root."""
import json, numpy as np
res = json.load(open("experiments/results/e8.json"))


def sci(x):
    if x is None:
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    return f"${m:.1f}\\cdot10^{{{e}}}$"


def tm(t):
    return "--" if t is None else (f"{t:.2f}" if t < 10 else f"{t:.1f}" if t < 100 else f"{t:.0f}")


L = []
w = L.append
w("\\begin{tabular}{r l r r r r r r r r}\\toprule")
w("$n$ & stage & iter. & $\\|g\\|$ & $\\rho_2-{\\rm LB}$ & $U-\\rho_2$ & $U-{\\rm LB}$ & $t_{10^{-2}}$ & $t_{10^{-3}}$ & total (s) \\\\\\midrule")
for key in sorted(res.get("tight", {}), key=int):
    r = res["tight"][key]; rho = r["rho"]
    std = [s for s in r["stages"] if s["phase"] == "standard"][-1]
    tight = [s for s in r["stages"] if s["phase"] == "tight"]
    rows = [("standard ($\\eta\\downarrow10^{-5}$)", sum(s["nit"] for s in r["stages"] if s["phase"] == "standard"), std)]
    rows += [(f"tight $\\eta={s['eta']:g}$".replace("e-0", "\\cdot10^{-").replace("0.001", "10^{-3}").replace("0.0003", "3\\cdot10^{-4}").replace("0.0001", "10^{-4}"), s["nit"], s) for s in tight]
    for i, (lab, nit, s) in enumerate(rows):
        first = f"{key}" if i == 0 else ""
        last = f"{tm(r['t_gap']['0.01'])} & {tm(r['t_gap']['0.001'])} & {r['total_time']:.0f}" if i == len(rows) - 1 else "& &"
        w(f"{first} & {lab} & {nit} & {sci(s['grad_norm'])} & {sci(rho - s['LB'])} & {sci(s['U'] - rho)} & {sci(s['bracket'])} & {last} \\\\")
    w("\\addlinespace")
w("\\bottomrule\\end{tabular}")
open("tables_e8b.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
