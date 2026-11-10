"""Aggregate results/bench_hybrid.jsonl and bench.jsonl -> tables_e12.tex (run from the repo root)."""
import json, collections, numpy as np
R = "experiments/results/"
base = {(r["family"], r["n"], r["seed"]): r for r in map(json.loads, open(R + "bench.jsonl"))}
rows = [json.loads(l) for l in open(R + "bench_hybrid.jsonl")]
by = {m: {(r["family"], r["n"], r["seed"]): r for r in rows if r["mode"] == m} for m in ("hybrid", "cold", "bm")}
def cell(mode, fam, n):
    sel = [r for k, r in by[mode].items() if k[0] == fam and k[1] == n]
    if not sel: return "--"
    tc = [r["t_cert"] for r in sel if r["t_cert"] is not None]
    return (f"{np.median(tc):.1f}" if np.median(tc) < 100 else f"{np.median(tc):.0f}") + f" ({len(tc)}/{len(sel)})" if tc else f"-- (0/{len(sel)})"
with open("tables_e12.tex", "w") as f:
    f.write("\\begin{tabular}{l r r r r r r}\\toprule\n")
    f.write("family & $n$ & default schedule & short schedule & BM + multipliers & BM $\\to$ our refinement & BM phase (s)\\\\\\midrule\n")
    for fam, ns in (("g05", (20, 30, 40)), ("pm1s", (20, 30, 40)), ("pm1d", (20, 30, 40)), ("torus", (16, 25, 36))):
        for n in ns:
            o = [base[k]["t_cert"] for k in base if k[0] == fam and k[1] == n]
            bmp = [r["t_bm"] for k, r in by["hybrid"].items() if k[0] == fam and k[1] == n]
            f.write(f"{fam} & {n} & {np.median(o):.1f} (5/5) & {cell('cold', fam, n)} & {cell('bm', fam, n)} & {cell('hybrid', fam, n)} & {np.median(bmp):.1f} \\\\\n")
    e3 = [json.loads(l) for l in open(R + "e3.jsonl")]
    for n, lb in ((50, 68), (60, 81)):
        r = [x for x in e3 if x["solver"] == "fo" and x["n"] == n][0]; h = np.array(r["hist"]); ok = np.where(np.floor(h[:, 1] + 1e-9) <= lb + 1e-9)[0]
        sel = [x for k, x in by["cold"].items() if k[0] == "reg3" and k[1] == n]; tc = [x["t_cert"] for x in sel if x["t_cert"] is not None]
        f.write(f"reg3 & {n} & {h[ok[0],0]:.0f} (1/1) & {np.median(tc):.0f} ({len(tc)}/{len(sel)}) & -- & -- & -- \\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
# summary numbers used in the text
sp = []
for k, r in by["cold"].items():
    if r["t_cert"] and k in base: sp.append(base[k]["t_cert"] / r["t_cert"])
print("short vs default speed-up: median %.1f  min %.1f  max %.1f over %d runs" % (np.median(sp), min(sp), max(sp), len(sp)))
print("short schedule certified:", sum(1 for r in by["cold"].values() if r["certified"]), "of", len(by["cold"]))
print("hybrid certified:", sum(1 for r in by["hybrid"].values() if r["certified"]), "of", len(by["hybrid"]))
print("BM+multipliers certified:", sum(1 for r in by["bm"].values() if r["certified"]), "of", len(by["bm"]))
hy_vs_cold = [(r["t_cert"], by["cold"][k]["t_cert"]) for k, r in by["hybrid"].items() if k in by["cold"] and r["t_cert"] and by["cold"][k]["t_cert"]]
print("hybrid faster than short schedule in", sum(1 for a, b in hy_vs_cold if a < b), "of", len(hy_vs_cold), "paired runs")
