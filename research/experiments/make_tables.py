import json, math, sys
sys.path.insert(0, ".")
import numpy as np
from msos import SpectralDual, problems as pb
R = "experiments/results/"
out = None
def w(s): out.write(s + "\n")
def openf(n):
    global out
    if out: out.close()
    out = open(f"tables_{n}.tex", "w")
def I(x): return f"{x:,}".replace(",", "{,}")

openf("e1")
rows = json.load(open(R + "e1.json"))
w("\\begin{tabular}{l c c c r r r r}\\toprule")
w("Problem & $r$ & $s$ & \\#blocks & true opt. & MOSEK (Lasserre) & MOSEK (+ cuts) & first-order $U$ \\\\\\midrule")
maxdiff = 0; maxcut_diff = 0
for r_ in rows:
    opt = "--" if r_["opt"] is None else f"{r_['opt']:.4f}"
    w(f"{r_['problem']} & {r_['r']} & {r_['sizes'][0]} & {r_['nblocks']} & {opt} & {r_['mosek_plain']:.5f} & {r_['mosek_cuts']:.5f} & {r_['FO']:.5f} \\\\")
    maxdiff = max(maxdiff, r_["FO"] - r_["mosek_cuts"]); assert r_["FO"] >= r_["mosek_cuts"] - 1e-6
    assert abs(r_["mosek_plain"] - r_["mosek_cuts"]) < 1e-5
w("\\bottomrule\\end{tabular}")
json.dump(dict(maxdiff=maxdiff), open(R + "e1_summary.json", "w"))

openf("e2")
sm = json.load(open(R + "e2_summary.json"))
def c(n, k, t):
    v = sm[f"{n}|{k}|{t}"]; return "--" if v is None else f"{v:.2f}" if v < 100 else f"{v:.0f}"
w("\\begin{tabular}{r r c c c c}\\toprule")
w("$n$ & gap & L-BFGS & accel.\\ grad. & subgradient & MOSEK (total) \\\\\\midrule")
import json as _j
for n in (16, 24):
    tm = _j.load(open(R + f"e2_n{n}.json"))["t_mosek"]
    for i, t in enumerate(("0.1", "0.01", "0.001")):
        w(f"{n if i == 0 else ''} & $10^{{{-(i+1)}}}$ & {c(n,'lbfgs',t)} & {c(n,'agd',t)} & {c(n,'sub',t)} & {'%.1f' % tm if i == 0 else ''} \\\\")
w("\\bottomrule\\end{tabular}")
openf("e3")
rows = [json.loads(l) for l in open(R + "e3.jsonl")]
bounds = json.load(open(R + "e3_bounds.json"))
fo = {r["n"]: r for r in rows if r["solver"] == "fo"}; mo = {r["n"]: r for r in rows if r["solver"] == "MOSEK"}
w("\\begin{tabular}{r r r r r r r r r r r}\\toprule")
w("$n$ & $s$ & \\#mult. & $U$ (ours) & time & mem. & MOSEK value & time & mem. & best cut & $\\lfloor U\\rfloor$ \\\\\\midrule")
for n in sorted(fo):
    pop = pb.maxcut(n, pb.graph("reg3", n, seed=1)); d = SpectralDual(pop, 2)
    f = fo[n]; m = mo.get(n)
    mt = f"{m['val']:.4f} & {m['time']:.1f}\\,s & {m['rss_gib']:.2f}" if m else ("\\multicolumn{3}{c}{killed (SIGKILL; Schur matrix $\\gtrsim$39\\,GiB)}" if n == 40 else "\\multicolumn{3}{c}{not attempted}")
    b = bounds.get(str(n))
    bc = f"{b['lb']} & {b['floor_ub']}" if b else "-- & --"
    tstr = f"{f['time']:.1f}\\,s"
    w(f"{n} & {d.sizes()[0]} & {I(d.dim)} & {f['val']:.4f} & {tstr} & {f['rss_gib']:.2f} & {mt} & {bc} \\\\")
w("\\bottomrule\\end{tabular}")

openf("e5")
rows = json.load(open(R + "e5.json"))
w("\\begin{tabular}{l l r r r r r r r}\\toprule")
w("Problem & relaxation & max block & \\#blocks & \\#mult. & $U$ (ours) & time & MOSEK value & time \\\\\\midrule")
for r_ in rows:
    kind = "sparse" if r_["sparse"] else "dense"
    ms = f"{r_['mosek']:.5f} & {r_['t_mosek']:.1f}\\,s" if "mosek" in r_ else "-- & --"
    w(f"{r_['name']} & {kind} & {r_['maxblock']} & {r_['nblocks']} & {I(r_['dim'])} & {r_['FO']:.5f} & {r_['t_fo']:.1f}\\,s & {ms} \\\\")
w("\\bottomrule\\end{tabular}")

openf("e4")
w("\\begin{tabular}{r c r r r r r r}\\toprule")
w("$n$ & $r$ & $s$ & \\#mult. & $U$ (ours) & true opt. & time & mem. \\\\\\midrule")
for n in (20, 24):
    d = json.load(open(R + f"e4_n{n}.json"))
    for run in d["runs"]:
        w(f"{n} & {run['r']} & {run['block']} & {I(run['dim'])} & {run['U']:.5f} & {d['opt']} & {run['time']:.0f}\\,s & {run['rss_gib']:.2f} \\\\")
w("\\bottomrule\\end{tabular}")
out.close(); print("tables written")
