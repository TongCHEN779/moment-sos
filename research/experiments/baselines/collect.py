"""Assemble experiments/results/baselines.json, tables_baselines.tex and figures/baselines.pdf from
experiments/results/baselines_raw.jsonl (+ our own histories from e3.jsonl / the n=16 rerun tagged 'ours').
Run from anywhere:  python experiments/baselines/collect.py
Metric: 'strict error' = max(|value - ref|, max constraint violation) for the infeasible-iterate baselines
(CGAL, BM, SCS: pri.res.); for our method the bound is certified so the error is U - ref >= 0.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import TOLS, time_to, times
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "experiments", "results")
NS = [16, 20, 26, 30, 40]
REF40 = 54.006490432611415

raw = [json.loads(l) for l in open(os.path.join(RES, "baselines_raw.jsonl"))]
e3 = {r["n"]: r for r in (json.loads(l) for l in open(os.path.join(RES, "e3.jsonl"))) if r["solver"] == "fo"}
ref = {d["n"]: d["val"] for d in raw if d.get("method") == "mosek" and d["tag"] == "ref"}
ref[40] = REF40
mosek = {d["n"]: d for d in raw if d.get("method") == "mosek" and d["tag"] == "ref"}
for d in raw:
    if d["tag"] == "ours":
        e3[d["n"]] = d          # fresh n=16 run (same format as e3.jsonl rows)

out = {"ref": ref, "note_n40": "reference at n=40 is our own best certified bound 54.0065 (MOSEK OOM-killed); true value in [54, 54.0065]",
       "instances": {}}
table = {}
for n in NS:
    R = ref[n]; inst = {}
    o = e3[n]
    ho = [[t, u, 0.0] for t, u in o["hist"]]
    inst["ours"] = dict(val=o["val"], err=o["val"] - R, time=o["time"], rss_gib=o["rss_gib"], iters=o["evals"],
                        times=times(ho, R), certified=True, hist=o["hist"],
                        source="e3.jsonl" if o.get("tag") != "ours" else "baselines_raw (rerun)")
    if n in mosek:
        inst["mosek"] = dict(val=mosek[n]["val"], time=mosek[n]["time"], rss_gib=mosek[n]["rss_gib"])
    for d in raw:
        if d["n"] != n or d["tag"] != "main" or "error" in d:
            continue
        m = d.get("method")
        if m == "scs":
            key = f"scs_{d['opts']['eps']:g}"
            tb = d["time_build"]
            inst[key] = dict(val=d["val"], err=d["val"] - R, time=d["time"], time_solve=d["time_solve"], time_build=tb,
                             rss_gib=d["rss_gib"], iters=d["iters"], status=d["status"], traj=d["traj"],
                             certified=False)
            # final-state criterion: the run itself ended with |err|<=tol
            # the SCS log prints the objective with 3 digits only, so the trajectory cannot resolve 1e-2/1e-3;
            # time-to-accuracy = total time of the run if its final value is within tol (an upper bound: no anytime iterate)
            inst[key]["times_final_run"] = {str(t): (d["time"] if abs(d["val"] - R) <= t else None) for t in TOLS}
            inst[key]["times"] = inst[key]["times_final_run"]
        else:
            key = "cgal" if m == "cgal" else f"bm_k{d['opts']['k']}"
            h = d["hist"]
            inst[key] = dict(val=d["val"], err=d["val"] - R, infeas_inf=d["infeas_inf"], time=d["time"], rss_gib=d["rss_gib"],
                             iters=d["iters"], status=d["status"], opts=d["opts"], times=times(h, R), times_loose=times(h, R, False),
                             certified=False, hist=h, load_start=d["load_start"], load_end=d["load_end"])
    if n == 40:      # alternative reference 54.0 (lower end of the interval [54, 54.0065] that contains the true value)
        for k, d in inst.items():
            if k == "mosek":
                continue
            h = d.get("hist") or [[t, v, pr] for it, t, v, pr, du, gap in d.get("traj", [])]
            if k.startswith("scs"):
                d["times_ref54"] = {str(t): (d["time"] if abs(d["val"] - 54.0) <= t else None) for t in TOLS}
                d["err_ref54"] = d["val"] - 54.0
                continue
            if k == "ours":
                h = [[t, u, 0.0] for t, u in d["hist"]]
            d["times_ref54"] = times(h, 54.0); d["err_ref54"] = d["val"] - 54.0
    out["instances"][str(n)] = inst
out["tuning"] = [{k: v for k, v in d.items() if k != "hist"} | dict(final_hist=d["hist"][-1] if "hist" in d else None)
                 for d in raw if d["tag"] == "tune"]
json.dump(out, open(os.path.join(RES, "baselines.json"), "w"))

# ---------------------------------------------------------------- table
def ft(t, total=None):
    if t is None:
        return "n.r."
    return f"{t:.1f}\\,s" if t < 100 else f"{t:.0f}\\,s"

names = [("ours", "ours (certified $U$)"), ("scs_0.001", "SCS $\\epsilon{=}10^{-3}$"), ("scs_1e-05", "SCS $\\epsilon{=}10^{-5}$"),
         ("cgal", "CGAL"), ("bm_k8", "BM $k{=}8$"), ("bm_k16", "BM $k{=}16$")]
L = ["\\begin{tabular}{r l r r r r r r r r}\\toprule",
     "$n$ & method & $t_{10^{-1}}$ & $t_{10^{-2}}$ & $t_{10^{-3}}$ & final value & signed err. & max viol. & total time & mem. \\\\\\midrule"]
for n in NS:
    inst = out["instances"][str(n)]; first = True
    rows = [(k, lab) for k, lab in names if k in inst]
    for k, lab in rows:
        d = inst[k]
        tt = d["times"]
        viol = "$0$" if k == "ours" else ("--" if k.startswith("scs") else f"{d['infeas_inf']:.1e}")
        L.append(f"{n if first else ''} & {lab} & {ft(tt['0.1'])} & {ft(tt['0.01'])} & {ft(tt['0.001'])} & {d['val']:.4f} & {d['err']:+.1e} & {viol} & {ft(d['time'])} & {d['rss_gib']:.2f} \\\\")
        first = False
    if n in mosek:
        mk = mosek[n]
        L.append(f" & MOSEK (IPM) & \\multicolumn{{3}}{{c}}{{(exact)}} & {mk['val']:.4f} & -- & -- & {ft(mk['time'])} & {mk['rss_gib']:.2f} \\\\")
    L.append("\\midrule" if n != NS[-1] else "")
L[-1] = "\\bottomrule\\end{tabular}"
open(os.path.join(ROOT, "tables_baselines.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
col = {"ours": "#1b6ca8", "cgal": "#2e8b57", "bm_k8": "#e08a00", "bm_k16": "#8e44ad", "scs": "#c0392b", "mosek": "#555555"}
panel = [20, 30, 40]
fig, ax = plt.subplots(1, len(panel), figsize=(7.4, 3.0), sharey=True)
for a, n in zip(ax, panel):
    inst = out["instances"][str(n)]; R = ref[n]
    h = np.array(inst["ours"]["hist"]); eo = h[:, 1] - R; keep = eo > 1e-6      # n=40: ref = our final bound, drop the trivial zero
    a.loglog(h[keep, 0], eo[keep], color=col["ours"], lw=1.6, label="ours (certified $U-$ref)")
    for k, lab in (("cgal", "CGAL"), ("bm_k8", "BM $k{=}8$"), ("bm_k16", "BM $k{=}16$")):
        if k in inst:
            h = np.array(inst[k]["hist"]); e = np.maximum(np.abs(h[:, 1] - R), h[:, 2])
            a.loglog(np.maximum(h[:, 0], 1e-3), np.maximum(e, 1e-7), color=col[k], lw=1.0, ls="--", label=lab)
    for k, mk in (("scs_0.001", "o"), ("scs_1e-05", "s")):
        if k in inst:
            d = inst[k]; a.loglog([d["time"]], [max(abs(d["err"]), 1e-7)], mk, color=col["scs"], ms=5,
                                  label="SCS " + ("$\\epsilon{=}10^{-3}$" if "0.001" in k else "$\\epsilon{=}10^{-5}$"))
    if "mosek" in inst:
        a.axvline(inst["mosek"]["time"], color=col["mosek"], ls=":", lw=1.2, label="MOSEK")
    a.set_title(f"$n={n}$, $r=2$"); a.set_xlabel("wall-clock time (s)"); a.set_ylim(1e-7, 1e2)
ax[0].set_ylabel("error vs. reference"); ax[2].set_title("$n=40$ (ref. $=$ our best bound 54.0065)", fontsize=7)
hd, lb = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels()
fig.legend(hd, lb, fontsize=6.5, ncol=7, loc="lower center", frameon=False)
plt.tight_layout(rect=(0, 0.09, 1, 1)); plt.savefig(os.path.join(ROOT, "figures", "baselines.pdf")); plt.close()
print(open(os.path.join(ROOT, "tables_baselines.tex")).read())
