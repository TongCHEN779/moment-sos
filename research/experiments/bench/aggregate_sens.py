"""results/sensitivity.jsonl -> tables_sens.tex (run from the repo root)."""
import json, collections, numpy as np
rows = [json.loads(l) for l in open("experiments/results/sensitivity.jsonl")]
g = collections.OrderedDict()
for r in rows: g.setdefault((r["eta0"], r["iters"], r["maxcor"]), []).append(r)
with open("tables_sens.tex", "w") as f:
    f.write("\\begin{tabular}{r r r c r r r}\\toprule\n$\\eta_0$ & iters/stage & L-BFGS memory & certified & $t_{\\rm cert}$ (s) & final gap & total (s)\\\\\\midrule\n")
    for (e0, ips, mc), v in g.items():
        tc = [r["t_cert"] for r in v if r["t_cert"] is not None]
        f.write(f"{e0:g} & {ips} & {mc} & {len(tc)}/{len(v)} & {(f'{np.median(tc):.1f}' if tc else '--')} & {np.median([r['gap'] for r in v]):.1e} & {np.median([r['total'] for r in v]):.0f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print(len(rows), "runs;", sum(r["t_cert"] is not None for r in rows), "certified")
