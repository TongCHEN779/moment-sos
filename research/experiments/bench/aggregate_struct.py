"""results/bench_struct.jsonl -> tables_struct.tex (run from the repo root)."""
import json, collections, numpy as np
rows = [json.loads(l) for l in open("experiments/results/bench_struct.jsonl")]
g = collections.OrderedDict()
for r in rows: g.setdefault((r["kind"], r["n"]), []).append(r)
name = {"boxqp": "box QP", "chain_sparse": "chain, sparse", "chain_dense": "chain, dense"}
sci = lambda x: f"{x:.1e}".replace("e-0", "e-").replace("e-", "\\!\\times\\!10^{-") + "}" if False else f"{x:.1e}"
with open("tables_struct.tex", "w") as f:
    f.write("\\begin{tabular}{l r r r r r r r r r}\\toprule\n")
    f.write("problem & $n$ & seeds & blocks / max & IPM (s) & smoothing gap & smoothing (s) & ALM gap & ALM (s) \\\\\\midrule\n")
    for (k, n), v in g.items():
        a = lambda key: np.array([r[key] for r in v])
        f.write(f"{name[k]} & {n} & {len(v)} & {v[0]['blocks']} / {v[0]['maxblock']} & {np.median(a('t_ipm')):.2f} & {np.median(a('gap_smooth')):.1e} ({a('gap_smooth').max():.0e}) & {np.median(a('t_smooth')):.1f} & {np.median(a('gap_alm')):.1e} ({a('gap_alm').max():.0e}) & {np.median(a('t_alm')):.1f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print(len(rows), "runs")
