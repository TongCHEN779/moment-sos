"""Aggregate results/bench.jsonl -> tables_bench.tex and figures/bench_profile.pdf (run from the repo root)."""
import json, collections, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open("experiments/results/bench.jsonl")]
groups = collections.OrderedDict()
for r in rows:
    groups.setdefault((r["family"], r["n"]), []).append(r)
med = lambda x: float(np.median(x))
fmt = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.1f}" if x < 100 else f"{x:.0f}")
with open("tables_bench.tex", "w") as f:
    f.write("\\begin{tabular}{l r r c r r r r r r}\\toprule\n")
    f.write("family & $n$ & $s$ & cert./runs & $t_{\\rm cert}$ med. [min,max] (s) & wall (s) & CPU (s) & mem.\\ (GiB) & IPM wall (s) & load \\\\\\midrule\n")
    for (fam, n), rs in groups.items():
        tc = [r["t_cert"] for r in rs if r["t_cert"] is not None]
        mw = [r["mosek_wall"] for r in rs if "mosek_wall" in r]
        f.write(f"{fam} & {n} & {rs[0]['s']} & {len(tc)}/{len(rs)} & "
                + (f"{fmt(med(tc))} [{fmt(min(tc))}, {fmt(max(tc))}]" if tc else "--")
                + f" & {fmt(med([r['wall'] for r in rs]))} & {fmt(med([r['cpu'] for r in rs]))} & {max(r['rss_gib'] for r in rs):.2f} & "
                + (fmt(med(mw)) if mw else "--") + f" & {med([r['load0'] for r in rs]):.0f} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
# performance profile: fraction of instances certified within time t, per size class
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
sizes = sorted({n for (_, n) in groups})
classes = [("small", [n for n in sizes if n in (16, 20)]), ("mid", [n for n in sizes if n in (25, 30)]), ("large", [n for n in sizes if n in (36, 40)])]
cols = {"g05": "#1b6ca8", "pm1s": "#e08a00", "pm1d": "#2a9d8f", "torus": "#8a8a8a"}
for a, (title, ns) in zip(ax, classes):
    for fam in cols:
        ts = [r["t_cert"] if r["t_cert"] is not None else np.inf for (fm, n), rs in groups.items() if fm == fam and n in ns for r in rs]
        if not ts:
            continue
        ts = np.sort(ts); y = np.arange(1, len(ts) + 1) / len(ts)
        a.step(ts[np.isfinite(ts)], y[np.isfinite(ts)], where="post", color=cols[fam], label=fam)
    a.set_xscale("log"); a.set_ylim(0, 1.05); a.grid(alpha=.3)
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    lo, hi = a.get_xlim()
    cand = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]
    tk = [v for v in cand if lo * 0.9 <= v <= hi * 1.1]
    a.set_xticks(tk); a.set_xticklabels([f"{v:g}" for v in tk], fontsize=7); a.xaxis.set_minor_formatter(NullFormatter())
    a.set_title(f"{title} ($n={'/'.join(map(str, ns))}$)", fontsize=8); a.set_xlabel("time to certified optimality (s)")
ax[0].set_ylabel("fraction of instances"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/bench_profile.pdf")
print(len(rows), "runs;", sum(1 for r in rows if r["certified"]), "certified")
