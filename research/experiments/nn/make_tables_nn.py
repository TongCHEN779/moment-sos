import json, glob, os
R = "experiments/results/"
w = lambda f, s: f.write(s + "\n")

# ---- N1
n1 = json.load(open(R + "nn_n1_new.json")); a1 = json.load(open(R + "nn_alm_n1.json"))
with open("tables_n1.tex", "w") as f:
    w(f, "\\begin{tabular}{l r r r r r r r}\\toprule")
    w(f, "net $(d;\\text{widths})$ & $n$ & unst. & \\multicolumn{4}{c}{certified radius $\\varepsilon$} \\\\\\cmidrule(lr){4-7}")
    w(f, " & & & IBP & order 1 & order 2 & exact \\\\\\midrule")
    for r, a in zip(n1, a1):
        rad = r["radius"]
        w(f, f"${r['d']};{r['widths'][0]}{{,}}{r['widths'][1]}$ & {r['n']} & {r['n_unstable']} & {rad['ibp']:.3f} & {rad['order1']:.3f} & {rad['order2']:.3f} & {rad['exact']:.3f} \\\\")
    w(f, "\\bottomrule\\end{tabular}\\par\\medskip")
    w(f, "\\begin{tabular}{l r r r r r r}\\toprule")
    w(f, "net & \\multicolumn{6}{c}{bound on $\\max(\\ell_{\\rm wrong}-\\ell_{\\rm true})$ at $\\varepsilon=0.97\\,\\varepsilon^\\star$} \\\\\\cmidrule(lr){2-7}")
    w(f, " & exact & order 1 (IPM) & order 2 (IPM) & order 1 (ALM) & order 2 (ALM) & order 2 (smooth.) \\\\\\midrule")
    for r, a in zip(n1, a1):
        w(f, f"${r['d']};{r['widths'][0]}{{,}}{r['widths'][1]}$ & {r['exact']:.4f} & {r['mosek1']:.4f} & {r['mosek2']:.4f} & {a['alm1']:.4f} & {a['alm2']:.4f} & {r['ours2']:.4f} \\\\")
    w(f, "\\bottomrule\\end{tabular}")

# ---- N2
n2 = json.load(open(R + "nn_n2_new.json")); a2 = json.load(open(R + "nn_alm_n2.json"))
with open("tables_n2.tex", "w") as f:
    w(f, "\\begin{tabular}{l r r r r r r r r}\\toprule")
    w(f, "network & $n$ & exact $L^2_{\\rm pat}$ & LipSDP & order 2 (IPM) & order 2 (ALM) & time & order 3 (ALM) & time \\\\\\midrule")
    for r, a in zip(n2, a2):
        name = f"{r['kind']} $d{{=}}{r['d']},w{{=}}{r['w']}$"
        ipm = f"{r['mosek2']:.5f}" if "mosek2" in r else "--"
        o3 = f"{a['alm3']:.5f} & {a['t_alm3']:.0f}\\,s" if "alm3" in a else "-- & --"
        w(f, f"{name} & {r['n']} & {r['exact_pattern']:.5f} & {r['lipsdp']:.5f} & {ipm} & {a['alm2']:.5f} & {a['t_alm2']:.0f}\\,s & {o3} \\\\")
    w(f, "\\bottomrule\\end{tabular}")

# ---- N3
rows = []
for tag, f1, f2 in [("A", "nn_n3_5_40-40_11_0.1.json", "nn_alm_5_40-40_11_0.1.json"), ("C", "nn_n3_6_48-48_12_0.1.json", "nn_alm_6_48-48_12_0.1.json"), ("B", "nn_n3_5_40-40_11_0.2.json", "nn_alm_5_40-40_11_0.2.json")]:
    if not (os.path.exists(R + f1) and os.path.exists(R + f2)):
        continue
    s, a = json.load(open(R + f1)), json.load(open(R + f2))
    rows.append((tag, s, a))
cols = []
for tag, s_, a in rows:
    ipm2 = f"{s_['mosek2_sparse']:.3f} ({s_['t_mosek2_sparse']:.0f}\\,s, {s_['rss_after_mosek_gib']:.1f}\\,GiB)" if "mosek2_sparse" in s_ else "not run"
    o2 = a.get("order2_sparse")
    cols.append(dict(case=f"{tag}: $d{{=}}{s_['d']}$, ${s_['widths'][0]}{{\\times}}2$, $\\varepsilon{{=}}{s_['eps']}$", n=s_["n"], unst=s_["n_unstable"], exact=f"{s_['exact']:.3f}",
                     o1i=f"{s_['mosek1']:.3f}", o1a=f"{a['order1']:.3f}", o2i=ipm2, o2a=(f"{o2:.3f}" if o2 is not None else "--"),
                     tm=(f"{a['t_order2_sparse']:.0f}\\,s ({a['rss_gib_order2_sparse']:.1f}\\,GiB)" if o2 is not None else "--"),
                     sm=(f"{s_['ours2_sparse']:.1f}" if "ours2_sparse" in s_ else "--")))
with open("tables_n3.tex", "w") as f:
    w(f, "\\begin{tabular}{l " + "r " * len(cols) + "}\\toprule")
    w(f, "case & " + " & ".join(c["case"] for c in cols) + " \\\\\\midrule")
    for key, lab in [("n", "$n$"), ("unst", "unstable neurons"), ("exact", "exact maximum"), ("o1i", "order 1 (IPM)"), ("o1a", "order 1 (ALM)"), ("o2i", "order 2 sparse (IPM)"),
                     ("o2a", "order 2 sparse (ALM)"), ("tm", "ALM time (memory)"), ("sm", "smoothing, order 2")]:
        w(f, lab + " & " + " & ".join(str(c[key]) for c in cols) + " \\\\")
    w(f, "\\bottomrule\\end{tabular}")
print("ok", [r[0] for r in rows])
