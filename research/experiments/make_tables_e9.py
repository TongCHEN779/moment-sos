"""Generate ../tables_e9a.tex and ../tables_e9b.tex (booktabs tabular only, no table environment) from results/e9.json.
Two tabulars separated by a blank line + comment: (A) cut comparison, (B) box QP formulation comparison."""
import json, sys

R = json.load(open("results/e9.json"))

def fmt(v, nd=5):
    return r"$+\infty$" if v is None else (v if isinstance(v, str) else f"{v:.{nd}f}")

def label(w):
    p = w["problem"]
    names = {"boxfree": r"box-free $x_1+x_2$, $x_1x_2=0$", "cubic_ball3": "cubic on ball ($n=3$)", "sphere": "sphere (eq.)",
             "petersen": "MAX-CUT Petersen"}
    if p in names:
        return names[p]
    if p.startswith("boxqp"):
        n = w["n"]; return f"box QP $n={n}$" + (" (+box)" if w["box_constraints"] else " (no box)")
    if p.startswith("chain6"):
        return "chain quartic $n=6$ (sparse)" + (" (+box)" if w["box_constraints"] else " (no box)")
    return p

lines = [r"\begin{tabular}{l c c r r r r r}\toprule",
         r"Problem & $r$ & \#blocks & $\rho_r$ & $\rho_r^{\rm tr}$ & $\rho_r^{\rm d}$ & smoothing $U$ & ALM-diag $U^{\rm d}$ \\\midrule"]
for w in R["A"]:
    lines.append(f"{label(w)} & {w['r']} & {w['smooth']['blocks']} & {fmt(w['rho_none'])} & {fmt(w['rho_trace'])} & {fmt(w['rho_diag'])} "
                 f"& {w['smooth']['U']:.5f} & {w['alm_diag']['U']:.5f} \\\\")
lines.append(r"\bottomrule\end{tabular}")
open("../tables_e9a.tex", "w").write("\n".join(lines) + "\n")
lines = []
lines.append("% Part B: box QP (seed 1), r=2 -- 'with box constraints + uniform shift' (n+1 blocks) vs 'no box constraints + diagonal shift' (1 block)")
lines.append(r"\begin{tabular}{c r r r r r r r r r r}\toprule")
lines.append(r" & \multicolumn{4}{c}{with box, uniform shift} & \multicolumn{4}{c}{no box, diagonal shift} & \multicolumn{2}{c}{bound} \\")
lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}")
lines.append(r"$n$ & blocks & mult. & smoothing $U$ (s) & ALM $U$ (s) & blocks & mult. & shifts & ALM $U^{\rm d}$ (s) & IPM $\rho_2^{\rm tr}$ & IPM $\rho_2^{\rm d}$ \\\midrule")
for w in R["B"]:
    sm, au, ad = w["smooth_box"], w["alm_uniform_box"], w["alm_diag_nobox"]
    lines.append(f"{w['n']} & {au['blocks']} & {au['dim']} & {sm['U']:.5f} ({sm['time']:.1f}) & {au['U']:.5f} ({au['time']:.1f}) "
                 f"& {ad['blocks']} & {ad['dim']} & {ad['n_shift_vars']} & {ad['U']:.5f} ({ad['time']:.1f}) & {fmt(w['ipm_box_trace'])} & {fmt(w['ipm_nobox_diag'])} \\\\")
lines.append(r"\bottomrule\end{tabular}")
open("../tables_e9b.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
