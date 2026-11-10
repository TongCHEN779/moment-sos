"""Generate ../tables_e7.tex from results/e7.json (booktabs tabular only).  Run from experiments/."""
import json, math
rows = json.load(open("results/e7.json"))


def sci(x):
    if x == 0:
        return "0"
    ex = math.floor(math.log10(abs(x)))
    m = x / 10 ** ex
    return f"${m:.1f}\\times10^{{{ex}}}$"


def tm(t):
    return f"{t:.3f}" if t < 0.1 else f"{t:.2f}"


with open("../tables_e7.tex", "w") as out:
    w = lambda s: out.write(s + "\n")
    w("\\begin{tabular}{l c r r r r r r r c}\\toprule")
    w("Problem & $r$ & $s$ & $U$ & $U_{\\rm rig}-U$ & $E$ & max eig.\\ slack & $t_{\\rm chol}$ [s] & $t_{\\rm res}$ [s] & certified? \\\\\\midrule")
    for r_ in rows:
        if r_["boolean"]:
            cert = ("yes" if r_["certified"] else "no") + f" ($\\lfloor U_{{\\rm rig}}\\rfloor={r_['floor_Urig']}$)"
        else:
            cert = "--"
        w(f"{r_['problem']} & {r_['r']} & {r_['s']} & {r_['U']:.6f} & {sci(r_['diff'])} & {sci(r_['E'])} & {sci(r_['max_slack'])} & {tm(r_['t_chol'])} & {tm(r_['t_res'])} & {cert} \\\\")
    w("\\bottomrule\\end{tabular}")
print("wrote ../tables_e7.tex")
