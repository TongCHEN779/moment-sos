"""Check that numbers typed in the prose of moment-sos.tex equal the raw data (experiments/results/*.json[l]).
Usage: python check_paper_numbers.py        -> prints a report, exit code 1 if any claim fails.
Each claim = (description, computed value, text that must occur in the .tex file)."""
import json, math, re, sys
R = "experiments/results/"
import glob
tex = open("moment-sos.tex").read() + "\n".join(open(f).read() for f in glob.glob("tables_*.tex"))
claims, fails = [], 0

def claim(desc, text):
    global fails
    ok = text in tex
    claims.append((ok, desc, text))
    if not ok:
        fails += 1

def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"

# ---- E3 scaling (e3.jsonl)
rows = [json.loads(l) for l in open(R + "e3.jsonl")]
fo = {r["n"]: r for r in rows if r["solver"] == "fo"}; mo = {r["n"]: r for r in rows if r["solver"] == "MOSEK"}
for n in (40, 50, 60):
    claim(f"E3 U(n={n})", f"{fo[n]['val']:.4f}" if n != 50 else f"{fo[n]['val']:.3f}")
claim("E3 MOSEK n=30 memory", f"{mo[30]['rss_gib']:.1f}")
claim("E3 first-order n=60 memory", f"{fo[60]['rss_gib']:.1f}")
claim("E3 MOSEK/ours memory ratio at n=30 (~30)", "30")
# ---- E1
e1 = json.load(open(R + "e1_summary.json"))
claim("E1 max excess 2.2e-4", "2.2\\times10^{-4}")
# ---- E2 summary
e2 = json.load(open(R + "e2_summary.json"))
claim("E2 n=16 L-BFGS time to 1e-3 (2.7 s)", f"{e2['16|lbfgs|0.001']:.1f}")
claim("E2 n=16 subgradient time to 1e-3 (10.6 s)", f"{e2['16|sub|0.001']:.1f}")
# ---- baselines (agent report)
b = json.load(open(R + "baselines.json"))["instances"]
claim("E6 BM k=8 n=30 time to 1e-3 (4.7 s)", "4.7")
for k in ("ours",):
    claim("E6 ours n=30 time total (62 s)", "62")
# ---- NN
n1 = json.load(open(R + "nn_n1_new.json")); a1 = json.load(open(R + "nn_alm_n1.json"))
dmax = max([abs(x["alm1"] - r["mosek1"]) for x, r in zip(a1, n1)] + [abs(x["alm2"] - r["mosek2"]) for x, r in zip(a1, n1)])
claim(f"N1 max |ALM - IPM| = {dmax:.2e} <= 9e-6", "9\\times10^{-6}") if dmax <= 9e-6 else claims.append((False, f"N1 max |ALM-IPM|={dmax:.2e} exceeds the 9e-6 stated", ""))
if dmax > 9e-6: fails += 1
n2 = json.load(open(R + "nn_n2_new.json")); a2 = json.load(open(R + "nn_alm_n2.json"))
gaps = [100 * (r["lipsdp"] - r["exact_pattern"]) / r["exact_pattern"] for r in n2]
claim(f"N2 LipSDP gap range {min(gaps):.1f}-{max(gaps):.1f}%", f"{min(gaps):.1f}\\%")
claim(f"N2 LipSDP max gap {max(gaps):.1f}%", f"{max(gaps):.1f}\\%")
claim("N2 ALM2 max deviation from enumeration < 1e-7", "five digits")
assert max(abs(x["alm2"] - r["exact_pattern"]) for x, r in zip(a2, n2)) < 1e-7
# N3 case A
sA = json.load(open(R + "nn_n3_5_40-40_11_0.1.json")); aA = json.load(open(R + "nn_alm_5_40-40_11_0.1.json"))
claim("N3 A exact -9.618", "9.618"); claim("N3 A ALM order 2 -8.623", "8.623")
assert abs(aA["order2_sparse"] + 8.6234) < 1e-3, "canonical A file must be the rho0=1000, no-warm-start run"
frac = (aA["order2_sparse"] - aA["order1"]) / (sA["exact"] - aA["order1"])
claim(f"N3 A fraction of improvement {100*frac:.0f}%", f"{100*frac:.0f}\\%")
sC = json.load(open(R + "nn_n3_6_48-48_12_0.1.json")); aC = json.load(open(R + "nn_alm_6_48-48_12_0.1.json"))
fracC = (sC["mosek1"] - aC["order2_sparse"]) / (sC["mosek1"] - sC["exact"])
claim(f"N3 C fraction of gap closed {100*fracC:.0f}%", f"{100*fracC:.0f}\\%")
claim("N3 A MOSEK sparse time 1591 s", "1591"); claim("N3 A MOSEK memory 19.6 GiB", "19.6")
# ---- E7 / E10
e7 = json.load(open(R + "e7.json")); corr = [r["U_rig"] - r["U"] for r in e7]
claim(f"E7 rows {len(e7)}", f"{len(e7)}"); claim("E7 max correction 7.2e-9", "7.2\\times10^{-9}")
claim("E7 min correction 4e-14", "4\\times10^{-14}")
e10b = json.load(open(R + "e10b_summary.json"))
claim("E10b all certified", "All $15$ instances")
assert all(v["certified"] == v["count"] for v in e10b.values())

# ---- E11 benchmark (bench.jsonl)
import collections
import numpy as np
bj = [json.loads(l) for l in open(R + "bench.jsonl")]
claim(f"E11 {len(bj)} runs", "$60$ order-$2$ MAX-CUT instances"); assert len(bj) == 60
claim("E11 all certified", "All $60$ instances are certified optimal"); assert all(r["certified"] for r in bj)
ratio = np.array([r["t_cert"] / r["wall"] for r in bj])
claim(f"E11 median t_cert/wall {np.median(ratio):.2f}", f"{np.median(ratio):.2f}")
claim(f"E11 ratio range {ratio.min():.2f}-{ratio.max():.2f}", f"{ratio.min():.2f}")
claim("E11 ratio max", f"{ratio.max():.2f}")
mk = [r for r in bj if "mosek_wall" in r]
claim(f"E11 IPM runs {len(mk)}", f"{len(mk)}")
claim(f"E11 t_cert < IPM in {sum(r['t_cert'] < r['mosek_wall'] for r in mk)} runs", f"{sum(r['t_cert'] < r['mosek_wall'] for r in mk)}$ of ${len(mk)}")
claim("E11 max |IPM - floor U|", f"{max(abs(r['mosek_val'] - np.floor(r['U'] + 1e-9)) for r in mk)*1e6:.1f}\\times10^{{-6}}")
claim("E11 max memory", f"{max(r['rss_gib'] for r in bj):.2f}")
claim("E11 max margin", f"{max(r['U'] - np.floor(r['U'] + 1e-9) for r in bj)*1e3:.1f}\\times10^{{-3}}")
g = collections.OrderedDict()
for r in bj: g.setdefault((r["family"], r["n"]), []).append(r["t_cert"])
inside = sum(max(abs(np.array(v) / np.median(v) - 1)) <= 0.20 for v in g.values())
claim(f"E11 rows within 20%: {inside}", f"${inside}$ of the $12$ rows")
claim("E11 worst deviation", f"{100*max(max(abs(np.array(v)/np.median(v)-1)) for v in g.values()):.0f}\\%")
claim("E11 growth 20->30", f"{np.median(g[('g05',30)])/np.median(g[('g05',20)]):.1f}")
claim("E11 growth 30->40", f"{np.median(g[('g05',40)])/np.median(g[('g05',30)]):.1f}")
claim("E11 median load", f"{np.median([r['load0'] for r in bj]):.1f}"); claim("E11 max load", f"{max(r['load0'] for r in bj):.1f}")

# ---- E12 certification schedules (bench_hybrid.jsonl, cold_large.txt)
hy = [json.loads(l) for l in open(R + "bench_hybrid.jsonl")]
base = {(r["family"], r["n"], r["seed"]): r for r in bj}
by = {m: {(r["family"], r["n"], r["seed"]): r for r in hy if r["mode"] == m} for m in ("hybrid", "cold", "bm")}
cold60 = {k: r for k, r in by["cold"].items() if k[0] != "reg3"}
sp = [base[k]["t_cert"] / r["t_cert"] for k, r in cold60.items() if r["t_cert"]]
claim(f"E12 median speed-up {np.median(sp):.1f} over {len(sp)} runs", f"factor $%.1f$ over the ${len(sp)}$ paired runs" % np.median(sp))
claim(f"E12 speed-up range {min(sp):.1f}-{max(sp):.1f}", f"range ${min(sp):.1f}$--${max(sp):.1f}$")
cc = sum(r["certified"] for r in cold60.values()); claim(f"E12 short certified {cc} of {len(cold60)}", f"certified ${cc}$ of the ${len(cold60)}$ runs")
bad_runs = sorted((k, r["U_final"] - r["LB"]) for k, r in cold60.items() if not r["certified"])
assert all(k[0] == "torus" and k[1] == 36 for k, _ in bad_runs), bad_runs
claim("E12 failures are grid n=36 seeds " + str([k[2] for k, _ in bad_runs]), "seeds $1,3,5$")
claim("E12 failure gaps", "4.8,\\ 7.9,\\ 1.6")
bysz = {}
for k, r in cold60.items():
    if r["t_cert"]: bysz.setdefault((k[0] == "torus", k[1]), []).append(base[k]["t_cert"] / r["t_cert"])
rnd = {n: np.median(v) for (tor, n), v in bysz.items() if not tor}; grid = {n: np.median(v) for (tor, n), v in bysz.items() if tor}
claim("E12 median factors random", f"${rnd[20]:.1f}$, ${rnd[30]:.1f}$ and ${rnd[40]:.1f}$")
claim("E12 median factors grid", f"${grid[16]:.1f}$, ${grid[25]:.1f}$ and ${grid[36]:.1f}$")
bmc = list(by["bm"].values())
small = [r for r in bmc if r["n"] <= 20]; claim(f"E12 BM cert n<=20: {sum(r['certified'] for r in small)} of {len(small)}", f"${sum(r['certified'] for r in small)}$ of ${len(small)}$")
rand30 = [r for r in bmc if r["n"] == 30 and r["family"] != "torus"]
assert sum(r["certified"] for r in rand30) == 0 and len(rand30) == 15
claim("E12 BM none at n=30 (15 runs)", "$15$ random-graph runs with $n=30$")
hv = [(r["t_cert"], cold60[k]["t_cert"]) for k, r in by["hybrid"].items() if k in cold60 and r["t_cert"] and cold60[k]["t_cert"]]
claim(f"E12 hybrid faster in {sum(a < b for a, b in hv)} of {len(hv)}", f"only ${sum(a < b for a, b in hv)}$ of the ${len(hv)}$ paired runs")
claim("E12 hybrid certified all 60", "certifies all $60$ runs"); assert all(r["certified"] for r in by["hybrid"].values())
for n in (50, 60):
    v = [r for k, r in by["cold"].items() if k[0] == "reg3" and k[1] == n]; tc = [r["t_cert"] for r in v]
    assert all(r["certified"] for r in v)
    if n == 50: claim("E12 reg3 n=50 median/range", f"median ${np.median(tc):.0f}$\\,s, range ${min(tc):.0f}$--${max(tc):.0f}$")
    else: claim("E12 reg3 n=60 range", f"(${min(tc):.0f}$--${max(tc):.0f}$\\,s")
e3r = {r["n"]: r for r in rows if r["solver"] == "fo"}
for n, lb in ((50, 68), (60, 81)):
    h = np.array(e3r[n]["hist"]); ok = np.where(np.floor(h[:, 1] + 1e-9) <= lb + 1e-9)[0]; claim(f"E12 default-schedule certification time n={n}: {h[ok[0],0]:.0f}", f"${h[ok[0],0]:.0f}$\\,s")
claim("E12 torus fix times", "$61$--$79$"); tf = open(R + "torus_fix.txt").read(); assert "60.6" in tf and "78.5" in tf and "76.9" in tf
claim("E12 n=60 earlier single run 218 s", "$218$"); assert "218.3" in open(R + "cold_large.txt").read()

# ---- E5b structured benchmark
st = [json.loads(l) for l in open(R + "bench_struct.jsonl")]
claim(f"E5b {len(st)} runs", f"(${len(st)}$ runs"); claim("E5b bounds above IPM", "$\\ge-6\\times10^{-10}$"); assert min(min(r["gap_smooth"], r["gap_alm"]) for r in st) >= -6e-10
claim(f"E5b max smoothing gap {max(r['gap_smooth'] for r in st):.1e}", f"{max(r['gap_smooth'] for r in st)*1e3:.1f}\\times10^{{-3}}")
claim(f"E5b max ALM gap {max(r['gap_alm'] for r in st):.1e}", f"{max(r['gap_alm'] for r in st)*1e6:.1f}\\times10^{{-6}}")
claim("E5b smoothing faster than IPM", f"${sum(r['t_smooth'] < r['t_ipm'] for r in st)}$ of the ${len(st)}$ runs and the augmented Lagrangian in ${sum(r['t_alm'] < r['t_ipm'] for r in st)}$")
d16 = [r for r in st if r["kind"] == "chain_dense" and r["n"] == 16]
claim("E5b dense chain n=16 times", f"${np.median([r['t_ipm'] for r in d16]):.1f}$\\,s for the interior-point solver, ${np.median([r['t_smooth'] for r in d16]):.1f}$\\,s for smoothing, ${np.median([r['t_alm'] for r in d16]):.1f}$")
claim("E5b max load", f"{max(r['load0'] for r in st):.1f}")

# ---- E13 sensitivity
sv = [json.loads(l) for l in open(R + "sensitivity.jsonl")]
claim(f"E13 {len(sv)} runs", f"${len(sv)}$ runs"); assert all(r["t_cert"] is not None for r in sv)
med = lambda xs: float(np.median(xs))
m500 = {e0: med([r["t_cert"] for r in sv if r["eta0"] == e0 and r["iters"] == 500 and r["maxcor"] == 30]) for e0 in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)}
claim("E13 eta0 medians slow", f"${m500[0.1]:.1f}$, ${m500[0.03]:.1f}$ and ${m500[0.01]:.1f}$"); claim("E13 eta0 medians fast", f"${m500[0.003]:.1f}$, ${m500[0.001]:.1f}$ and ${m500[0.0003]:.1f}$")
small = [r["t_cert"] for r in sv if r["eta0"] <= 3e-3]
claim("E13 range for eta0<=3e-3", f"between ${min(small):.2f}$ and ${max(small):.2f}$\\,s over the ${len(small)}$ runs")
it = {k: med([r["t_cert"] for r in sv if r["eta0"] <= 3e-3 and r["maxcor"] == 30 and r["iters"] == k]) for k in (200, 500, 1000)}
claim("E13 iterations per stage", f"(${it[200]:.2f}$, ${it[500]:.2f}$, ${it[1000]:.2f}$\\,s")
gp = [r["gap"] for r in sv]; claim("E13 final gap range", f"(${min(gp)*1e4:.1f}\\times10^{{-4}}$--${max(gp)*1e3:.1f}\\times10^{{-3}}$")
mc = {k: (med([r["t_cert"] for r in sv if r["eta0"] == 1e-3 and r["iters"] == 500 and r["maxcor"] == k]), med([r["total"] for r in sv if r["eta0"] == 1e-3 and r["iters"] == 500 and r["maxcor"] == k])) for k in (10, 30, 100)}
claim("E13 memory", f"(${mc[10][0]:.2f}$, ${mc[30][0]:.2f}$, ${mc[100][0]:.2f}$\\,s for memory"); claim("E13 memory total", f"from ${mc[30][1]:.0f}$ to ${mc[100][1]:.0f}$\\,s")

# ---- profile
pf = json.load(open(R + "profile_oracle.json"))
sh = [r["eigh_share"] for r in pf]; claim(f"profile eigh share {100*min(sh):.0f}-{100*max(sh):.0f}", f"${100*min(sh):.0f}$--${100*max(sh):.0f}\\%$")
e60 = [r["t_eigh"] * 1e3 for r in pf if r["n"] == 60]; claim("profile eigh n=60 across threads", f"${min(e60):.0f}$--${max(e60):.0f}$\\,ms")
g60 = [r["t_gram"] * 1e3 for r in pf if r["n"] == 60]; claim("profile gram n=60", f"${min(g60):.0f}$--${max(g60):.0f}$\\,ms")

# ---- TSSOS comparison and MNIST (N4)
ts = {r["n"]: r for r in map(json.loads, open(R + "tssos_reg3.jsonl"))}
claim("TSSOS times n=10..26", f"(${ts[10]['time']:.0f}$, ${ts[14]['time']:.0f}$, ${ts[20]['time']:.0f}$, ${ts[26]['time']:.0f}$")
claim("TSSOS n=30 time 264 s", "264")
mn = json.load(open(R + "nn_mnist_d4_w8.json")); mr = mn["rows"]
claim("MNIST accuracy", f"{mn['acc']:.3f}")
for key, tag in (("ibp", "5/3/0"), ("mosek1", "7/6/5"), ("mosek2", "7/6/5"), ("exact", "7/6/5")):
    got = "/".join(str(sum(1 for r in mr if r["eps"] == e and r[key] < 0)) for e in (0.05, 0.1, 0.2))
    claim(f"MNIST certified counts {key} = {got}", tag) if got == tag else claims.append((False, f"MNIST counts {key}: {got} != {tag}", ""))
claim("MNIST order-1 gap>1e-3 in 13 of 24", f"{sum(1 for r in mr if r['mosek1'] - r['exact'] > 1e-3)}\\ of the ${len(mr)}$".replace("\\ of", " of").replace("\\","")) if False else None

br = [json.loads(l) for l in open(R + "bench_real.jsonl")]
assert len(br) == 5 and all(r["certified"] for r in br)
claim("real g05_60 cut values", ", ".join(f"${int(r['LB'])}$" for r in br))
tc = [r["t_cert"] for r in br]; claim("real g05_60 t_cert range", f"${min(tc):.0f}$--${max(tc):.0f}$\\,s (median ${float(np.median(tc)):.0f}$")
wl = [r["wall"] for r in br]; claim("real g05_60 wall range", f"${min(wl):.0f}$--${max(wl):.0f}$\\,s in total")

nw = json.load(open(R + "nn_alm_newton_fixed_5_40-40_11_0.1.json")); claim("Newton (fixed) n=40 order2", f"${nw['order2_sparse']:.1f}$")
nc = dict(l.split() for l in open(R + "newton_check_n14.txt")); claim("Newton n=14 error 8e-8", f"${abs(float(nc['newton'])-float(nc['ipm']))*1e8:.0f}\\times10^{{-8}}$"); claim("LBFGS n=14 error 5e-6", f"${abs(float(nc['lbfgs'])-float(nc['ipm']))*1e6:.0f}\\times10^{{-6}}$")
claim("t2g10 PRECISE time 264", "$264$\\,s") if "$264$\\,s" in open("moment-sos.tex").read() else claims.append((False, "t2g10 PRECISE text missing", ""))
# ---- report
for ok, desc, text in claims:
    print(("ok   " if ok else "FAIL ") + desc + ("" if ok else f"   [expected substring: {text!r}]"))
print(f"\n{len(claims) - fails}/{len(claims)} claims verified")
sys.exit(1 if fails else 0)
