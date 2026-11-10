"""Triangle walk: topk vs nms (and random) on G(20, 0.3), seed 1, 3 rounds, K = n = 20 triples per round. python run_walk.py out.json"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import numpy as np, networkx as nx
import moment_sos as po

out = []
for seed in (1,):
    G = nx.gnp_random_graph(20, 0.3, seed=seed)
    n = 20
    s = po.variables(n)
    P = po.Problem(sum((1 - s[i] * s[j]) / 2 for i, j in G.edges()), boolean=True)
    r2 = po.solve_alm(P, order=2)
    best = po.solve(P, order=1).round()[1]
    rec = dict(seed=seed, edges=G.number_of_edges(), order2=r2.bound, best_cut=best, runs={})
    for mode, rule in (("cuts", "nms"), ("cuts", "topk"), ("cuts", "random"), ("basis", "nms"), ("basis", "topk")):
        if True:
            t = time.time()
            w = po.walk(P, rounds=3, rule=rule, mode=mode)
            rec["runs"][f"{mode}/{rule}"] = dict(bounds=w.bounds, sizes=w.sizes, times=w.times, stopped=w.stopped, wall=time.time() - t)
            print(seed, mode, rule, [round(b, 4) for b in w.bounds], w.sizes, round(time.time() - t, 1), flush=True)
    out.append(rec)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
