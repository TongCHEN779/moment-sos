"""MAX-CUT on the 5-cycle plus chords: order-1 (Goemans-Williamson/Shor) vs order-2 and order-3 bounds."""
import itertools
import moment_sos as po

n = 5
s = po.variables(n)                                   # spins x_i in {-1,+1}
edges = list(itertools.combinations(range(n), 2))     # complete graph K5, max cut = 6
cut = sum((1 - s[i] * s[j]) / 2 for i, j in edges)
problem = po.Problem(cut, boolean=True)               # boolean=True: x_i^2 = 1 rewriting rule
for order in (1, 2, 3):
    res = po.solve(problem, order=order)
    x, val = res.round()
    print(f"order {order}: upper bound {res.bound:.5f}   best cut found {val:.0f}")
