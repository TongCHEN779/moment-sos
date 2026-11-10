"""Certified bounds for a small non-convex polynomial optimization problem."""
import moment_sos as po

x, y = po.variables(2)
f = x**2 * y + x * y**2 - x - y                      # objective (to be maximised)
problem = po.Problem(f, [x**2 + y**2 <= 1], sense="max", box=(-1, 1))

for order in (2, 3):                                   # need 2*order >= deg(f) = 3
    res = po.solve(problem, order=order)
    pt, val = res.round()                              # best feasible point found = lower bound
    print(f"order {order}: certified upper bound {res.bound:.6f}   feasible value {val:.6f} at {pt}")
