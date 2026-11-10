# Command line and JSON format

Any language can use `moment-sos` through a JSON problem file and the command line tool.

## JSON problem format

```json
{"n": 2, "sense": "max", "name": "demo",
 "box": {"lo": -1, "hi": 1},
 "boolean": false,
 "objective": [{"c": 1.0, "m": [[0, 2], [1, 1]]}, {"c": 1.0, "m": [[0, 1], [1, 2]]},
               {"c": -1.0, "m": [[0, 1]]}, {"c": -1.0, "m": [[1, 1]]}],
 "constraints": [{"kind": "ineq", "terms": [{"c": 1.0, "m": []}, {"c": -1.0, "m": [[0, 2]]}, {"c": -1.0, "m": [[1, 2]]}]}]}
```

- A term `{"c": c, "m": [[i, p], ...]}` is $c\prod x_i^{p}$ with 0-based variable indices; `"m": []` is a constant.
- `kind` is `"ineq"` (poly $\ge0$) or `"eq"` (poly $=0$).
- `box` holds scalars or length-$n$ lists and must contain the feasible set; omit it for Boolean problems.
- `boolean` is `true`, `false` or a length-$n$ list of booleans.

From Python:

```python
import moment_sos as po

x, y = po.variables(2)
problem = po.Problem(x**2*y + x*y**2 - x - y, [x**2 + y**2 <= 1], box=(-1, 1))
d = po.problem_to_dict(problem)            # or po.save_problem(problem, "problem.json")
problem2 = po.problem_from_dict(d)
print(problem2)
```

## Command line

```bash
moment-sos solve problem.json --order 2 --out result.json
moment-sos solve problem.json --method alm --cliques auto --time-limit 600 --rigorous
moment-sos version
```

| option | meaning |
|---|---|
| `--order r` | relaxation order (default: smallest admissible) |
| `--method smooth \| alm` | smoothing + L-BFGS (default) or dual augmented Lagrangian |
| `--cliques dense \| auto \| JSON` | e.g. `'[[0,1,2],[2,3]]'` |
| `--time-limit s` | seconds |
| `--rigorous` | also compute the verified exact-arithmetic bound |
| `--no-round` | skip primal recovery |
| `--out path` | result JSON (`-` = stdout, the default) |

The result file holds `bound`, `bound_kind`, `order`, `blocks`, `largest_block`, `time`, `evaluations`, `history`
(`[time, evaluations, best bound]` triples in the original units), `x` and `value` (the best feasible point and its
value, `null` if none) and, with `--rigorous`, a `rigorous` object.
