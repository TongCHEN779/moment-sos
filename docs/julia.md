# Julia: `MomentSOS.jl`

`MomentSOS.jl` (directory `julia/MomentSOS` of the repository) lets you write problems with
[DynamicPolynomials.jl](https://github.com/JuliaAlgebra/DynamicPolynomials.jl) and solve them with `moment-sos`. It
serializes the problem to the JSON format of {doc}`guide/cli_json` and runs the `moment-sos` command, so **Python with
the `moment-sos` package must be installed**. There is no Julia implementation of the solver.

## Setup

```julia
using Pkg
Pkg.develop(path = "julia/MomentSOS")        # from the repository root; a registered package is future work
using MomentSOS
MomentSOS.set_python!("/path/to/python")     # interpreter with moment-sos installed (or set MOMENT_SOS_PYTHON)
```

## Example

```julia
using MomentSOS, DynamicPolynomials

@polyvar x y
res = solve(x^2*y + x*y^2 - x - y, [x, y]; ineq = [1 - x^2 - y^2], box = (-1, 1), order = 2)
res.bound        # certified upper bound
res.x, res.value # best feasible point found and its value
```

MAX-CUT over spins:

```julia
@polyvar s[1:4]
cut = sum((1 - s[i]*s[mod1(i + 1, 4)]) / 2 for i in 1:4)
r = solve(cut, s; boolean = true, order = 2, rigorous = true)
r.bound, r.value, r.rigorous
```

## Reference

`solve(f, vars; ineq=[], eq=[], sense=:max, box=nothing, boolean=false, order=nothing, method=:smooth, cliques="dense", time_limit=nothing, rigorous=false)`

returns a `Result` with fields `bound`, `bound_kind` (`:upper` or `:lower`), `order`, `x`, `value`, `time`, `history`,
`rigorous` and `raw` (the full result dictionary). `problem_dict` builds the JSON-ready dictionary without solving.
`cliques` is `"dense"`, `"auto"` or a vector of 0-based index vectors.

:::{note}
Each call starts a new Python process (a few seconds of start-up, mostly importing PyTorch), so `MomentSOS.jl` is
suited to solving a moderate number of larger problems, not thousands of tiny ones.
:::
