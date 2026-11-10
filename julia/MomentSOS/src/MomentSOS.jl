"""
    MomentSOS

Julia interface to the Python package `moment-sos`: certified first-order bounds for polynomial optimization
problems through the moment-SOS hierarchy. Problems are written with DynamicPolynomials, serialised to the JSON
problem format of `moment_sos.serialize` and solved by the `moment-sos` command line tool (so Python with the
`moment-sos` package must be installed; set the interpreter with `set_python!` or the environment variable
`MOMENT_SOS_PYTHON`).

```julia
using MomentSOS, DynamicPolynomials
@polyvar x y
res = solve(x^2*y + x*y^2 - x - y, [x, y]; ineq = [1 - x^2 - y^2], box = (-1, 1), order = 2)
res.bound      # certified upper bound
res.x          # best feasible point found
```
"""
module MomentSOS

using JSON
using DynamicPolynomials
using MultivariatePolynomials: terms, coefficient, monomial, degree

export solve, Result, set_python!, problem_dict

const PYTHON = Ref{String}(get(ENV, "MOMENT_SOS_PYTHON", "python3"))

"Set the Python interpreter used to run `moment-sos` (it must have the `moment-sos` package installed)."
set_python!(path::AbstractString) = (PYTHON[] = String(path); nothing)

"Result of [`solve`](@ref): `bound` is certified (upper for `:max`, lower for `:min`); `x`, `value` is the best feasible point found."
struct Result
    bound::Float64
    bound_kind::Symbol
    order::Int
    x::Union{Nothing,Vector{Float64}}
    value::Union{Nothing,Float64}
    time::Float64
    history::Vector{Vector{Float64}}
    rigorous::Union{Nothing,Dict{String,Any}}
    raw::Dict{String,Any}
end

Base.show(io::IO, r::Result) = print(io, "MomentSOS.Result($(r.bound_kind) bound = $(r.bound), order $(r.order), $(round(r.time; digits=2)) s)")

function _terms(p, vars)
    out = Vector{Dict{String,Any}}()
    for t in terms(p)
        m = monomial(t)
        pw = [[i - 1, degree(m, v)] for (i, v) in enumerate(vars) if degree(m, v) > 0]
        push!(out, Dict("c" => Float64(coefficient(t)), "m" => pw))
    end
    return out
end

"""
    problem_dict(f, vars; ineq=[], eq=[], sense=:max, box=nothing, boolean=false, name="")

Build the JSON-ready problem dictionary: maximise (or minimise) `f` over `vars` subject to `g >= 0` for `g in ineq`
and `h == 0` for `h in eq`; `box = (lo, hi)` (scalars or vectors) must contain the feasible set; `boolean` marks spin
variables `x^2 = 1` (`true` for all, or a vector of `Bool`).
"""
function problem_dict(f, vars; ineq = [], eq = [], sense::Symbol = :max, box = nothing, boolean = false, name = "")
    sense in (:max, :min) || throw(ArgumentError("sense must be :max or :min"))
    cons = Dict{String,Any}[]
    for g in ineq
        push!(cons, Dict("kind" => "ineq", "terms" => _terms(g, vars)))
    end
    for h in eq
        push!(cons, Dict("kind" => "eq", "terms" => _terms(h, vars)))
    end
    d = Dict{String,Any}("format" => 1, "n" => length(vars), "sense" => String(sense), "name" => name,
                         "boolean" => boolean isa AbstractVector ? collect(Bool, boolean) : Bool(boolean),
                         "objective" => _terms(f, vars), "constraints" => cons)
    box === nothing || (d["box"] = Dict("lo" => box[1], "hi" => box[2]))
    return d
end

"""
    solve(f, vars; ineq=[], eq=[], sense=:max, box=nothing, boolean=false, order=nothing, method=:smooth,
          cliques="dense", term_sparsity=nothing, time_limit=nothing, rigorous=false)

Solve with the `moment-sos` command line tool and return a [`Result`](@ref). `method` is `:smooth` (entropic smoothing +
L-BFGS, anytime) or `:alm` (dual augmented Lagrangian); `cliques` is `"dense"`, `"auto"` or a vector of index vectors
(0-based, as in Python).
"""
function solve(f, vars; ineq = [], eq = [], sense::Symbol = :max, box = nothing, boolean = false, order = nothing,
               method::Symbol = :smooth, cliques = "dense", term_sparsity = nothing, time_limit = nothing, rigorous::Bool = false, name = "")
    d = problem_dict(f, vars; ineq, eq, sense, box, boolean, name)
    mktempdir() do dir
        pin, pout = joinpath(dir, "problem.json"), joinpath(dir, "result.json")
        write(pin, JSON.json(d))
        args = String["-m", "moment_sos.cli", "solve", pin, "--out", pout, "--method", String(method)]
        order === nothing || append!(args, ["--order", string(order)])
        append!(args, ["--cliques", cliques isa AbstractString ? cliques : JSON.json(cliques)])
        term_sparsity === nothing || append!(args, ["--term-sparsity", string(term_sparsity)])
        time_limit === nothing || append!(args, ["--time-limit", string(time_limit)])
        rigorous && push!(args, "--rigorous")
        err = IOBuffer()
        ok = success(pipeline(Cmd(`$(PYTHON[]) $args`); stderr = err))
        ok || error("moment-sos failed:\n" * String(take!(err)))
        r = JSON.parsefile(pout)
        x = r["x"] === nothing ? nothing : Float64.(r["x"])
        return Result(r["bound"], Symbol(r["bound_kind"]), r["order"], x, r["value"] === nothing ? nothing : Float64(r["value"]),
                      r["time"], [Float64.(h) for h in r["history"]], get(r, "rigorous", nothing), r)
    end
end

end # module
