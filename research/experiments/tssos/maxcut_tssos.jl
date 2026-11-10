# Usage: julia maxcut_tssos.jl graph.txt order [CS: false|MF]   (graph: "n m" then "i j w", 1-based; MAX-CUT, +-1 spins)
using Mosek, MosekTools
using TSSOS, DynamicPolynomials, JSON
lines = readlines(ARGS[1]); n, m = parse.(Int, split(lines[1])); d = parse(Int, ARGS[2])
TS = length(ARGS) >= 3 ? ARGS[3] : "false"
@polyvar x[1:n]
f = 0.0 * x[1]^2
for l in lines[2:m+1]
    i, j, w = split(l); i = parse(Int, i); j = parse(Int, j); w = parse(Float64, w)
    global f += w * (1 - x[i] * x[j]) / 2          # cut value
end
pop = [-f]                                   # spins handled by nb = n (x_i^2 = 1)
t0 = time()
CSopt = (TS == "false") ? false : "MF"
opt, sol, data = cs_tssos(pop, x, d, nb=n, CS=CSopt, TS=false, solution=false, QUIET=true)
println(JSON.json(Dict("bound" => -opt, "time" => time() - t0, "n" => n, "order" => d, "TS" => TS)))
