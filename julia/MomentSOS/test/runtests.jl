using Test, MomentSOS, DynamicPolynomials

@testset "MomentSOS" begin
    @polyvar x y
    d = problem_dict(x^2 * y - x, [x, y]; ineq = [1 - x^2 - y^2], box = (-1, 1))
    @test d["n"] == 2 && d["sense"] == "max" && length(d["constraints"]) == 1

    # constrained polynomial problem: same instance as the Python quickstart (order 2 and 3 give 1.0000)
    res = solve(x^2 * y + x * y^2 - x - y, [x, y]; ineq = [1 - x^2 - y^2], box = (-1, 1), order = 2)
    @test res.bound_kind == :upper
    @test 0.99 < res.bound < 1.01
    @test res.x !== nothing && res.value <= res.bound + 1e-6

    # 4-cycle MAX-CUT over spins: optimum 4, certified bound must be >= 4 and tight
    @polyvar s[1:4]
    cut = sum((1 - s[i] * s[mod1(i + 1, 4)]) / 2 for i in 1:4)
    r2 = solve(cut, s; boolean = true, order = 2, rigorous = true)
    @test 3.99 < r2.bound < 4.02
    @test r2.value == 4.0
    @test r2.rigorous !== nothing

    @test_throws ArgumentError problem_dict(x, [x]; sense = :foo)
end
