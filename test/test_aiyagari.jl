
@testitem "params" begin
    p = Params()
    @test isa(p, Params)
    @test p.γ == 1
    p1 = Params(ρ = 0.5)
    @test p1.ρ == 0.5
end


@testitem "Aiyagari constructor" begin
    p = Params()
    m = SSJ.Aiyagari(p)
    @test size(m.m) == (p.n_a + 1, p.n_e)
    @test size(m.m0) == (p.n_a, p.n_e)
    @test size(m.mplus) == (p.n_a, p.n_e)
    @test size(m.c_m) == (p.n_a + 1, p.n_e)
    @test size(m.Π) == (p.n_e, p.n_e)
    @test size(m.shockgrid) == (p.n_e, )


end

@testitem "Aiyagari helpers" begin
    p = Params()
    m = SSJ.Aiyagari(p)

    @test SSJ.aggregate_labor(m) == 1.0

    f1 = SSJ.firm(0.05,m)
    @test f1[2].r == 0.05
    @test f1[1].Y == 0

    f2 = SSJ.firm(2.1,3.1,m)
    @test f2[1].Y ≠ 0
    @test f2[2].w == (1 - p.α) * p.Z * (2.1 / 3.1) ^ p.α
end


@testitem "Aiyagari EGMstep" begin
    p = Params()
    m = SSJ.Aiyagari(p)
    aggs,prices = SSJ.firm(0.05,m)
    SSJ.EGMstep!(m,prices)
    @test all(m.m0 .== (m.agrid .+ m.c0_m))
end

@testitem "Aiyagari EGM runs" begin
    p = Params()
    m = SSJ.Aiyagari(p)
    aggs,prices = SSJ.firm(0.05,m)
    SSJ.EGM!(m,prices)

    @test all(m.consumption .> 0)
    @test all(m.c0_m .> 0)
    @test all(m.c_m[1,:] .== 0)
end



@testitem "Euler Equation Error" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    aggs,prices = SSJ.firm(0.05,a)
    SSJ.EGM!(a,prices)

    # first sanity check on solution
    @test all( (a.m0 - a.c0_m) .≈ a.agrid )

    # now test that `policies` are consistent

	μ = zeros(p.n_a)  # vector of marginal utilities
	cimplied = zeros(p.n_a,p.n_e)  # matrix of inverse marginal utilities

    # at each grid point compute 2 consumption values: one from your solution (c) and one implied. 
    # cimp = g ^ (-1/p.γ)
    # g is accurately computed rhs of euler equation, using the c function
	mnext = zeros(p.n_a)

	for ie in 1:p.n_e
		fill!(μ , 0.0)
		for je in 1:p.n_e
			pr = a.Π[ie,je]  # transprob

			# get next period consumption if state is je
            # cprime = (1+r) policygrid + w * shock[je] - policy[je]
            # (p.n_a by 1)
            mnext = ((1 + prices.r) * a.agrid) .+ (prices.w .* a.shockgrid[je])
            cint = SSJ.linear_interpolation(a.m[:,je], a.c_m[:,je], extrapolation_bc = SSJ.Line())
            cnext = cint(mnext)

			# Expected marginal utility at each state of tomorrow's income shock
    		global μ += pr * (cnext .^ ((-1) * p.γ))
		end
	   # RHS of euler equation
    	rhs = p.β * (1 + prices.r) * μ
		# today's consumption: inverse marginal utility over RHS
    	cimplied[:,ie] = rhs .^ ((-1)/p.γ)
	end
    endogrid = a.agrid .+ cimplied
    endogrid = vcat(reshape(zeros(p.n_e),1,p.n_e) , endogrid)
    endoc = vcat(reshape(zeros(p.n_e),1,p.n_e) , cimplied)

    # consumption on asset grid
    newcons = zeros(p.n_a, p.n_e)

    for ie in 1:p.n_e
        cint = SSJ.linear_interpolation(endogrid[:,ie], endoc[:,ie], extrapolation_bc = SSJ.Line())
        newcons[:,ie] = cint(a.agrid .+ prices.w * a.shockgrid[ie])  # a + we = m
    end
    
	@test maximum(abs,((cimplied - a.c0_m) ./ a.c0_m)) < 1e-8
	@test maximum(abs,((newcons - a.consumption) ./ a.consumption)) < 1e-8
end


@testitem "SingleRun r" begin
    r = 0.02
    p = Params()
    a = SSJ.Aiyagari(p)
    o = SSJ.SingleRun(r,a)
    @test isa(o, SSJ.SteadyState)
    @test o.prices.r == r
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
end

@testitem "SingleRun β,K,Z" begin
    Z = 0.86
    β = 0.95
    K = 3.1
    p = Params()
    a = SSJ.Aiyagari(p)

    o = SSJ.SingleRun(β,K,Z,a)
    @test isa(o, SSJ.SteadyState)
    @test o.aggregates.K == K
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
    @test a.params.β == β
    @test a.params.Z == Z
end

@testitem "SteadyState r" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    o = SSJ.solve_SteadyState(a)
    @test isa(o, SSJ.SteadyState)
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
end

@testitem "SteadyState β" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    o = SSJ.solve_SteadyState(a)
    @test isa(o, SSJ.SteadyState)
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
end

@testitem "solve SS r" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    s = SSJ.solve_SteadyState(a)
    @test isa(s, SSJ.SteadyState)
    @test sum(s.D) ≈ 1
    @test all(sum(s.Λ, dims = 1) .≈ 1)
    @test SSJ.get_residual(s.prices.r,a) ≈ 0.0 atol = 1e-6 
end


@testitem "solve SS β" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    r = 0.01
    y = 1.0
    s = SSJ.solve_SteadyState_r(r,y,a)
    @test isa(s, SSJ.SteadyState)
    @test sum(s.D) ≈ 1
    @test all(sum(s.Λ, dims = 1) .≈ 1)
    @test s.prices.r ≈ r
    @test s.aggregates.Y ≈ y

    F = zeros(3)
    SSJ.residual!(F,a.params.β,s.aggregates.K,a.params.Z,r,y,a)

    @test SSJ.norm(F, Inf) < 1e-8

end