
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
    @test size(m.s) == (p.n_a , p.n_e)
    @test size(m.s1) == (p.n_a , p.n_e)
    @test size(m.cplus) == (p.n_a, p.n_e)
    @test size(m.Π) == (p.n_e, p.n_e)
    @test size(m.shockgrid) == (p.n_e, )
    @test size(m.shockmat) == (p.n_a,p.n_e)


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
    SSJ.EGMstep!(m.s, m.s1, m, prices)
    @test true
end

@testitem "Aiyagari EGM runs" begin
    p = Params()
    m = SSJ.Aiyagari(p)
    aggs,prices = SSJ.firm(0.05,m)
    policies = SSJ.EGM(m,prices)

    @test all(policies.consumption .> 0)
    @test all(policies.saving .>= 0)
    
end



@testitem "Euler Equation Error" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    aggs,prices = SSJ.firm(0.05,a)
    policies = SSJ.EGM(a,prices)

    # first sanity check on solution
    @test all( policies.saving .≈ ((1 + prices.r) * a.amat) + (prices.w * a.shockmat) - policies.consumption)

    μ = zeros(p.n_a)  # vector of marginal utilities
	iuprime = zeros(p.n_a,p.n_e)  # matrix of inverse marginal utilities

    # basically does `consumptiongrid`
	for ie in 1:p.n_e
		fill!(μ , 0.0)
		for je in 1:p.n_e
			pr = a.Π[ie,je]  # transprob

			# get next period consumption if state is je
            # cprime = (1+r) policygrid + w * shock[je] - policy[je]
            # (p.n_a by 1)
            cprimevec = ((1 + prices.r) * a.agrid) .+ (prices.w .* a.shockgrid[je]) .- policies.saving[:,je]

			# Expected marginal utility at each state of tomorrow's income shock
    		global μ += pr * (cprimevec .^ ((-1) * p.γ))
		end
	   # RHS of euler equation
    	rhs = p.β * (1 + prices.r) * μ
		# today's consumption: inverse marginal utility over RHS
    	iuprime[:,ie] = rhs .^ ((-1)/p.γ)
	end

    impliedgrid = (1 / (1 + prices.r)) * (iuprime - (prices.w .* a.shockmat) + a.amat)
    # newguess = Matrix{Float64}(undef, size(impliedgrid))

    saving = zeros(p.n_a, p.n_e)

    for i in axes(impliedgrid, 2)
        linpolate = SSJ.linear_interpolation(impliedgrid[:,i], a.amat[:,i], extrapolation_bc = SSJ.Flat())
        saving[:,i] = linpolate(a.amat[:,i])
    end

    cons = ((1 + prices.r) * a.amat) + (prices.w * a.shockmat) - saving

    # checks reverse engineering of `consumptiongrid`
	@test maximum(abs,(cons ./ policies.consumption) .- 1) < 0.0000001
	# @test maximum(abs,(saving ./ policies.saving) .- 1) < 0.0000001
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


@testitem "SingleRun r 1" begin
    r = 0.01
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
    o = SSJ.solve_SteadyState(a,guess = (0.02, 0.05))
    @test isa(o, SSJ.SteadyState)
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
end

@testitem "SteadyState β" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    o = SSJ.solve_SteadyState_r(0.01,1.0,a)
    @test isa(o, SSJ.SteadyState)
    @test sum(o.D) ≈ 1
    @test all(sum(o.Λ, dims = 1) .≈ 1)
end

@testitem "solve SS r" begin
    p = Params()
    a = SSJ.Aiyagari(p)
    s = SSJ.solve_SteadyState(a,guess = (0.02, 0.05))
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

    @test SSJ.norm(F, Inf) < 1e-7

end