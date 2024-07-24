
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

    @test all(m.consumption .== (m.agrid .+ m.savings))
end



# @testitem "Euler Error" begin

#     # defining the parameters of the model
#     rho = 0.966
#     s = 0.5
#     sig = s * sqrt(1 - rho^2)
#     params = SSJ.Params(0.96, 1.0, sig, rho, 0.025, 0.11, 0.0001, [0.0, 200.0], 200, 7, 300)

#     # set an r
#     r = 0.04
    
#     # Setting up the model
#     BaseModel = SSJ.setup_Aiyagari(params)

#     agg_labor = SSJ.aggregate_labor(BaseModel.Π, BaseModel.shockgrid)
#     aggregates, prices = SSJ.get_AggsPrices(r, agg_labor, BaseModel)
#     policies = SSJ.EGM(BaseModel, prices)

#     # now test that `policies` are consistent

# 	μ = zeros(params.n_a)  # vector of marginal utilities
# 	iuprime = zeros(params.n_a,params.n_e)  # matrix of inverse marginal utilities
	
#     # basically does `consumptiongrid`
# 	for ie in 1:params.n_e
# 		fill!(μ , 0.0)
# 		for je in 1:params.n_e
# 			pr = BaseModel.Π[ie,je]  # transprob

# 			# get next period consumption if state is je
#             # cprime = (1+r) policygrid + w * shock[je] - policy[je]
#             # (params.n_a by 1)
#             cprimevec = ((1 + prices.r) * BaseModel.policygrid) .+ (prices.w .* BaseModel.shockgrid[je]) .- policies.saving[:,je]

# 			# Expected marginal utility at each state of tomorrow's income shock
#     		global μ += pr * (cprimevec .^ ((-1) * params.γ))
# 		end
# 	   # RHS of euler equation
#     	rhs = params.β * (1 + prices.r) * μ
# 		# today's consumption: inverse marginal utility over RHS
#     	iuprime[:,ie] = rhs .^ ((-1)/params.γ)
# 	end

#     savings = SSJ.policyupdate(prices,BaseModel.policymat,BaseModel.shockmat,iuprime)
#     cons = ((1 + prices.r) * BaseModel.policymat) + (prices.w * BaseModel.shockmat) - savings

#     # checks reverse engineering of `consumptiongrid`
# 	@test maximum(abs,(cons ./ policies.consumption) .- 1) < 0.00001
# end