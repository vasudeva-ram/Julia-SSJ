
@testitem "Euler Error" begin

    # defining the parameters of the model
    rho = 0.966
    s = 0.5
    sig = s * sqrt(1 - rho^2)
    params = SSJ.Params(0.96, 1.0, sig, rho, 0.025, 0.11, 0.0001, [0.0, 200.0], 200, 7, 300)

    # set an r
    r = 0.05
    
    # Setting up the model
    BaseModel = SSJ.setup_Aiyagari(params)

    agg_labor = SSJ.aggregate_labor(BaseModel.Π, BaseModel.shockgrid)
    aggregates, prices = SSJ.get_AggsPrices(r, agg_labor, BaseModel)
    policies = SSJ.EGM(BaseModel, prices)

    # now test that `policies` are consistent

	μ = zeros(params.n_a)  # vector of marginal utilities
	iuprime = zeros(params.n_a,params.n_e)  # matrix of inverse marginal utilities
	
    # basically does `consumptiongrid`
	for ie in 1:params.n_e
		fill!(μ , 0.0)
		for je in 1:params.n_e
			pr = BaseModel.Π[ie,je]  # transprob

			# get next period consumption if state is je
            # cprime = (1+r) policygrid + w * shock[je] - policy[je]
            # (params.n_a by 1)
            cprimevec = ((1 + prices.r) * BaseModel.policygrid) .+ (prices.w .* BaseModel.shockgrid[je]) .- policies.saving[:,je]

			# Expected marginal utility at each state of tomorrow's income shock
    		global μ += pr * (cprimevec .^ ((-1) * params.γ))
		end
	   # RHS of euler equation
    	rhs = params.β * (1 + prices.r) * μ
		# today's consumption: inverse marginal utility over RHS
    	iuprime[:,ie] = rhs .^ ((-1)/params.γ)
	end
    # println([iuprime  (policies.consumption)])

    implied = ((1 + prices.r) * BaseModel.policymat) .+ (prices.w .* BaseModel.shockmat) .- policies.saving

    # checks last penultimate line in `EGM`:
    @test all(implied .== policies.consumption)

    # checks reverse engineering of `consumptiongrid`
	@test maximum(abs,(iuprime ./ policies.consumption) .- 1) < 0.0001
end