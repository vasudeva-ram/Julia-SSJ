# Implementing the Krussell-Smith model as in Auclert et. al. (2021)
include("SSJ.jl")



function get_prices(aggs::Aggregates, 
    model::AiyagariModel)

    # Unpack parameters
    @unpack agg_ks, agg_labor = aggs
    @unpack α, δ = model.params 
    r = α * (agg_ks / agg_labor)^(α-1) - δ
    w = (1-α) * (agg_ks / agg_labor)^α
    # Note: we are assuming the value of the shock Z in steady state is 1.

    return Prices(r, w)
end


function steadystateKS_KBrent(BaseModel::AiyagariModel;
    ϵ::Float64 = 1e-6,
    itermax::Int64 = 1000,
    printsol::Bool = false)

    # Unpack parameters
    @unpack params, policygrid, initialguess, shockgrid, Π = BaseModel
    prices = Prices(0.0, 0.0)
    policies = (saving = zeros(size(initialguess)), consumption = zeros(size(initialguess)))
    D = zeros(size(initialguess))
    agg_labor = aggregate_labor(Π, shockgrid)
    aggregates = Aggregates(0.0, 0.0)
    Λ = spzeros(prod(size(initialguess)))

    # Start a solution loop
    k_max = 100.0
    k_min = 0.1

    function get_optimum(ks)
        # Get the implied aggregate labor supply and aggregate capital demand
        aggregates = Aggregates(ks, agg_labor)
        # Get the implied prices
        prices = get_prices(aggregates, BaseModel)
        # Solve the household's policies
        policies = EGM(BaseModel, prices)
        # Get the implied stationary distribution of wealth
        Λ = distribution_transition(policies.saving, BaseModel.policygrid, BaseModel.Π)
        D = invariant_dist(Λ')
        # Get the implied aggregate capital supply
        spolicy = vcat(policies.saving...) 
        agg_kdmat = D' * spolicy
        agg_kd = agg_kdmat[1,1]
        return norm(agg_kd - ks)
    end

    sol = Optim.optimize(get_optimum, k_min, k_max, abs_tol=ϵ)

    return sol    
end


function steadystateKS_K(BaseModel::AiyagariModel;
    ϵ::Float64 = 1e-6,
    itermax::Int64 = 1000,
    printsol::Bool = false)

    # Unpack parameters
    @unpack params, policygrid, initialguess, shockgrid, Π = BaseModel
    prices = Prices(0.0, 0.0)
    policies = (saving = zeros(size(initialguess)), consumption = zeros(size(initialguess)))
    D = zeros(size(initialguess))
    agg_labor = aggregate_labor(Π, shockgrid)
    aggregates = Aggregates(0.0, 0.0)
    Λ = spzeros(prod(size(initialguess)))

    # Start a solution loop
    crit = 100.0
    iter = 0
    k_max = 100.0
    k_min = 0.1

    while (crit > ϵ) && (iter < itermax)
        # Get the first guess of interest rate from range
        ks = (k_max + k_min) / 2
        # Get the implied aggregate labor supply and aggregate capital demand
        aggregates = Aggregates(ks, agg_labor)
        # Get the implied prices
        prices = get_prices(aggregates, BaseModel)
        # Solve the household's policies
        policies = EGM(BaseModel, prices)
        # Get the implied stationary distribution of wealth
        Λ = distribution_transition(policies.saving, BaseModel.policygrid, BaseModel.Π)
        D = invariant_dist(Λ')
        # Get the implied aggregate capital supply
        spolicy = vcat(policies.saving...) 
        agg_kdmat = D' * spolicy
        agg_kd = agg_kdmat[1,1]
        # Check if the implied aggregate capital supply equals the aggregate capital demand
        crit = abs(agg_kd - aggregates.agg_ks)
        # Update the range of interest rate
        if agg_kd > ks
            k_min = ks
        else
            k_max = ks
        end
        iter += 1
            
    end
    
    if printsol
        println("Iteration: $iter, \n
            Interest rate: $(prices.r), \n
            Aggregate capital: $(aggregates.agg_ks), \n
            Aggregate labor: $(aggregates.agg_labor)")
    end

    steadystate = SteadyState(prices, policies, D, aggregates, Λ)
    return steadystate
    
end



function steadystateKS_r(BaseModel::AiyagariModel;
    ϵ::Float64 = 1e-6,
    itermax::Int64 = 1000,
    printsol::Bool = false)

    # Unpack parameters
    @unpack params, policygrid, initialguess, shockgrid, Π = BaseModel
    prices = Prices(0.0, 0.0)
    policies = (saving = zeros(size(initialguess)), consumption = zeros(size(initialguess)))
    D = zeros(size(initialguess))
    agg_labor = aggregate_labor(Π, shockgrid)
    aggregates = Aggregates(0.0, 0.0)
    Λ = spzeros(prod(size(initialguess)))

    # Start a solution loop
    crit = 100.0
    iter = 0
    high_r = (1/params.β) -1
    low_r = 0.0

    while (crit > ϵ) && (iter < itermax)
        # Get the first guess of interest rate from range
        r = (high_r + low_r) / 2
        # Get the implied aggregate labor supply and aggregate capital demand
        aggregates = get_aggregates(r, agg_labor, BaseModel)
        # Get the implied prices
        prices = get_prices(r, aggregates, params)
        # Solve the household's policies
        policies = EGM(BaseModel, prices)
        # Get the implied stationary distribution of wealth
        Λ = distribution_transition(policies.saving, BaseModel.policygrid, BaseModel.Π)
        D = invariant_dist(Λ')
        # Get the implied aggregate capital supply
        spolicy = vcat(policies.saving...) 
        agg_ksmat = D' * spolicy
        agg_ks = agg_ksmat[1,1]
        # Check if the implied aggregate capital supply equals the aggregate capital demand
        crit = abs(agg_ks - aggregates.agg_ks)
        # Update the range of interest rate
        if agg_ks > aggregates.agg_ks
            high_r = r
        else
            low_r = r
        end
        iter += 1
            
    end
    
    if printsol
        println("Iteration: $iter, \n
            Interest rate: $(prices.r), \n
            Aggregate capital: $(aggregates.agg_ks), \n
            Aggregate labor: $(aggregates.agg_labor)")
    end

    steadystate = SteadyState(prices, policies, D, aggregates, Λ)
    return steadystate
    
end

function getJacobian(BaseModel::AiyagariModel,
    steadystate::SteadyState,
    𝔼::Vector{Vector{Float64}},
    input::Char)

    # Get the policies (yso) and associated transition matrices (Λso)
    yso = get_yso(BaseModel, steadystate, input)
    Λso = get_Λso(BaseModel, yso)
    
    # Get yso and Λso for the "ghost run"
    # TODO: don't actually need to do the ghost run for each input 'r' and 'w' since it's the same for both. Or is it?
    ỹso = get_yso(BaseModel, steadystate, input, dx = 0.00) 
    ĩso = get_Λso(BaseModel, ỹso)
    
    # Get the curlyYs and curlyDs
    curlyYs = getCurlyYs(yso, ỹso, steadystate, BaseModel.params.dx)
    curlyDs = getCurlyDs(Λso, ĩso, steadystate, BaseModel.params.dx)

    # Create the fake news matrix
    fakeNews = createFakeNewsMatrix(curlyYs, curlyDs, 𝔼)

    # Create the Jacobian
    Jacobian = createJacobian(fakeNews)

    return fakeNews, Jacobian
end

function solveKS(BaseModel::AiyagariModel,
    steadystate::SteadyState)
   
    @unpack β, α, δ, γ, ρ, σ = BaseModel.params
    @unpack agg_ks, agg_labor = steadystate.aggregates
    @unpack r, w = steadystate.prices

    # Get the expectation vectors
    𝔼 = expectationVectors(steadystate, BaseModel.params.T)
    
    # Get the Jacobian and Fake News Matrices for each input
    fakeNews_r, Jacobian_r = getJacobian(BaseModel, steadystate, 𝔼, 'r')
    fakeNews_w, Jacobian_w = getJacobian(BaseModel, steadystate, 𝔼, 'w')

    # Solve the derivatives
    ∂r_∂K = α*(α-1) * (agg_ks/agg_labor)^(α-2) * (1/agg_labor)
    ∂w_∂K = α * (1-α) * (agg_ks/agg_labor)^(α-1) * (1/agg_labor)
    ∂r_∂Z = α * (agg_ks/agg_labor)^(α-1)
    ∂w_∂Z = (1-α) * (agg_ks/agg_labor)^α
    derivatives = Derivatives(∂r_∂K, ∂w_∂K, ∂r_∂Z, ∂w_∂Z)

    solution = Solution(fakeNews_r,
                        fakeNews_w,
                        Jacobian_r, 
                        Jacobian_w, 
                        derivatives)

    return solution
end


function generate_ar1(n::Int, 
    ρ::Float64; 
    z0::Float64 = 0.01)
    
    z = Vector{Float64}(undef, n)
    z[1] = z0
    for t in 2:n
        z[t] = ρ * z[t-1]
    end

    return z
end

function generateIRFs(solution::Solution,
    steadystate::SteadyState,
    dZ::Vector{Float64})
    
    @unpack rjacobian, wjacobian = solution
    @unpack ∂r_∂K, ∂w_∂K, ∂r_∂Z, ∂w_∂Z = solution.derivatives
    Hk = (rjacobian * ∂r_∂K) + (wjacobian * ∂w_∂K) - Matrix{Float64}(I, size(rjacobian)...)
    Hz = (rjacobian * ∂r_∂Z) + (wjacobian * ∂w_∂Z)
    invHk = inv(Hk)
    dK = -invHk * Hz * dZ
    #TODO: Need to fix the percentage deviations vs. deviations bit
    
    return dK
end


function mainKS()

    # defining the parameters of the model
    rho = 0.966
    s = 0.5
    sig = s * sqrt(1 - rho^2)
    params = Params(0.96, 1.0, sig, rho, 0.025, 0.11, 0.0001, [0.0, 200.0], 200, 7, 300)
    
    # Setting up the model
    BaseModel = setup_Aiyagari(params)
    
    # Solving for the steady state
    ss = solve_SteadyState(BaseModel, guess=(0.01, 0.10))

    # Solve the KS model
    solution = solveKS(BaseModel, ss)

    # Plot the fake news matrix and Jacobian
    p1 = plot(solution.rfakeNews[:, [1, 25, 50, 75, 100]], title = "Fake News Matrix", label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    display(p1)

    p2 = plot(solution.rjacobian[:, [1, 25, 50, 75, 100]], title = "Jacobian", label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    display(p2)

    # plot IRFs 
    dZs = Matrix{Float64}(undef, 300, 4)
    irfPlot = plot(title = "Impulse Response Functions", xlabel = "Quarters", ylabel = "Percent Deviation from SS")
    for ρ in [0.3, 0.5, 0.7, 0.9]
        dZ = generate_ar1(BaseModel.params.T, ρ)
        irfs = generateIRFs(solution, ss, dZ)
        plot!(irfPlot, irfs[1:50], label = "ρ = $ρ")
    end
    display(irfPlot)

end

