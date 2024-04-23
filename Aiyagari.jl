using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra, Plots, Distributions, SparseArrays, 
UnPack, BenchmarkTools, Profile, ProfileView

import Optim, IterativeSolvers, Interpolations, NLsolve

include("ShockDiscretization.jl")


"""
    EGM(model::AiyagariModel,
    prices::Prices; 
    ϵ::Float64 = 1e-6, 
    itermax::Int64 = 1000)
    

    This function implement the Endogenous Gridpoint Method of Carroll (2006).
    The method uses a time invariant grid of policy choices, and endogenously 
    generates a grid of states implied by those policies. 
"""
function EGM(model::AiyagariModel,
    prices::Prices; 
    ϵ::Float64 = 1e-6, 
    itermax::Int64 = 1000)
    
    @unpack params, policygrid, initialguess, shockgrid, Π, policymat, shockmat = model

    crit = 100.0
    iter = 0
    currentguess::Matrix{Float64} = initialguess # n_a x n_e matrix
    cmat = Matrix{Float64}(undef, size(currentguess)) # initialize cmat

    #TODO: implement Brent's method here
    while (crit > ϵ) && (iter < itermax)
        cmat = consumptiongrid(prices, policymat, shockmat, currentguess, Π, params)
        newguess = policyupdate(prices, policymat, shockmat, cmat)
        crit = norm(newguess - currentguess)
        currentguess = newguess
        iter += 1
    end
    
    cons = ((1 + prices.r) * policymat) + (prices.w * shockmat) - currentguess
    
    return (saving = currentguess, consumption = cons)

end


"""
    consumptiongrid(prices::Prices, 
    policymat::Matrix{Float64}, 
    shockmat::AbstractArray{Float64}, 
    currentguess::Matrix{Float64}, 
    Π::Matrix{Float64}, 
    params::Params)

    This function computes the implied consumption grid for one iteration of
    the EGM loop.
"""
function consumptiongrid(prices::Prices, 
    policymat::Matrix{Float64}, 
    shockmat::AbstractArray{Float64}, 
    currentguess::Matrix{Float64}, 
    Π::Matrix{Float64}, 
    params::Params)
    
    cprimemat = ((1 + prices.r) .* policymat) + (prices.w .* shockmat) - currentguess
    exponent = -1 * params.γ
    eulerlhs = params.β * (1 + prices.r) * ((cprimemat .^ exponent) * Π')
    cmat = eulerlhs .^ (1 / exponent)
    
    return cmat
    
end


"""
    policyupdate(prices::Prices, 
    policymat::Matrix{Float64}, 
    shockmat::AbstractArray{Float64}, 
    cmat::Matrix{Float64})

    This function updates the policy function using the Euler equation.
"""
function policyupdate(prices::Prices, 
    policymat::Matrix{Float64}, 
    shockmat::AbstractArray{Float64}, 
    cmat::Matrix{Float64})
    
    impliedstate = (1 / (1 + prices.r)) * (cmat - (prices.w .* shockmat) + policymat)
    newguess = Matrix{Float64}(undef, size(impliedstate))

    for i in axes(impliedstate, 2)
        linpolate = Interpolations.linear_interpolation(impliedstate[:,i], policymat[:,i], extrapolation_bc = Interpolations.Flat())
        newguess[:,i] = linpolate.(policymat[:,i])
    end

    return newguess

end




"""
    aggregate_labor(Π::Matrix{Float64}, 
    shockgrid::Vector{Float64})

    Computes the aggregate labor supply in the economy, given the
    transition matrix for the exogenous shock process and the grid
    of shocks.
"""
function aggregate_labor(Π::Matrix{Float64}, 
    shockgrid::Vector{Float64})
    
    invardist = invariant_dist(Π)
    explabor = shockgrid' * invardist
    
    return explabor[1,1]
end


"""
    distribution_transition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64})

    This function computes the transition matrix for the distribution of
    savings choices. It takes as inputs the savings policy function, the
    grid of savings choices, and the transition matrix for the exogenous
    shock process.
    Note that the output is a (n_a * n_e) x (n_a * n_e) sparse matrix, where n_a is the
    size of the savings grid and n_e is the size of the shock grid.
    TODO: Consider returning the transposed Λ matrix to stay consistent with the
    Auclert et. al. (Econmetrica, 2021) paper.
"""
function distribution_transition_Old(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64}) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
    n_a, n_e = size(savingspf)
    n_m = n_a * n_e
    Λ = spzeros(n_m, n_m)
    col = 1
    
    for j in 1:n_e # iterating first through the columns of the savings policy function
        for i in 1:n_a # iterating next through the rows of the savings policy function
            temp_transition = zeros(n_a,n_e)
            polval = savingspf[i,j]
            m = findfirst(x->x>=polval, policygrid)
            if m == 1
                temp_transition[m, :] = 1.0 .* Π[j,:] 
            else
                w = (polval - policygrid[m-1]) / (policygrid[m] - policygrid[m-1]) 
                temp_transition[m-1, :] = (1.0 - w) .* Π[j,:]
                temp_transition[m, :] = w .* Π[j,:]
            end
            # temp_transition = temp_transition * spdiagm(0 => Π[j,:])
            temp_vec = vcat(temp_transition...)
            Λ[:,col] = temp_vec
            col +=1
        end
    end

    return Λ

end



function distribution_transition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64}) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
    n_a, n_e = size(savingspf)
    n_m = n_a * n_e
    policy = vcat(savingspf...)
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]

    for col in eachindex(policy)
        m = findfirst(x->x>=policy[col], policygrid)
        j = div(col - 1, n_a) + 1
        if m == 1
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, n_e))
            append!(Vs, 1.0 .* Π[j,:])
        else
            append!(Is, (m-1) .+ Jbases)
            append!(Is, m .+ Jbases)
            append!(Js, fill(col, 2*n_e))
            w = (policy[col] - policygrid[m-1]) / (policygrid[m] - policygrid[m-1])
            append!(Vs, (1.0 - w) .* Π[j,:])
            append!(Vs, w .* Π[j,:])
        end
    end

    # println("Is: $(Is[end-10:end]), Js: $(Js[end-10:end]), Vs: $(Vs[end-10:end])")

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end

"""
    get_prices(r::Float64,
    aggregates::Aggregates,
    params::Params)

    Given the interest rate, the aggregate capital and labor,
    returns a Prices type with the interest rate and implied
    wage.
"""
function get_prices(r::Float64,
    aggregates::Aggregates,
    params::Params)
    
    @unpack agg_labor, agg_ks = aggregates
    w = (1 - params.α)*(agg_ks / agg_labor)^params.α
    prices = Prices(r, w)
    
    return prices
    
end


"""
    get_aggregates(r::Float64,
    agg_labor::Float64,
    Model::AiyagariModel)

    Given the interest rate and the aggregate labor supply
    (which is exogenously determined), returns an Aggregates 
    type with the implied aggregate capital supply from the 
    Firm sector.
"""
function get_aggregates(r::Float64,
    agg_labor::Float64,
    Model::AiyagariModel)
    
    params = Model.params
    agg_ks = (((r + params.δ)/params.α)^(1/(params.α - 1))) * agg_labor
    
    return Aggregates(agg_ks, agg_labor) 
end


"""
    setup_Aiyagari(params::Params;
    a_min::Float64 = 0.0,
    a_max::Float64 = 200.0)

    This function sets up an Aiyagari model. It takes as input the
    parameters of the model (Params type) and the parameters of the solution
    (SolutionParams type) and returns an instance of the AiyagariModel type.
"""
function setup_Aiyagari(params::Params;
    a_min::Float64 = 0.0,
    a_max::Float64 = 200.0)
    
    @unpack σ, ρ, n_a, n_e = params
    shockgrid, Π = normalized_shockprocess(σ, ρ, n_e)
    a_values = range(a_min, stop=a_max, length=n_a)
    policygrid = collect(a_values)
    initialguess = zeros(length(policygrid), length(shockgrid))
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)
    BaseModel = AiyagariModel(params, policygrid, policymat, initialguess, shockgrid, shockmat, Π)
    
    return BaseModel
end


"""
    steady_state(BaseModel::AiyagariModel;
    ϵ::Float64 = 1e-6,
    itermax::Int64 = 1000,
    printsol::Bool = false)

    This function computes the steady state of the Aiyagari model.
    It takes as input an instance of the AiyagariModel type, and
    returns the steady state policies, the stationary distribution
    of wealth, the prices, and the aggregate capital and labor.
"""
function steady_state(BaseModel::AiyagariModel;
    ϵ::Float64 = 1e-6,
    itermax::Int64 = 1000,
    printsol::Bool = false)
    """
    
    """
    @unpack params, policygrid, initialguess, shockgrid, Π = BaseModel
    # initializing the solutions
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
        agg_kdmat = D' * spolicy
        agg_kd = agg_kdmat[1,1]
        # Check if the implied aggregate capital supply equals the aggregate capital demand
        crit = abs(agg_kd - aggregates.agg_ks)
        # Update the range of interest rate
        if agg_kd > aggregates.agg_ks
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


"""
    main(printsol::Bool = false)

    Defining parameters and solving the Aiyagari model.
"""
function main(printsol::Bool = false)
    
    # defining the parameters of the model
    rho = 0.966
    s = 0.5
    sig = s * sqrt(1 - rho^2)
    params = Params(0.96, 1.0, sig, rho, 0.025, 0.11, 0.0001, 200, 7, 300)
    
    # Setting up the model
    BaseModel = setup_Aiyagari(params, a_min=0.0, a_max=200.0)
    
    # Solving for the steady state
    sol = steady_state(BaseModel, printsol = printsol)

    return sol
end




