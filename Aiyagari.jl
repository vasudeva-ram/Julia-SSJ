using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra, Plots, Distributions, SparseArrays, 
UnPack, BenchmarkTools, Profile, ProfileView, Roots

import IterativeSolvers, Interpolations

include("GeneralStructures.jl")
include("HelperFunctions.jl")
include("Household.jl")

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
    get_aggregates(r::Float64,
    agg_labor::Float64,
    Model::AiyagariModel)

    Given the interest rate and the aggregate labor supply
    (which is exogenously determined), returns an Aggregates 
    type with the implied aggregate capital supply from the 
    Firm sector.
"""
function get_AggsPrices(r::Float64,
    agg_labor::Float64,
    Model::AiyagariModel)
    
    params = Model.params
    agg_ks = (((r + params.δ)/params.α)^(1/(params.α - 1))) * agg_labor
    w = (1 - params.α)*(agg_ks / agg_labor)^params.α

    return Aggregates(agg_ks, agg_labor), Prices(r, w)
end


"""
    setup_Aiyagari(params::Params;
    a_min::Float64 = 0.0,
    a_max::Float64 = 200.0)

    This function sets up an Aiyagari model. It takes as input the
    parameters of the model (Params type) and the parameters of the solution
    (SolutionParams type) and returns an instance of the AiyagariModel type.
"""
function setup_Aiyagari(params::Params)
    
    @unpack σ, ρ, n_a, n_e, gridx = params
    a_min, a_max = gridx
    Π, _, shockgrid = normalized_shockprocess(n_e, ρ, σ, method=1)
    #a_values = range(a_min, stop=a_max, length=n_a)
    policygrid = make_DoubleExponentialGrid(a_min, a_max, n_a)
    initialguess = zeros(length(policygrid), length(shockgrid))
    policymat = repeat(policygrid, 1, length(shockgrid)) # making this n_a x n_e matrix
    shockmat = repeat(shockgrid, 1, length(policygrid))' # making this n_a x n_e matrix (note the transpose)
    BaseModel = AiyagariModel(params, policygrid, policymat, initialguess, shockgrid, shockmat, Π)
    
    return BaseModel
end



"""
    SingleRun(r::Float64,
    BaseModel::AiyagariModel)

Given a guess for the interest rate, this function computes all the
    policies of all the agents.
    Note: This is not the *steady state* solution; just the solution of all 
    agents given an interest rate.
"""
function SingleRun(r::Float64,
    BaseModel::AiyagariModel)
    
    agg_labor = aggregate_labor(BaseModel.Π, BaseModel.shockgrid)
    aggregates, prices = get_AggsPrices(r, agg_labor, BaseModel)
    policies = EGM(BaseModel, prices)
    Λ = distribution_transition(policies.saving, BaseModel.policygrid, BaseModel.Π)
    D = invariant_dist(Λ')

    return SteadyState(prices, policies, D, aggregates, Λ)
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
function solve_SteadyState(BaseModel::AiyagariModel;
    guess = (0.01, 0.10))
    
    # internal function to obtain residual
    function get_residual(r_guess::Float64)
        ss = SingleRun(r_guess, BaseModel)
        agg_ks = ss.aggregates.agg_ks
        spolicy = vcat(ss.policies.saving...)
        agg_kd = (ss.D' * spolicy)[1,1]
        return agg_kd - agg_ks
    end

    r = find_zero(get_residual, guess, Bisection())
    
    solution = SingleRun(r, BaseModel)
    
    return solution
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
    params = Params(0.96, 1.0, sig, rho, 0.025, 0.11, 0.0001, [0.0, 200.0], 200, 7, 300)
    
    # Setting up the model
    BaseModel = setup_Aiyagari(params)
    
    # Solving for the steady state
    sol = solve_SteadyState(BaseModel, guess=(0.01, 0.10))

    if printsol
        println("Steady state interest rate: ", sol.prices.r)
        println("Steady state wage rate: ", sol.prices.w)
        println("Steady state aggregate capital: ", sol.aggregates.agg_ks)
        println("Steady state aggregate labor: ", sol.aggregates.agg_labor)
    end

    return sol
end




