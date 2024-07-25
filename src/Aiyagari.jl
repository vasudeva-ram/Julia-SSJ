# this file contains type defs for parameter and Aiyagari model
# and all functions needed to solve for it's steady state.


"""
Params struct: contains the parameters of the model.
"""
@with_kw mutable struct Params @deftype Float64
    β = 0.968 # discount factor
    γ = 1.0 # coefficient of relative risk aversion
    ρ = 0.966 # persistence of the shock process
    σ = 0.5 # shock process std dev
    δ = 0.025# depreciation rate
    α = 0.11 # share of capital in production
    dx = 0.0001 # size of infinitesimal shock for numerical differentiation
    mina = 0.0 # lower bounds for the savings grid
    maxa = 200.0 # upper bounds for the savings grid
    n_a::Int = 200 # number of grid points for the savings grid
    n_e::Int = 7 # number of grid points for the shock grid
    T::Int = 300 # number of periods for the transition path
    Z = 0.8 # TFP
    curvature_a = 4.0 # parameter for scaling assset grid towards zero
    tol = 1e-8  # numerical tolerance on EGM convergence loop
end


"""
AiyagariModel struct: contains all the objects needed to solve the
Aiyagari (1994) model.
"""
struct Aiyagari
    params::Params # parameters of the model
    Π::Matrix{Float64}   # icnome shock trans matrix
    Π_stationary::Vector{Float64}   # stationary distribution of Π
    shockgrid::Vector{Float64} # grid of shocks

    agrid::Vector{Float64} # end of period fixed asset grid

    # EGM related grids defined on cash on hand m = a + y
    m::Matrix{Float64} # endogenous grid for cash on hand on: n_a + 1 by n_e
    m0::Matrix{Float64} # iteration object n_a by n_e
    mplus::Matrix{Float64} # next period cash on hand
    c_m::Matrix{Float64} # optimal consumption on m: n_a + 1 by n_e
    c0_m::Matrix{Float64} # ieration helper on m: n_a by n_e
    cplus::Matrix{Float64} # optimal consumption on m: n_a by n_e
    # c0_m::Matrix{Float64} # optimal consumption on m: n_a by n_e

    consumption::Matrix{Float64} # optimal consumption on a : na by ne
    savings::Matrix{Float64} # optimal consumption on a : na by ne
    Eμ::Vector{Float64} # Expected marginal utility of consumption given income shock

    function Aiyagari(p)
        # income shock
        mc = get_RouwenhorstDiscretization(p.n_e,p.ρ,p.σ)
        Π = mc[1]
        Π_stationary = mc[2]
        shockgrid = mc[3]

        # asset grids
        agrid_ = agrid(p.maxa, p.mina, p.n_a)
        m = zeros(p.n_a + 1, p.n_e)

        # initialize endogenous grid to something similar to end of asset grid
        # making sure the first row is all zeros!
        m[2:end, :] = reshape(repeat(agrid_ .+ 0.001, outer = p.n_e), p.n_a, p.n_e)
        c_m = deepcopy(m)

        @assert m[1,:][:] == zeros(p.n_e)
        
        return new(p,Π,Π_stationary,shockgrid,agrid_,m,zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),c_m,zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a))
    end
end

"""
agrid function copied from SSJ paper
julia 1.11 will have the logrange function for this purposeo

"""
function agrid(amax,amin,n)
    pivot = abs(amin) + 0.25
    grid = 10 .^ (range(log(10, amin + pivot), log(10, amax + pivot), length=n)) .- pivot
    grid[1] = amin
    grid
end

"""
    get_RouwenhorstDiscretization(n::Int64, # dimension of state-space
    ρ::Float64, # persistence of AR(1) process
    σ::Float64)

Discretizes an AR(1) process using the Rouwenhorst method.
See Kopecky and Suen (2009) for details: http://www.karenkopecky.net/Rouwenhorst_WP.pdf
Better than Tauchen (1986) method especially for highly persistent processes.

This implementation is identical to the SSJ toolbox, so we use this.
"""
function get_RouwenhorstDiscretization(n::Int64, # dimension of state-space
    ρ::Float64, # persistence of AR(1) process
    σ::Float64) # standard deviation of AR(1) process

    # Construct the transition matrix
    p = (1 + ρ)/2
    
    Π = [p 1-p; 1-p p]
    
    for i = 3:n
        Π_old = Π
        Π = zeros(i, i)
        Π[1:i-1, 1:i-1] += p * Π_old
        Π[1:i-1, 2:end] += (1-p) * Π_old
        Π[2:end, 1:i-1] += (1-p) * Π_old
        Π[2:end, 2:end] += p * Π_old
        Π[2:i-1, 1:end] /= 2
    end

    # Obtain the stationary distribution
    #TODO: should Π be transposed here? What does Rouwenhorst return? 
    #SOLVED: No, Π should not be transposed here; it gets transposed (correctly) within the invariant_dist function  
    D = invariant_dist(Π) 

    # Construct the state-space
    α = 2 * (σ/sqrt(n-1))
    z = exp.(α * collect(0:n-1))
    z = z ./ sum(z .* D) # normalize the distribution to have mean of 1
    
    #TODO: Based on this construction Zᵢⱼ has a mean of 1. But HHᵢ's wage income equals Zᵢⱼᵥ * Wⱼ. Should Zᵢⱼᵥ
    # have a mean of 1 instead? #UPDATE: for IMPALight, it is sufficient for Zᵢⱼ to have a mean of 1.

    return Π, D, z

end


"""
    invariant_dist(Π::AbstractMatrix;
    method::Int64 = 1,
    ε::Float64 = 1e-9,
    itermax::Int64 = 50000,
    initVector::Union{Nothing, Vector{Float64}}=nothing,
    verbose::Bool = false
    )

Calculates the invariant distribution of a Markov chain with transition matrix Π.
"""
function invariant_dist(Π::AbstractMatrix;
    method::Int64 = 1,
    ε::Float64 = 1e-9,
    itermax::Int64 = 50000,
    initVector=nothing,
    verbose::Bool = false
    )

    # Function to generate an initial vector if there isn't one already
    function generate_initVector(Π::AbstractMatrix)
        m = size(Π,1)
        D = (1/m) .* ones(m)
        return D
    end

    ΠT = Π' # transpose to avoid creating an adjoint at each step

    # https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/8
    if method == 1 # solve the system of equations
        D = [1; (I - ΠT[2:end, 2:end]) \ Vector(ΠT[2:end,1])]
    
    elseif method == 2 # iteration
        crit = 1.0
        iter = 0
        D = isnothing(initVector) ? generate_initVector(Π) : initVector
        while crit > ε && iter < itermax
            newD = ΠT * D 
            crit = norm(newD - D)
            D = newD
            iter += 1
        end        
        
        if verbose
            println("Converged in $iter iterations.")
        end

        if iter == itermax
            println("Warning: invariant distribution did not converge.")
        end
        
    elseif method == 3 # inverse power method
        λ, D = IterativeSolvers.powm!(ΠT, D, tol= ε, maxiter = itermax, verbose=verbose) # Given that the approximate eigenvalue is not really necssary, could we just use something like D = IterativeSolvers.powm!(Π', D, tol = ε, maxiter = itermax)[2]?
        
    elseif method == 4 # Anderson mixing
        D = isnothing(initVector) ? generate_initVector(Π) : initVector
        func(x) = ΠT * x
        D = NLsolve.fixedpoint(func, D, ftol=ε, iterations=itermax).zero        
    
    else
        error("Method choice must be between 
        1: Sparse-Direct Linear Solution (default), 
        2: Iteration, 
        3: Inverse Power method, 
        4: Fixed-point with Anderson Mixing")
    end

    return D ./ sum(D) # return normalized to sum to 1.0
end


aggregate_labor(a::Aiyagari) = a.shockgrid' * a.Π_stationary


function firm(r::Float64,a::Aiyagari)
    L = aggregate_labor(a)
    p = a.params
    capital_demand = (((r + p.δ) /p.α )^(1/(p.α - 1))) * L
    wage = (1 - p.α) * (capital_demand / L)^p.α

    return Aggregates(capital_demand,L,0), Prices(r,wage)
end

function firm(K,L,a::Aiyagari)
    p = a.params
    r = p.α * p.Z * (K / L) ^ (p.α-1) - p.δ
    w = (1 - p.α) * p.Z * (K / L) ^ p.α
    Y = p.Z * K ^ p.α * L ^ (1 - p.α)
    return Aggregates(K,L,Y), Prices(r,w)
end


# solving the household problem
# =============================

"""
Solves the household problem via EGM
"""
function EGM!(a::Aiyagari,prices::Prices)
    p = a.params
    diff = 1e12
	count = 0

    # pre-compute tomorrow's cash on hand, which stays constant throughout iteration below
    a.mplus[:] = [(1 + prices.r) * a.agrid[ia] + prices.w * a.shockgrid[ie] for ia in 1:p.n_a, ie in 1:p.n_e]

    while diff > p.tol
        EGMstep!(a,prices)  
        diff = max(maximum(abs,a.m[2:end, :] - a.m0),
                   maximum(abs,a.c_m[2:end, :] - a.c0_m))
        # update objects with new values
        # notice that we keep the first row always equal to zero to represent the 
        # borrowing constraint.
        a.m[2:end, :]   = a.m0
        a.c_m[2:end, :] = a.c0_m
		count += 1
    end

    # interpolate cons onto end of period asset grid.
    for ie in 1:p.n_e
        cint = linear_interpolation(a.m[:,ie], a.c_m[:,ie], extrapolation_bc = Line())
        a.consumption[:,ie] = cint(a.agrid .+ prices.w * a.shockgrid[ie])  # a + we = m
    end

    # on agrid
    a.savings[:] = [ia + prices.w * iy for ia in a.agrid, iy in a.shockgrid] .- a.consumption
end


"Single Step in EGM algorithm - one policy function update"
function EGMstep!(a::Aiyagari,prices::Prices)
    p = a.params

    # get next period's consumption policy
    # this requires interpolation of consumption function on next period's
    # cash on hand grid:
    for ie in 1:p.n_e
        # TODO this can be made much faster by exploiting monotonicity in m and c
        itp = linear_interpolation(a.m[:, ie], a.c_m[:,ie], extrapolation_bc = Line())
        a.cplus[:,ie] = itp(a.mplus[:,ie])  # cplus is a,e
    end

    # construct expect marginal utility
    Eμ = (a.cplus .^ ((-1) * p.γ))  * a.Π'

    # RHS of the euler equation
    rhs = p.β * (1 + prices.r) * Eμ

    # inverse of LHS of EE is current consumption
    a.c0_m[:] = rhs .^ ((-1) / p.γ)

    # current consumption plus end of period asset grid is endogenous grid.
    a.m0[:]   = repeat(a.agrid,1,p.n_e) + a.c0_m
   
end






"""
    distribution_transition(savingspf::Matrix{Float64}, # savings policy function
    policygrid::Vector{Float64}, # savings grid
    Π::Matrix{Float64})

Obtains the transition matrix for the household between period t and t+1 via
    the Young (2010) method. In essence, this method uses the policy function
    of the household to obtain transition probabilities between grid positions
    using a "lottery" approach. This is then composed with the transition
    probabilities of the exogenous employment process to obtain the full
    transition matrix.
"""
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

    Λ = sparse(Is, Js, Vs, n_m, n_m)

    return Λ
end

# Partial Equilibrium Model Runs

"""
    SingleRun(r::Float64, a::Aiyagari)

Model Solution for a given interest rate
"""
function SingleRun(r::Float64, a::Aiyagari)
    # p = Params()
    # a = Aiyagari(p)
    aggs,prices = firm(r,a)
    EGM!(a,prices)

    Λ = distribution_transition(a.savings, a.agrid, a.Π)
    D = invariant_dist(Λ')

    return SteadyState(prices, (saving = a.savings,consumption = a.consumption), D, aggs, Λ)
    
end

"""
    SingleRun(β,K,Z, a::Aiyagari)

Model Solution for given values of (β,K,Z). This is used then targeting certain
    values for interest rate and Output.
"""
function SingleRun(β,K,Z,a::Aiyagari)

    a.params.β = β
    a.params.Z = Z
    L = aggregate_labor(a)

    aggs,prices = firm(K,L,a)
    EGM!(a,prices)

    Λ = distribution_transition(a.savings, a.agrid, a.Π)
    D = invariant_dist(Λ')

    return SteadyState(prices, (saving = a.savings,consumption = a.consumption ), D, aggs, Λ)
end



"""
solve_SteadyState(a::Aiyagari;
    guess = (0.01, 0.10))

    This function computes the steady state of the Aiyagari model.
    It takes as input an instance of the Aiyagari type, and
    returns the steady state policies, the stationary distribution
    of wealth, the prices, and the aggregate capital and labor.
"""
function solve_SteadyState(a::Aiyagari;
    guess = (0.01, 0.10))
    
    r = find_zero(x -> get_residual(x,a), guess, Roots.A42())
    
    solution = SingleRun(r, a)
    
    return solution
end

# internal function to obtain residual
function get_residual(r_guess::Float64,a::Aiyagari)
    ss = SingleRun(r_guess, a)
    agg_ks = ss.aggregates.K
    spolicy = vcat(ss.policies.saving...)
    agg_kd = (ss.D' * spolicy)[1,1]
    return agg_kd - agg_ks
end


"""
    solve_SteadyState_r(rtarget::Float64, ytarget::Float64, a::Aiyagari)

This function computes the steady state of the Aiyagari model.
It chooses β while trying to hit a target on r and Y.

this yields identical results to the KS notebook of the original SSJ project.


"""
function solve_SteadyState_r(rtarget::Float64, ytarget::Float64, a::Aiyagari)
    
    # need to hit target r

    init_x = [0.98, 3.0, 0.85]  # β, K, Z

    # does not work
    # result = nlsolve((out,x) -> residual!(out,x[1],x[2],x[3],rtarget,1.0,BaseModel), init_x, autodiff = :forward)
    # super slow
    result = nlsolve((out,x) -> residual!(out,x[1],x[2],x[3],rtarget,ytarget,a), init_x, show_trace = false, ftol = 1e-7)

    return SingleRun(result.zero...,a)

end

# internal function to obtain residual
function residual!(F,β,K,Z,rtarget,ytarget, a::Aiyagari)
    ss = SingleRun(β,K,Z,a)
    agg_ks = ss.aggregates.K
    spolicy = vcat(ss.policies.saving...)
    agg_kd = (ss.D' * spolicy)[1,1]
    F[1] = agg_kd - agg_ks # capital_market 
    F[2] = ss.prices.r - rtarget # r target
    F[3] = ss.aggregates.Y - ytarget  # Y target
end

# Some Plotting functions


function plotcons_single()
    p = Params()
    a = Aiyagari(p)
    aggs,prices = firm(0.01,a)
    EGM!(a,prices)
    plot(a.agrid, a.consumption)
end
function run_single()
    p = Params()
    a = Aiyagari(p)
    aggs,prices = firm(0.01,a)
    for i in 1:100
        EGM!(a,prices)
    end
end

function runSS1()
    p = Params(n_a = 500)
    a = Aiyagari(p)
    solve_SteadyState(a)
end

function runSS2()
    p = Params(n_a = 500)
    r = 0.01
    y = 1.0
    a = Aiyagari(p)
    solve_SteadyState_r(r,y,a)
end


function plotconsSS1()
    p = Params()
    a = Aiyagari(p)
    s = solve_SteadyState(a)
    plot(a.agrid, s.policies.consumption, title = "r = $(round(s.prices.r,digits =4))",ylab = "Consumption", xlab = "Assets",leg = false)
end

function plotconsSS2()
    p = Params(n_a = 500)
    a = Aiyagari(p)
    r = 0.01
    y = 1.0
    s = SSJ.solve_SteadyState_r(r,y,a)
    plot(a.agrid, s.policies.consumption, title = "r = $(round(s.prices.r,digits =4))",ylab = "Consumption", xlab = "Assets",leg = false)
end