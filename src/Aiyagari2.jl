


"""
Params struct: contains the parameters of the model.
"""
@with_kw mutable struct Params @deftype Float64
    β = 0.968 # discount factor
    γ = 1.0 # coefficient of relative risk aversion
    s = 0.5 # parameter in getting shock process std dev
    ρ = 0.966 # persistence of the shock process
    σ = s * sqrt(1 - ρ^2)
    δ = 0.025# depreciation rate
    α = 0.11 # share of capital in production
    dx = 0.0001 # size of infinitesimal shock for numerical differentiation
    gridx::Vector{Float64} = [0.0, 200.0] # [a_min, a_max] bounds for the savings grid
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
        mc = rouwenhorst(p.n_e,p.ρ,p.σ,0.0)
        Π = mc.p
        Π_stationary = stationary_distributions(mc)[1]
        shockgrid = exp.(mc.state_values) / sum(Π_stationary .* exp.(mc.state_values))

        # asset grids
        agrid = collect(exp.(p.curvature_a * (range(p.gridx[1], log(p.gridx[2])/p.curvature_a, length = p.n_a))) .- 1)  # end of period fixed grid
        m = zeros(p.n_a + 1, p.n_e)

        # initialize endogenous grid to something similar to end of asset grid
        # making sure the first row is all zeros!
        m[2:end, :] = reshape(repeat(agrid .+ 0.001, outer = p.n_e), p.n_a, p.n_e)
        c_m = deepcopy(m)

        @assert m[1,:][:] == zeros(p.n_e)
        
        return new(p,Π,Π_stationary,shockgrid,agrid,m,zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),c_m,zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a, p.n_e),zeros(p.n_a))
    end
end

struct Aggregates
    K
    L
    Y
end

struct Prices
    r
    w
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

"Single Step in EGM algorithm - one policy function update"
function EGMstep!(a::Aiyagari,prices::Prices)
    p = a.params

    # get next period's consumption policy

    # this requires interpolation
    # prepe an empty object of size n_e, we need one policy per shock
	cfuns = Vector{Interpolations.Extrapolation{Float64, 1, Interpolations.GriddedInterpolation{Float64, 1, Vector{Float64}, Gridded{Linear{Throw{OnGrid}}}, Tuple{Vector{Float64}}}, Gridded{Linear{Throw{OnGrid}}}, Line{Nothing}}}(undef,p.n_e)

    Threads.@threads for ie in 1:p.n_e
        cfuns[ie] = linear_interpolation(a.m[:, ie], a.c_m[:,ie], extrapolation_bc = Line())
        a.cplus[:,ie] = cfuns[ie](a.mplus[:,ie])  # cplus is a,e
    end

    # loop over today's income state
    for ie in 1:p.n_e
        fill!(a.Eμ, 0.0)  #n_a

        # loop over tomorrow's income state
        # to compute conditional expected marginal utilities given e = ie
        for je in 1:p.n_e
            pr = a.Π[ie,je]
            a.Eμ[:] = a.Eμ[:] .+ (pr .* (a.cplus[:,je] .^ ((-1) * p.γ)))
        end

        rhs = p.β * (1 + prices.r) * a.Eμ
        a.c0_m[:,ie] = rhs .^ ((-1) / p.γ)
        a.m0[:,ie]   = a.agrid .+ a.c0_m[:,ie]
    end

    # done
end

"""
Solves the model via EGM
"""
function EGM!(a::Aiyagari,prices::Prices)
    p = a.params
    diff = 1e12
	count = 0
    a.mplus[:] = [(1 + prices.r) * a.agrid[ia] + prices.w * a.shockgrid[ie] for ia in 1:p.n_a, ie in 1:p.n_e]

    while diff > p.tol
        EGMstep!(a,prices)  
        diff = max(maximum(abs,a.m[2:end, :] - a.m0),
                   maximum(abs,a.c_m[2:end, :] - a.c0_m))
                   println(diff)
        # update objects with new values
        # notice that we keep the first row always equal to zero to represent the 
        # borrowing constraint.
        a.m[2:end, :] = a.m0
        a.c_m[2:end, :] = a.c0_m
		count += 1
    end

    # interpolate cons and savings onto end of period asset grid.
    for ie in 1:p.n_e
        cint = linear_interpolation(a.m[:,ie], a.c_m[:,ie], extrapolation_bc = Line())
        a.consumption[:,ie] = cint(a.agrid)
    end
    a.savings[:] = [ia + prices.w * iy for ia in a.agrid, iy in a.shockgrid] .- a.consumption
end


function main()
    p = Params()
    a = Aiyagari(p)
    aggs,prices = firm(0.05,a)
    EGM!(a,prices)
    a
end

