# Script with helper functions to assist with shock process discretization

"""
    tauchendisc(σ::Float64,
    ρ::Float64;
    n::Int64 = 7)

    Discretizes a first-order autoregressive process using Tauchen's method.
    Returns a vector of states and a transition matrix.
    See Tauchen (Economic Letters, 1986) for details.
"""
function tauchendisc(σ::Float64,
    ρ::Float64,
    n::Int64)
    
    σy = σ / sqrt(1 - ρ^2)
    w = (6*σy/(n-1))
    states = collect(-3*σy:w:3*σy)
    Π = zeros(n,n)
    for j in eachindex(states)
        for k in eachindex(states)
            if k == 1
                Π[j,k] = cdf(Normal(0,σ), states[1] - ρ*states[j] + w/2)
            elseif k == n
                Π[j,k] = 1 - cdf(Normal(0,σ), states[n] - ρ*states[j] - w/2)
            else
                Π[j,k] = cdf(Normal(0,σ), states[k] - ρ*states[j] + w/2) - 
                cdf(Normal(0,σ), states[k] - ρ*states[j] - w/2)
            end
        end
    end
    
    return states, Π
end


"""
    normalized_shockprocess(σ::Float64, 
    ρ::Float64)

    Wrapper for the tauchendisc function.
    Returns a normalized distribution in levels when the shock process
    is specified as an AR(1) in logs.
"""
function normalized_shockprocess(σ::Float64, 
    ρ::Float64,
    n::Int64)
    
    logshockgrid, Π = tauchendisc(σ, ρ, n)
    shockgrid = exp.(logshockgrid) # n_a x 1 vector
    
    return shockgrid, Π
    
end