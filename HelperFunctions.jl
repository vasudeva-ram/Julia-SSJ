using LinearAlgebra
import IterativeSolvers

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


"""
    make_DoubleExponentialGrid(amin::Float64, 
    amax::Float64, 
    n_a::Int64)

Produces a double-exponential grid of asset holdings.
Compared to a uniform grid, the double-exponential grid is more dense around the origin.
This provides more precision for the asset holdings of the poorest households,
    where nonlinearities are most prevalent.
"""
function make_DoubleExponentialGrid(amin::Float64, 
    amax::Float64, 
    n_a::Int64)
    
    # Find maximum 𝕌 corresponding to amax
    𝕌 = log(1 + log(1 + amax- amin))

    # Create the uniform grid
    𝕌grid = range(0, 𝕌, n_a)

    # Transform the uniform grid to the double-exponential grid
    agrid = amin .+ exp.(exp.(𝕌grid) .- 1) .- 1

    return agrid
end



"""
    get_RouwenhorstDiscretization(n::Int64, # dimension of state-space
    ρ::Float64, # persistence of AR(1) process
    σ::Float64)

Discretizes an AR(1) process using the Rouwenhorst method.
See Kopecky and Suen (2009) for details: http://www.karenkopecky.net/Rouwenhorst_WP.pdf
Better than Tauchen (1986) method especially for highly persistent processes.
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
    initVector::Union{Nothing, Vector{Float64}}=nothing,
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

