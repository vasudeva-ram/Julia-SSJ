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
    ϵ::Float64 = 1e-9, 
    itermax::Int64 = 5000)
    
    @unpack params, policygrid, initialguess, shockgrid, Π, policymat, shockmat = model

    # @infiltrate 
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
    
    cons::Matrix{Float64} = ((1 + prices.r) * policymat) + (prices.w * shockmat) - currentguess
    
    return (saving = currentguess, consumption =  cons)

end

function EGM(model::AiyagariModel,
    prices::Prices, β; 
    ϵ::Float64 = 1e-9, 
    itermax::Int64 = 5000)
    
    @unpack params, policygrid, initialguess, shockgrid, Π, policymat, shockmat = model

    # @infiltrate 
    crit = 100.0
    iter = 0
    currentguess::Matrix = initialguess # n_a x n_e matrix
    cmat = similar(currentguess) # initialize cmat

    #TODO: implement Brent's method here
    while (crit > ϵ) && (iter < itermax)
        cmat = consumptiongrid(prices, policymat, shockmat, currentguess, Π, params,β)
        newguess = policyupdate(prices, policymat, shockmat, cmat)
        crit = norm(newguess - currentguess)
        currentguess = newguess
        iter += 1
    end
    
    cons= ((1 + prices.r) * policymat) + (prices.w * shockmat) - currentguess
    
    return (saving = currentguess, consumption =  cons)

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

function consumptiongrid(prices::Prices, 
    policymat::Matrix, 
    shockmat::AbstractArray, 
    currentguess::Matrix, 
    Π::Matrix, 
    params::Params,
    β)
    
    cprimemat = ((1 + prices.r) .* policymat) + (prices.w .* shockmat) - currentguess
    exponent = -1 * params.γ
    eulerlhs = β * (1 + prices.r) * ((cprimemat .^ exponent) * Π')
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
    policymat::Matrix, 
    shockmat::AbstractArray, 
    cmat::Matrix)
    
    impliedstate = (1 / (1 + prices.r)) * (cmat - (prices.w .* shockmat) + policymat)
    newguess = similar(impliedstate)

    for i in axes(impliedstate, 2)
        linpolate = Interpolations.linear_interpolation(impliedstate[:,i], policymat[:,i], extrapolation_bc = Interpolations.Flat())
        newguess[:,i] = linpolate.(policymat[:,i])
    end

    return newguess

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

function distribution_transition(savingspf::Matrix, # savings policy function
    policygrid::Vector, # savings grid
    Π::Matrix) # transition matrix for the exogenous shock process (get from `normalized_shockprocess()` function)
    
    n_a, n_e = size(savingspf)
    n_m = n_a * n_e
    policy = vcat(savingspf...)
    Jbases = [(ne -1)*n_a for ne in 1:n_e]
    Is = Int64[]
    Js = Int64[]
    Vs = []

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

