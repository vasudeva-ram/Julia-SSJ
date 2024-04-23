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
