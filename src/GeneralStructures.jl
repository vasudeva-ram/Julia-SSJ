"""
This file defines a set of structs that will be used in solving
the Aiyagari (1994) and Krussell Smith (1998) models.
The structures are defined such that commonly used objects 
can be easily shared and passed between functions.
Making them immutable structs allows us to use the full power of Julia's 
multiple dispatch to write more efficient code.
"""


struct Aggregates
    K
    L
    Y
end

struct Prices
    r
    w
end

"""
SteadyState struct: In our model, we define a steady state as the
prices, policies, distribution of wealth, aggregate capital and labor,
and the transition matrix for the distribution of wealth that satisfy
the equilibrium conditions of the model. 
Defining such a struct allows us to easily pass around the steady state
when we need to compute transitions in the KS model.
"""
struct SteadyState{TF<:Float64}
    prices::Prices # steady state prices
    policies::NamedTuple{(:saving, :consumption), Tuple{Matrix{TF}, Matrix{TF}}} # savings and consumption policies
    D::Vector{TF} # steady state distribution of wealth
    aggregates::Aggregates # steady state aggregate capital and labor
    Λ::SparseMatrixCSC{TF,Int64} # steady state transition matrix for the distribution of wealth
end


"""
Derivatives struct: contains the derivatives of the model with respect to
the `inputs` of the model.
In our case, the inputs will be the prices of the model, i.e. the interest rate
and the wage rate.
"""
struct Derivatives{TF<:Float64}
    ∂r_∂K::TF
    ∂w_∂K::TF
    ∂r_∂Z::TF
    ∂w_∂Z::TF
end


"""
Solution struct: contains the solution of the model, i.e. the prices of the model,
the Jacobian of the model, and the derivatives of the model.
These are the objects necessary to compute the transition path of the model and 
obtain the impulse responses.
"""
struct Solution{TF<:Float64}
    rfakeNews::Matrix{TF}
    wfakeNews::Matrix{TF}
    rjacobian::Matrix{TF}
    wjacobian::Matrix{TF}
    derivatives::Derivatives{TF}
end


