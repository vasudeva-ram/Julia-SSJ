

"""
    generate_ar1(n::Int, 
    ρ::Float64; 
    z0::Float64 = 0.01)

Generates an array with the first `n` elements of an AR(1) process with persistence `ρ`.
"""
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