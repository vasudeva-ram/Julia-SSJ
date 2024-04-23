



function Output(K::Float64, 
    N::Float64, 
    Z::Float64, 
    params::Parameters)

    @unpack α = params

    Y = Z * K^α * N^(1-α)
    
    return Y
end


function LaborDemand(Y::Float64, 
    N::Float64, 
    MC::Float64, 
    params::Parameters)

    @unpack α = params

    W = (1-α) * (Y/N) * MC
    
    return W
end


function Investment(K::Float64, 
    K_1::Float64, 
    δ::Float64)

    I = K_1 - (1-δ) * K
    
    return I
end