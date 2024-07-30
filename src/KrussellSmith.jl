# Implementing the Krussell-Smith model as in Auclert et. al. (2021)




"""
    getJacobian(BaseModel::Aiyagari,
    steadystate::SteadyState,
    𝔼::Vector{Vector{Float64}},
    input::Char)

Obtains the Jacobian and the FakeNews matrix of the Krussell Smith model.
Corresponds to the procedure described in Section 3.2 of the Auclert et. al. (2021) paper.
Note: The Jacobian here refers only to the matrix of first-order derivatives of 
capital supply (Kˢ) w.r.t. the two inputs `r` and `w`. 
The analytical derivatives of these inputs w.r.t. the aggregate capital supply (K) 
and the shock (Z) are calculated in the `solveKS` function.
"""
function getJacobian(BaseModel::Aiyagari,
    steadystate::SteadyState,
    𝔼::Vector{Vector{Float64}},
    input::Char)

    # Get the policies (yso) and associated transition matrices (Λso)
    yso = get_yso(BaseModel, steadystate, input)
    Λso = get_Λso(BaseModel, yso)
    
    # Get yso and Λso for the "ghost run"
    # TODO: don't actually need to do the ghost run for each input 'r' and 'w' since it's the same for both. Or is it?
    ỹso = get_yso(BaseModel, steadystate, input, dx = 0.00) 
    ĩso = get_Λso(BaseModel, ỹso)
    
    # Get the curlyYs and curlyDs
    curlyYs = getCurlyYs(yso, ỹso, steadystate, BaseModel.params.dx)
    curlyDs = getCurlyDs(Λso, ĩso, steadystate, BaseModel.params.dx)

    # Create the fake news matrix
    fakeNews = createFakeNewsMatrix(curlyYs, curlyDs, 𝔼)

    # Create the Jacobian
    Jacobian = createJacobian(fakeNews)

    return fakeNews, Jacobian
end


"""
    solveKS(BaseModel::Aiyagari,
    steadystate::SteadyState)

Solves the Krussell-Smith model to first order. 
Note: The "solution" here refers to all the derivatives necessary 
to compute the impulse response functions. 
"""
function solveKS(BaseModel::Aiyagari,
    steadystate::SteadyState)
   
    @unpack β, α, δ, γ, ρ, σ = BaseModel.params
    @unpack K, L = steadystate.aggregates
    @unpack r, w = steadystate.prices

    # Get the expectation vectors
    𝔼 = expectationVectors(steadystate, BaseModel.params.T)
    
    # Get the Jacobian and Fake News Matrices for each input
    fakeNews_r, Jacobian_r = getJacobian(BaseModel, steadystate, 𝔼, 'r')
    fakeNews_w, Jacobian_w = getJacobian(BaseModel, steadystate, 𝔼, 'w')

    # Solve the derivatives
    ∂r_∂K = α*(α-1) * (K/L)^(α-2) * (1/L)
    ∂w_∂K = α * (1-α) * (K/L)^(α-1) * (1/L)
    ∂r_∂Z = α * (K/L)^(α-1)
    ∂w_∂Z = (1-α) * (K/L)^α
    derivatives = Derivatives(∂r_∂K, ∂w_∂K, ∂r_∂Z, ∂w_∂Z)

    solution = Solution(fakeNews_r,
                        fakeNews_w,
                        Jacobian_r, 
                        Jacobian_w, 
                        derivatives)

    return solution
end


"""
    generateIRFs(solution::Solution,
    dZ::Vector{Float64})

Uses the Jacobian to generate the impulse response functions of the model.
"""
function generateIRFs(solution::Solution,
    steadystate::SteadyState,
    dZ::Vector{Float64})
    
    @unpack rjacobian, wjacobian = solution
    @unpack ∂r_∂K, ∂w_∂K, ∂r_∂Z, ∂w_∂Z = solution.derivatives
    Hk = (rjacobian * ∂r_∂K) + (wjacobian * ∂w_∂K) - Matrix{Float64}(I, size(rjacobian)...)
    Hz = (rjacobian * ∂r_∂Z) + (wjacobian * ∂w_∂Z)
    invHk = inv(Hk)
    dK = -invHk * Hz * dZ
    #TODO: Need to fix the percentage deviations vs. deviations bit
    
    return dK
end

"exact same SS for KS as in Auclert et al"
function AuclertKS_SS()
    p = Params(β = 0.981952788061795,
    Z = 0.8816460975214567,
    n_a = 500)
    a = Aiyagari(p)
    SingleRun(p.β,3.142857142857143,p.Z,a),a
end

function mainAuclertKS()

    ss,a = AuclertKS_SS()  # get their SS
    solution = solveKS(a, ss) # Solve the KS model

    # numbers here should be identical to J_ha['A']['r'] in their notebook?
    @assert solution.rjacobian[1,1] == 3.04707181e+00


end

function mainKS()

    p = Params(n_a = 200, dx = 0.01)
    BaseModel = Aiyagari(p)
    
    ss = solve_SteadyState_r(0.01,1.0,BaseModel) # Solving for the steady state
    return ss, BaseModel
    solution = solveKS(BaseModel, ss) # Solve the KS model
    return solution
    # Plot the fake news matrix and Jacobian
    p1 = plot(solution.rfakeNews[:, [1, 25, 50, 75, 100]], 
                title = "Fake News Matrix", 
                label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    # display(p1)
    p2 = plot(solution.rjacobian[:, [1, 25, 50, 75, 100]], 
                title = "Jacobian", 
                label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    # display(p2)

    # plot IRFs 
    irfPlot = plot(title = "Impulse Response Functions", xlabel = "Quarters", ylabel = "Percent Deviation from SS")
    for ρ in [0.3, 0.5, 0.7, 0.9]
        dZ = generate_ar1(BaseModel.params.T, ρ)
        irfs = generateIRFs(solution, ss, dZ)
        plot!(irfPlot, irfs[1:50], label = "ρ = $ρ")
    end
    return p1,p2,irfPlot

end

