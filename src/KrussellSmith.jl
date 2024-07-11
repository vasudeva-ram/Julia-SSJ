# Implementing the Krussell-Smith model as in Auclert et. al. (2021)

"""
    get_prices(aggs::Aggregates, 
    model::AiyagariModel)

Given aggregate capital and labor supply, and the model parameters,
    returns the prices - `r` and `w` - of the model.
"""
function get_prices(aggs::Aggregates, 
    model::AiyagariModel)

    # Unpack parameters
    @unpack agg_ks, agg_labor = aggs
    @unpack α, δ = model.params 
    r = α * (agg_ks / agg_labor)^(α-1) - δ
    w = (1-α) * (agg_ks / agg_labor)^α
    # Note: we are assuming the value of the shock Z in steady state is 1.

    return Prices(r, w)
end


"""
    getJacobian(BaseModel::AiyagariModel,
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
function getJacobian(BaseModel::AiyagariModel,
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
    solveKS(BaseModel::AiyagariModel,
    steadystate::SteadyState)

Solves the Krussell-Smith model to first order. 
Note: The "solution" here refers to all the derivatives necessary 
to compute the impulse response functions. 
"""
function solveKS(BaseModel::AiyagariModel,
    steadystate::SteadyState)
   
    @unpack β, α, δ, γ, ρ, σ = BaseModel.params
    @unpack agg_ks, agg_labor = steadystate.aggregates
    @unpack r, w = steadystate.prices

    # Get the expectation vectors
    𝔼 = expectationVectors(steadystate, BaseModel.params.T)
    
    # Get the Jacobian and Fake News Matrices for each input
    fakeNews_r, Jacobian_r = getJacobian(BaseModel, steadystate, 𝔼, 'r')
    fakeNews_w, Jacobian_w = getJacobian(BaseModel, steadystate, 𝔼, 'w')

    # Solve the derivatives
    ∂r_∂K = α*(α-1) * (agg_ks/agg_labor)^(α-2) * (1/agg_labor)
    ∂w_∂K = α * (1-α) * (agg_ks/agg_labor)^(α-1) * (1/agg_labor)
    ∂r_∂Z = α * (agg_ks/agg_labor)^(α-1)
    ∂w_∂Z = (1-α) * (agg_ks/agg_labor)^α
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


function mainKS()

    # defining the parameters of the model
    rho = 0.966
    s = 0.5
    sig = s * sqrt(1 - rho^2)
    params = Params(0.98, 1.0, sig, rho, 0.025, 0.11, 0.0001, [0.0, 200.0], 200, 7, 300)
    
    # Solving the model
    BaseModel = setup_Aiyagari(params) # Setting up the model
    ss = solve_SteadyState(BaseModel, guess=(0.01, 0.10)) # Solving for the steady state
    solution = solveKS(BaseModel, ss) # Solve the KS model

    # Plot the fake news matrix and Jacobian
    p1 = plot(solution.rfakeNews[:, [1, 25, 50, 75, 100]], 
                title = "Fake News Matrix", 
                label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    display(p1)
    p2 = plot(solution.rjacobian[:, [1, 25, 50, 75, 100]], 
                title = "Jacobian", 
                label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    display(p2)

    # plot IRFs 
    irfPlot = plot(title = "Impulse Response Functions", xlabel = "Quarters", ylabel = "Percent Deviation from SS")
    for ρ in [0.3, 0.5, 0.7, 0.9]
        dZ = generate_ar1(BaseModel.params.T, ρ)
        irfs = generateIRFs(solution, ss, dZ)
        plot!(irfPlot, irfs[1:50], label = "ρ = $ρ")
    end
    display(irfPlot)

end

