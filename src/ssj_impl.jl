
function get_yso(BaseModel::Aiyagari,
    steadystate::SteadyState,
    input::Char;
    dx::Union{Nothing, Float64} = nothing,
    shock_T_minus = 1)

    p = BaseModel.params

    if isnothing(dx)
        dx = BaseModel.params.dx
    end

    T = BaseModel.params.T

    # Creating the vector X^s_0: TODO: too verbose, make it more elegant
    if input == 'r'
        dxprice = fill(steadystate.prices.r, T) 
        dxprice[T-shock_T_minus] = steadystate.prices.r + dx
        price_tuples = collect(zip(dxprice, fill(steadystate.prices.w, T)))
    elseif input == 'w'
        dxprice = fill(steadystate.prices.w, T) 
        dxprice[T-shock_T_minus] = steadystate.prices.w + dx
        price_tuples = collect(zip(fill(steadystate.prices.r, T), dxprice))
    else
        error("input must be either 'r' or 'w'")
    end
    # @infiltrate
    # Creating the Jacobian
    yso = [zeros(p.n_a, p.n_e) for i in 1:T]
    # caution
    # julia> A = fill(Matrix{Float64}(undef, 2,2), 3)
    # 3-element Vector{Matrix{Float64}}:
    # [0.0 0.0; 0.0 0.0]
    # [0.0 0.0; 0.0 0.0]
    # [0.0 0.0; 0.0 0.0]

    # julia> A[1] === A[2]
    # true

    # very subtle but the s1[:,j] does *not* create a new array hence modifies always the same one.


    yso[T] = steadystate.policies.saving
    
    for i in 1:T-1
        prices = Prices(price_tuples[T-i]...)
        EGMstep!(yso[T-i+1],yso[T-i],BaseModel,prices)
    end
    
    return yso
end


function get_Λso(BaseModel::Aiyagari,
    yso::Vector{Matrix{Float64}})

    T = BaseModel.params.T
    Λso = Array{SparseMatrixCSC{Float64,Int64}}(undef, T)
    #TODO: Λso[T] is not assigned; probably doesn't matter since curlyDs[t] is zero anyway
    # but check if it should in fact be zero
    for i in 1:T-1
        Λso[T-i] = distribution_transition(yso[T-i], 
                            BaseModel.agrid, 
                            BaseModel.Π)
    end

    return Λso
end


function getCurlyYs(yso::Vector{Matrix{Float64}},
    ỹso::Vector{Matrix{Float64}},
    steadystate::SteadyState,
    dx::Float64)

    T = length(yso)
    curlyYs = zeros(T)
    for i in 1:T-1
        dyso = (yso[T-i] - ỹso[T-i]) ./ dx
        curlyYs[i] = vcat(dyso...)' * steadystate.D
    end

    return curlyYs
end


function getCurlyDs(Λso::Array{SparseMatrixCSC{Float64,Int64}},
    ĩso::Array{SparseMatrixCSC{Float64,Int64}},
    steadystate::SteadyState,
    dx::Float64)

    T = length(Λso)
    curlyDs = fill(zeros(size(steadystate.D)), T)
    for i in 1:T-1
        dΛso = (Λso[T-i] - ĩso[T-i]) ./ dx
        curlyDs[i] = dΛso * steadystate.D # Note: dΛso is not transposed because it is already transposed by construction
    end

    return curlyDs
end


"""
    expectationVectors(steadystate::SteadyState,
    T::Int)

TBW
"""
function expectationVectors(steadystate::SteadyState,
    T::Int)

    Λss = steadystate.Λ
    yss = vcat(steadystate.policies.saving...)
    
    # danger
    # 𝔼 = fill(Vector{Float64}(undef, size(steadystate.D)), T-1)
    𝔼 = [ zeros(size(steadystate.D)) for i in 1:(T-1) ]
    𝔼[1] = yss
    for i in 2:T-1
        𝔼[i] = Λss' * 𝔼[i-1]
    end

    return 𝔼
end


"""
    createFakeNewsMatrix(curlyYs::Vector{Float64},
    curlyDs::Vector{Matrix{Float64}},
    𝔼::Vector{Matrix{Float64}})

TBW
"""
function createFakeNewsMatrix(curlyYs::Vector{Float64},
    curlyDs::Vector{Vector{Float64}},
    𝔼::Vector{Vector{Float64}})

    T = length(curlyYs)

    # Create the fake news matrix
    fakeNews = Matrix{Float64}(undef, T, T)
    fakeNews[1,:] = curlyYs
    for j in eachindex(curlyDs) # Julia is column-major, so we iterate over columns first
        for i in eachindex(𝔼)
            fN = 𝔼[i]' * curlyDs[j]
            fakeNews[i+1,j] = fN[1,1]
        end
    end
    
    return fakeNews
end


"""
    createJacobian(fakeNews::Matrix{Float64})

TBW
"""
function createJacobian(fakeNews::Matrix{Float64})
    
    T = size(fakeNews,1)
    # Initialize the Jacobian
    Jacobian = Matrix{Float64}(undef, T, T)
    Jacobian[1,:] = fakeNews[1,:]
    Jacobian[:,1] = fakeNews[:,1]
    for s in 2:T # Julia is column-major, so we iterate over columns first
        for t in 2:T
            Jacobian[t,s] = Jacobian[t-1,s-1] + fakeNews[t,s]
        end
    end

    return Jacobian
end


"""
    mainSSJ()

Main Function that generates the fake news matrix, the Jacobian, and the impulse response functions
for a Krussell-Smith model using the Sequence-Space Jacobian method.
"""
function mainSSJ()
    p = Params(dx = 0.1)
    a = Aiyagari(p)
    
    steadystate = solve_SteadyState(a,guess = (0.011,0.05)); # find the steady state

    # Get the policies (yso) and associated transition matrices (Λso)
    yso_r = get_yso(a, steadystate, 'r')
    Λso_r = get_Λso(a, yso_r)
    
    # Get yso and Λso for the "ghost run"
    ỹso_r = get_yso(a, steadystate, 'r', dx = 0.00) 
    ĩso_r = get_Λso(a, ỹso_r)
    
    # Get the curlyYs and curlyDs
    curlyYs = getCurlyYs(yso_r, ỹso_r, steadystate, a.params.dx)
    curlyDs = getCurlyDs(Λso_r, ĩso_r, steadystate, a.params.dx)

    # Get the expectation vectors
    𝔼 = expectationVectors(steadystate, a.params.T)

    # # Create the fake news matrix
    fakeNews = createFakeNewsMatrix(curlyYs, curlyDs, 𝔼)

    # # Create the Jacobian
    Jacobian = createJacobian(fakeNews)

    # # Plot the fake news matrix and the Jacobian
    p1 = plot(fakeNews[:, [1, 25, 50, 75, 100]], title = "Fake News Matrix", label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    # display(p1)
    p2 = plot(Jacobian[:, [1, 25, 50, 75, 100]], title = "Jacobian", label = ["t = 1" "t = 25" "t = 50" "t = 75" "t = 100"])
    # display(p2)

    return fakeNews, Jacobian, p1,p2

end


