module SSJ

greet() = print("Hello World!")

using LinearAlgebra, Plots, Distributions, SparseArrays, UnPack, Roots
using NLsolve
using Infiltrator
# using OptimizationPRIMA, Optimization
# using NonlinearSolve
# using SteadyStateDiffEq

import IterativeSolvers, Interpolations


include("GeneralStructures.jl")
include("HelperFunctions.jl")
include("Household.jl")
include("Aiyagari.jl")
include("KrussellSmith.jl")
include("ssj_impl.jl")

export mainKS, main, mainSSJ

end # module SSJ
