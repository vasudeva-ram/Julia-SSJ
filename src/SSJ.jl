
# this is a rewrite based on https://github.com/vasudeva-ram/Julia-SSJ

module SSJ


using Parameters
using Interpolations
using LinearAlgebra
using Plots
using SparseArrays
using Roots
using NLsolve
using Infiltrator


include("GeneralStructures.jl")
include("Aiyagari.jl")
include("ssj_impl.jl")
include("KrussellSmith.jl")
include("HelperFunctions.jl")

export Params




end # module SSJ
