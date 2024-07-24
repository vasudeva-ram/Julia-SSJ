
# this is a rewrite based on https://github.com/vasudeva-ram/Julia-SSJ

module SSJ


using QuantEcon: rouwenhorst, stationary_distributions
using Parameters
using Interpolations

include("Aiyagari2.jl")

export Params




end # module SSJ
