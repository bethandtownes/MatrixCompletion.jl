module Utilities
using ..Concepts

import Distributions
import LinearAlgebra




# export ErrorMatrix,
#     relative_error,
#     total_error



include("./Misc.jl")
include("./Diagnostics.jl")
include("./ExponentialFamily.jl")
include("./FastEigen.jl")
include("./RandomMatrices.jl")
include("./Indexing.jl")
include("./Sampling.jl")
include("./BatchUtils.jl")
include("./PrettyPrinter.jl")


# include("./TestModule.jl")

end
