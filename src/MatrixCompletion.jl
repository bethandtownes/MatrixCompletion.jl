module MatrixCompletion



macro SET_API(as,of)
    eval(Expr(:toplevel,:($as=$of),:(export $as)))
end


include("./Concepts.jl")
include("./Utilities/Utilities.jl")
include("./Losses.jl")
include("./Convex/Convex.jl")






@SET_API provide            Concepts.provide


@SET_API Loss               Concepts.Loss
@SET_API ExponentialFamily  Concepts.ExponentialFamily
@SET_API AbstractGamma      Concepts.AbstractGamma
@SET_API AbstractBinomial   Concepts.AbstractBinomial
@SET_API AbstractGaussian   Concepts.AbstractGaussian
@SET_API AbstractPoisson    Concepts.AbstractPoisson
@SET_API Diagnostics        Concepts.Diagnostics




end # module

