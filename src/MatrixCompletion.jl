module MatrixCompletion

global VERBOSE_MODE = true







macro api(as,of)
    eval(Expr(:toplevel,:($as=$of),:(export $as)))
end

function dbg(x)
    if VERBOSE_MODE == true
        @show x
    end
end

function see(x)
    if VERBOSE_MODE == true
        display(x)
    end
end


function format(str)
    return rpad(str,70,"⋅")
end

export dbg,see,format

include("./Concepts.jl")
include("./Utilities/Utilities.jl")
#include("./Losses.jl")
#include("./Convex/Convex.jl")




@api MaybeMissing      Concepts.MaybeMissing
@api VecOrMatOf        Concepts.VecOrMatOf
 




#@api convert            Base.convert
@api provide            Concepts.provide


@api Loss               Concepts.Loss
@api ExponentialFamily  Concepts.ExponentialFamily
@api Gamma              Concepts.AbstractGamma
@api Binomial           Concepts.AbstractBinomial
@api Gaussian           Concepts.AbstractGaussian
@api Poisson            Concepts.AbstractPoisson



@api AbstractSamplingModels     Concepts.AbstractSamplingModels




@api Sampler            Utilities.Sampler
@api BernoulliModel     Concepts.BernoulliModel
@api UniformModel       Concepts.UniformModel
# @api NonUniformModel    Concepts.NonUniformModel




@api Diagnostics        Concepts.Diagnostics
@api LpSpace            Concepts.LpSpace
@api ErrorMetric        Concepts.ErrorMetric



# @api RelativeError      Utilities.RelativeError
# @api AbsoluteError      Utilities.AbsoluteError
# @api within_radius      Utilities.within_radius






end # module

