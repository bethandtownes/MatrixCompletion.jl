module MatrixCompletion

global VERBOSE_MODE = true
global DEBUG_MODE   = true


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
    return rpad(str,70,".")
end

export dbg,see,format

include("./Concepts.jl")
include("./MGF.jl")
include("./Estimator.jl")
include("./ModelFitting.jl")
include("./Utilities/Utilities.jl")
include("./Losses.jl")

#include("./Convex/Convex.jl")




@api MaybeMissing               Concepts.MaybeMissing
@api VecOrMatOf                 Concepts.VecOrMatOf
@api UnivariateDistributions    Concepts.UnivariateDistributions 




@api provide            Concepts.provide
@api check              Concepts.check
@api predict            Concepts.predict
@api evaluate           Concepts.evaluate
@api estimator          Concepts.estimator
@api choose             Concepts.choose



@api NotOverLoadedException       Concepts.NotOverLoadedException
@api UnrecognizedSymbolException  Concepts.UnrecognizedSymbolException



#==============================================================================#
#                               Model Fitting                                  #
#==============================================================================#





#==============================================================================#
#                                Comparator                                    #
#==============================================================================#
@api Comparator        Concepts.Comparator




#==============================================================================#
#                                Loss Functions                                #
#==============================================================================#
@api AbstractLoss       Concepts.AbstractLoss
@api Loss               Losses.Loss
@api train              Losses.train



#==============================================================================#
#                                 Estimator                                    #
#==============================================================================#
#@api EstimationProcedure    Estimator.EstimationProcedure
@api ProfileLikelihood       Estimator.ProfileLikelihood
@api MLE                     Estimator.MLE



#==============================================================================#
#                              Exponential Family                              #
#==============================================================================#
@api ExponentialFamily  Concepts.ExponentialFamily
@api Gamma              Concepts.AbstractGamma
@api Binomial           Concepts.AbstractBinomial
@api Gaussian           Concepts.AbstractGaussian
@api Poisson            Concepts.AbstractPoisson
@api Bernoulli          Concepts.AbstractBernoulli
@api NegativeBinomial   Concepts.AbstractNegativeBinomial
@api GaussianMatrix     Utilities.GaussianMatrix
@api PoissonMatrix      Utilities.PoissonMatrix
@api BernoulliMatrix    Utilities.BernoulliMatrix
@api GammaMatrix        Utilities.GammaMatrix
@api forward_map        Concepts.forward_map


@api AbstractSamplingModels     Concepts.AbstractSamplingModels
@api AbstractFixedRankMatrix      Concepts.AbstractFixedRankMatrix
@api FixedRankMatrix              Utilities.FixedRankMatrix


#==============================================================================#
#                                     Tracker                                  #
#==============================================================================#
@api IndexTracker     Utilities.IndexTracker





@api Continuous       Concepts.Continuous
@api Categorical      Concepts.Categorical



@api MGF             BetterMGF.MGF
@api SampleMGF       BetterMGF.SampleMGF





@api Sampler            Utilities.Sampler
@api BernoulliModel     Concepts.BernoulliModel
@api UniformModel       Concepts.UniformModel
# @api NonUniformModel    Concepts.NonUniformModel




@api Diagnostics        Concepts.Diagnostics
@api LpSpace            Concepts.LpSpace
@api ErrorMetric        Concepts.ErrorMetric



@api RelativeError      Utilities.RelativeError
@api AbsoluteError      Utilities.AbsoluteError
@api within_radius      Utilities.within_radius






end # module


