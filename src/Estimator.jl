module Estimator

using ..Concepts



import Distributions
import StatsBase


export MLE,
  ProfileLikelihood,
  MOM

abstract type EstimationProcedure{T<:Any} end


## not super important
struct MLE{T<:Any} <:Concepts.AbstractEstimator
    of::T 
    function MLE{T}() where T<:Any
        return new{T}()
    end

    function MLE{T}(of::T) where T<:Any
        return new{T}(of)
    end
end

## not super important
struct MOM{T<:Any} <:Concepts.AbstractEstimator
    of::T 
    function MOM{T}() where T<:Any
        return new{T}()
    end

    function MOM{T}(of::T) where T<:Any
        return new{T}(of)
    end
end


struct ProfileLikelihood <: EstimationProcedure{MLE} end


function Concepts.estimator(name::MOM{AbstractNegativeBinomial}, data::AutoboxedArray{T}) where T<:Real
  EX = StatsBase.mean(data)
  VarX = StatsBase.var(data)
  pₘₒₘ = EX / VarX
  rₘₒₘ = EX * (pₘₒₘ) / (1 - pₘₒₘ)
  return Dict{Symbol, Float64}(:p => pₘₒₘ, :r => rₘₒₘ)
end
                            



const MLE(of::Union{T,Symbol}) where T<:Any =
    !isa(of,Symbol) ? MLE{T}(of) : MLE(convert(of))





@overload
function Concepts.estimator(name::MLE{AbstractGaussian}, data::AutoboxedArray{T};
                            method::EstimationProcedure{MLE} = ProfileLikelihood()) where T<:Real

    if method == ProfileLikelihood()
        # for now, we use Distributions.jl. Will be replaced in the future.
        est = Distributions.fit_mle(Distributions.Gaussian,data)
        return Dict(:μ => est.μ, :σ => est.σ)
    end
end


@overload
function Concepts.estimator(name::MLE{AbstractGamma},data::AutoboxedArray{T};
                            method::EstimationProcedure{MLE} = ProfileLikelihood()) where T<:Real
    # default uses profile likelihood method
    if method == ProfileLikelihood()
        # for now, we use Distributions.jl. Will be replaced in the future.
        est = Distributions.fit_mle(Distributions.Gamma,data)
        return Dict(:α => est.α, :θ => est.θ)
    end
end


# data layout needs to be implemented properly

@overload
function Concepts.estimator(name::MLE{AbstractPoisson},data::AutoboxedArray{T};
                            method::EstimationProcedure{MLE} = ProfileLikelihood(),
                            data_layout::Symbol = :flatten) where T<:Real
    # default uses profile likelihood method
    if method == ProfileLikelihood()
           λ = sum(data)/length(data)
        return Dict(:λ => λ)
    end
end


@overload
function Concepts.estimator(name::MLE{AbstractBernoulli},data::AutoboxedArray{T};
                            method::EstimationProcedure{MLE} = ProfileLikelihood(),
                            data_layout::Symbol = :flatten) where T<:Real
    # default uses profile likelihood method
    if method == ProfileLikelihood()
           est = Distributions.fit_mle(Distributions.Bernoulli,data)
        return Dict(:p => est.p)
    end
end




end
