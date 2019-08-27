module Concepts
import LinearAlgebra
#import Distributions
#import Utilities.Indexing


# serves as a marker
macro overload() end

export @overload





const Optional{T}       = Union{T,Nothing}


const VecOrMatOfNumbers  = VecOrMat{<:Number}
const VecOrMatOfReals    = VecOrMat{<:Real}
const VecOrMatOfFloats   = VecOrMat{<:AbstractFloat}
const VecOrMatOfIntegers = VecOrMat{<:Integer}
const VecOrMatOf{T}      = VecOrMat{<:T} where T<:Any

const MaybeMissing{T}    = Union{Missing,T} where T<:Any

export Optional,
    VecOrMatOfReals,
    VecOrMatOfFloats,
    VecOrMatOfIntegers,
    VecOrMatOf,
    MaybeMissing




abstract type ExponentialFamily end
struct AbstractBinomial         <: ExponentialFamily         end
struct AbstractGaussian         <: ExponentialFamily         end
struct AbstractPoisson          <: ExponentialFamily         end
struct AbstractGamma            <: ExponentialFamily         end
struct AbstractExponential      <: ExponentialFamily         end
struct AbstractNegativeBinomial <: ExponentialFamily         end
struct AbstractGeometric        <: ExponentialFamily         end


struct Diagnostics{T<:Any} end
export Diagnostics


struct Loss{T} end
export Loss





@enum DIST_FLAGS Bernoulli Poisson Gaussian Gamma NegativeBinomial

export ExponentialFamily,
    AbstractBinomial,
    AbstractGaussian,
    AbstractGamma,
    AbstractExponential,
    AbstractPoisson,
    DIST_FLAGS,
    Bernoulli,
    Poisson,
    Gamma,
    NegativeBinomial,
    Gaussian


struct ErrorMetric end


struct LpSpace
    p::Real
    norm
    function LpSpace(p::Real)
        return new(p,
                   x -> LinearAlgebra.norm(x,p))
    end
end

struct SchattenClass end
struct BoundedLinearOperator end

export LpSpace,
    SchattenClass,
    BoundedLinearOperator


################################################################################
#                              Sampling Models                                 #
################################################################################


abstract type AbstractSamplingModels end
export AbstractSamplingModels,
    BernoulliModel,
    UniformModel,
    NonUniformModel

struct  BernoulliModel  <: AbstractSamplingModels
    rate::AbstractFloat
    function BernoulliModel()
        # AbstractType Constructor
        return new()
    end

    function BernoulliModel(rate::T) where T<:AbstractFloat
        return new(rate);
    end
end



struct UniformModel <: AbstractSamplingModels
    rate::AbstractFloat
    function UniformModel()
        #Abstract Type Constructor
        return new()
    end

    function UniformModel(rate::T) where T<:AbstractFloat
        return new(rate)
    end

end


struct NonUniformModel <: AbstractSamplingModels
    rate::AbstractFloat
    function NonUniformModel()
        #Abstract Type Constructor
        return new()
    end

    function NonUniformModel(rate::T) where T<:AbstractFloat
        return new(rate)
    end

end


abstract type AbstractRandomLowRankMatrix end









export provide,
    convert


function provide(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(DomainError("[provide]: calling abstract method. Concrete implementation needed"))
    return 0;
end



function is(is_object::T,arg...;kwargs...) where T<:Any
    throw(DomainError("[is]: calling abstract method. Concrete implementation needed"))
end



@overload
function Base.convert(::Type{MaybeMissing{S}},x::VecOrMatOf{T}) where {T<:Any,S<:Any}
    ret = nothing;
    if isa(x,Vector)
        ret = Vector{MaybeMissing{S}}(undef,length(x))
        for i in 1:length(x)
            ret[i] = x[i]
        end
    end
    if isa(x,Matrix)
#       print("haha")
        row, col = size(x)
        ret = Matrix{MaybeMissing{S}}(undef,row,col)
#        display(ret)
        for i in row
            for j in col
                ret[i,j] = x[i,j]
            end
        end
    end
    return ret;
end






end # Module: Concept

# function convert_to_flag(from::T) where T<:ExponentialFamily
#     if isa(from,AbstractBinomial)
#         return Indexing.Binomial
#     end
#     if isa(from,AbstractGamma)
#         return Indexing.Gamma
#     end
#     if isa(from,AbstractNegativeBinomial)
#         return Indexing.NegativeBinomial
#     end
#     if isa(from,AbstractGaussian)
#         return Indexing.Gaussian
#     end
#     if isa(from,AbstractBinomial)
#         return Indexing.Binomial
#     end
# end



# function check(match_type::AbstractGaussian;input::Array{Float64,2})

# end

# function check(match_type::AbstractBinomial;input::Array{Float64,2})

# end


# function check(match_type::AbstractPoisson;input::Array{Float64,2})

# end


# function check(match_type::AbstractNegativeBinomial;input::Array{Float64,2})

# end

# function check(match_type::AbstractGamma;input::Array{Float64,2})

# end
