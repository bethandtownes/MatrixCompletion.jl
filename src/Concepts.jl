module Concepts
import LinearAlgebra
import Distributions

using Printf


#==============================================================================#
#                         Customized Exceptions                                #
#==============================================================================#
struct UnrecognizedSymbolException <: Exception
    symbol_name::String
end

Base.showerror(io::IO,e::UnrecognizedSymbolException) =
    @printf(io,"[%s]: Unrecognized symbol",e.symbol_name)


struct NotOverLoadedException <: Exception
    fcn_name::String
end

Base.showerror(io::IO,e::NotOverLoadedException) =
    @printf(io,"[%s]: Abstract method called. Concrete implementation is required.",e.fcn_name)


struct Unimplemented <: Exception
    fcn_name::String
end

Base.showerror(io::IO,e::Unimplemented) =
    @printf(io,"[%s]: Notimplemented",e.fcn_name)


export UnrecognizedSymbolException,
    NotOverLoadedException,
    Unimplemented




#==============================================================================#
#                            Customized Macros                                 #
#==============================================================================#
# Serves as a annotator. So far it is sufficient. We can always add actual defi-
# nitions later if we want to enforce the property. I don't think Julia currently
# supports private functions.. It maybe possible with some dark-art style trick..
macro private() end
macro overload() end
macro abstract_instance() end


export @overload,
    @abstract_instance,
    @private




#==============================================================================#
#                           Type Declearations                                 #
#==============================================================================#
const Optional{T}        = Union{T,Nothing}
const VecOrMatOfNumbers  = VecOrMat{<:Number}
const VecOrMatOfReals    = VecOrMat{<:Real}
const VecOrMatOfFloats   = VecOrMat{<:AbstractFloat}
const VecOrMatOfIntegers = VecOrMat{<:Integer}
const VecOrMatOf{T}      = VecOrMat{<:T} where T<:Any
const MaybeMissing{T}    = Union{Missing,T} where T<:Any


const AutoboxedArray{T}  = Union{S,Array{S}} where S<:T

export Optional,
    VecOrMatOfReals,
    VecOrMatOfFloats,
    VecOrMatOfIntegers,
    VecOrMatOf,
    MaybeMissing,
    AutoboxedArray


global SYMBOL_LIST = Set([:Poisson,:Gaussian,:Gamma,:Bernoulli,:NegativeBinomial,
                          :Count,:Binary,
                          :dimension,:rank,:l2diff,:l2difference])


global AVAILABLE_DATA_LAYOUT = Set([:flatten,:bycol,:byrow,:asmatrix,:astensor])



#==============================================================================#
#                            Exponential Family                                #
#==============================================================================#
abstract type ExponentialFamily end
struct AbstractBernoulli        <: ExponentialFamily   end
struct AbstractBinomial         <: ExponentialFamily   end
struct AbstractGaussian         <: ExponentialFamily   end
struct AbstractPoisson          <: ExponentialFamily   end
struct AbstractGamma            <: ExponentialFamily   end
struct AbstractExponential      <: ExponentialFamily   end
struct AbstractNegativeBinomial <: ExponentialFamily   end
struct AbstractGeometric        <: ExponentialFamily   end



export ExponentialFamily,
    AbstractBernoulli,
    AbstractBinomial,
    AbstractGaussian,
    AbstractGamma,
    AbstractExponential,
    AbstractNegativeBinomial,
    AbstractPoisson


function forward_map(distribution::T,args...;kwargs...) where T<:Any
    throw(NotOverLoadedException("forward_map"))
end

#------------------------------------------------------------------------------#
struct Diagnostics{T<:Any} end
export Diagnostics


abstract type AbstractLoss end
export AbstractLoss 



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



#==============================================================================#
#                                  Tracker                                     #
#==============================================================================#
abstract type AbstractTracker end
abstract type AbstractView    end
struct Continuous   end
struct Categorical  end
struct Binary       end
struct Support      end 




export Continuous,
    Categorical,
    AbstractTracker,
    Binary,
    Support,
    AbstractView



    



#==============================================================================#
#                              Sampling Models                                 #
#==============================================================================#
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

#==============================================================================#
#                          Frequency Domain Objects                            #
#==============================================================================#
abstract type FrequencyDomainObjects end
abstract type AbstractMGF <: FrequencyDomainObjects end

export FrequencyDomainObjects,
    AbstractMGF


#==============================================================================#
#                                 Estimator                                    #
#==============================================================================#
abstract type AbstractEstimator end

function estimator(of::T=nothing,arg...;kwargs...) where T<:Any
    throw(NotOverLoadedException("estimator"))
end

export AbstractEstimator,
    estimator



#==============================================================================#
#                                 Comparator                                   #
#==============================================================================#
abstract type AbstractComparator end

struct Comparator{T<:Any} <:AbstractComparator
    field::Optional{Dict}
    
    @abstract_instance
    function Comparator{T}() where T<:Any
        return new{T}(nothing)
    end

    function Comparator{T}(by::Type{T}) where T<:Any
        return new{T}(nothing)
    end
    
    function Comparator{T}(by::T) where T<:Any
        return new{T}(nothing)
    end

    function Comparator{T}(by::T;eval_at::Optional{AutoboxedArray{Real}}=nothing) where T<:AbstractMGF
        return new{T}(Dict(:eval_at => eval_at))
    end
end

const Comparator(of::T;eval_at::Optional{AutoboxedArray{Real}}=nothing) where T<:AbstractMGF = begin
    Comparator{T}(of,eval_at=eval_at)
end

const Comparator(of::T) where T<:Any = begin
    if !isa(of,Symbol)
        if isa(of,Type)
            Comparator{of}(of)
        else
            Comparator{T}(of)
        end
    else
        Comparator(convert(of))
    end
end

export AbstractComparator,
    Comparator

#==============================================================================#
#                              Random Structures                               #
#==============================================================================#
const UnivariateDistributions =
    Distributions.Distribution{Distributions.Univariate,S} where S<:Distributions.ValueSupport

abstract type AbstractFixedRankMatrix end 


export UnivariateDistributions,
    AbstractFixedRankMatrix

#==============================================================================#
#                              Abstract Inteface                               #
#==============================================================================#
export provide,
    convert,
    predict,
    check,
    evaluate,
    choose


function provide(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(DomainError("[provide]: calling abstract method. Concrete implementation needed"))
    return 0;
end


function is(is_object::T,arg...;kwargs...) where T<:Any
    throw(DomainError("[is]: calling abstract method. Concrete implementation needed"))
end


function check(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(DomainError("[check]: calling abstract method. Concrete implementation needed"))
end


function predict(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(NotOverLoadedException("predict"))
end


function evaluate(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(NotOverLoadedException("evaluate"))
end


function choose(arg...;kwargs...) where T<:Any
    throw(NotOverLoadedException("evaluate"))
end


#==============================================================================#
#                             Base Overrides                                   #
#==============================================================================#
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
        row, col = size(x)
        ret = Matrix{MaybeMissing{S}}(undef,row,col)
        for i in row
            for j in col
                ret[i,j] = x[i,j]
            end
        end
    end
    return ret;
end


# emacs indentation BUGGGGGGGGGG
@overload
function Base.convert(::Type{ExponentialFamily},x::Symbol)
    if x == :Poisson || x == :Count
        return AbstractPoisson()
    end
    if x == :Gaussian || x == :Normal
        return AbstractGaussian()
    end
    if x == :Bernoulli || x == :Binary
        return AbstractBernoulli()
    end
    if x == :Gamma
        return AbstractGamma()
    end
    if x == :NegativeBinomial
        return AbstractNegativeBinomial()
    end
    throw(InexactError())
end


@overload
function Base.convert(x::Symbol) 
    try return Base.convert(ExponentialFamily,x)      catch  end
    try return Base.convert(FrequencyDomainObjects,x) catch  end
    throw(UnrecognizedSymbolException(String(x)))
end





end # Module: Concept


