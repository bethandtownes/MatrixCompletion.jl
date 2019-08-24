module Concepts

#import Distributions
#import Utilities.Indexing


abstract type ExponentialFamily end 
struct AbstractBinomial         <: ExponentialFamily         end
struct AbstractGaussian         <: ExponentialFamily         end
struct AbstractPoisson          <: ExponentialFamily         end
struct AbstractGamma            <: ExponentialFamily         end
struct AbstractExponential      <: ExponentialFamily         end
struct AbstractNegativeBinomial <: ExponentialFamily         end
struct AbstractGeometric        <: ExponentialFamily         end


abstract type AbstractDiagnostics end

struct Diagnostics{T} end
struct Loss{T} end


export Loss 



struct ErrorMetric{T} end



struct LpSpace
    p::Real
    function LpSpace(p::Real)
        return new(p)
    end
end


export LpSpace


#struct L0Distance <: ErrorMetric end
#struct L2Distance <: ErrorMetric end
#struct L1Distance <: ErrorMetric end


@enum DIST_FLAGS Bernoulli Poisson Gaussian Gamma NegativeBinomial


# Abstract Interface #




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


export ErrorMetric,
    L1Distance,
    L2Distance,
    L0Distance,
    Diagnostics

#export convert_to_flag,
#    check
        
    

function provide(object::T=nothing,arg...;kwargs...) where T<:Any
    throw(DomainError("[provide]: calling abstract method. Concrete implementation needed"))
    return 0;
end



end


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





