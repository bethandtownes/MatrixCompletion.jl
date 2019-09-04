module BetterMGF
using ..Concepts
import LinearAlgebra


struct MGF{T<:Any}  <: AbstractMGF
    OPTION_LOG_SCALE::Optional{Bool}

    
    function MGF{T}() where T<:Any
        return new{T}()
    end
    
    function MGF{T}(object::T;logscale::Optional{Bool} = nothing) where T<:Any
        if isnothing(logscale)
            return new{T}()
        end
        return new{T}(logscale)
    end
end


const MGF() = MGF{Any}()

const MGF(object::Union{T, Symbol};logscale::Optional{Bool}=nothing) where T<:Any =
    !isa(object,Symbol) ? MGF{T}(object,logscale=logscale) : MGF(convert(object),logscale=logscale)
                                         

struct SampleMGF <: AbstractMGF
    OPTION_LOG_SCALE::Optional{Bool}

    
    function SampleMGF(;logscale::Optional{Bool} = nothing)
        if isnothing(logscale)
            return new()
        end
        return new(logscale)
    end
end

export MGF,
    SampleMGF

@overload
function Base.convert(::Type{FrequencyDomainObjects},x::Symbol) 
    if x == :MGF
        return MGF
    end
end


@overload
function Concepts.evaluate(object::MGF{AbstractPoisson},t::AutoboxedArray{Real};
                           λ::S) where S<:Real
    if λ <= 0
        throw(DomainError("λ should be a positive number"))
    end
    if !isnothing(object.OPTION_LOG_SCALE) && object.OPTION_LOG_SCALE == true
        @warn "[evaluate(MGF{Poisson}): using log-scale]"
        return λ .* (exp.(t) .- 1)
    end
    return exp.(λ .* (exp.(t) .- 1))
end


@overload
function Concepts.evaluate(object::MGF{AbstractBernoulli},t::AutoboxedArray{Real};
                           p::S) where S<:Real
    if p < 0 || p >1 
        throw(DomainError("p should be a positive number in [0,1]"))
    end
    if !isnothing(object.OPTION_LOG_SCALE) && object.OPTION_LOG_SCALE == true
        @warn "[evaluate(MGF{Bernoulli}): using log-scale]"
        return log.((1 .- p) .+ p .* exp.(t))
    end
    return (1 .- p) .+ p .* exp.(t)
end


@overload
function Concepts.evaluate(object::MGF{AbstractGamma},t::AutoboxedArray{Real};
                           α::S,θ::S) where S<:Real
    if α <= 1e-5 || θ <= 1e-5
        throw(DomainError("α and θ should be a positive number"))
    end
    if !isnothing(object.OPTION_LOG_SCALE) && object.OPTION_LOG_SCALE == true
        @warn "[evaluate(MGF{Gamma}): using log-scale]"
        return (-α) .* log.(1 .- t .* θ)
    end
    return (1 .- t .* θ) .^ (-α)
end

@overload
function Concepts.evaluate(object::MGF{AbstractGaussian},t::AutoboxedArray{Real};
                           μ::S = 0,σ::S=1) where S<:Real
    if abs(σ) <= 1e-4
        throw(DomainError("σ should be non-zero. Otherwise, it will be degenerate."))
    end
    if !isnothing(object.OPTION_LOG_SCALE) && object.OPTION_LOG_SCALE == true
        @warn "[evaluate(MGF{Gaussian}): using log-scale]"
        return t .* μ .+ σ^2 / 2 .* t.^2
    end
    return exp.(t .* μ .+ σ^2 / 2 .* t.^2)
end

import Distributions
@overload
function Concepts.evaluate(object::MGF{AbstractNegativeBinomial},t::AutoboxedArray{Real};
                           r::S = 0,p::S=1) where S<:Real
    if r <= 1e-5 || p <= 1e-5 || p >=1
        throw(DomainError("[Negative Binomial]: r should be > 0, p should be in (0,1)"))
    end
    if !isnothing(object.OPTION_LOG_SCALE) && object.OPTION_LOG_SCALE == true
        @warn "[evaluate(MGF{NegativeBinomial})]: using log-scale"
        return (r .* (log.(1 .- p) .+ t)) .- (r .* log.(1 .- p .* exp.(t)))
    end
    return ((1 .- p) .* exp.(t)).^r ./ (1 .- p .* exp.(t)).^r
end



@overload
function Concepts.evaluate(object::SampleMGF,t::VecOrMat{T};data::VecOrMatOf{Real},order::Integer = 8) where T <: Real
    if isa(t,Matrix) || isa(data,Matrix)
        throw(DomainError("Unimplemented"))
    end 
    return dp_get_sample_basis(t,order) * dp_get_sample_mean(data,order)
end


@private
function dp_get_denominator(order::Integer = 10)
    denom = ones(order+1)
    for i = 2:order+1
        denom[i] = denom[i-1]*(i-1)
    end
    return 1 ./ denom
end

@private
function dp_get_numerator(t::Array{S,1},order::Integer = 10) where S<:Real
    n = length(t)
    T = ones(n,order+1)
    for i = 2: order +1
        T[:,i] = T[:,i-1] .* t
    end
    return T
end

@private 
function dp_get_sample_basis(t::Array{S,1}, order::Integer = 10) where S<:Real
    num = dp_get_numerator(t,order)
    denom = dp_get_denominator(order)
    for i = 1:size(num)[1]
       num[i,:] = num[i,:] .* denom
    end
    return num
end

@private 
function dp_get_sample_mean(data::Array{S,1}, order::Integer = 10) where S<: Real
    ret = ones(order + 1)
    copy_of_data = deepcopy(data)
    n = length(copy_of_data)
    for i = 2:order + 1
        ret[i] = sum(copy_of_data)/n
        copy_of_data = copy_of_data .* data
    end
    return ret
end

end
