module ModelFitting
using ..Concepts
using ..BetterMGF
using ..Estimator
import LinearAlgebra

abstract type FittingMethod end

@overload
function Concepts.check(object::Union{Type{Val{:continuous}},Continuous},
                        data::Array{T}) where T<:Real
    dist_to_integral_cast = LinearAlgebra.norm(data .- round.(data),2)
    return dist_to_integral_cast > 1e-2
end


@overload
function Concepts.check(object::Union{Type{Val{:integral}},Categorical},
                        data::Array{T}) where T<:Real
    dist_to_integral_cast = LinearAlgebra.norm(data .- round.(data),2)
    return dist_to_integral_cast <= 1e-2
end


@overload
function Concepts.check(object::Union{Type{Binary},
                                      Type{Val{:binary}},
                                      Binary},
                        data::Array{T}) where T<:Real
    return length(unique(data)) == 2
end


@overload
function Concepts.check(object::Union{Type{Support},
                                      Type{Val{:support}},
                                      Support},
                        data::Array{T};
                        layout::Symbol = :flatten) where T<:Real
    if layout == :flatten
        return (min.(data[:]), max.(data[:]))
    end
    # TODO: implement for other data layouts
end

# for now we only compare with Gamma. More smooth distributions will be added later.
@overload
function Concepts.check(object::Union{Type{Val{:Gaussian}},
                                      Type{AbstractGaussian},
                                      AbstractGaussian},
                        data::Array{T}) where T<:Real
    if check(:continuous,data) == false
        return false
    end
    # check against gamma when we get positive support
    if check(:support,data) > (0,0) && choose(Gaussian,Gamma) == :Gamma
        return false
    end
    return true
end


@overload
function Concepts.check(object::Union{Type{Val{:Gamma}},
                                      Type{AbstractGamma},
                                      AbstractGamma},
                        data::Array{T}) where T<:Real
    if check(:continuous,data) == false
        return false
    end
    if check(:support,data) < (0,0)
        return false
    end
    if choose(Gaussian,Gamma) == :Gaussian
        return false
    end
    return true
end


# For now we let non binary categorical array be Poisson. We will refine this later.
@overload
function Concepts.check(object::Union{Type{Val{:Poisson}},
                                      Type{AbstractPoisson},
                                      AbstractPoisson},
                        data::Array{T}) where T<:Real
    if check(:integral,data) == false
        return false
    end
    if check(:binary,data) == true
        return false
    end
    return true
end


@overload
function Concepts.check(object::Union{Type{Val{:Bernoulli}},
                                      Type{AbstractBernoulli},
                                      AbstractBernoulli},
                        data::Array{T}) where T<:Real
    if check(:integral,data) == false
        return false
    end
    if check(:binary,data) == false
        return false
    end
    return true
end


@overload
function Concepts.check(object::Union{Type{Val{:NegativeBinomial}},
                                      Type{AbstractNegativeBinomial},
                                      AbstractNegativeBinomial},
                        data::Array{T}) where T<:Real
    return false #TODO: implement this properly
end



@overload
function Concepts.choose(a::Union{AbstractPoisson,
                                  Type{AbstractPoisson},
                                  Type{Val{:Poisson}}},
                         b::Union{AbstractNegativeBinomial,
                                  Type{AbstractNegativeBinomial},
                                  Type{Val{:NegativeBinomial}}};
                         data::AutoboxedArray{Real} = nothing,
                         comp::Comparator = Comparator{MGF}(MGF()))
    # for now we return poisson all the time
    return :Poisson
end


# we choose to return symbol for maximum flexibility
@overload
function Concepts.choose(a::Union{AbstractGaussian,
                                  Type{AbstractGaussian},
                                  Type{Val{:Gaussian}}},
                         b::Union{AbstractGamma,
                                  Type{AbstractGamma},
                                  Type{Val{:Gamma}}};
                         data::AutoboxedArray{Real} = nothing,
                         comp::Comparator = Comparator{MGF}(MGF()))
    mle_est_gaussian = estimator(MLE{AbstractGaussian}(),data)
    mle_est_gamma    = estimator(MLE{AbstractGamma}(),data)
    t = nothing
    if isnothing(comp.field[:eval_at])
        if mle_est_gamma[:α] <= 15 && mle_est_gamma[:θ] <= 15
            t = collect(0.01:0.001:0.02)
        else
            t = collect(0:0.0001:0.001)
        end
    else
        t = comp.field[:eval_at]
    end
    empirical_mgf = evaluate(SampleMGF(),t, data=data, order=20)
    gaussian_mgf  = evaluate(MGF(:Gaussian),t,μ = mle_est_gaussian[:μ],σ=mle_est_gaussian[:σ])
    gamma_mgf     = evaluate(MGF(:Gamma),t,α=mle_est_gamma[:α],θ=mle_est_gamma[:θ])
    if check(:l2diff,empirical_mgf, gaussian_mgf) < check(:l2diff,empirical_mgf,gamma_mgf)
        return :Gaussian
    end
    return :Gamma
end


end
