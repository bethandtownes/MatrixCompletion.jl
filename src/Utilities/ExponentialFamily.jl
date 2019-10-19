@overload
function Concepts.forward_map(distribution::Union{AbstractPoisson,Type{Val{:Poisson}}},
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing) where T<:Real
  if !isnothing(non_canonical_parameter)
    ## TODO
  end
  return exp.(canonical_parameter)    
end

@overload
function Concepts.forward_map(distribution::Union{AbstractGamma,Type{Val{:Gamma}}},
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing) where T<:Real
  if !isnothing(non_canonical_parameter)
    ## TODO
  end


  canonical_parameter = 1 ./ exp.(canonical_parameter) 
  return canonical_parameter
  # canonical_parameter = -1 .* abs.(canonical_parameter)
  # return -1 ./ canonical_parameter
  # return 1 ./ canonical_parameter
end

@overload
function Concepts.forward_map(distribution::Union{AbstractNegativeBinomial,Type{Val{:NegativeBinomial}}},
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing,
                              r_estimate = nothing) where T<:Real
  if !isnothing(non_canonical_parameter)
    ## TODO
  end
  return r_estimate ./ (exp.(exp.(canonical_parameter)) .- 1)
end

@overload
function Concepts.forward_map(distribution::Union{AbstractBernoulli,Type{Val{:Bernoulli}}},
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing) where T<:Real
  if !isnothing(non_canonical_parameter)
    ## TODO
  end
  ex = exp.(canonical_parameter)
  return ex ./ (1 .+ ex)
  #    return (Int.(sign.(canonical_parameter)) .+ 1) ./ 2
end

@overload
function Concepts.forward_map(distribution::Union{AbstractGaussian,Type{Val{:Gaussian}}},
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing) where T<:Real
  if !isnothing(non_canonical_parameter)
    ## TODO
  end
  return canonical_parameter
end


@overload
function Concepts.forward_map(distribution::Symbol,
                              canonical_parameter::Array{T};
                              non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                              non_canonical_map = nothing) where T<:Real
  return Concepts.forward_map(Val{distribution},canonical_parameter,
                              non_canonical_map=non_canonical_map,
                              non_canonical_parameter=non_canonical_parameter)
end

@overload
function Concepts.predict(distribution::Union{AbstractPoisson,Type{Val{:Poisson}}},mean::Any;
                          custom_prediction_function=nothing)
  if !isnothing(custom_prediction_function)
    return -1.0
  end
  return round.(mean)
end

@overload
function Concepts.predict(distribution::Union{AbstractBernoulli,Type{Val{:Bernoulli}}},mean::Any;
                          custom_prediction_function=nothing)
  if !isnothing(custom_prediction_function)
    return -1.0
  end
  return Int.(mean .> 0.5)
end

@overload
function Concepts.predict(distribution::Union{AbstractGaussian,Type{Val{:Gaussian}}},mean::Any;
                          custom_prediction_function=nothing)
  if !isnothing(custom_prediction_function)
    return -1.0
  end
  return mean
end

@overload
function Concepts.predict(distribution::Union{AbstractGamma,Type{Val{:Gamma}}},mean::Any;
                          custom_prediction_function=nothing)
  if !isnothing(custom_prediction_function)
    return -1.0
  end
  return mean
end

@overload
function Concepts.predict(distribution::Union{AbstractNegativeBinomial,Type{Val{:NegativeBinomial}}},mean::Any;
                          custom_prediction_function=nothing)
  if !isnothing(custom_prediction_function)
    return -1.0
  end
  return round.(mean)
end


@overload
const Concepts.predict(obj::Symbol,arg1;custom_prediction_function=nothing) =
  Concepts.predict(Val{obj},arg1;custom_prediction_function=custom_prediction_function)
