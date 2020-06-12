module Losses

#==============================================================================#
#                          MODULE OPTIONS & FLAGS                              #
const DEBUG_MODE   = false
#==============================================================================#
const VERBOSE_MODE = true
#------------------------------------------------------------------------------#



using Printf
import Random, AutoGrad, Distributions
using ..Concepts
using ..Utilities.BatchUtils
# export BatchFactory

# struct SGD end



####################### HELPER METHODS ###################
# vectorized sigmoid function
σ(z) = 1.0 ./ (1.0 .+ exp.(-z))
##########################################################



# function Poisson()
#     L(x,y,c,ρ) = sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
#     return L;
# end


# function Logistic()
#     L(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);



#     return L;
# end


export Loss,
  train,
  sgd_train,
  subgrad_train,
  negative_binomial_train

struct Loss{T} <: AbstractLoss where T<:Any
  function Loss{T}() where T<:Any
    @abstract_instance
    return new{T}()
  end
  
  #    function Loss{AbstractPoisson}(of::Union{AbstractPoisson,Type{Val{:Poisson}}})
  function Loss{T}(of::T) where T<:ExponentialFamily
    return new{T}()
  end
end


# kind of hackish.. the second one just defined in the first.
const Loss(of::Union{T,Symbol}) where T<:ExponentialFamily =
  typeof(of) <: ExponentialFamily ? Loss{T}(of) : Loss(type_conversion(ExponentialFamily, of))

loss_logistic(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);



 #==============================================================================#
#                             Gaussian Loss                                    #
#==============================================================================#
function Concepts.provide(loss::Loss{AbstractGaussian})    
  #TODO
end


function Concepts.evaluate(loss::Loss{AbstractGaussian},
                  x,y,c,ρ)
  #TODO
  return sum(0.5 .* (y .- x).^2) + sum(ρ .* (x .- c).^2);
end


function grad(loss::Loss{AbstractGaussian} ,
              x,y,c,ρ)
  #TODO
  return -(y .- x) .+ (2*ρ) .* (x .- c)
end
#==============================================================================#
#                         Bernoulli (Logistic) Loss                            #
#==============================================================================#
function Concepts.provide(loss::Loss{AbstractBernoulli})
  L(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);
  return L
end


function Concepts.evaluate(loss::Loss{AbstractBernoulli},
                           x,y,c,ρ)
  return  -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x)))
  .+ sum(ρ .* (x .- c).^2);
end


function grad(loss::Loss{AbstractBernoulli},
              x,y,c,ρ)
  ex = exp.(x)
  inv_ex1 = 1 ./(ex .+ 1)
  return inv_ex1 .* (-y + (1 .-y) .* ex) .+ (2*ρ) .* (x .- c)
end

#==============================================================================#
#                             Poisson Loss                                     #
#==============================================================================#
@overload
function Concepts.provide(loss::Loss{AbstractPoisson})
  L(x,y,c,ρ) = sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
  return L;
end


function Concepts.evaluate(loss::Loss{AbstractPoisson},
                           x,y,c,ρ)
  return sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
end


function grad(loss::Loss{AbstractPoisson},
              x,y,c,ρ)
  return exp.(x) .- y .+ (2*ρ) .* (x .- c)
end

#==============================================================================#
#                              Gamma Loss                                      #
#==============================================================================#
function Concepts.provide(loss::Loss{AbstractGamma})
  L(x,y,c,ρ) = sum(-x .* y .- log.(-x))+ sum(ρ .* (x .- c).^2)
  return L
end


# function Concepts.evaluate(loss::Loss{AbstractGamma},x,y,c,ρ)
#     return sum(x .* y .- log.(x))+ sum(ρ .* (x .- c).^2)
# end


function Concepts.evaluate(loss::Loss{AbstractGamma},x,y,c,ρ)
  return sum(y .* exp.(x) .- x) + sum(ρ .* (x .- c).^2)
end

# function _evaluate(loss::Loss{AbstractGamma},x,y,c,ρ)
#   x₋ = x[findall(a -> a < 0, x)]
#   y₋ = y[findall(a -> a < 0, x)]
#   x₊ = x[findall(a -> a > 0, x)]
#   y₊ = x[findall(a -> a > 0, x)]
#   return (-1) * sum(x₋ .* y₋ .+ log.(-x₋)) + (-1) * sum(-x₊ .* y₊ .+ log.(x₊)) + sum(ρ .* (x .- c).^2)
# end


## Use the reciprocal link instead of the negative reciprocal link
# function grad(loss::Loss{AbstractGamma}, x, y, c, ρ)
#     return y .- (1 ./ x) .+ (2*ρ) .* (x .- c)
# end


function grad(loss::Loss{AbstractGamma}, x, y, c, ρ)
  return y .* exp.(x) .- 1 + (2*ρ) .* (x .- c)
end



function subgrad(loss::Loss{AbstractGamma}, x, y, c, ρ)
  ∇ = zeros(length(x))
  pos_id = findall(a -> a > 0, x)
  neg_id = findall(a -> a < 0, x)
  ∇[neg_id] = (-1) .* (y[neg_id] - (1 ./ x[neg_id])) .+ (2*ρ) .* (x[neg_id] .- c[neg_id])
  ∇[pos_id] = (-1) .* (-y[pos_id] + (1 ./ x[pos_id])) .+ (2*ρ) .* (x[pos_id] .- c[pos_id])
  return ∇
end


function grad_logistic(x,y,c,ρ)
  ex = exp.(x)
  inv_ex1 = 1 ./(ex .+ 1);
  return inv_ex1 .* (-y + (1 .-y) .* ex) .+ (2*ρ) .* (x .- c);
  # return (-y .* inv_ex1 + (1 .- y) .* (ex .* inv_ex1)) .+ (2*ρ) .* (x.-c);
end

#==============================================================================#
#                         Negative Binomial Loss                               #
#==============================================================================#
function Concepts.provide(loss::Loss{AbstractNegativeBinomial})
  ## to implement
  return nothing
end

function Concepts.evaluate(loss::Loss{AbstractNegativeBinomial}, x, y, c, ρ; r_estimate)
  return sum(y .* exp.(x) - r_estimate .* log.(1 .- exp.(-exp.(x)))) .+ sum(ρ .* (x .- c).^2)
  # originally it is. However, this is a constrained optimization problem.
  # return sum(-y .* x .- r_estimate .* log.(1 .- exp.(x))) + sum(ρ .* (x .- c).^2)
end

function grad(loss::Loss{AbstractNegativeBinomial}, x, y, c, ρ; r_estimate = nothing)
  return y .* exp.(x) .- r_estimate .* exp.(x) ./ (exp.(exp.(x)) .- 1)  + (2*ρ) .* (x .- c)
end




function train(loss;fx,y,c,ρ,γ=0.02,iter=20,verbose=false)
  DEBUG_MODE && @info "Gradient Descent with Autograd"
  ∇ = AutoGrad.grad(loss);
  curFx = fx;
  for i = 1:iter
    curFx = curFx .- γ .* ∇(curFx,y,c,ρ);
  end
  return curFx;
end


function train(native_loss::Loss{T};
               fx, y, c, ρ, γ=0.02, iter=20, verbose=false, subgrad = false) where T<:ExponentialFamily
  DEBUG_MODE && @info "Gradient Descent with native differentitaion"
  curFx = fx;
  for i = 1:iter 
    curFx .-= γ * grad(native_loss, curFx, y, c, ρ);
    # curFx .-= γ * grad(native_loss, curFx, y, c, ρ);
    # if project == true
    #   curFx = abs.(curFx)
    # end
    if verbose == true
      @printf("loss:%f\n", Concepts.evaluate(native_loss,curFx,y,c,ρ ))
    end
  end
  return curFx;
end



# specialization for negative binomial loss
function negative_binomial_train(;fx, y, c, ρ, γ=0.02, iter=20, verbose=false, r_estimate = nothing) 
  DEBUG_MODE && @info "Gradient Descent with native differentitaion"
  curFx = fx;
  for i = 1:iter
    # @show("here")
    curFx .-= γ * grad(Loss{AbstractNegativeBinomial}(), curFx, y, c, ρ;r_estimate = r_estimate);
    if verbose == true
      @printf("loss:%f\n", Concepts.evaluate(Loss{AbstractNegativeBinomial}(),
                                             curFx,y,c,ρ; r_estimate = r_estimate))
    end
  end
  return curFx;
end



function subgrad_train(native_loss::Loss{T};
                       fx, y, c, ρ, γ=0.02, iter=20, verbose=false) where T<:ExponentialFamily
  DEBUG_MODE && @info "Gradient Descent with native differentitaion"
  curFx = fx;
  for i = 1:iter 
    curFx .-= γ * subgrad(native_loss, curFx, y, c, ρ);
    if verbose == true
      @printf("loss:%f\n",_evaluate(native_loss,curFx,y,c,ρ ))
    end
  end
  return curFx;
end




function sgd_train(native_loss::Loss{T};
                   fx, y, c, ρ, α, ρ₁, ρ₂, batch_size, epoch) where T<:ExponentialFamily
  n = length(fx)
  curFx = fx
  batch = BatchFactory{SequentialScan}(size = batch_size)
  initialize(batch, fx)
  s = zeros(batch_size)
  r = zeros(batch_size)
  ŝ = zeros(batch_size)
  r̂ = zeros(batch_size)
  for i in 1:epoch
    while has_next(batch)
      cur_batch = next(batch)
      ∇ₛₐₘₚₗₑ = grad(native_loss, curFx[cur_batch], y[cur_batch], c[cur_batch], ρ)
      s .= (ρ₁ .* s) .+ (1 .- ρ₁) .* ∇ₛₐₘₚₗₑ
      r .= (ρ₂ .* r) .+ (1 .- ρ₂) .* (∇ₛₐₘₚₗₑ.^2)
      ŝ .= s ./ (1 - ρ₁^i)
      r̂ .= r ./ (1 - ρ₂^i)
      # @show(r̂)
      curFx[cur_batch] = curFx[cur_batch] - α ./ sqrt.(r̂) .* ŝ
    end
    # @show(Concepts.evaluate(native_loss,curFx,y,c,ρ))
    reset(batch)
  end
  return curFx
end




function sgd_subgrad_train(native_loss::Loss{T};
                           fx, y, c, ρ, α, ρ₁, ρ₂, batch_size, epoch) where T<:ExponentialFamily
  n = length(fx)
  curFx = fx
  batch = BatchFactory{SequentialScan}(size = batch_size)
  initialize(batch, fx)
  s = zeros(batch_size)
  r = zeros(batch_size)
  ŝ = zeros(batch_size)
  r̂ = zeros(batch_size)
  for i in 1:epoch
    while has_next(batch)
      cur_batch = next(batch)
      ∇ₛₐₘₚₗₑ = subgrad(native_loss, curFx[cur_batch], y[cur_batch], c[cur_batch], ρ)
      s .= (ρ₁ .* s) .+ (1 .- ρ₁) .* ∇ₛₐₘₚₗₑ
      r .= (ρ₂ .* r) .+ (1 .- ρ₂) .* (∇ₛₐₘₚₗₑ.^2)
      ŝ .= s ./ (1 - ρ₁^i)
      r̂ .= r ./ (1 - ρ₂^i)
      # @show(r̂)
      curFx[cur_batch] = curFx[cur_batch] - α ./ sqrt.(r̂) .* ŝ
    end
    # @show(Concepts.evaluate(native_loss,curFx,y,c,ρ))
    reset(batch)
  end
  return curFx
end

end
