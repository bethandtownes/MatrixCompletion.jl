module ADMM

using Printf
using LinearAlgebra
using SparseArrays



using ..Concepts
using ..ModelFitting
using ..Utilities
using ..Utilities.FastEigen

using ..Losses
using ..MathLib

import LinearAlgebra:norm



export complete






mutable struct RunHistory
  primfeas::Array{Float64,1}
  dualfeas::Array{Float64,1}
  
  function RunHistory()
    new(Array{Float64,1}(),Array{Float64,1}())
  end
end



import KrylovKit


# function project(v,e)
#     return e * diagm(0 => v) * e';
# end


function sdpProjection(mat::Array{Float64, 2})
  λ, X    = eigs(KrylovMethods(), mat, nev = 20)
  # @show(λ)
  # @show(X)
  # @show(size(λ))
  # @show(size(X))
  return project(λ, X)
  # posEigenValuesIndex = findall(x -> x > 0, λ);
  # posEigenValues      = λ[posEigenValuesIndex];
  # posEigenVectors     = X[:,posEigenValuesIndex];
  # projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
  # return projectedMatrix;
end

function sdpProjection0(data)
  eigDecomposition    = eigen(data);
  posEigenValuesIndex = findall(x -> x>0,eigDecomposition.values);
  posEigenValues      = eigDecomposition.values[posEigenValuesIndex];
  posEigenVectors     = eigDecomposition.vectors[:,posEigenValuesIndex];
  projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
  return projectedMatrix;
end

function logisticLoss(x,y)
  f_x = Losses.σ.(x);
  return -sum(y .* log.(f_x) + (1 .- y) .* log.(1 .- f_x));
end

function l1BallProjection(v,b)
  if (norm(v,1) <= b);
    return v
  end
  n = length(v);
  nvec = hcat(1:n...)'[:];
  vv = sort(abs.(v),lt = (x,y) -> !isless(x,y));
  idxsort = sortperm(abs.(v),lt = (x,y) -> !isless(x,y));
  vsum = cumsum(vv);
  tmp  = vv .-(vsum .- b)./nvec;
  idx  = findall(x->x>0,tmp);
  if !isempty(idx);
    k = maximum(idx);
  else
    println("something is wrong")
  end
  lam = (vsum[k] .- b) ./ k;
  xx = zeros(length(idxsort));
  xx[idxsort,1] = max.(vv .- lam,0);
  x = sign.(v).*xx;
  return x;
end

@private
function initialize_warmup(tracker)
  warmup = Dict{Symbol, Array{Float64}}()
  if haskey(tracker, :Bernoulli) && length(tracker[:Bernoulli][:Observed]) > 0
    warmup[:Bernoulli] = rand(length(tracker[:Bernoulli][:Observed]), 1)
  end
  if haskey(tracker, :Poisson) && length(tracker[:Poisson][:Observed]) > 0
    warmup[:Poisson] = rand(length(tracker[:Poisson][:Observed]), 1)
  end
  if haskey(tracker, :Gamma) && length(tracker[:Gamma][:Observed]) > 0
    warmup[:Gamma] = rand(length(tracker[:Gamma][:Observed]), 1)
  end
  return warmup
end

@private 
function update(::Type{Val{:Gaussian}}, A::Array{MaybeMissing{Float64}},
                Y12::Array{Float64}, tracker::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}}, ρ::Float64)
  if haskey(tracker, :Gaussian) && length(tracker[:Gaussian][:Observed]) > 0
    Y12[tracker[:Gaussian][:Observed]] .= (1 / (1 + ρ)) * (A[tracker[:Gaussian][:Observed]] + ρ * Y12[tracker[:Gaussian][:Observed]])
  end
end


@private
function update(::Type{Val{:Bernoulli}}, A::Array{MaybeMissing{Float64}},
                Y12::Array{Float64}, tracker::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ::Float64, gd_iter::Int64, warmup::Dict{Symbol, Array{Float64}}, γ::Float64, use_autodiff::Bool)
  if haskey(tracker, :Bernoulli) && length(tracker[:Bernoulli][:Observed]) > 0
    Y12[tracker[:Bernoulli][:Observed]] = train(use_autodiff ? provide(Loss{AbstractBernoulli}()) : Loss{AbstractBernoulli}(),
                                                fx   = warmup[:Bernoulli],
                                                y    = A[tracker[:Bernoulli][:Observed]],
                                                c    = Y12[tracker[:Bernoulli][:Observed]],
                                                ρ    = ρ,
                                                iter = gd_iter,
                                                γ    = 0.2)
    warmup[:Bernoulli] = Y12[tracker[:Bernoulli][:Observed]]
  end
end

@private
function update(::Type{Val{:Poisson}},
                A            ::Array{MaybeMissing{Float64}},
                Y12          ::Array{Float64},
                tracker      ::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ            ::Float64,
                gd_iter      ::Int64,
                warmup       ::Dict{Symbol, Array{Float64}},
                γ            ::Float64,
                use_autodiff ::Bool)
  if haskey(tracker, :Poisson) && length(tracker[:Poisson][:Observed]) > 0
    Y12[tracker[:Poisson][:Observed]] = train(use_autodiff ? provide(Loss{AbstractPoisson}()) : Loss{AbstractPoisson}(),
                                              fx   = warmup[:Poisson],
                                              y    = A[tracker[:Poisson][:Observed]],
                                              c    = Y12[tracker[:Poisson][:Observed]],
                                              ρ    = ρ,
                                              iter = gd_iter,
                                              γ    = 0.1)
    warmup[:Poisson] = Y12[tracker[:Poisson][:Observed]]
  end
end

@private
function update(::Type{Val{:Gamma}}, A::Array{MaybeMissing{Float64}},
                Y12::Array{Float64}, tracker::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ::Float64, gd_iter::Int64, warmup::Dict{Symbol, Array{Float64}}, γ::Float64, use_autodiff::Bool)
  if haskey(tracker, :Gamma) && length(tracker[:Gamma][:Observed]) > 0
    Y12[tracker[:Gamma][:Observed]] = train(use_autodiff ? provide(Loss{AbstractGamma}()) : Loss{AbstractGamma}(),
                                                fx   = warmup[:Gamma],
                                                y    = A[tracker[:Gamma][:Observed]],
                                                c    = Y12[tracker[:Gamma][:Observed]],
                                                ρ    = ρ,
                                                iter = gd_iter,
                                                γ    = 0.1)
    warmup[:Gamma] = Y12[tracker[:Gamma][:Observed]]
  end
end

@private
function update(::Type{Val{:ρ}}, primal_feasibility::Float64, dual_feasibility::Float64, current::Float64 = nothing)
  if (primal_feasibility < 0.5 * dual_feasibility)
    return 0.7 * current
  elseif (primal_feasibility > 2 * dual_feasibility)
    return 1.3 * current
  end
  return current
end

@private
function balance_gap(ρ::Float64, primal_feasibility::Float64, dual_feasibility::Float64)
  if (primal_feasibility < 0.5 * dual_feasibility)
    return 0.7 * ρ
  elseif (primal_feasibility > 2 * dual_feasibility)
    return 1.3 * ρ
  end
  return ρ
end

# update
const update(arg::Symbol, args...) = update(Val{arg}, args...)

@private
function calculate_primal_and_dual_residual(X, Z, W, C, Xinput, ρ)
  Fnorm = x -> norm(x,2);
  Xdual    = -ρ * (X - Xinput)
  Zdual    = ρ * (Z - C)
  normX    = 1 + Fnorm(X)
  primfeas = Fnorm(X - Z) / normX
  err1     = 1/ρ * Fnorm(W - Xdual)
  err2     = 1/ρ * Fnorm(W - Zdual)
  dualfeas = maximum([err1, err2]) / normX
  return primfeas, dualfeas
end

function set_diagonal(mat::Array{Float64, 2}, val::Array{Float64, 1})
  [mat[i, i] = val[i] for i in 1:length(val)]
end

function calculate_Z12_update(A, C,tracker, ρ, α, warmup, use_autodiff, gd_iter)
  d1, d2 = size(A)
  Z12 = C[1:d1, (d1+1):(d1+d2)]
  update(:Gaussian,  A, Z12, tracker, ρ)
  update(:Bernoulli, A, Z12, tracker, ρ, gd_iter, warmup, 0.2, use_autodiff)
  update(:Poisson,   A, Z12, tracker, ρ, 20, warmup, 0.05, use_autodiff)
  update(:Gamma,     A, Z12, tracker, ρ, 1000, warmup, 0.005, use_autodiff)
  project!(ClosedInterval{Float64}(-α, α), Z12)
  # Z12 .= project(ClosedInterval{Float64}(-α, α), Z12)
  # @. Z12      = max.(-α, min.(Z12, α))
  return Z12
end

function set_block_12(mat, d1, d2, val)
  @. mat[1:d1, (d1+1):(d1+d2)] = val
end

function set_block_21(mat, d1, d2, val)
  @. mat[(d1+1):(d1+d2), 1:d1] = val
end

@private
function initialize_trackers(A::Array{MaybeMissing{Float64}}, type_assignment)
  type_tracker = Utilities.IndexTracker{Symbol}()
  if type_assignment == nothing
    disjoint_join(type_tracker, Concepts.fit(DistributionDeduction(), A))
  else
    disjoint_join(type_tracker, type_assignment)
  end
  disjoint_join(type_tracker, Concepts.fit(ObservedOrMissing(), A))
  return groupby(type_tracker, [:Observed, :Missing]), type_tracker
  # tracker = groupby(type_tracker, [:Observed, :Missing])
  # return tracker 
end

@private
function ensure_feasible(A::Array{MaybeMissing{Float64}})
    if isnothing(A)
    @error("please provide data matrix")
    throw(MethodError())
  end
end


# function (str)
#     return rpad(str,70,".")
# end
function append_both_ends(str::String, token::String)
  return token * str * token
end

@overload
function Base.similar(str::String, token::Char)
  return token^length(str)
end

function toprule(header_list::Array{String})
  # new_rule = map(x -> Base.repeat("-", append_both_ends(str, " ")), header_list)
  new_rule = map(x -> Base.similar(append_both_ends(x, " "), '-'), header_list)
  push!(new_rule, "") 
  new_rule2 = foldl((x, y) -> x * "+" *y, new_rule, init="")
  @printf("%s\n", new_rule2)
end



@private
function print_optimization_log(iter,A, X, Z, Z12, W, II, Rp, Rd, ρ, λ, μ, tracker)

  
  R = abs.(maximum(diag(Z)))
  gaussian_loss = norm(Z12[tracker[:Gaussian][:Observed]] -  A[tracker[:Gaussian][:Observed]])^2

  # header_list = ["Iter", "R(primal)", " R(dual)",  "ℒ(Gaussian)", "ℒ(Bernoulli)", "ℒ(Poisson)", "ℒ(Gamma)", "λ‖diag(Z)‖ᵢ", "μ⟨I, X⟩", "‖Z₁₂‖ᵢ"]
  

  # toprule(header_list)
  # @printf("+------+-----------+---------+-------------+--------------+------------+----------+-------------+---------+--------+\n")
  # @printf("| Iter | R(primal) | R(dual) | ℒ(Gaussian) | ℒ(Bernoulli) | ℒ(Poisson) | ℒ(Gamma) | λ‖diag(Z)‖ᵢ | μ⟨I, X⟩ | ‖Z₁₂‖ᵢ |\n")
  # @printf("+------+-----------+---------+-------------+--------------+------------+----------+-------------+---------+--------+\n")
  # @printf("| %3.0f |", iter)
  # @printf("Loss{Gaussian}: %3.2e\n", gaussian_loss)

  
  # obj1 = norm(Z12[tracker[:Gaussian][:Observed]] -  A[tracker[:Gaussian][:Observed]])^2
  #             + logisticLoss(Z12[tracker[:Bernoulli][:Observed]], A[tracker[:Bernoulli][:Observed]]);
  obj1 = 0
  obj2 = λ * R
  obj3 = μ * tr(II * X)
  maxZ12 = maximum(abs.(Z12))
  @printf("\n %3.0f %3.2e %3.2e| %3.2e %5.2f %5.2f %3.2e| %3.2e|", iter, Rp, Rd, λ, R, maxZ12, μ, ρ)
  @printf("| obj1: %3.2e obj2: %3.2e obj3: %3.2e|", obj1, obj2, obj3)
end

function complete(;A::Array{MaybeMissing{Float64}}   = nothing,
                  α::Float64         = maximum(A[findall(x -> !ismissing(x),A)]),
                  λ::Float64         = 5e-1,
                  μ::Float64         = 5e-4,
                  ρ::Float64         = 0.3,
                  τ::Float64         = 1.618,
                  maxiter::Int64     = 200,
                  stoptol::Float64   = 1e-5,
                  use_autodiff::Bool = false,
                  gd_iter::Int64     = 50,
                  debug_mode::Bool   = false,
                  interactive_plot   = false,
                  type_assignment    = nothing,
                  dynamic_ρ          = true)
  ensure_feasible(A)
  d1, d2                               = size(A);
  Z::Array{Float64, 2}                 = zeros(d1 + d1,  d1 + d2)
  X::Array{Float64, 2}                 = zeros(d1 + d1,  d1 + d2)
  W::Array{Float64, 2}                 = zeros(d1 + d1,  d1 + d2)
  C::Array{Float64, 2}                 = zeros(d1 + d1,  d1 + d2)
  Xinput::Array{Float64, 2}            = zeros(d1 + d1,  d1 + d2)
  II                                   = sparse(1.0I, d1 + d2, d1 + d2)
  tracker, type_tracker                = initialize_trackers(A, type_assignment)
  warmup::Dict{Symbol, Array{Float64}} = initialize_warmup(tracker)

  for iter = 1:maxiter
    @. Xinput = Z + W/ρ
    # step 1
    @time X = project(SemidefiniteCone(rank = 20), Z + W / ρ - (μ / ρ) * II)
    # Step 2
    @. C = X - 1/ρ * W; @. Z = C
    Z12 = calculate_Z12_update(A, C, tracker, ρ, α, warmup, use_autodiff, gd_iter)
    set_block_12(Z, d1, d2, Z12)
    set_block_21(Z, d1, d2, Z12')
    set_diagonal(Z, diag(C) - (λ / ρ) * l1BallProjection(diag(C) * ρ / λ, 1))
    # step 3
    @. W = W + τ * ρ * (Z - X)
    # calculation of dual program
    # primfeas, dualfeas = calculate_primal_and_dual_residual(X, Z, W, C, Xinput, ρ)
    if rem(iter, 10)==1
      primfeas, dualfeas = calculate_primal_and_dual_residual(X, Z, W, C, Xinput, ρ)
      print_optimization_log(iter, A, X, Z, Z12, W, II, primfeas, dualfeas, ρ, λ, μ, tracker)
      # R = abs.(maximum(diag(Z)))
      # # obj1 = norm(Z12[tracker[:Gaussian][:Observed]] -  A[tracker[:Gaussian][:Observed]])^2
      # #             + logisticLoss(Z12[tracker[:Bernoulli][:Observed]], A[tracker[:Bernoulli][:Observed]]);
      # obj1 = 0
      # obj2 = λ * R
      # obj3 = μ * tr(II * X)
      # maxZ12 = maximum(abs.(Z12))
      # @printf("\n %3.0f %3.2e %3.2e| %3.2e %5.2f %5.2f %3.2e| %3.2e|", iter, primfeas, dualfeas, λ, R, maxZ12, μ, ρ)
      # @printf("| obj1: %3.2e obj2: %3.2e obj3: %3.2e|", obj1, obj2, obj3)

      if dynamic_ρ
        ρ = balance_gap(ρ, primfeas, dualfeas)
      end
    end
    # if dynamic_ρ && (rem(iter, 10) == 0)
    #   ρ = balance_gap(ρ, primfeas, dualfeas)
    # end
  end
  completedMatrix = C[1:d1, (d1+1):(d1+d2)]
  return completedMatrix, type_tracker, tracker
end

end




# early exit
    # if (max(primfeas, dualfeas) < stoptol) | (iter == maxiter) 
    #   breakyes = 1
    # end
    # if (max(primfeas, dualfeas) < sqrt(stoptol)) & (dualfeas > 1.5 * minimum(runhist.dualfeas[max(iter - 49, 1):iter])) & (iter > 150)
    #   breakyes = 2
    # end


# tune ρ
    # if (ρReset > 0) & (rem(iter, 10)==0)
    #   if (primfeas < 0.5 * dualfeas)
    #     ρ = 0.7 * ρ
    #   elseif (primfeas > 2 * dualfeas)
    #     ρ = 1.3 * ρ
    #   end
    # end
    # if (breakyes > 0)
    #   @printf("\n break = %1.0f\n", breakyes)
    #   break;
    # end
