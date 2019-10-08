module ADMM

using Printf
using LinearAlgebra
using SparseArrays



using ..Concepts
using ..ModelFitting
using ..Utilities
using ..Losses

export complete

mutable struct RunHistory
  primfeas::Array{Float64,1}
  dualfeas::Array{Float64,1}
  
  function RunHistory()
    new(Array{Float64,1}(),Array{Float64,1}())
  end
end


function sdpProjection(data)
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
    Y12[tracker[:Gaussian][:Observed]] = (1 / (1 + ρ)) * (A[tracker[:Gaussian][:Observed]] + ρ * Y12[tracker[:Gaussian][:Observed]])
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
function update(::Type{Val{:Poisson}}, A::Array{MaybeMissing{Float64}},
                Y12::Array{Float64}, tracker::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ::Float64, gd_iter::Int64, warmup::Dict{Symbol, Array{Float64}}, γ::Float64, use_autodiff::Bool)
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





# update
const update(arg::Symbol, args...) = update(Val{arg}, args...)


function complete(;A::Array{MaybeMissing{Float64}}   = nothing,
                  α::Float64         = maximum(A[findall(x -> !ismissing(x),A)]),
                  λ::Float64         = 5e-1,
                  μ::Float64         = 5e-4,
                  σ::Float64         = 0.3,
                  τ::Float64         = 1.618,
                  maxiter::Int64     = 200,
                  stoptol::Float64   = 1e-5,
                  use_autodiff::Bool = false,
                  gd_iter::Int64     = 50,
                  debug_mode::Bool   = false,
                  interactive_plot   = false,
                  type_assignment    = nothing)
  if isnothing(A)
    @error("please provide data matrix")
    throw(MethodError())
  end
  d1,d2 = size(A);
  Y, X, W, C, R = [zeros(d1 + d2, d1 + d2) for i = 1:5]
  II = sparse(1.0I, d1 + d2, d1 + d2)
  Fnorm = x -> norm(x,2);
  _tracker = Utilities.IndexTracker{Symbol}()
  if type_assignment == nothing
    disjoint_join(_tracker, Concepts.fit(DistributionDeduction(), A))
  else
    disjoint_join(_tracker, type_assignment)
  end
  disjoint_join(_tracker, Concepts.fit(ObservedOrMissing(), A))
  @show(keys(_tracker.indices))
  tracker = groupby(_tracker, [:Observed, :Missing])
  # @show(tracker[:Poisson])
  σReset    = 1
  breakyes  = 0;
  runhist = RunHistory()
  warmup = initialize_warmup(tracker)
  
  for iter = 1:maxiter
    σInv     = 1/σ
    Xinput   = Y + σInv * W
    # Xinput   = Y + W / σ
    X        = sdpProjection(Y + W / σ - (1/σ * μ) * II)
    Xdual    = -σ * (X - Xinput)
    # Step 2
    C     = X - σInv * W
    Y12   = C[1:d1, (d1+1):(d1+d2)]
    update(:Gaussian,  A, Y12, tracker, σ)
    update(:Bernoulli, A, Y12, tracker, σ, gd_iter, warmup, 0.2, use_autodiff)
    update(:Poisson,   A, Y12, tracker, σ, 20, warmup, 0.05, use_autodiff)
    update(:Gamma,   A, Y12, tracker, σ, 200, warmup, 0.05, use_autodiff)
    Y12      = max.(-α, min.(Y12, α))
    ϵ        = λ * σInv
    # diagC = diag(C)
    # diagYtmp =  diag(C) - ϵ * l1BallProjection(diag(C) / ϵ, 1)
    Y11      = C[1:d1, 1:d1]
    Y22      = C[d1+1:d1+d2, d1+1:d1+d2]
    Y        = [Y11  Y12;
                Y12' Y22]
    Y        = Y+spdiagm(0 => diag(C) - ϵ * l1BallProjection(diag(C) / ϵ, 1) - diag(Y))
    Ydual    = σ * (Y - C)
    # step 3
    W = W + τ * σ * (Y - X)

    
    normX    = 1 + Fnorm(X)
    primfeas = Fnorm(X - Y) / normX
    err1     = σInv * Fnorm(W - Xdual)
    err2     = σInv * Fnorm(W - Ydual)
    dualfeas = maximum([err1, err2]) / normX
    push!(runhist.primfeas, primfeas)
    push!(runhist.dualfeas, dualfeas)
    if (max(primfeas, dualfeas) < stoptol) | (iter == maxiter)
      breakyes = 1
    end
    if (max(primfeas, dualfeas) < sqrt(stoptol)) & (dualfeas > 1.5 * minimum(runhist.dualfeas[max(iter - 49, 1):iter])) & (iter > 150)
      breakyes = 2
    end
    if (rem(iter,20)==1) | (breakyes > 0)
      R = abs.(maximum(diag(Y)))
      # obj1 = norm(Y12[tracker[:Gaussian][:Observed]] -  A[tracker[:Gaussian][:Observed]])^2
      #             + logisticLoss(Y12[tracker[:Bernoulli][:Observed]], A[tracker[:Bernoulli][:Observed]]);
      obj1 = 0
      obj2 = λ * R
      obj3 = μ * tr(II * X)
      maxY12 = maximum(abs.(Y12))
      @printf("\n %3.0f %3.2e %3.2e| %3.2e %5.2f %5.2f %3.2e| %3.2e|", iter, primfeas, dualfeas, λ, R, maxY12, μ, σ)
      @printf("| obj1: %3.2e obj2: %3.2e obj3: %3.2e|", obj1, obj2, obj3)
    end
    if (σReset > 0) & (rem(iter, 10)==0)
      if (primfeas < 0.5 * dualfeas)
        σ = 0.7 * σ
      elseif (primfeas > 2 * dualfeas)
        σ = 1.3 * σ
      end
    end
    if (breakyes > 0)
      @printf("\n break = %1.0f\n", breakyes)
      break;
    end
  end
completedMatrix = C[1:d1, (d1+1):(d1+d2)]
return completedMatrix, _tracker

end
end
