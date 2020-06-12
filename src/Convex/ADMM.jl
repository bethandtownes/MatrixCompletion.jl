module ADMM

using Printf
using LinearAlgebra
using SparseArrays
using ..Concepts
using ..Estimator

using ..ModelFitting
using ..Utilities
using ..Utilities.FastEigen
using ..Utilities.PrettyPrinter
using ..Losses
using ..MathLib
using Logging


import  StatsBase
import LinearAlgebra:norm

# export complete

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



const header_list = ["Iter",
                     " R(dual)",
                     "R(primal)",
                     "ℒ(Gaussian)",
                     "ℒ(Bernoulli)",
                     "ℒ(Poisson)",
                     "ℒ(NegBin) ",
                     " ℒ(Gamma) ",
                     "λ‖diag(Z)‖ᵢ",
                     " μ⟨I, X⟩ ",
                     " ‖Z₁₂‖ᵢ "]



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
function initialize_warmup(tracker, A)
    warmup = Dict{Symbol, Array{Float64}}()
    if haskey(tracker, :Bernoulli) && length(tracker[:Bernoulli][:Observed]) > 0
        warmup[:Bernoulli] = rand(length(tracker[:Bernoulli][:Observed]), 1)
    end
    if haskey(tracker, :Gaussian) && length(tracker[:Gaussian][:Observed]) > 0
        warmup[:Gaussian] = rand(length(tracker[:Gaussian][:Observed]), 1)
    end
    if haskey(tracker, :Poisson) && length(tracker[:Poisson][:Observed]) > 0
        warmup[:Poisson] = rand(length(tracker[:Poisson][:Observed]), 1)
    end
    if haskey(tracker, :Gamma) && length(tracker[:Gamma][:Observed]) > 0
        warmup[:Gamma] = rand(length(tracker[:Gamma][:Observed]), 1)
    end
    if haskey(tracker, :NegativeBinomial) && length(tracker[:NegativeBinomial][:Observed]) > 0
        warmup[:NegativeBinomial] = rand(length(tracker[:NegativeBinomial][:Observed]), 1)
    end
    return warmup
end

@private
function ensure_feasible_data_type(set_of_types)
    @info(set_of_types)
    if !issubset(set_of_types, Set([:Gaussian,
                                    :Poisson,
                                    :NegativeBinomial,
                                    :Gamma,
                                    :Bernoulli]))
        @error("Unrecognized data type.")
        return nothing
    end
end


@private
function ensure_valid_param_estimates(::Type{Val{:Gaussian}}, value::T) where T
    return true
end

@private
function ensure_valid_param_estimates(::Type{Val{:Bernoulli}}, value::T) where T
    return true
end

@private
function ensure_valid_param_estimates(::Type{Val{:Gamma}}, value::T) where T
    return true
end

@private
function ensure_valid_param_estimates(::Type{Val{:Posson}}, value::T) where T
    return true
end

@private
function ensure_valid_param_estimates(::Type{Val{:NegativeBinomial}}, value::T) where T
    if keys(value) != Set((:p, :r))
        @error("Unrecognized parameter name for Negative Binomial. Expected (:p, :r)")
        return false
    end
    if !(0 < value[:p] < 1)
        @error("Estimator for p in Negative Binomial distribution should be in (0, 1)")
        return false
    end
    if value[:r] < 0
        @error("Estimator for r in Negative Binomial distribution should be in (0, ∞)")
        return false
    end
    return true
end

@private
function preprocess_user_input_estimators(input::Optional{Dict{Symbol, Dict{Symbol, T}}}) where T<:Real
    if isnothing(input)
        return Dict{Symbol, Any}()
    end
    ensure_feasible_data_type(keys(input))
    processed_input = Dict{Symbol, Dict{Symbol, Any}}()
    
    for dist in [:NegativeBinomial, :Gaussian, :Poisson, :Bernoulli, :Gamma]
        if haskey(input, dist)
            # @show(dist)
            processed_input[dist] = Dict{Symbol, Any}()
            processed_input[dist] = merge(processed_input[dist],
                                          preprocess_user_input_param_estimates(Val{dist}, input[dist]))
        end
    end
    return processed_input
end

@private
function preprocess_user_input_param_estimates(::Type{T}, value::S) where {T, S}
    ensure_valid_param_estimates(T, value)
    return Dict{Symbol, Any}(:user_input => value)
end

@private
function initialize_estimators(tracker, A, user_input_estimators)
    estimators = Dict{Symbol, Dict{Symbol, Any}}()
    for sym in [:Gaussian, :Bernoulli, :Poisson, :Gamma, :NegativeBinomial]
        estimators[sym] = Dict{Symbol, Any}()
    end
    if haskey(tracker, :NegativeBinomial) && length(tracker[:NegativeBinomial][:Observed]) > 0
        @info("Found negative binomial items. Use MOM for r and p")
        estimators[:NegativeBinomial][:MOM] = Concepts.estimator(MOM{AbstractNegativeBinomial}(),
                                                                 convert(Array{Float64},
                                                                         A[tracker[:NegativeBinomial][:Observed]]))
        @info(estimators[:NegativeBinomial][:MOM])
    end
    return merge(estimators, preprocess_user_input_estimators(user_input_estimators))
end

@private 
function update(::Type{Val{:Gaussian}},
                A            ::Array{MaybeMissing{Float64}},
                Y12          ::Array{Float64},
                tracker      ::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ            ::Float64,
                gd_iter      ::Int64,
                warmup       ::Dict{Symbol, Array{Float64}},
                γ            ::Float64,
                use_autodiff ::Bool,
                closed_form  ::Bool)
    if closed_form == true
        if haskey(tracker, :Gaussian) && length(tracker[:Gaussian][:Observed]) > 0
            Y12[tracker[:Gaussian][:Observed]] .= (1 / (1 + ρ)) * (A[tracker[:Gaussian][:Observed]] + ρ * Y12[tracker[:Gaussian][:Observed]])
        end
    else
        if haskey(tracker, :Gaussian) && length(tracker[:Gaussian][:Observed]) > 0
            Y12[tracker[:Gaussian][:Observed]] = train(use_autodiff ? provide(Loss{AbstractGaussian}()) : Loss{AbstractGaussian}(),
                                                       fx   = warmup[:Gaussian],
                                                       y    = A[tracker[:Gaussian][:Observed]],
                                                       c    = Y12[tracker[:Gaussian][:Observed]],
                                                       ρ    = ρ,
                                                       iter = gd_iter,
                                                       γ    = 0.2)
            warmup[:Gaussian] = Y12[tracker[:Gaussian][:Observed]]
        end
    end
end

@private
function update(::Type{Val{:Bernoulli}},
                A            ::Array{MaybeMissing{Float64}},
                Y12          ::Array{Float64},
                tracker      ::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ            ::Float64,
                gd_iter      ::Int64,
                warmup       ::Dict{Symbol, Array{Float64}},
                γ            ::Float64,
                use_autodiff ::Bool)
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
function update(::Type{Val{:Gamma}},
                A       ::Array{MaybeMissing{Float64}},
                Y12     ::Array{Float64},
                tracker ::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ       ::Float64,
                gd_iter ::Int64,
                warmup  ::Dict{Symbol, Array{Float64}},
                γ       ::Float64, use_autodiff::Bool)
    if haskey(tracker, :Gamma) && length(tracker[:Gamma][:Observed]) > 0
        Y12[tracker[:Gamma][:Observed]] = train(use_autodiff ? provide(Loss{AbstractGamma}()) : Loss{AbstractGamma}(),
                                                fx   = warmup[:Gamma],
                                                y    = A[tracker[:Gamma][:Observed]],
                                                c    = Y12[tracker[:Gamma][:Observed]],
                                                ρ    = ρ,
                                                iter = gd_iter,
                                                γ    = 0.2)
        warmup[:Gamma] = Y12[tracker[:Gamma][:Observed]]
    end
end

@private
function update(::Type{Val{:NegativeBinomial}},
                A            ::Array{MaybeMissing{Float64}},
                Y12          ::Array{Float64},
                tracker      ::Dict{Symbol, Dict{Symbol, Array{<:CartesianIndex}}},
                ρ            ::Float64,
                gd_iter      ::Int64,
                warmup       ::Dict{Symbol, Array{Float64}},
                γ            ::Float64,
                use_autodiff ::Bool,
                estimator    ::Dict{Symbol, Any})
    if haskey(tracker, :NegativeBinomial) && length(tracker[:NegativeBinomial][:Observed]) > 0
        local r_est = nothing
        if haskey(estimator, :user_input)
            r_est = estimator[:user_input][:r]
        else
            r_est = estimator[:MOM][:r]
        end
        Y12[tracker[:NegativeBinomial][:Observed]] = negative_binomial_train(fx         = warmup[:NegativeBinomial],
                                                                             y          = A[tracker[:NegativeBinomial][:Observed]],
                                                                             c          = Y12[tracker[:NegativeBinomial][:Observed]],
                                                                             ρ          = ρ,
                                                                             iter       = gd_iter,
                                                                             γ          = 0.2,
                                                                             r_estimate = r_est)
        warmup[:NegativeBinomial] = Y12[tracker[:NegativeBinomial][:Observed]]
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

function calculate_Z12_update(A, C,tracker, ρ, α, warmup, use_autodiff, gd_iter, estimators, closed_form)
    d1, d2 = size(A)
    Z12 = C[1:d1, (d1+1):(d1+d2)]
    update(:Gaussian,         A, Z12, tracker, ρ, gd_iter, warmup, 0.2,   use_autodiff, closed_form)
    update(:Bernoulli,        A, Z12, tracker, ρ, gd_iter, warmup, 0.2,   use_autodiff)
    update(:Poisson,          A, Z12, tracker, ρ, gd_iter, warmup, 0.2,  use_autodiff)
    update(:Gamma,            A, Z12, tracker, ρ, gd_iter, warmup, 0.005, use_autodiff)
    update(:NegativeBinomial, A, Z12, tracker, ρ, gd_iter, warmup, 0.005, use_autodiff, estimators[:NegativeBinomial])
    project!(ClosedInterval{Float64}(-α, α), Z12)
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
end

@private
function ensure_feasible(A::Array{MaybeMissing{Float64}})
    if isnothing(A)
        @error(io, "please provide data matrix")
        throw(MethodError())
    end
end

function format_log_data(x)
    if x == -10000000
        return "N/A"
    end
    return @sprintf("%3.2e", x)
end


@private
function print_optimization_log(iter,A, X, Z, Z12, W, II, Rp, Rd, ρ, λ, μ, tracker, estimators, io)
    R = abs.(maximum(diag(Z)))
    local gaussian_loss          = -10000000
    local bernoulli_loss         = -10000000
    local poisson_loss           = -10000000
    local gamma_loss             = -10000000
    local negative_binomial_loss = -10000000
    if haskey(tracker, :Gaussian)
        # gaussian_loss = norm(Z12[tracker[:Gaussian][:Observed]] -  A[tracker[:Gaussian][:Observed]])^2
        gaussian_loss = evaluate(Loss{AbstractGaussian}(),
                                 Z12[tracker[:Gaussian][:Observed]],
                                 A[tracker[:Gaussian][:Observed]],
                                 similar(Z12[tracker[:Gaussian][:Observed]]),
                                 0)
    end
    if haskey(tracker, :Bernoulli)
        bernoulli_loss = evaluate(Loss{AbstractBernoulli}(),
                                  Z12[tracker[:Bernoulli][:Observed]],
                                  A[tracker[:Bernoulli][:Observed]],
                                  similar(Z12[tracker[:Bernoulli][:Observed]]),
                                  0)
    end

    if haskey(tracker, :NegativeBinomial)
        local r_est = nothing
        if haskey(estimators[:NegativeBinomial], :user_input)
            r_est = estimators[:NegativeBinomial][:user_input][:r]
        else
            r_est = estimators[:NegativeBinomial][:MOM][:r]
        end
        negative_binomial_loss = evaluate(Loss{AbstractNegativeBinomial}(),
                                          Z12[tracker[:NegativeBinomial][:Observed]],
                                          A[tracker[:NegativeBinomial][:Observed]],
                                          similar(Z12[tracker[:NegativeBinomial][:Observed]]),
                                          0,
                                          r_estimate = r_est)
    end
    if haskey(tracker, :Poisson)
        poisson_loss = evaluate(Loss{AbstractPoisson}(),
                                Z12[tracker[:Poisson][:Observed]],
                                A[tracker[:Poisson][:Observed]],
                                similar(Z12[tracker[:Poisson][:Observed]]),
                                0)
    end
    if haskey(tracker, :Gamma)
        gamma_loss = evaluate(Loss{AbstractGamma}(),
                              Z12[tracker[:Gamma][:Observed]],
                              A[tracker[:Gamma][:Observed]],
                              similar(Z12[tracker[:Gamma][:Observed]]),
                              0)
        # gamma_loss = 0
    end

    data = [iter,
            Rp,
            Rd,
            gaussian_loss,
            bernoulli_loss,
            poisson_loss,
            negative_binomial_loss,
            gamma_loss,
            maximum(abs.(Z12)),
            μ * tr(II * X),
            abs.(maximum(diag(Z)))
            ]
    new_data = map(x -> format_log_data(x) ,data)
    new_data[1] = string(iter)
    add_row(header_list, data=new_data, io = io)
end

#TODO
# function standardize(A, tracker, estimators)
#   A[tracker[:Gaussian][:Observed]] .= (A[tracker[:Gaussian][:Observed]] .- estimators[:Gaussian][:μ]) ./ estimators[:Gaussian][:σ]
# end


#TODO
# function destandardize(A, type_tracker, estimators)
#   A[tracker[:Gaussian]]] .= (A[tracker] .* estimators[:Gaussian][:σ]) .+ estimators[:Gaussian][:μ]
# end



function Concepts.complete(;A::Array{MaybeMissing{Float64}} = nothing,
                           α::Float64                       = maximum(A[findall(x -> !ismissing(x),A)]),
                           λ::Float64                       = 5e-1,
                           μ::Float64                       = 5e-4,
                           ρ::Float64                       = 0.3,
                           τ::Float64                       = 1.618,
                           maxiter::Int64                   = 200,
                           stoptol::Float64                 = 1e-5,
                           use_autodiff::Bool               = false,
                           gd_iter::Int64                   = 50,
                           debug_mode::Bool                 = false,
                           interactive_plot                 = false,
                           type_assignment                  = nothing,
                           dynamic_ρ                        = true,
                           user_input_estimators            = nothing,
                           project_rank                     = nothing,
                           io::IO                           = Base.stdout,
                           eigen_solver                     = KrylovMethods(),
                           closed_form_update               = false)
    logger = SimpleLogger(io)
    global_logger(logger)
    if isnothing(project_rank)
        @info("Using Full Eigen Decomposition.")
    else
        @info("Using Fast Eigen")
    end
    ensure_feasible(A)
    d1, d2                               = size(A);
    Z::Array{Float64, 2}                 = zeros(d1 + d2,  d1 + d2)
    X::Array{Float64, 2}                 = zeros(d1 + d2,  d1 + d2)
    W::Array{Float64, 2}                 = zeros(d1 + d2,  d1 + d2)
    C::Array{Float64, 2}                 = zeros(d1 + d2,  d1 + d2)
    Xinput::Array{Float64, 2}            = zeros(d1 + d2,  d1 + d2)
    II                                   = sparse(1.0I, d1 + d2, d1 + d2)
    tracker, type_tracker                = initialize_trackers(A, type_assignment)
    # initialize warmup input for various gradient descent procedures
    warmup::Dict{Symbol, Array{Float64}} = initialize_warmup(tracker, A)
    # initialize various estimators 
    estimators::Dict{Symbol, Any}        = initialize_estimators(tracker, A, user_input_estimators)
    # print optimization path table header
    table_header(header_list, io = io)
    for iter = 1:maxiter
        @. Xinput = Z + W/ρ
        # step 1
        # try
        if isnothing(project_rank)
            X = project(SemidefiniteCone(), Z + W / ρ - (μ / ρ) * II)
        else
            X = project(SemidefiniteCone(rank = project_rank), Z + W / ρ - (μ / ρ) * II,
                        eigs_implementation = eigen_solver)
        end
        # catch 
        #   @warn("Manual fix for numerical instability")
        #   fix =  Z + W / ρ - (μ / ρ) * II
        #   fix[findall(x -> x== Inf || isnan(x), fix)] .= rand()
        #   X = project(SemidefiniteCone(rank = 20), fix)
        # end
        # Step 2
        @. C = X - 1/ρ * W; @. Z = C
        Z12 = calculate_Z12_update(A, C, tracker, ρ, α, warmup, use_autodiff, gd_iter, estimators, closed_form_update)
        set_block_12(Z, d1, d2, Z12)
        set_block_21(Z, d1, d2, Z12')
        set_diagonal(Z, diag(C) - (λ / ρ) * l1BallProjection(diag(C) * ρ / λ, 1))
        # step 3
        @. W = W + τ * ρ * (Z - X)
        if rem(iter, 10)==1
            primfeas, dualfeas = calculate_primal_and_dual_residual(X, Z, W, C, Xinput, ρ)
            print_optimization_log(iter, A, X, Z, Z12, W, II, primfeas, dualfeas, ρ, λ, μ, tracker, estimators, io)
            if dynamic_ρ
                ρ = balance_gap(ρ, primfeas, dualfeas)
            end
        end
    end
    completedMatrix = C[1:d1, (d1+1):(d1+d2)]
    return completedMatrix, type_tracker, tracker
end

end
