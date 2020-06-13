include("./lowrankmodels/LowRankModels.jl")

module ALM

using Printf
import ..Concepts
import ..Concepts: MaybeMissing, type_conversion
import ..Utilities
import ..LowRankModels
import ..ModelFitting.ObservedOrMissing
using Logging

export ChainedALM, OneShotALM

struct GLRMLosses end

struct ChainedALM end

struct OneShotALM end

struct MeanImputation end

struct ChainedEquations end

struct ChainedTruncatedSVD end



function sample_without_replacement!(collection, n::Int64) where T<:Any
    sample = []
    for i in 1:n
        push!(sample, splice!(collection,rand(eachindex(collection))))
    end
    return sample
end


function update_observed_entries!(observed_entries, sampled_entries)
    for index in sampled_entries
        Base.push!(observed_entries, index)
    end
end

function update_imputed_entries!(cur_imputed, sampled_entries, X, Y)
    XtY = X' * Y
    for index in sampled_entries
        cur_imputed[index] = XtY[index]
    end
end


function Concepts.type_conversion(::Type{GLRMLosses}, x::Symbol)
    if x == :Gaussian
        return LowRankModels.QuadLoss()
    else
        return nothing
    end
end

function initialize_trackers(A::Array{MaybeMissing{Float64}}, type_assignment)
    type_tracker = Utilities.IndexTracker{Symbol}()
    if type_assignment == nothing
        Concepts.disjoint_join(type_tracker, Concepts.fit(DistributionDeduction(), A))
    else
        Concepts.disjoint_join(type_tracker, type_assignment)
    end
    Concepts.disjoint_join(type_tracker, Concepts.fit(ObservedOrMissing(), A))
    result = Concepts.groupby(type_tracker, [:Observed, :Missing])
    result[:Observed][:Total] = Base.findall(!ismissing, A)
    result[:Missing][:Total] = Base.findall(ismissing, A)
    result[:SingleView] = type_tracker
    return result
end

function prepare_loss_functions(type_assignment)
    return Base.map(x -> type_conversion(GLRMLosses ,x), type_assignment[1, :])
end

function Concepts.complete(model::OneShotALM;
                           A::Array{MaybeMissing{Float64}},
                           type_assignment,
                           rx = LowRankModels.QuadReg(0),
                           ry = LowRankModels.QuadReg(0),
                           target_rank,
                           initialX = nothing,
                           initialY = nothing,
                           proximal_params = nothing)
    row, col = size(A)
    tracker = initialize_trackers(A, type_assignment)
    missing_entries = deepcopy(tracker[:Missing][:Total])
    observed_entries = deepcopy(tracker[:Observed][:Total])
    loss = prepare_loss_functions(type_assignment)
    imputed = A
    @info @sprintf("total number of entries: %d\n", row * col)
    @info @sprintf("total number of missing entries: %d (%.4f%%)\n", length(missing_entries), length(missing_entries) / (row * col))
    @info @sprintf("target rank: %d\n", target_rank)
    if isnothing(initialX)
        @info "initial state of X in undeteced, using randomized initilization"
        initialX = randn(target_rank, row)
    end
    if isnothing(initialY)
        @info "initial state of Y in undeteced, using randomized initilization"
        initialY = randn(target_rank, col)
    end
    glrm = LowRankModels.GLRM(imputed, loss, rx, ry, target_rank, obs = observed_entries, X = initialX, Y = initialY);
    local X, Y, ch
    if isnothing(proximal_params)
        X, Y, ch = LowRankModels.fit!(glrm)
    else
        X, Y, ch = LowRankModels.fit!(glrm, proximal_params)
    end
    update_imputed_entries!(imputed, missing_entries, X, Y)
    return imputed, X, Y, tracker
end
                           

function Concepts.complete(model::ChainedALM;
                           A::Array{MaybeMissing{Float64}},
                           type_assignment,
                           block_size,
                           imputation_round::Union{Int64, Nothing} = nothing,
                           rx = LowRankModels.QuadReg(0),
                           ry = LowRankModels.QuadReg(0),
                           target_rank,
                           initialX = nothing,
                           initialY = nothing,
                           proximal_params = nothing)
    row, col = size(A)
    tracker  = initialize_trackers(A, type_assignment)
    missing_entries = deepcopy(tracker[:Missing][:Total])
    observed_entries = deepcopy(tracker[:Observed][:Total])
    if !isnothing(imputation_round)
        block_size = trunc(Int64, Base.ceil(length(missing_entries) / imputation_round))
        @info @sprintf("imputation round enforced, current block size is %d.\n", block_size)
    end 
    @info @sprintf("total number of entries: %d\n", row * col)
    @info @sprintf("total number of missing entries: %d (%.4f%%)\n", length(missing_entries), length(missing_entries) / (row * col))
    @info @sprintf("target rank: %d\n", target_rank)
    @info @sprintf("block size: %d\n", block_size)
    @info @sprintf("rounds of completion expected: %d\n", Base.ceil(length(missing_entries) / block_size))
    if isnothing(initialX)
        @info "initial state of X in undeteced, using randomized initilization"
        initialX = randn(target_rank, row)
    end
    if isnothing(initialY)
        @info "initial state of Y in undeteced, using randomized initilization"
        initialY = randn(target_rank, col)
    end
    imputed = A
    warmup = false
    prevX = []; prevY=[];
    loss = prepare_loss_functions(type_assignment)
    warmup = false
    round = 0
    while length(missing_entries) > 0
        round += 1
        println(round)
        block_samples = sample_without_replacement!(missing_entries, min(block_size, length(missing_entries)))
        if (warmup == false)
            glrm = LowRankModels.GLRM(imputed, loss, rx, ry, target_rank, obs = observed_entries, X = initialX, Y = initialY);
        else
            glrm = LowRankModels.GLRM(imputed, loss, rx, ry, target_rank, obs = observed_entries, X = prevX, Y = prevY)
        end
        warmup = true
        local X, Y, ch
        if isnothing(proximal_params)
            X, Y, ch = LowRankModels.fit!(glrm)
        else
            X, Y, ch = LowRankModels.fit!(glrm, proximal_params)
        end
        prevX = X
        prevY = Y
        update_observed_entries!(observed_entries, block_samples)
        update_imputed_entries!(imputed, block_samples, X, Y)
    end
    return imputed,  prevX, prevY, tracker
end


end
