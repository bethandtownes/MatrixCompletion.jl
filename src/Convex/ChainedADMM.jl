struct ChainedADMM end

function sample_without_replacement!(collection, n::Int64) where T<:Any
    sample = []
    for i in 1:n
        push!(sample, splice!(collection,rand(eachindex(collection))))
    end
    return sample
end

function initialize_chained_trackers(A::Array{MaybeMissing{Float64}}, type_assignment)
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

function Concepts.complete(model::ChainedADMM;
                           block_size = nothing,
                           imputation_round::Union{Int64, Nothing} = 10,
                           A::Array{MaybeMissing{Float64}} = nothing,
                           α::Float64 = maximum(A[findall(x -> !ismissing(x),A)]),
                           λ::Float64 = 5e-1,
                           μ::Float64 = 5e-4,
                           ρ::Float64 = 0.3,
                           τ::Float64 = 1.618,
                           maxiter::Int64 = 200,
                           stoptol::Float64 = 1e-5,
                           use_autodiff::Bool = false,
                           gd_iter::Int64 = 50,
                           debug_mode::Bool = false,
                           interactive_plot = false,
                           type_assignment = nothing,
                           warmup = nothing,
                           dynamic_ρ = true,
                           user_input_estimators = nothing,
                           project_rank = nothing,
                           io::IO = Base.stdout,
                           eigen_solver = KrylovMethods(),
                           closed_form_update = false)
    o_tracker, o_type_tracker   = initialize_trackers(A, type_assignment)
    tracker = initialize_chained_trackers(A, type_assignment)
    missing_entries = deepcopy(tracker[:Missing][:Total])
    # @show(missing_entries[1:10])
    observed_entries = deepcopy(tracker[:Observed][:Total])
    # @show(observed_entries[1:10])
    if !isnothing(imputation_round)
        block_size = trunc(Int64, Base.ceil(length(missing_entries) / imputation_round))
        @info @sprintf("imputation round enforced, current block size is %d.\n", block_size)
    end 

    # @show(imputation_round)
    if isnothing(warmup)
        warmup::Dict{Symbol, Array{Float64}} = initialize_warmup(tracker, A)
    end
    imputed = A
    round = 0
    local result_completed_matrix = nothing
    local result_type_tracker = nothing
    local result_tracker = nothing
    local result_last_var = nothing
    while length(missing_entries) > 0
        round = round + 1
        @info(round)
        block_samples = sample_without_replacement!(missing_entries, min(block_size, length(missing_entries)))
        result_completed_matrix, result_type_tracker, result_tracker, result_last_var = complete(A = imputed,
                                                                                                 α = α,
                                                                                                 λ = λ,
                                                                                                 μ = μ,
                                                                                                 ρ = ρ,
                                                                                                 τ = τ,
                                                                                                 maxiter = maxiter,
                                                                                                 stoptol = stoptol,
                                                                                                 use_autodiff = use_autodiff,
                                                                                                 gd_iter = gd_iter,
                                                                                                 start_var = result_last_var,
                                                                                                 debug_mode = debug_mode,
                                                                                                 interactive_plot = interactive_plot,
                                                                                                 type_assignment = type_assignment,
                                                                                                 # warmup = warmup,
                                                                                                 dynamic_ρ = dynamic_ρ,
                                                                                                 user_input_estimators = user_input_estimators,
                                                                                                 project_rank = project_rank,
                                                                                                 io = io,
                                                                                                 eigen_solver = eigen_solver,
                                                                                                 closed_form_update = closed_form_update)
        # predicted_matrix = predict(MatrixCompletionModel(),
        #                            completed_matrix = result_completed_matrix,
        #                            type_tracker     = result_type_tracker)

        # @show("update imputed")
        for index in block_samples
            imputed[index] = result_completed_matrix[index]
        end
        
        # if haskey(tracker, :Bernoulli) && length(tracker[:Bernoulli][:Observed]) > 0
                    
        # if haskey(tracker, :Bernoulli) && length(tracker[:Bernoulli][:Observed]) > 0
        # warmup[:Bernoulli] = 
        # end
    end
    return result_completed_matrix, o_type_tracker, o_tracker, imputed
end

