# using MatrixCompletion

const TEST_CHAINED_ALM_SMALL_RANDOM = true
const TEST_ONESHOT_ALM_SMALL_RANDOM = false


import LinearAlgebra: norm
import Random

function run_chained_alm_randomized(;m = nothing, n = nothing, k = nothing,
                                    impute_round = 10, sample_rate = nothing, data = nothing, sampled_data = nothing, 
                                    prox_params = nothing,
                                    init_X = nothing, init_Y = nothing)
    local truth_matrix
    if isnothing(data)
        truth_matrix = randn(m, k) * randn(k, n)
    else
        truth_matrix = data
        m, n = Base.size(data)
    end
    if isnothing(sample_rate) && !isnothing(sampled_data)
        input_matrix = sampled_data
    else 
        sample_model = provide(Sampler{BernoulliModel}(), rate = sample_rate / 100)
        input_matrix = sample_model.draw(truth_matrix)
    end
    manual_type_matrix = Array{Symbol}(undef, m, n)
    manual_type_matrix .= :Gaussian
    imputed, X, Y, tracker = complete(ChainedALM(),
                                      A = input_matrix,
                                      type_assignment = manual_type_matrix,
                                      block_size = Int64(500 * 500 / 10),
                                      rx = MatrixCompletion.LowRankModels.QuadReg(0),
                                      ry = MatrixCompletion.LowRankModels.QuadReg(0),
                                      target_rank = k,
                                      imputation_round = impute_round,
                                      initialX = init_X,
                                      initialY = init_Y,
                                      proximal_params = prox_params)
    
    return truth_matrix, imputed, X, Y, tracker
end

function run_one_shot_alm_randomized(;m = nothing, n = nothing, k = nothing,
                                     sample_rate = nothing, data = nothing, prox_params = nothing,
                                     sampled_data = nothing, 
                                     init_X = nothing, init_Y = nothing)
    local truth_matrix, input_matrix
    if isnothing(data)
        truth_matrix = randn(m, k) * randn(k, n)
    else
        truth_matrix = data
        m, n = Base.size(data)
    end
    if isnothing(sample_rate) && !isnothing(sampled_data)
        input_matrix = sampled_data
    else 
        sample_model = provide(Sampler{BernoulliModel}(), rate = sample_rate / 100)
        input_matrix = sample_model.draw(truth_matrix)
    end
    manual_type_matrix = Array{Symbol}(undef, m, n)
    manual_type_matrix .= :Gaussian
    imputed, X, Y, tracker = complete(OneShotALM(),
                                      A = input_matrix,
                                      type_assignment = manual_type_matrix,
                                      target_rank = k,
                                      initialX = init_X,
                                      initialY = init_Y,
                                      proximal_params = prox_params)
    return truth_matrix, imputed, X, Y, tracker
end

function get_diagnostic(A, A_imputed, X, Y, tracker)
    ret = Dict{Symbol, Any}()
    ret[:L2_total_error] = norm(A[tracker[:Missing][:Total]] - A_imputed[tracker[:Missing][:Total]]) ^ 2
    ret[:L2_relative_error] = ret[:L2_total_error] / norm(A[tracker[:Missing][:Total]]) ^ 2
    ret[:MissingEntries] = tracker[:Missing][:Total]
    ret[:Truth] = A
    ret[:Imputed] = Base.convert(Array{Float64, 2}, A_imputed)
    ret[:X] = X
    ret[:Y] = Y
    return ret
end


# @testset "$(format("Algorithm: OneShotALM[Randomized, Small]"))" begin
#     for i in 1:1
#         @test get_diagnostic(run_one_shot_alm_randomized(m = 200, n = 200, k = 10, sample_rate = 80)...)[:L2_relative_error] < 0.05
#     end
# end

# @testset "$(format("Algorithm: ChainedALM[Randomized, Small]"))" begin
#     for i in 1:1 
#        @test get_diagnostic(run_chained_alm_randomized(m = 200, n = 200, k = 10, sample_rate = 80)...)[:L2_relative_error] < 0.05
#     end
# end
 
let
    Random.seed!(65536)
    m = 500
    n = 500
    for k in collect(10:10:500)
        for sample_rate in collect(10:1:99)
            RESULTS_DIR  = GLOBAL_SIMULATION_RESULTS_DIR *
                "random_continuous/small_500x500/" *
                "rank" * string(k) * "/"  *
                "sample" * string(sample_rate) * "/"
            DATA_FILE_NAME_ONESHOT = "oneshot_saved_variables.h5"
            DATA_FILE_PATH_ONESHOT = RESULTS_DIR * DATA_FILE_NAME_ONESHOT
            DATA_FILE_NAME_CHAINED = "chained_saved_variables.h5"
            DATA_FILE_PATH_CHAINED = RESULTS_DIR * DATA_FILE_NAME_CHAINED
            Base.Filesystem.mkpath(RESULTS_DIR)
            
            truth_matrix = randn(m, k) * randn(k, n)
            sample_rate = 80
            sample_model = provide(Sampler{BernoulliModel}(), rate = sample_rate / 100)
            input_matrix = sample_model.draw(truth_matrix)
            initX = randn(k, m)
            initY = randn(k, n)
            param_oneshot = ProxGradParams(max_iter = 200)
            param_chained = ProxGradParams(max_iter = 200)
            
            result_oneshot_alm = get_diagnostic(run_one_shot_alm_randomized(data = deepcopy(truth_matrix),
                                                                            sampled_data = deepcopy(input_matrix),
                                                                            k = k,
                                                                            init_X = deepcopy(initX),
                                                                            init_Y = deepcopy(initY),
                                                                            prox_params = param_oneshot)...)
            
            result_chained_alm = get_diagnostic(run_chained_alm_randomized(data = deepcopy(truth_matrix),
                                                                           sampled_data = deepcopy(input_matrix),
                                                                           impute_round = 5,
                                                                           init_X = deepcopy(initX),
                                                                           init_Y = deepcopy(initY),
                                                                           k = k,
                                                                           prox_params = param_chained)...)
            pickle(DATA_FILE_PATH_ONESHOT, result_oneshot_alm)
            pickle(DATA_FILE_PATH_CHAINED, result_chained_alm)
        end
    end
end

