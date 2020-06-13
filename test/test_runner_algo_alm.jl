# using MatrixCompletion

const TEST_CHAINED_ALM_SMALL_RANDOM = true
const TEST_ONESHOT_ALM_SMALL_RANDOM = false

import LinearAlgebra: norm

function run_chained_alm_randomized(;m = nothing, n = nothing, k = nothing, sample_rate = nothing, data = nothing)
    local truth_matrix
    if isnothing(data)
        truth_matrix = randn(m, k) * randn(k, n)
    else
        truth_matrix = data
    end
    sample_model = provide(Sampler{BernoulliModel}(), rate = sample_rate / 100)
    input_matrix = sample_model.draw(truth_matrix)
    manual_type_matrix = Array{Symbol}(undef, m, n)
    manual_type_matrix .= :Gaussian
    imputed, X, Y, tracker = complete(ChainedALM(),
                                      A = input_matrix,
                                      type_assignment = manual_type_matrix,
                                      block_size = Int64(500 * 500 / 10),
                                      rx = MatrixCompletion.LowRankModels.QuadReg(0.01),
                                      ry = MatrixCompletion.LowRankModels.QuadReg(0.01),
                                      target_rank = k,
                                      imputation_round = 10,
                                      initialX = randn(k, m),
                                      initialY = randn(k, n))
    
    return truth_matrix, imputed, X, Y, tracker
end


function run_one_shot_alm_randomized(;m = nothing, n = nothing, k = nothing, sample_rate = nothing, data = nothing)
    local truth_matrix
    if isnothing(data)
        truth_matrix = randn(m, k) * randn(k, n)
    else
        truth_matrix = data
    end
    sample_model = provide(Sampler{BernoulliModel}(), rate = sample_rate / 100)
    input_matrix = sample_model.draw(truth_matrix)
    manual_type_matrix = Array{Symbol}(undef, m, n)
    manual_type_matrix .= :Gaussian
    imputed, X, Y, tracker = complete(OneShotALM(),
                                          A = input_matrix,
                                          type_assignment = manual_type_matrix,
                                          target_rank = k)
    return truth_matrix, imputed, X, Y, tracker
end

function get_diagnostic(A, A_imputed, X, Y, tracker)
    ret = Dict{Symbol, Float64}()
    ret[:L2_total_error] = norm(A[tracker[:Missing][:Total]] - A_imputed[tracker[:Missing][:Total]]) ^ 2
    ret[:L2_relative_error] = ret[:L2_total_error] / norm(A[tracker[:Missing][:Total]]) ^ 2
    return ret
end


@testset "$(format("Algorithm: OneShotALM[Randomized, Small]"))" begin
    for i in 1:1
        @test get_diagnostic(run_one_shot_alm_randomized(m = 200, n = 200, k = 10, sample_rate = 80)...)[:L2_relative_error] < 0.05
    end
end

@testset "$(format("Algorithm: ChainedALM[Randomized, Small]"))" begin
    for i in 1:1 
       @test get_diagnostic(run_chained_alm_randomized(m = 200, n = 200, k = 10, sample_rate = 80)...)[:L2_relative_error] < 0.05
    end
end

    



