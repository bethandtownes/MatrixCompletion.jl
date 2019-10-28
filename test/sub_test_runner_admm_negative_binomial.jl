import Distributions
using MatrixCompletion
import Random
using TimerOutputs

const to = TimerOutput()

@testset "$(format("ADMM Algorithm: Small Input[Negative Binomial User Input]"))" begin
  let
    for i = 5:2:200
      @printf("[Small] Currently doing rank %d\n", i)
      Random.seed!(65536)
      io = open("./test_result/negbin/small/rank"*string(i)*".txt", "w")
      r_input  = 6
      input_size = 200
      input_rank = i
      truth_matrix = rand([(FixedRankMatrix(Distributions.NegativeBinomial(r_input, 0.8),
                                            rank = input_rank),
                            input_size,
                            input_size)])
      sample_model = provide(Sampler{BernoulliModel}(), rate = 0.8)
      input_matrix = sample_model.draw(truth_matrix)
      user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>r_input, :p=>0.8))
      manual_type_input = Array{Symbol}(undef, input_size, input_size)
      manual_type_input .= :NegativeBinomial
      # display(input_matrix)
      @timeit to "neg small"  completed_matrix, type_tracker = complete(A            = input_matrix,
                                                                        maxiter      = 200,
                                                                        ρ            = 0.3,
                                                                        use_autodiff = false,
                                                                        gd_iter      = 3,
                                                                        debug_mode   = false,
                                                                        type_assignment = manual_type_input,
                                                                        user_input_estimators = user_input_estimators,
                                                                        project_rank = input_rank,
                                                                        io           = io)
      # display(completed_matrix[1:10, 1:10])
      predicted_negative_binomial = predict(NegativeBinomial(),
                                            forward_map(NegativeBinomial(),
                                                        completed_matrix[type_tracker[:NegativeBinomial]],
                                                        r_estimate = r_input))
      truth_negative_binomial = truth_matrix[type_tracker[:NegativeBinomial]]
      summary_negative_binomial = provide(Diagnostics{Any}(),
                                          reference = truth_negative_binomial,
                                          input_data = predicted_negative_binomial)
      show(io, MIME("text/plain"), summary_negative_binomial)
      print(io, "\n")
      show(io, MIME("text/plain"), to)
      close(io)
    end
  end
end




@testset "$(format("ADMM Algorithm: Medium Input[Negative Binomial User Input]"))" begin
  let
    for i = 5:20:500
      @printf("[Medium] Currently doing rank %d\n ", i)
      Random.seed!(65536)
      io = open("./test_result/negbin/medium/rank"*string(i)*".txt", "w")
      # io = stdout
      r_input  = 6
      input_size = 2000
      input_rank = i
      truth_matrix = rand([(FixedRankMatrix(Distributions.NegativeBinomial(r_input, 0.8),
                                            rank = input_rank),
                            input_size,
                            input_size)])
      sample_model = provide(Sampler{BernoulliModel}(), rate = 0.8)
      input_matrix = sample_model.draw(truth_matrix)
      user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>r_input, :p=>0.8))
      manual_type_input = Array{Symbol}(undef, input_size, input_size)
      manual_type_input .= :NegativeBinomial
      # display(input_matrix)
      @timeit to "neg medium"  completed_matrix, type_tracker = complete(A            = input_matrix,
                                                                         maxiter      = 200,
                                                                         ρ            = 0.3,
                                                                         use_autodiff = true,
                                                                         gd_iter      = 3,
                                                                         debug_mode   = false,
                                                                         type_assignment = manual_type_input,
                                                                         user_input_estimators = user_input_estimators,
                                                                         project_rank = input_rank,
                                                                         io           = io)
      predicted_negative_binomial = predict(NegativeBinomial(),
                                            forward_map(NegativeBinomial(),
                                                        completed_matrix[type_tracker[:NegativeBinomial]],
                                                        r_estimate = r_input))
      truth_negative_binomial = truth_matrix[type_tracker[:NegativeBinomial]]
      summary_negative_binomial = provide(Diagnostics{Any}(),
                                          reference = truth_negative_binomial,
                                          input_data = predicted_negative_binomial)
      show(io, MIME("text/plain"), summary_negative_binomial)
      print(io, "\n")
      show(io, MIME("text/plain"), to)
      close(io)
    end
  end
end
