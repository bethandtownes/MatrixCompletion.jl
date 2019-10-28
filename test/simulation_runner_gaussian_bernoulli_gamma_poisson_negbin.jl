





# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Bernoulli + Poisson + Gamma + NegativeBinomial]"))" begin
#   let
#     Random.seed!(65536)
#     for i in 1:200
#       @printf("small case: rank = %d\n", i)
#       io = open("./test_result/gaussian_bernoulli/small/rank"*string(2*i)*".txt", "w")
#       truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = i), 400, 200),
#                                (FixedRankMatrix(Distributions.Bernoulli(0.5),  rank = i),  400, 200)])
#       sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#       input_matrix     = sample_model.draw(truth_matrix)
#       @timeit to  "Gaussian + Bernoulli" completed_matrix, type_tracker = complete(A            = input_matrix,
#                                                                                    maxiter      = 200,
#                                                                                    ρ            = 0.3,
#                                                                                    use_autodiff = false,
#                                                                                    gd_iter      = 3,
#                                                                                    debug_mode   = false,
#                                                                                    project_rank = 2 * i,
#                                                                                    io           = io)
#       log_simulation_result(Bernoulli(), completed_matrix, truth_matrix, type_tracker, io = io)
#       log_simulation_result(Gaussian(),  completed_matrix, truth_matrix, type_tracker, io = io)
#       show(io, MIME("text/plain"), to)
#       close(io)
#     end
#   end
# end
