# include("admm_test.jl")




# test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
# test_admm_without_autodiff_smallinput(gd_iter=3)
#test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)

using LinearAlgebra

# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Bernoulli]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 2), 200, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 2), 200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               σ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false)
#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     display(summary_bernoulli)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     display(summary_gaussian)
#   end
# end


# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Poisson]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 5), 200, 100),
#                              (FixedRankMatrix(Distributions.Poisson(5), rank = 5), 200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               σ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false)
#     predicted_poisson = predict(Poisson(),
#                                 forward_map(Poisson(), completed_matrix[type_tracker[:Poisson]]))
#     truth_poisson = truth_matrix[type_tracker[:Poisson]]
#     summary_poisson = provide(Diagnostics{Any}(),
#                               reference = truth_poisson,
#                               input_data = predicted_poisson)
#     display(summary_poisson)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     display(summary_gaussian)
#   end
# end



function relative_l2_error(x, y)
  return LinearAlgebra.norm(x -  y, 2)^2 / LinearAlgebra.norm(y)^2
end


@testset "$(format("ADMM Algorithm: Small Input[Gamma]"))" begin
  using MatrixCompletion
  import Distributions
  import LinearAlgebra
  truth_matrix     = rand([(FixedRankMatrix(Distributions.Gamma(5, 0.5), rank = 5), 200, 200)])
  sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
  input_matrix     = sample_model.draw(truth_matrix)
  manual_type_input = Array{Symbol}(undef, 200, 200)
  manual_type_input .= :Gamma
  completed_matrix, type_tracker = complete(A            = input_matrix,
                                            maxiter      = 200,
                                            σ            = 0.3,
                                            use_autodiff = false,
                                            gd_iter      = 3,
                                            debug_mode   = false,
                                            type_assignment = manual_type_input)
  predicted_matrix = predict(Gamma(), forward_map(Gamma(), completed_matrix))
  error_matrix = abs.(predicted_matrix - truth_matrix)
  total_l2_error = LinearAlgebra.norm(error_matrix, 2)^2
  # total_l2_relative_error = LinearAlgebra.norm(norm ) 
end




# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Gamma]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 5), 200, 100),
#                              (FixedRankMatrix(Distributions.Gamma(5, 0.5), rank = 5), 200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     manual_type_input = Array{Symbol}(undef, 200, 200)
#     manual_type_input[:, 1:100] .= :Gaussian
#     manual_type_input[:, 101:200] .= :Gamma
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               σ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false,
#                                               type_assignment = manual_type_input)
#     predicted_gamma = predict(Gamma(),
#                                 forward_map(Gamma(), completed_matrix[type_tracker[:Gamma]]))
#     truth_gamma = truth_matrix[type_tracker[:Gamma]]
#     display(truth_gamma - predicted_gamma)
#     error_matrix = abs.(truth_gamma - predicted_gamma)
#     relative_error = LinearAlgebra.norm(error_matrix,2)^2 / LinearAlgebra.norm(truth_gamma) ^ 2
    
#     @show(relative_l2_error(predicted_gamma, truth_gamma))
#     # summary_gamma = provide(Diagnostics{Any}(),
#     #                           reference = truth_gamma,
#     #                           input_data = predicted_gamma)
#     # display(summary_gamma)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     display(summary_gaussian)
#   end
# end



# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Bernoulli, AutoDiff]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 2), 200, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 2), 200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               σ            = 0.3,
#                                               use_autodiff = true,
#                                               gd_iter      = 3,
#                                               debug_mode   = false)
#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     display(summary_bernoulli)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     display(summary_gaussian)
#   end
# end
