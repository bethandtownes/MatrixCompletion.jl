# include("admm_test.jl")




# test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
# test_admm_without_autodiff_smallinput(gd_iter=3)
#test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)

using LinearAlgebra


# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Bernoulli]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 5), 200, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 5),  200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     @time completed_matrix, type_tracker = complete(A      = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
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
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 10), 200, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 5,
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


# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Poisson + Bernoulli]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 5), 300, 100),
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 5), 300, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 5), 300, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 5,
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

#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     display(summary_bernoulli)
#   end
# end

# @testset "$(format("ADMM Algorithm: Medium Input[Gaussian + Poisson + Bernoulli]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 5), 300, 100),
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 5), 300, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 5), 300, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false)
#     predicted_poisson = predict(Poisson(),
#                                 forward_map(Poisson(), completed_matrix[type_tracker[:Poisson]]))
#     truth_poisson = truth_matrix[type_tracker[:Poisson]]
#     summary_poisson = provide(Diagnostics{Any}(),
#                               reference = truth_poisson,
#                               input_data = predicted_poisson)

#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)


#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     @info("Gaussian")
#     display(summary_gaussian)
#     @info("Poisson")
#     display(summary_poisson)
#     @info("Bernoulli")
#     display(summary_bernoulli)
#   end
# end


function relative_l2_error(x, y)
  return LinearAlgebra.norm(x -  y, 2)^2 / LinearAlgebra.norm(y)^2
end




# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Poisson + Bernoulli + Gamma]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 3), 400, 100),
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 3), 400, 100),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 3), 400, 100),
#                              (FixedRankMatrix(Distributions.Gamma(10, 2), rank = 3), 400, 100)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false)
#     predicted_poisson = predict(Poisson(),
#                                 forward_map(Poisson(), completed_matrix[type_tracker[:Poisson]]))
#     truth_poisson = truth_matrix[type_tracker[:Poisson]]
#     summary_poisson = provide(Diagnostics{Any}(),
#                               reference = truth_poisson,
#                               input_data = predicted_poisson)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     predicted_gamma = predict(Gamma(),
#                               forward_map(Gamma(), completed_matrix[type_tracker[:Gamma]]))
#     truth_gamma = truth_matrix[type_tracker[:Gamma]]
#     summary_gamma = provide(Diagnostics{Any}(),
#                             reference = truth_gamma,
#                             input_data = predicted_gamma)
#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     @info("Gaussian")
#     display(summary_gaussian)
#     @info("Poisson")
#     display(summary_poisson)
#     @info("Bernoulli")
#     display(summary_bernoulli)
#     @info("Gamma")
#     display(summary_gamma)
#   end
# end

# # 1. rectangular matrices sufficient condition
# # 2. big scale simulation
# # 3. fix the small dimension
# # 4. time performance
# # 5. beroulli / uniform
# # 6. weighted missing pattern

# @testset "$(format("ADMM Algorithm: Small Input[Gaussian + Poisson + Bernoulli + Gamma]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 4), 400, 15),
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 4), 400, 15),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 4), 400, 15),
#                              (FixedRankMatrix(Distributions.Gamma(10, 2), rank = 4), 400, 15)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 10,
#                                               debug_mode   = false)
#     predicted_poisson = predict(Poisson(),
#                                 forward_map(Poisson(), completed_matrix[type_tracker[:Poisson]]))
#     truth_poisson = truth_matrix[type_tracker[:Poisson]]
#     summary_poisson = provide(Diagnostics{Any}(),
#                               reference = truth_poisson,
#                               input_data = predicted_poisson)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     predicted_gamma = predict(Gamma(),
#                               forward_map(Gamma(), completed_matrix[type_tracker[:Gamma]]))
#     truth_gamma = truth_matrix[type_tracker[:Gamma]]
#     summary_gamma = provide(Diagnostics{Any}(),
#                             reference = truth_gamma,
#                             input_data = predicted_gamma)
#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     @info("Gaussian")
#     display(summary_gaussian)
#     @info("Poisson")
#     display(summary_poisson)
#     @info("Bernoulli")
#     display(summary_bernoulli)
#     @info("Gamma")
#     display(summary_gamma)
#   end
# end



# @testset "$(format("ADMM Algorithm: Medium Input[Gaussian + Poisson + Bernoulli + Gamma]"))" begin
#   let
#     truth_matrix     = rand([(FixedRankMatrix(Distributions.Gaussian(5, 10), rank = 3), 1600, 400),
#                              (FixedRankMatrix(Distributions.Poisson(10), rank = 3), 1600, 400),
#                              (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 3), 1600, 400),
#                              (FixedRankMatrix(Distributions.Gamma(10, 2), rank = 3), 1600, 400)])
#     sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#     input_matrix     = sample_model.draw(truth_matrix)
#     display(input_matrix)
#     manual_type_input = Array{Symbol}(undef, 1600, 1600)
#     manual_type_input[:, 1:400]     .= :Gaussian
#     manual_type_input[:, 401:800]   .= :Poisson
#     manual_type_input[:, 801:1200]  .= :Bernoulli
#     manual_type_input[:, 1201:1600] .= :Gamma
#     completed_matrix, type_tracker = complete(A            = input_matrix,
#                                               maxiter      = 200,
#                                               ρ            = 0.3,
#                                               use_autodiff = false,
#                                               gd_iter      = 3,
#                                               debug_mode   = false,
#                                               type_assignment = manual_type_input)

#     predicted_poisson = predict(Poisson(),
#                                 forward_map(Poisson(), completed_matrix[type_tracker[:Poisson]]))
#     truth_poisson = truth_matrix[type_tracker[:Poisson]]
#     summary_poisson = provide(Diagnostics{Any}(),
#                               reference = truth_poisson,
#                               input_data = predicted_poisson)
#     predicted_gaussian = predict(Gaussian(),
#                                  forward_map(Gaussian(), completed_matrix[type_tracker[:Gaussian]]))
#     truth_gaussian = truth_matrix[type_tracker[:Gaussian]]
#     summary_gaussian = provide(Diagnostics{Any}(),
#                                reference = truth_gaussian,
#                                input_data = predicted_gaussian)
#     predicted_gamma = predict(Gamma(),
#                               forward_map(Gamma(), completed_matrix[type_tracker[:Gamma]]))
#     truth_gamma = truth_matrix[type_tracker[:Gamma]]
#     summary_gamma = provide(Diagnostics{Any}(),
#                             reference = truth_gamma,
#                             input_data = predicted_gamma)
#     predicted_bernoulli = predict(Bernoulli(),
#                                   forward_map(Bernoulli(), completed_matrix[type_tracker[:Bernoulli]]))
#     truth_bernoulli = truth_matrix[type_tracker[:Bernoulli]]
#     summary_bernoulli = provide(Diagnostics{Any}(),
#                                 reference = truth_bernoulli,
#                                 input_data = predicted_bernoulli)
#     @info("Gaussian")
#     display(summary_gaussian)
#     @info("Poisson")
#     display(summary_poisson)
#     @info("Bernoulli")
#     display(summary_bernoulli)
#     @info("Gamma")
#     display(summary_gamma)
#   end
# end


# @testset "$(format("ADMM Algorithm: Small Input[Gamma]"))" begin
#   using MatrixCompletion
#   import Distributions
#   import LinearAlgebra
#   truth_matrix     = rand([(FixedRankMatrix(Distributions.Gamma(10, 2), rank = 5), 1600, 400)])
#   sample_model     = provide(Sampler{BernoulliModel}(), rate = 0.8)
#   input_matrix     = sample_model.draw(truth_matrix)
#   manual_type_input = Array{Symbol}(undef, 1600, 400)
#   manual_type_input .= :Gamma
#   completed_matrix, type_tracker = complete(A            = input_matrix,
#                                             maxiter      = 200,
#                                             ρ            = 0.3,
#                                             use_autodiff = false,
#                                             gd_iter      = 10,
#                                             debug_mode   = false,
#                                             type_assignment = manual_type_input)

#   predicted_matrix = predict(Gamma(), forward_map(Gamma(), completed_matrix))
#   error_matrix = abs.(predicted_matrix - truth_matrix)
#   total_l2_error = LinearAlgebra.norm(error_matrix, 2)^2
#   relative_error = relative_l2_error(predicted_matrix, truth_matrix)
#   @info("completed")
#   display(completed_matrix[1:10, 1:10])
#   @info("predicted")
#   display(predicted_matrix[1:10, 1:10])
#   @info("truth")
#   display(truth_matrix[1:10, 1:10])
#   @info("error")
#   display(error_matrix[1:10, 1:10])
#   @show(relative_error)
#   @show(total_l2_error)

# end




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



include("./sub_test_runner_admm_negative_binomial.jl")
# include("./sub_test_runner_admm_bernoulli.jl")
# include("./sub_test_runner_admm_gaussian.jl")
# include("./sub_test_runner_admm_poisson.jl")
# include("./sub_test_runner_admm_gamma.jl")


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
