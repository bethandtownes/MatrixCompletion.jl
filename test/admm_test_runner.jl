
# include("admm_test.jl") 
# @time test_train_logistic()
# @time test_train_logistic_optimized()
# @time test_train_logistic(size=1000*1000)
# @time test_train_logistic_optimized(size=1000*1000)





# function test_admm_without_autodiff_smallinput(;gd_iter = 3,dbg = false)
#     admm_test_matrix1 = rand([(Distributions.Bernoulli(0.7), 100 => 50, 3),(Distributions.Gaussian(3, 1), 100 => 50, 3)])
#     admm_test_matrix_missing1 = sample(BernoulliModel(), x = admm_test_matrix1, rate = 0.8)
#     @time admm_test_matrix_output_1 = complete(A = admm_test_matrix_missing1, maxiter = 200, use_autodiff = false, gd_iter = gd_iter, debug_mode = dbg)
#     gaussian_acc = accuracyImputedContinuousPart(truth = admm_test_matrix1, completedMatrix = admm_test_matrix_output_1)
#     bernoulli_acc = accuracyImputedBinaryPart(truth = admm_test_matrix1, completedMatrix = admm_test_matrix_output_1)
#     @printf("gaussian acc: %f\n", gaussian_acc)
#     @printf("bernoulli acc: %f\n",bernoulli_acc)
# end



@testset "$(format("ADMM Algorithm: Small Input[Gaussian + Bernoulli]"))" begin
  let
    truth_matrix = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1), rank = 5), 200, 100),
                         (FixedRankMatrix(Distributions.Bernoulli(0.5), rank = 5), 200, 100)])
    sample_model = provide(Sampler{BernoulliModel}, rate = 0.8)
    input_matrix = sample_model.draw(truth_matrix)
    
  end
end




# test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
# test_admm_without_autodiff_smallinput(gd_iter=3)
# test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)
#@time POISSON_SMALL_TEST_SET_LOOSE()
