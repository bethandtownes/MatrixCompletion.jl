using Test, Random, Distributions
import MatrixCompletion.Losses


struct ExponentialFamily end 


# function predict(



function unit_test_train_poisson(;sz = 500,ρ = 0,step_size = 0.1,max_iter = 100)
    y = Random.rand(Distributions.Poisson(10), sz) * 1.0
    mle_x = Losses.train(Losses.Poisson(), Random.rand(sz), y, zeros(sz), ρ, iter = max_iter, γ = step_size);
    recoveredX = round.(exp.(mle_x));
    errRate = sum(abs.(recoveredX .- y) .> 1) / sz;
    return 1- errRate;
end

################################################################################
#                                TEST SETS                                     #
################################################################################


function POISSON_SMALL_TEST_SET_LOOSE()
    @test unit_test_train_poisson(sz=1000,ρ=0,step_size=0.1,max_iter=200) > 0.9
    @test unit_test_train_poisson(sz=3000,ρ=0,step_size=0.1,max_iter=200) > 0.9
    @test unit_test_train_poisson(sz=5000,ρ=0,step_size=0.1,max_iter=200) > 0.9
    @test unit_test_train_poisson(sz=1000,ρ=0.2,step_size=0.1,max_iter=200) > 0.9
    @test unit_test_train_poisson(sz=3000,ρ=0.2,step_size=0.1,max_iter=200) > 0.9
    @test unit_test_train_poisson(sz=5000,ρ=0.2,step_size=0.1,max_iter=200) > 0.9
end



function main()
    POISSON_SMALL_TEST_SET_LOOSE()
end

main()


