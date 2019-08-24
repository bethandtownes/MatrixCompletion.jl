
@testset "$(_gen("Optimizer: Poisson Loss [Small][Forgiving][AutoGrad]"))"  begin
    @timeit to "Optimizer: Poisson Loss [Small][Forgiving][AutoGrad]"  begin
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 1000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 3000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter          = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 5000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 1000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 3000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,max_iter = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = provide(Loss{AbstractPoisson}()),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 5000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
    end
end



@testset "$(_gen("Optimizer: Poisson Loss [Small][Forgiving][Native]"))" begin
    @timeit to "Optimizer: Poisson Loss [Small][Forgiving][Native]" begin
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 1000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 3000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter          = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 5000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 1000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 3000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,max_iter = 100) > 0.9
        @test unit_test_train_subloss(gradient_eval      = Loss{AbstractPoisson}(),
                                      input_distribution = Distributions.Poisson(10),
                                      input_size         = 5000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
    end
end

