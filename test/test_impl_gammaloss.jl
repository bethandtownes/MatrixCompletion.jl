@testset "$(_gen("Optimizer: Gamma Loss [Small][Forgiving][Native]"))" begin
    @timeit to "Optimizer: Gamma Loss [Small][Forgiving][Native]" begin
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 1000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 3000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 5000,
                                      ρ                  = 0,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 1000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 3000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,max_iter = 100) > 0.9
        @test unit_test_train_subloss(AbstractGamma(),
                                      gradient_eval      = Loss{AbstractGamma}(),
                                      input_distribution = Distributions.Gamma(5,0.5),
                                      input_size         = 5000,
                                      ρ                  = 0.2,
                                      step_size          = 0.1,
                                      max_iter           = 100) > 0.9
    end
end
