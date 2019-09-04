@testset "$(format("Loss: Bernoulli[construction]"))" begin
    @test typeof(Loss{Bernoulli}(Bernoulli())) == Loss{Bernoulli}
    @test typeof(Loss(Bernoulli())) == Loss{Bernoulli}
    @test typeof(Loss(:Bernoulli)) == Loss{Bernoulli}
end


@testset "$(format("Optimizer: Bernoulli Loss [Small][Forgiving][AutoGrad]"))"  begin
    @timeit to "Optimizer: Bernoulli Loss [Small][Forgiving][AutoGrad]"  begin
        let
            is_admissible(result)  = begin
                if result["relative-error[#within-radius(1e-5)]"] < 0.1
                    return true
                end
                @warn @sprintf("expected %f, got %f",0.1,result["relative-error[#within-radius(1e-5)]"])
                return false
            end
            @test !isnothing(provide(Loss(Bernoulli())))
            @test !isnothing(provide(Loss(:Bernoulli)))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval = provide(Loss(Bernoulli())),
 
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 1000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter           = 200))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(Bernoulli())),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 3000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter          = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(Bernoulli())),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 5000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(Bernoulli())),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 1000,
                                                        ρ                  = 0.1,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(Bernoulli())),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 3000,
                                                        ρ                  = 0.1,
                                                        step_size          = 0.1,
                                                        max_iter = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(Bernoulli())),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 5000,
                                                        ρ                  = 0.2,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 1000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 3000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter          = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 5000,
                                                        ρ                  = 0,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 1000,
                                                        ρ                  = 0.1,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 3000,
                                                        ρ                  = 0.1,
                                                        step_size          = 0.1,
                                                        max_iter = 100))
            @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = provide(Loss(:Bernoulli)),
                                                        input_distribution = Distributions.Bernoulli(0.6),
                                                        input_size         = 5000,
                                                        ρ                  = 0.2,
                                                        step_size          = 0.1,
                                                        max_iter           = 100))
        end
    end
end



@testset "$(format("Optimizer: Bernoulli Loss [Small][Forgiving][Native]"))" begin
    @timeit to "Optimizer: Bernoulli Loss [Small][Forgiving][Native]" begin
        is_admissible(result)  = begin
                if result["relative-error[#within-radius(1e-5)]"] < 0.1 
                    return true
                end
                @warn @sprintf("expected %f, got %f",0.1,result["relative-error[#within-radius(1e-5)]"])
                return false
            end
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 1000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 3000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter          = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 5000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter           = 100)) 
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 1000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 3000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(Bernoulli()),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 5000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 1000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 3000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter          = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 5000,
                                                    ρ                  = 0,
                                                    step_size          = 0.1,
                                                    max_iter           = 100)) 
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 1000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 3000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter = 100))
        @test is_admissible(unit_test_train_subloss(Bernoulli(),gradient_eval      = Loss(:Bernoulli),
                                                    input_distribution = Distributions.Bernoulli(0.6),
                                                    input_size         = 5000,
                                                    ρ                  = 0.2,
                                                    step_size          = 0.1,
                                                    max_iter           = 100))
    end
end

