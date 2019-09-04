@testset "$(format("Loss: Gamma[construction]"))" begin
    @test typeof(Loss{Gamma}(Gamma())) == Loss{Gamma}
    @test typeof(Loss(Gamma())) == Loss{Gamma}
    @test typeof(Loss(:Gamma)) == Loss{Gamma}
end

@testset "$(format("Optimizer: Gamma Loss [Small][Forgiving][Native][]"))"  begin
    @timeit to "Optimizer: Gamma Loss [Small][Forgiving][Native]"  begin
        let
            is_admissible(result)  = begin
                if result["relative-error[L2]"] < 1000
                    return true
                end
                @warn @sprintf("expected %f, got %f",0.1,result["relative-error[L2]"])
                return false
            end
            @test !isnothing(Loss(Gamma()))
            @test !isnothing(provide(Loss(:Gamma)))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 1000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 3000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 5000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 1000,
                                                        ρ                     = 0.1,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 3000,
                                                        ρ                     = 0.1,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(Gamma()) ,
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 5000,
                                                        ρ                     = 0.2,
                                                        step_size             = 0.005,
                                                        max_iter              = 100))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 1000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 3000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 5000,
                                                        ρ                     = 0,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 1000,
                                                        ρ                     = 0.1,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 3000,
                                                        ρ                     = 0.1,
                                                        step_size             =0.005,
                                                        max_iter              = 200))
            @test is_admissible(unit_test_train_subloss(Gamma(),gradient_eval = Loss(:Gamma),
                                                        input_distribution    = Distributions.Gamma(5,0.5),
                                                        input_size            = 5000,
                                                        ρ                     = 0.2,
                                                        step_size             = 0.005,
                                                        max_iter              = 200))
        end
    end
end

@warn "Autograd version of the gamma case is yet to be implemented due to numerical instability (forward pass goes into complex domain...)"

