using MatrixCompletion
import Distributions



function unit_test_relative_error(metric,input,reference;base_metric = metric)
    @test abs(
        provide(RelativeError(),input,reference;metric = metric,base_metric=base_metric)
              - metric(input-reference)/base_metric(reference)) <1e-5
end


function unit_test_absolute_error(metric,input,reference)
    @test abs(provide(AbsoluteError(),input,reference;metric=metric) -  metric(input-reference)) <1e-5
end



function unit_test_diagnostics(;input =nothing, reference = nothing)
    
end





@testset "$(format("Diagnostics: Absolute Error[LpMetric]"))" begin
    # test lp norm metric for arrays
    [unit_test_absolute_error(metric,rand(100),rand(100)) for metric in [x ->  LinearAlgebra.norm(x,p) for p in 1:0.5:10]]
    # test lp norm metric for metrices
    [unit_test_relative_error(metric,rand(100,100),rand(100,100)) for metric in [x ->  LinearAlgebra.norm(x,p) for p in 1:0.5:10]]
    # test within_radius metric for arrays
end

@testset "$(format("Diagnostics: Absolute Error[WithinRadius]"))" begin
    let 
        for i = 1:10
            tc = rand(100) .+ 1000
            mask = rand(Distributions.Bernoulli(0.6),100)
            num_of_non_zeros = sum(mask)
            unit_test_absolute_error(x -> within_radius(x), mask,zeros(100))
            unit_test_absolute_error(x -> within_radius(x), tc .* mask, tc) 
        end
    end
    # test within_radius metric for matrices    
    let 
        for i = 1:10
            tc = rand(100,100) .+ 1000
            mask = rand(Distributions.Bernoulli(0.6),100,100)
            num_of_non_zeros = sum(mask)
            unit_test_absolute_error(x -> within_radius(x), mask,zeros(100,100))
            unit_test_absolute_error(x -> within_radius(x), tc .* mask, tc) 
        end
    end
end

@testset "$(format("Diagnostics: Relative Error[LpMetric]"))" begin
    # test lp norm metric for arrays
    [unit_test_relative_error(metric,rand(100),rand(100)) for metric in [x ->  LinearAlgebra.norm(x,p) for p in 1:0.5:10]]
    # test lp norm metric for matrices
    [unit_test_relative_error(metric,rand(100,100),rand(100,100)) for metric in [x ->  LinearAlgebra.norm(x,p) for p in 1:0.5:10]]
end


@testset "$(format("Diagnostics: Relative Error[WithinRadius]"))" begin
    # for arrays 
    let 
        for i = 1:10
            tc = rand(100) .+ 1000
            mask = rand(Distributions.Bernoulli(0.6),100)
            num_of_non_zeros = sum(mask)
            unit_test_relative_error(x -> within_radius(x), mask,zeros(100);base_metric = x -> LinearAlgebra.norm(mask,0))
            unit_test_relative_error(x -> within_radius(x), tc .* mask, tc;base_metric = x -> LinearAlgebra.norm(mask,0))
        end
    end
    # for matrices
    let 
        for i = 1:10
            tc = rand(100,100) .+ 1000
            mask = rand(Distributions.Bernoulli(0.6),100,100)
            num_of_non_zeros = sum(mask)
            unit_test_relative_error(x -> within_radius(x), mask,zeros(100,100);base_metric = x -> LinearAlgebra.norm(mask,0))
            unit_test_relative_error(x -> within_radius(x), tc .* mask, tc;base_metric = x -> LinearAlgebra.norm(mask,0))
        end
    end
end



@testset "$(format("Diagnostics: Construction[For Arrays]"))" begin

    # test dispatcher
    @test isa(provide(Diagnostics{Gamma()}(),input_data = [1.1], reference = [1.1]),Dict)

    # test arg parser
    @test_throws DomainError provide(Diagnostics{Gamma()}())

    # test for arrays 
    let
        input_data = [1,1,2,1,0] * 1.0
        reference = deepcopy(input_data)
        diagnostic = provide(Diagnostics{Gamma()}(),
                             input_data = input_data,reference = reference);
        @test diagnostic["relative-error[#within-radius(1e-5)]"] == sum(Int.(abs.(input_data - reference) .> 1e-5))/length(input_data)
        @test diagnostic["absolute-error[#within-radius(1e-5)]"] == sum(Int.(abs.(input_data - reference) .> 1e-5))
        @test diagnostic["relative-error[L1]"] == 0
        @test diagnostic["relative-error[L2]"] == 0
        @test diagnostic["absolute-error[L1]"] == 0
        @test diagnostic["absolute-error[L2]"] == 0
        @test LinearAlgebra.norm(diagnostic["error-matrix"] - [0,0,0,0,0],2) <1e-5
    end
    let
        input_data = [1,1,3,1,0] * 1.0
        reference  = [1,0,1,0,0] * 1.0
        diagnostic = provide(Diagnostics{Gamma()}(),
                             input_data = input_data,reference = reference);
        @test diagnostic["relative-error[#within-radius(1e-5)]"] == 3/5
        @test diagnostic["absolute-error[#within-radius(1e-5)]"] == 3
        @test diagnostic["relative-error[L1]"] == 2
        @test diagnostic["relative-error[L2]"] == sqrt(6) / sqrt(2)
        @test diagnostic["absolute-error[L1]"] == 4
        @test diagnostic["absolute-error[L2]"] == sqrt(6)
        @test LinearAlgebra.norm(diagnostic["error-matrix"] - [0,1,2,1,0],2) <1e-5
    end
end



@testset "$(format("Diagnostics: Construction[For Matrices]"))" begin

    # test dispatcher
    @test isa(provide(Diagnostics{Gamma()}(),input_data = ones(2,2), reference = ones(2,2)),Dict)
    # zeros 
    let
        input_data = rand(10,10)
        reference = deepcopy(input_data)
        diagnostic = provide(Diagnostics{Gamma()}(),
                             input_data = input_data, reference = reference);
        @test diagnostic["relative-error[#within-radius(1e-5)]"] == 0
        @test diagnostic["absolute-error[#within-radius(1e-5)]"] == 0
        @test diagnostic["relative-error[L1]"] == 0
        @test diagnostic["relative-error[L2]"] == 0
        @test diagnostic["absolute-error[L1]"] == 0
        @test diagnostic["absolute-error[L2]"] == 0
        @test LinearAlgebra.norm(diagnostic["error-matrix"] - zeros(10,10) ,2) <1e-5
    end
    # given test 
    let
        input_data = [1 0 2;
                      0 2 0;
                      0 0 1]
        reference = [1 0 0;
                     0 0 1;
                     0 0 1]                 
        diagnostic = provide(Diagnostics{Gamma()}(),
                             input_data = input_data,reference = reference);
        @test diagnostic["relative-error[#within-radius(1e-5)]"] == 3/9
        @test diagnostic["absolute-error[#within-radius(1e-5)]"] == 3
        @test diagnostic["relative-error[L1]"] == 5 /3 
        @test diagnostic["relative-error[L2]"] == 3/sqrt(3)
        @test diagnostic["absolute-error[L1]"] == 5
        @test diagnostic["absolute-error[L2]"] == 3
        @test LinearAlgebra.norm(diagnostic["error-matrix"] - [0 0 2;0 2 1;0 0 0],2) <1e-5
    end
end


