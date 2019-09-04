import LinearAlgebra


@testset "$(format("Misc: check[:rank]"))" begin
    let
        tc = ones(5,5) 
        @test check(Val{:rank},tc,1)
        @test check(:rank,tc,1)
        @test check(:rank,tc) == 1
        @test check(:rank,tc,2) == false
    end
end

@testset "$(format("Misc: check[:dimension]"))" begin
    let
        tc = rand(5,5)
        @test check(:dimension,tc) == (5,5)
        @test check(:dimension,tc,(5,5)) == true 
        @test check(:dimension,tc,(4,5)) == false
    end
end


@testset "$(format("Misc: check[:l2difference]"))" begin
    # number case
    let
        for i = 1:5
            tc_a = rand() * 10
            tc_b = rand() * 10
            @test abs(check(:l2difference,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2difference,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
            @test abs(check(:l2diff,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2diff,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
        end
    end
    # vector case
    let
        for i = 1:5
            tc_a = rand(10) .* 10
            tc_b = rand(10) .* 10
            @test abs(check(:l2difference,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2difference,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
            @test abs(check(:l2diff,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2diff,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
            
        end
    end
    # matrix case
    let
        for i = 1:5
            tc_a = rand(10,10) .* 10 
            tc_b = rand(10,10) .* 10
            @test abs(check(:l2difference,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2difference,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
            @test abs(check(:l2diff,tc_a,tc_b) - LinearAlgebra.norm(tc_a-tc_b,2)) < 1e-5
            @test check(:l2diff,tc_a,tc_b,LinearAlgebra.norm(tc_a-tc_b,2))
        end
    end
end


@testset "$(format("Misc: zeros"))" begin
    let
        # matrix case
        tc = rand(100,100)
        @test check(:l2diff,zeros(tc),zeros(100,100)) < 1e-3
        
    end
end

