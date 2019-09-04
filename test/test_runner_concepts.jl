using MatrixCompletion
import LinearAlgebra 

function unit_test_lpspace(p)
    tc = LpSpace(p)
    test_vec = rand(100)
    @test tc.p==p && tc.norm(test_vec) == LinearAlgebra.norm(test_vec,tc.p)
end




@testset "$(format("Concepts: LpSpace[Construction]"))" begin
    [unit_test_lpspace(i) for i in 1:10]
end


@testset "$(format("Concepts: Type Convertsion[Symbol->Exponential Family]"))" begin
    @test typeof(convert(ExponentialFamily,:Poisson)) == typeof(Poisson())
end


@testset "$(format("Concepts: Comparator[construction]"))" begin
    @test typeof(Comparator{Int64}()) == Comparator{Int64}
    @test typeof(Comparator(MGF(:Gaussian)))  == Comparator{MGF{Gaussian}}
    @test typeof(Comparator(MGF)) == Comparator{MGF}
    @test typeof(Comparator(:MGF)) == Comparator{MGF}
    let
        tc = Comparator{MGF}(MGF(),eval_at = [1,2,3])
        @test tc.field[:eval_at] == [1,2,3]  
    end
end
