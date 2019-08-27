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

