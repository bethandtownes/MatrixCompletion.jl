# using MatrixCompletion
using Test

# @testset "IndexTools Test" begin include("./Indexing_test.jl") end



@testset "SparseEigen" begin
    # include("./Indexing_test.jl")
    include("./sparse_eigen_test_runner.jl")
end
