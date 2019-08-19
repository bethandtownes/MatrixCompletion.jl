# using MatrixCompletion
using Test

# @testset "IndexTools Test" begin include("./Indexing_test.jl") end

const FLAG_TEST_INDEXING_TOOLS            = false
const FLAG_TEST_SPARSE_EIGEN              = false
const FLAG_TEST_ADMM                      = false
const FLAG_TEST_LOSS_OPTIMIZER_POISSON    = true
const FLAG_TEST_LOSS_OPTIMIZER_LOGISTIC   = false
const FLAG_TEST_LOSS_OPTIMIZER_GAMMA      = false
const FLAG_TEST_LOSS_OPTIMIZER_BINOMIAL   = false


if FLAG_TEST_SPARSE_EIGEN
    @testset "SparseEigen" begin
        include("./sparse_eigen_test_runner.jl")
    end
end


if FLAG_TEST_LOSS_OPTIMIZER_POISSON
    @testset "Poisson Loss Optimizer" begin
        include("./test_runner_poisson_loss.jl")
    end
end
