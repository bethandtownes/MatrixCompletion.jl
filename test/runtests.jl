# using MatrixCompletion
using Test,TimerOutputs, Printf





const FLAG_TEST_CONCEPTS                = false
const FLAG_TEST_DIAGNOSTICS             = false
const FLAG_TEST_INDEXING_TOOLS          = false
const FLAG_TEST_SPARSE_EIGEN            = false
const FLAG_TEST_ADMM_SMALL_INPUT        = true
const FLAG_TEST_LOSS_OPTIMIZER_POISSON  = false
const FLAG_TEST_LOSS_OPTIMIZER_LOGISTIC = false
const FLAG_TEST_LOSS_OPTIMIZER_GAMMA    = false
const FLAG_TEST_LOSS_OPTIMIZER_BINOMIAL = false

const to = TimerOutput()

function _gen(str)
    return rpad(str,80,"⋅")
end


# function format(str)
#     return rpad(str,75,"⋅")
# end








FLAG_TEST_INDEXING_TOOLS ?
    include("test_runner_indexing.jl")         : @printf("Skipped: Indexing Tool\n")   

FLAG_TEST_CONCEPTS ?
    include("test_runner_concepts.jl")         : @printf("Skipped: Concepts Test\n")

FLAG_TEST_DIAGNOSTICS ?
    include("test_runner_diagnostics.jl")      : @printf("Skipped: Diagnostics Test\n")

FLAG_TEST_LOSS_OPTIMIZER_POISSON ?
    include("test_runner_poisson_loss.jl")     : @printf("Skipped: Poisson Test\n")

FLAG_TEST_ADMM_SMALL_INPUT ?
    include("test_runner_admm_small_input.jl") : @printf("Skipped: ADMM Small Input Test\n")








# if FLAG_TEST_SPARSE_EIGEN
#     @testset "SparseEigen" begin
#         include("./sparse_eigen_test_runner.jl")
#     end
# end


# if FLAG_TEST_LOSS_OPTIMIZER_POISSON
#     include("./test_runner_poisson_loss.jl")
# end

println()
println(to)
