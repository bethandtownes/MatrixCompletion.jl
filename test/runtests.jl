# using MatrixCompletion
using Test,TimerOutputs, Printf




const FLAG_TEST_DIAGNOSTICS               = true
const FLAG_TEST_INDEXING_TOOLS            = false
const FLAG_TEST_SPARSE_EIGEN              = false
const FLAG_TEST_ADMM_SMALL_INPUT          = false
const FLAG_TEST_LOSS_OPTIMIZER_POISSON    = false
const FLAG_TEST_LOSS_OPTIMIZER_LOGISTIC   = false
const FLAG_TEST_LOSS_OPTIMIZER_GAMMA      = false
const FLAG_TEST_LOSS_OPTIMIZER_BINOMIAL   = false

const to = TimerOutput()

function _gen(str)
    return rpad(str,80,"⋅")
end






FLAG_TEST_DIAGNOSTICS ?
    include("test_runner_diagnostics.jl") : @printf("Skipped: Diagnostics Test\n")

FLAG_TEST_LOSS_OPTIMIZER_POISSON ?
    include("test_runner_poisson_loss.jl") : @printf("Skipped: Poisson Test\n")

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
