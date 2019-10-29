module Tst

include("abstract_unittest_functions.jl")

using MatrixCompletion


# legacy code to be refactored soon!!

const to = TimerOutput()





#==============================================================================#
#                               TEST OPTIONS                                   #
#==============================================================================#
const TEST_OPTION_PRINT_TIMER    = false
const TEST_OPTION_SMALL_INPUT    = true
const TEST_OPTION_MEDIUM_INPUT   = false
const TEST_OPTION_LARGE_INPUT    = false
const TEST_OPTION_USE_AUTOGRAD   = true
const TEST_OPTION_TEST_R_WRAPPER = false


#==============================================================================#
#                              SUBMODULE FLAGS                                 #
#==============================================================================#
const FLAG_TEST_CONCEPTS                         = false
const FLAG_TEST_SAMPLING                         = false
const FLAG_TEST_MISC                             = false
const FLAG_TEST_RANDOM_OBJECTS                   = false
const FLAG_TEST_DIAGNOSTICS                      = false
const FLAG_TEST_EXPONENTIAL_FAMILY               = false
const FLAG_TEST_INDEXING_TOOLS                   = false
const FLAG_TEST_SPARSE_EIGEN                     = false
const FLAG_TEST_BETTER_MGF                       = false
const FLAG_TEST_ESTIMATOR_MLE                    = false
const FLAG_TEST_ESTIMATOR_MOM                    = false
const FLAG_TEST_MODEL_FITTING                    = false

const FLAG_TEST_LOSS_OPTIMIZER_POISSON           = false
const FLAG_TEST_LOSS_OPTIMIZER_BERNOULLI         = false
const FLAG_TEST_LOSS_OPTIMIZER_GAMMA             = false
const FLAG_TEST_LOSS_OPTIMIZER_GAUSSIAN          = false
const FLAG_TEST_LOSS_OPTIMIZER_NEGATIVE_BINOMIAL = false
const FLAG_TEST_LOSS_OPTIMIZER_MULTINOMIAL       = false
const FLAG_TEST_ALGO_ADMM                        = false
const FLAG_TEST_LIB_MATH                         = false
const FLAG_TEST_PRETTY_PRINTER                   = false
const FLAG_TEST_UTILITY_BATCHUTILS               = false
const FLAG_TEST_SGD_BERNOULLI                    = false
const FLAG_TEST_SGD_GAMMA                        = false

const FLAG_TEST_ALGO_ADMM_PARALLELL              = false
const FLAG_TEST_ALGO_GLRM                        = false
const FLAG_TEST_ALGO_SVT                         = false
const FLAG_TEST_ALGO_ONEBIT                      = false
const FLAG_TEST_ALGO_OPTSPACE                    = false


#==============================================================================#
#                             SIMULATION FLAGS                                 #
#==============================================================================#
const FLAG_SIMULATION_ADMM_GAMMA                 = true
const FLAG_SIMULATION_ADMM_BERNOULLI             = false
const FLAG_SIMULATION_ADMM_GAUSSIAN              = false
const FLAG_SIMULATION_ADMM_POISSON               = false
const FLAG_SIMULATION_ADMM_GAUSSIAN_BERNOULLI    = false


#==============================================================================#
#                             VISUALIZATION FLAGS                              #
#==============================================================================#
const FLAG_VISUAL_RANDOM_OBJECTS = false




#==============================================================================#
#                             SIMULATION SCRIPTS                               #
#==============================================================================#
FLAG_SIMULATION_ADMM_GAMMA ?
  include("simulation_runner_gamma.jl")              : @info @sprintf("Skipped: Simulation[vary rank] Gamma")

FLAG_SIMULATION_ADMM_BERNOULLI ?
  include("simulation_runner_bernoulli.jl")          : @info @sprintf("Skipped: Simulation[vary rank] Bernoulli")

FLAG_SIMULATION_ADMM_GAUSSIAN ?
  include("simulation_runner_gaussian.jl")           : @info @sprintf("Skipped: Simulation[vary rank] Gaussian")

FLAG_SIMULATION_ADMM_POISSON ?
  include("simulation_runner_poisson.jl")            : @info @sprintf("Skipped: Simulation[vary rank] Poisson")

# FLAG_SIMULATION_ADMM_NEGATIVE_BINOMIAL ?
#   include("simulation_runner_negbin.jl")             : @info @sprintf("Skipped: Simulation[vary rank] NegativeBinomial")


FLAG_SIMULATION_ADMM_GAUSSIAN_BERNOULLI ?
  include("simulation_runner_gaussian_bernoulli.jl") : @info @sprintf("Skipped: Simulation Gaussian + Bernoulli")





#==============================================================================#
#                                 TEST SCRIPTS                                 #
#==============================================================================#
FLAG_TEST_MISC ?
  include("test_runner_misc.jl")               : @info @sprintf("Skipped: Miscellaneous Test\n")

FLAG_TEST_RANDOM_OBJECTS ?
  include("test_runner_random_objects.jl")     : @info @sprintf("Skipped: Random Objects Test\n")

FLAG_TEST_SAMPLING ?
  include("test_runner_sampling.jl")           : @info @sprintf("Skipped: Sampling Test\n")

FLAG_TEST_SPARSE_EIGEN ?
  include("test_runner_sparse_eigen.jl")       : @info @sprintf("Skipped: Sparse Eigen Test\n")

FLAG_TEST_INDEXING_TOOLS ?
  include("test_runner_indexing.jl")           : @info @sprintf("Skipped: Indexing Tracker Test\n")   

FLAG_TEST_CONCEPTS ?
  include("test_runner_concepts.jl")           : @info @sprintf("Skipped: Concepts Test\n")

FLAG_TEST_DIAGNOSTICS ?
  include("test_runner_diagnostics.jl")        : @info @sprintf("Skipped: Diagnostics Test\n")

FLAG_TEST_EXPONENTIAL_FAMILY ?
  include("test_runner_exponential_family.jl") : @info @sprintf("Skipped: Exponential Family Test\n")

FLAG_TEST_BETTER_MGF ?
  include("test_runner_better_mgf.jl")         : @info @sprintf("Skipped: MGF Test\n")

FLAG_TEST_ESTIMATOR_MLE ?
  include("test_runner_estimator_mle.jl")      : @info @sprintf("Skipped: MLE Test\n")

FLAG_TEST_ESTIMATOR_MOM ?
  include("test_runner_estimator_mom.jl")      : @info @sprintf("Skipped: MOM Test\n")

FLAG_TEST_MODEL_FITTING ?
  include("test_runner_model_fitting.jl")      : @info @sprintf("Skipped: Model Fitting Test\n")

FLAG_TEST_LOSS_OPTIMIZER_POISSON ?
  include("test_runner_poisson_loss.jl")       : @info @sprintf("Skipped: Poisson Loss Test\n")

FLAG_TEST_LOSS_OPTIMIZER_BERNOULLI ?
  include("test_runner_bernoulli_loss.jl")     : @info @sprintf("Skipped: Bernoulli Loss Test\n")

FLAG_TEST_LOSS_OPTIMIZER_GAMMA ?
  include("test_runner_gamma_loss.jl")         : @info @sprintf("Skipped: Gamma Loss Test\n")

FLAG_TEST_LOSS_OPTIMIZER_NEGATIVE_BINOMIAL ?
  include("test_runner_negative_binomial_loss.jl") : @info @sprintf("Skipped: Bernoulli Loss Test\n")

FLAG_TEST_ALGO_ADMM ?
  include("test_runner_admm_small_input.jl")   : @info @sprintf("Skipped: ADMM Small Input Test\n")

FLAG_TEST_LIB_MATH ?
  include("test_runner_lib_math.jl")           : @info @sprintf("Skipped: Math Library Test\n")

FLAG_TEST_PRETTY_PRINTER ?
  include("test_runner_pretty_printer.jl")     : @info @sprintf("Skipped: Pretty Printer\n")

FLAG_TEST_UTILITY_BATCHUTILS ?
  include("test_runner_batch_utils.jl")        : @info @sprintf("Skipped: Utility[Batch Utils]")

FLAG_TEST_SGD_BERNOULLI ?
  include("test_runner_sgd_bernoulli.jl")      : @info @sprintf("Skipped: SGD[Bernoulli]")

FLAG_TEST_SGD_GAMMA ?
  include("test_runner_sgd_gamma.jl")          : @info @sprintf("Skipped: SGD[Gamma]")
#==============================================================================#
#                                VISUAL SCRIPTS                                #
#==============================================================================#

FLAG_VISUAL_RANDOM_OBJECTS ?
  include("visual_random_objects.jl")        : nothing


if TEST_OPTION_PRINT_TIMER 
  println()
  println(to)
end

end
