const FLAG_SIMULATION_POISSON_VARY_RANK_SMALL                 = false
const FLAG_SIMULATION_POISSON_VARY_RANK_MEDIUM                = true
const FLAG_SIMULATION_POISSON_VARY_RANK_LARGE                 = false

const FLAG_SIMULATION_POISSON_VARY_MISSING_PERCENTAGE_SMALL   = false
const FLAG_SIMULATION_POISSON_VARY_MISSING_PERCENTAGE_MEDIUM  = false
const FLAG_SIMULATION_POISSON_VARY_MISSING_PERCENTAGE_LARGE   = false


FLAG_SIMULATION_POISSON_VARY_RANK_SMALL ?
  include("simulation_poisson_vary_rank_small.jl")  : nothing

FLAG_SIMULATION_POISSON_VARY_RANK_MEDIUM ?
  include("simulation_poisson_vary_rank_medium.jl")  : nothing


# @testset "$(format("ADMM Algorithm: Small Input Simulation [Poisson 400 x 400]"))" begin
#   if SIMULATION_STATUS_POISSON_VARY_RANK_SMALL == false
#     let
#       Random.seed!(65536)
#       ROW = 400
#       COL = 400
#       for input_rank in 1:2:400
#         @printf("small case: rank = %d\n", input_rank)
#         dist = Poisson()
#         timer = TimerOutput()
#         Base.Filesystem.mkpath("./test_result/poisson/small(400x400)")
#         io = open("./test_result/poisson/small(400x400)/rank"*string(input_rank)*".log", "w")
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Poisson(5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Poisson
#         @timeit timer  "Poisson(400x400)" * "| rank="*string(input_rank) begin
#           completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
#                                                              maxiter         = 200,
#                                                              ρ               = 0.3,
#                                                              use_autodiff    = false,
#                                                              gd_iter         = 3,
#                                                              debug_mode      = false,
#                                                              project_rank    = nothing,
#                                                              io              = io,
#                                                              type_assignment = manual_type_matrix)
#         end
        
#         log_simulation_result(Poisson(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#   end
# end



# @testset "$(format("ADMM Algorithm: Medium Input Simulation [Poisson 2000 x 2000]"))" begin
#   if SIMULATION_STATUS_POISSON_VARY_RANK_MEDIUM == false
#     let
#       Random.seed!(65536)
#       ROW = 2000
#       COL = 2000
#       for input_rank in union(1, collect(10:10:500))
#         @printf("medium case: rank = %d\n", input_rank)
#         dist = Poisson()
#         timer = TimerOutput()
#         Base.Filesystem.mkpath("./test_result/poisson/medium(2000x2000)")
#         io = open("./test_result/poisson/medium(2000x2000)/rank"*string(input_rank)*".log", "w")
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Poisson(5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Poisson
#         @timeit timer  "Poisson(2000x2000)" * "| rank="*string(input_rank) begin
#           completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
#                                                              maxiter         = 200,
#                                                              ρ               = 0.3,
#                                                              use_autodiff    = false,
#                                                              gd_iter         = 3,
#                                                              debug_mode      = false,
#                                                              project_rank    = input_rank,
#                                                              io              = io,
#                                                              type_assignment = manual_type_matrix)
#         end
#         log_simulation_result(Poisson(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#   end
# end

