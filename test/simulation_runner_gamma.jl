using MatrixCompletion

const FLAG_SIMULATION_GAMMA_VARY_RANK_SMALL                 = false
const FLAG_SIMULATION_GAMMA_VARY_RANK_MEDIUM                = true
const FLAG_SIMULATION_GAMMA_VARY_RANK_LARGE                 = false

const FLAG_SIMULATION_GAMMA_VARY_MISSING_PERCENTAGE_SMALL   = false
const FLAG_SIMULATION_GAMMA_VARY_MISSING_PERCENTAGE_MEDIUM  = false
const FLAG_SIMULATION_GAMMA_VARY_MISSING_PERCENTAGE_LARGE   = false


FLAG_SIMULATION_GAMMA_VARY_RANK_SMALL ?
  include("simulation_gamma_vary_rank_small.jl") : nothing

FLAG_SIMULATION_GAMMA_VARY_RANK_MEDIUM ?
  include("simulation_gamma_vary_rank_medium.jl") : nothing



  

# @testset "$(format("ADMM Algorithm: Small Input Simulation [Gamma 400 x 400]"))" begin
#   if SIMULATION_STATUS_GAMMA_VARY_MISSING_PERCENTAGE_LARGE == false
#     @info("Running Simulation: [Gamma vary rank][SMALL]")
#     let
#       Random.seed!(65536)
#       ROW = 400
#       COL = 400
#       for input_rank in 1:2:400
#         @printf("small case: rank = %d\n", input_rank)
#         dist = Gamma()
#         timer = TimerOutput()
#         Base.Filesystem.mkpath("./test_result/gamma/small(400x400)")
#         io = open("./test_result/gamma/small(400x400)/rank"*string(input_rank)*".log", "w")
#         # io = Base.stdout
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Gamma(10, 0.5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Gamma
#         @timeit timer  "Gamma(400x400)" * "| rank="*string(input_rank) begin
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
        
#         log_simulation_result(Gamma(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#   else
#     @info("Already Completed Simualtion[Gamma vary rank][SMALL]")
#   end
# end

# @testset "$(format("ADMM Algorithm: Medium Input Simulation [Gamma 2000 x 2000]"))" begin
#   if SIMULATION_STATUS_GAMMA_VARY_RANK_MEDIUM == false
#     @info("Running Simulation: [Gamma vary rank][MEDIUM]")
#     let
#       Random.seed!(65536)
#       ROW = 2000
#       COL = 2000
#       for input_rank in collect(400:10:500)
#         @printf("medium case: rank = %d\n", input_rank)
#         dist = Gamma()
#         timer = TimerOutput()
#         Base.Filesystem.mkpath("./test_result/gamma/medium(2000x2000)")
#         io = open("./test_result/gamma/medium(2000x2000)/rank"*string(input_rank)*".log", "w")
#         # io = Base.stdout
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Gamma(10, 0.5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Gamma
#         @timeit timer  "Gamma(2000x2000)" * "| rank="*string(input_rank) begin
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
        
#         log_simulation_result(Gamma(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#     else
#       @info("Already Completed Simualtion[Gamma vary rank][MEDIUM]")
#     end
# end




  # @testset "$(format("ADMM Algorithm: Large Input Simulation [Gamma 4000 x 4000]"))" begin
  #   let
  #     Random.seed!(65536)
  #     ROW = 2000
  #     COL = 2000
  #     for input_rank in union(1, collect(10:20:1000))
  #       @printf("medium case: rank = %d\n", input_rank)
  #       dist = Gamma()
  #       timer = TimerOutput()
  #       Base.Filesystem.mkpath("./test_result/gamma/medium(2000x2000)")
  #       io = open("./test_result/gamma/medium(2000x2000)/rank"*string(input_rank)*".log", "w")
  #       # io = Base.stdout
  #       truth_matrix        = rand([(FixedRankMatrix(Distributions.Gamma(10, 0.5), rank = input_rank), ROW, COL)])
  #       sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
  #       input_matrix        = sample_model.draw(truth_matrix)
  #       manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
  #       manual_type_matrix .= :Gamma
  #       @timeit timer  "Gamma(2000x2000)" * "| rank="*string(input_rank) begin
  #         completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
  #                                                            maxiter         = 200,
  #                                                            ρ               = 0.3,
  #                                                            use_autodiff    = false,
  #                                                            gd_iter         = 3,
  #                                                            debug_mode      = false,
  #                                                            project_rank    = nothing,
  #                                                            io              = io,
  #                                                            type_assignment = manual_type_matrix)
  #       end

  #       log_simulation_result(Gamma(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
  #       show(io, MIME("text/plain"), timer)
  #     end
  #   end
  # end


