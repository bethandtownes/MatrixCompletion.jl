const FLAG_SIMULATION_BERNOULLI_VARY_RANK_SMALL                 = true
const FLAG_SIMULATION_BERNOULLI_VARY_RANK_MEDIUM                = false
const FLAG_SIMULATION_BERNOULLI_VARY_RANK_LARGE                 = false

const FLAG_SIMULATION_BERNOULLI_VARY_MISSING_PERCENTAGE_SMALL   = false
const FLAG_SIMULATION_BERNOULLI_VARY_MISSING_PERCENTAGE_MEDIUM  = false
const FLAG_SIMULATION_BERNOULLI_VARY_MISSING_PERCENTAGE_LARGE   = false




FLAG_SIMULATION_BERNOULLI_VARY_RANK_SMALL ?
  include("simulation_bernoulli_vary_rank_small.jl")  : nothing

FLAG_SIMULATION_BERNOULLI_VARY_RANK_MEDIUM ?
  include("simulation_bernoulli_vary_rank_medium.jl") : nothing

FLAG_SIMULATION_BERNOULLI_VARY_RANK_LARGE ?
  include("simulation_bernoulli_vary_rank_large.jl") : nothing




# @testset "$(format("ADMM Algorithm: Small Input Simulation [Bernoulli 400 x 400]"))" begin
#   if SIMULATION_STATUS_BERNOULLI_VARY_RANK_SMALL == false
#     let
#       Random.seed!(65536)
#       ROW = 400
#       COL = 400
#       for input_rank in 1:2:400
#         @printf("small case: rank = %d\n", input_rank)
#         dist = Bernoulli()
#         timer = TimerOutput()
#         Base.Filesystem.mkpath("./test_result/bernoulli/small(400x400)")
#         io = open("./test_result/bernoulli/small(400x400)/rank"*string(input_rank)*".log", "w")
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Bernoulli(0.5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Bernoulli
#         @timeit timer  "Bernoulli(400x400)" * "| rank="*string(input_rank) begin
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
#         log_simulation_result(Bernoulli(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#   end
# end


# @testset "$(format("ADMM Algorithm: Medium Input Simulation [Bernoulli 2000 x 2000]"))" begin
#   if SIMULATION_STATUS_BERNOULLI_VARY_RANK_MEDIUM == false
#     let
#       Random.seed!(65536)
#       ROW = 2000
#       COL = 2000
#       # for input_rank in union(1, collect(10:10:500))
#       for input_rank in collect(300:10:500)
#         @printf("medium case: rank = %d\n", input_rank)
#         dist = Bernoulli()
#         timer = TimerOutput()
#         RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR * "bernoulli/medium(2000x2000)(vary_rank)/" * "rank" * string(input_rank) * "/"
#         LOG_FILE_NAME  = "io.log"
#         DATA_FILE_NAME = "saved_variables.h5"
#         LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
#         DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
#         Base.Filesystem.mkpath(RESULTS_DIR)
#         io = open(LOG_FILE_PATH, "w")
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Bernoulli(0.5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Bernoulli
#         @timeit timer  "Bernoulli(2000x2000)" * "| rank=" * string(input_rank) begin
#           completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
#                                                              maxiter         = 200,
#                                                               ρ               = 0.3,
#                                                              use_autodiff    = false,
#                                                              gd_iter         = 3,
#                                                              debug_mode      = false,
#                                                              project_rank    = input_rank,
#                                                              io              = io,
#                                                              type_assignment = manual_type_matrix)
#         end

#         predicted_matrix = predict(MatrixCompletionModel(),
#                                    completed_matrix = completed_matrix,
#                                    type_tracker     = type_tracker)

#         summary_object   = summary(MatrixCompletionModel(),
#                                    predicted_matrix = predicted_matrix,
#                                    truth_matrix     = truth_matrix,
#                                    type_tracker     = type_tracker,
#                                    tracker          = tracker)
#         pickle(DATA_FILE_PATH,
#                "missing_idx"      => type_tracker[:Missing],
#                "completed_matrix" => completed_matrix,
#                "predicted_matrix" => predicted_matrix,
#                "truth_matrix"     => truth_matrix,
#                "summary"          => summary_object)


#         log_simulation_result(Bernoulli(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         print(io, JSON.json(summary_object, 4))
#         print(io, timer)
#         close(io)
#       end
#     end
#   else
#     @info("Already Completed Simualtion[Bernoulli vary rank][MEDIUM]")
#   end
# end




# @testset "$(format("ADMM Algorithm: Medium Input Simulation [Bernoulli 2000 x 2000]"))" begin
#   if SIMULATION_STATUS_BERNOULLI_VARY_RANK_MEDIUM == false
#     let
#       Random.seed!(65536)
#       ROW = 100
#       COL = 100
#       for input_rank in union(1, collect(10:10:500))
#         @printf("medium case: rank = %d\n", input_rank)
#         dist = Bernoulli()
#         timer = TimerOutput()
#         # Base.Filesystem.mkpath("./test_result/bernoulli/medium(2000x2000)")
#         # io = open("./test_result/bernoulli/medium(2000x2000)/rank"*string(input_rank)*".log", "w")
#         io = Base.stdout
#         truth_matrix        = rand([(FixedRankMatrix(Distributions.Bernoulli(0.5), rank = input_rank), ROW, COL)])
#         sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
#         input_matrix        = sample_model.draw(truth_matrix)
#         manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
#         manual_type_matrix .= :Bernoulli
#         @timeit timer  "Bernoulli(2000x2000)" * "| rank="*string(input_rank) begin
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
#         log_simulation_result(Bernoulli(), completed_matrix, truth_matrix, type_tracker,tracker, io = io)
#         show(io, MIME("text/plain"), timer)
#       end
#     end
#   end
# end
