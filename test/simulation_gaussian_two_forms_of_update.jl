include("abstract_unittest_functions.jl")
@testset "$(format("ADMM Algorithm: Small Input Simulation [Gaussian 400 x 400]"))" begin
  let
    Random.seed!(65536)
    ROW = 400
    COL = 400
    for input_rank in 1:400
      input_rank = 100
      @printf("small case: rank = %d\n", input_rank)
      dist = Gaussian()
      timer = TimerOutput()
      # RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR * "gaussian/small(400x400)(vary_rank)/" * "rank" * string(input_rank) * "/"
      # LOG_FILE_NAME  = "io.log"
      # DATA_FILE_NAME = "saved_variables.h5"
      # LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
      # DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
      # Base.Filesystem.mkpath(RESULTS_DIR)
      # io = open(LOG_FILE_PATH, "w")
      io = stdout
      _truth_matrix        = rand([(FixedRankMatrix(Distributions.Gaussian(10, 1), rank = input_rank), ROW, COL)])
      truth_matrix       = _truth_matrix .- 10
      sample_model        = provide(Sampler{BernoulliModel}(), rate = 0.8)
      input_matrix        = sample_model.draw(truth_matrix)
      manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
      manual_type_matrix .= :Gaussian
      @timeit timer  "Gaussian(400x400)" * "| rank=" * string(input_rank) begin
        completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
                                                           maxiter         = 200,
                                                           ρ               = 0.3,
                                                           use_autodiff    = false,
                                                           gd_iter         = 3,
                                                           debug_mode      = false,
                                                           project_rank    = nothing,
                                                           io              = io,
                                                           type_assignment = manual_type_matrix)
      end

      predicted_matrix = predict(MatrixCompletionModel(),
                                 completed_matrix = completed_matrix,
                                 type_tracker     = type_tracker)

      summary_object   = summary(MatrixCompletionModel(),
                                 predicted_matrix = predicted_matrix .+ 10,
                                 truth_matrix     = _truth_matrix,
                                 type_tracker     = type_tracker,
                                 tracker          = tracker)


      summary_object[:Gaussian]
 
      @timeit timer  "Gaussian(400x400)" * "closed" * string(input_rank) begin
        completed_matrix, type_tracker, tracker = complete(A               = input_matrix,
                                                           maxiter         = 200,
                                                           ρ               = 0.3,
                                                           use_autodiff    = false,
                                                           gd_iter         = 3,
                                                           debug_mode      = false,
                                                           project_rank    = nothing,
                                                           io              = io,
                                                           type_assignment = manual_type_matrix,
                                                           closed_form     = true)
      end

      predicted_matrix = predict(MatrixCompletionModel(),
                                 completed_matrix = completed_matrix,
                                 type_tracker     = type_tracker)

      summary_object   = summary(MatrixCompletionModel(),
                                 predicted_matrix = predicted_matrix,
                                 truth_matrix     = truth_matrix,
                                 type_tracker     = type_tracker,
                                 tracker          = tracker)

      summary_object[:Gaussian]
      
      pickle(DATA_FILE_PATH,
             "missing_idx"      => type_tracker[:Missing],
             "completed_matrix" => completed_matrix,
             "predicted_matrix" => predicted_matrix,
             "truth_matrix"     => truth_matrix,
             "summary"          => summary_object)
      print(io, JSON.json(summary_object, 4))
      print(io, timer)
      close(io)
    end
  end
else
  @info("Already Completed Simualtion[Gaussian vary rank][SMALL]")
end

