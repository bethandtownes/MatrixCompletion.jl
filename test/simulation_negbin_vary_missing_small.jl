include("abstract_unittest_functions.jl")
@info("Simulation: Vary Missing [NegBin, Small]")

let
  Random.seed!(65536)
  ROW = 400
  COL = 400
  for input_rank in collect(25:25:400)
    for input_sample in union(collect(40:5:99))
      # try
      @printf("small case: rank = %d | sample = %d%%\n", input_rank, input_sample)
      timer = TimerOutput()
      RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR *
        "negbin/small_400x400_vary_missing/" *
        "rank" * string(input_rank) * "/"  *
        "sample" * string(input_sample) * "/"
      LOG_FILE_NAME  = "io.log"
      DATA_FILE_NAME = "saved_variables.h5"
      LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
      DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
      Base.Filesystem.mkpath(RESULTS_DIR)
      io = open(LOG_FILE_PATH, "w")
      # io = stdout
      truth_matrix        = rand([(FixedRankMatrix(Distributions.NegativeBinomial(6,0.8), rank = input_rank), ROW, COL)])
      sample_model        = provide(Sampler{BernoulliModel}(), rate = input_sample / 100)
      input_matrix        = sample_model.draw(truth_matrix)
      manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
      manual_type_matrix .= :NegativeBinomial
      user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>6, :p=>0.8))
      
      @timeit timer  "NegBin(400x400)" * "| rank=" * string(input_rank) * "| sample=" * string(input_sample) begin
        completed_matrix, type_tracker, tracker = complete(A                     = input_matrix,
                                                           maxiter               = 200,
                                                           ρ                     = 0.3,
                                                           use_autodiff          = false,
                                                           gd_iter               = 3,
                                                           debug_mode            = false,
                                                           project_rank          = nothing,
                                                           io                    = io,
                                                           user_input_estimators = user_input_estimators,
                                                           type_assignment       = manual_type_matrix)
      end

      predicted_matrix = predict(MatrixCompletionModel(),
                                 completed_matrix = completed_matrix,
                                 type_tracker     = type_tracker,
                                 estimators       = user_input_estimators)

      summary_object   = summary(MatrixCompletionModel(),
                                 predicted_matrix = predicted_matrix,
                                 truth_matrix     = truth_matrix,
                                 type_tracker     = type_tracker,
                                 tracker          = tracker)
      pickle(DATA_FILE_PATH,
             "missing_idx"      => type_tracker[:Missing],
             "completed_matrix" => completed_matrix,
             "predicted_matrix" => predicted_matrix,
             "truth_matrix"     => truth_matrix,
             "summary"          => summary_object)
      print(io, JSON.json(summary_object, 4))
      print(io, timer)
      close(io)
      # catch
      #   @printf("ERROR!!!! rank = %d | sample = %d%%\n", input_rank, input_sample)
      # end
    end
  end
end
