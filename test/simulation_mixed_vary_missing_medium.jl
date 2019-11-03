include("abstract_unittest_functions.jl")


@info("Simulation: Vary Missing [Mixed, Medium]")
let
  Random.seed!(65536)
  ROW = 2000
  COL = 2000
  # for input_rank in union(1,collect(10:10:100))
  for input_rank in union(80)
    for input_sample in union(collect(50:5:99))
      # try
      @printf("medium case: rank = %d | sample = %d%%\n", input_rank, input_sample)
      timer = TimerOutput()
      RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR *
        "mixed/medium(2000x2000)(vary_missing)_standardized/" *
        "rank" * string(input_rank) * "/"  *
        "sample" * string(input_sample) * "/"
      LOG_FILE_NAME  = "io.log"
      DATA_FILE_NAME = "saved_variables.h5"
      LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
      DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
      Base.Filesystem.mkpath(RESULTS_DIR)
      io = open(LOG_FILE_PATH, "w")
      truth_matrix      = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1),           rank = input_rank), 2000, 400),
                                (FixedRankMatrix(Distributions.Bernoulli(0.5),           rank = input_rank), 2000, 400),
                                (FixedRankMatrix(Distributions.Gamma(10, 0.5),           rank = input_rank), 2000, 400),
                                (FixedRankMatrix(Distributions.Poisson(5),               rank = input_rank), 2000, 400),
                                (FixedRankMatrix(Distributions.NegativeBinomial(6, 0.8), rank = input_rank), 2000, 400)])
      sample_model       = provide(Sampler{BernoulliModel}(), rate = input_sample / 100)
      input_matrix       = sample_model.draw(truth_matrix)
      manual_type_matrix = Array{Symbol}(undef, ROW, COL)
      manual_type_matrix[:, 1:400]   .= :Gaussian
      manual_type_matrix[:, 401:800] .= :Bernoulli
      manual_type_matrix[:, 801:1200] .= :Gamma
      manual_type_matrix[:, 1201:1600] .= :Poisson
      manual_type_matrix[:, 1601:2000] .= :NegativeBinomial
      user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>6, :p=>0.8))

      @timeit timer  "Mixed(2000x2000)" * "| rank=" * string(input_rank) * "| sample=" * string(input_sample) begin
        completed_matrix, type_tracker, tracker = complete(A                     = input_matrix,
                                                           maxiter               = 200,
                                                           ρ                     = 0.3,
                                                           use_autodiff          = false,
                                                           gd_iter               = 3,
                                                           debug_mode            = false,
                                                           user_input_estimators = user_input_estimators,
                                                           project_rank          = input_rank * 10 + 1,
                                                           io                    = io,
                                                           type_assignment       = manual_type_matrix,
                                                           closed_form_update    = true)
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
      # end
      # catch
      #   @printf("ERROR!!! rank = %d | sample = %d%%\n", input_rank, input_sample)
      # end
    end
  end
end
