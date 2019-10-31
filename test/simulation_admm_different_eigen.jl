include("abstract_unittest_functions.jl")

import MatrixCompletion.Utilities.FastEigen:ARPACK


@info("Simulation: Vary Missing [Mixed, Small]")
let
  Random.seed!(65536)
  ROW = 500
  COL = 500
  # for input_rank in union(1,collect(10:10:100))
  #   for input_sample in union(1, collect(5:5:99))
  # try
  # @printf("small case: rank = %d | sample = %d%%\n", input_rank, input_sample)
  input_rank = 10
  input_sample = 80
  timer = TimerOutput()
  # RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR *
  #   "mixed/small(500x500)(vary_missing)/" *
  #   "rank" * string(input_rank) * "/"  *
  #   "sample" * string(input_sample) * "/"
  # LOG_FILE_NAME  = "io.log"
  # DATA_FILE_NAME = "saved_variables.h5"
  # LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
  # DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
  # Base.Filesystem.mkpath(RESULTS_DIR)
  # io = open(LOG_FILE_PATH, "w")
  io = stdout
  truth_matrix      = rand([(FixedRankMatrix(Distributions.Gaussian(10, 5),          rank = input_rank), 500, 100),
                            (FixedRankMatrix(Distributions.Bernoulli(0.5),           rank = input_rank), 500, 100),
                            (FixedRankMatrix(Distributions.Gamma(10, 0.5),           rank = input_rank), 500, 100),
                            (FixedRankMatrix(Distributions.Poisson(5),               rank = input_rank), 500, 100),
                            (FixedRankMatrix(Distributions.NegativeBinomial(6, 0.8), rank = input_rank), 500, 100)])
  sample_model       = provide(Sampler{BernoulliModel}(), rate = input_sample / 100)
  input_matrix       = sample_model.draw(truth_matrix)
  manual_type_matrix = Array{Symbol}(undef, ROW, COL)
  manual_type_matrix[:, 1:100]   .= :Gaussian
  manual_type_matrix[:, 101:200] .= :Bernoulli
  manual_type_matrix[:, 201:300] .= :Gamma
  manual_type_matrix[:, 301:400] .= :Poisson
  manual_type_matrix[:, 401:500] .= :NegativeBinomial
  user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>6, :p=>0.8))


  @timeit timer  "ARPACK" * string(input_sample) begin
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
                                                       eigen_solver          = ARPACK())
  end

  @timeit timer  "KrylovKit" * string(input_sample) begin
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
                                                       eigen_solver          = KrylovMethods())
  end


  @timeit timer  "FullEigen" * string(input_sample) begin
    completed_matrix, type_tracker, tracker = complete(A                     = input_matrix,
                                                       maxiter               = 200,
                                                       ρ                     = 0.3,
                                                       use_autodiff          = false,
                                                       gd_iter               = 3,
                                                       debug_mode            = false,
                                                       user_input_estimators = user_input_estimators,
                                                       project_rank          = nothing,
                                                       io                    = io,
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
  # end
  # catch
  #   @printf("ERROR!!! rank = %d | sample = %d%%\n", input_rank, input_sample)
  # end
end
end
end
