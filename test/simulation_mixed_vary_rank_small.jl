include("abstract_unittest_functions.jl")
@info("Simulation: Vary Rank [Mixed, Small]")
let
  Random.seed!(65536)
  ROW = 500
  COL = 500
  for input_rank in 1:50
    @printf("small case: rank = %d\n", input_rank)
    timer = TimerOutput()
    RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR * "mixed/small_500x500_vary_rank_sample80_lanzcos/" * "rank" * string(input_rank) * "/"
    LOG_FILE_NAME  = "io.log"
    DATA_FILE_NAME = "saved_variables.h5"
    LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
    DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
    Base.Filesystem.mkpath(RESULTS_DIR)
    io = open(LOG_FILE_PATH, "w")
    truth_matrix         = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1),           rank = input_rank), 500, 100),
                                 (FixedRankMatrix(Distributions.Bernoulli(0.5),           rank = input_rank), 500, 100),
                                 (FixedRankMatrix(Distributions.Gamma(10, 0.5),           rank = input_rank), 500, 100),
                                 (FixedRankMatrix(Distributions.Poisson(5),               rank = input_rank), 500, 100),
                                 (FixedRankMatrix(Distributions.NegativeBinomial(6, 0.8), rank = input_rank), 500, 100)])
    sample_model          = provide(Sampler{BernoulliModel}(), rate = 0.8)
    input_matrix          = sample_model.draw(truth_matrix)
    manual_type_matrix    = Array{Symbol}(undef, ROW, COL)
    manual_type_matrix[:, 1:100]   .= :Gaussian
    manual_type_matrix[:, 101:200] .= :Bernoulli
    manual_type_matrix[:, 201:300] .= :Gamma
    manual_type_matrix[:, 301:400] .= :Poisson
    manual_type_matrix[:, 401:500] .= :NegativeBinomial
    user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>6, :p=>0.8))

    @timeit timer  "Mixed(500x500)" * "| rank=" * string(input_rank) begin
      completed_matrix, type_tracker, tracker = complete(A                     = input_matrix,
                                                         maxiter               = 200,
                                                         ρ                     = 0.3,
                                                         use_autodiff          = false,
                                                         gd_iter               = 3,
                                                         debug_mode            = false,
                                                         user_input_estimators = user_input_estimators,
                                                         project_rank          = input_rank * 10 + 1,
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
  end
end
