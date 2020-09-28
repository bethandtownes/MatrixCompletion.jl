@testset "$(format("Chained ADMM: Bernoulli"))"  begin
    let
        Random.seed!(65536)
        ROW = 100
        COL = 100
        for input_rank in collect(20:5:40)
            for input_sample in collect(50:5:90)
                try
                    @show(Pair(input_rank, input_sample))
                    truth_matrix        = rand([(FixedRankMatrix(Distributions.Bernoulli(0.5), rank = input_rank), ROW, COL)])
                    sample_model        = provide(Sampler{BernoulliModel}(), rate = input_sample / 100)
                    input_matrix        = sample_model.draw(truth_matrix)
                    manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
                    manual_type_matrix .= :Bernoulli
                    let 
                        RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR *
                            "bernoulli/small(" * string(ROW) * "x" * string(COL) * ")" * "(vary_missing_200_iter)/" *
                            "rank" * string(input_rank) * "/"  *
                            "sample" * string(input_sample) * "/oneshot/"
                        LOG_FILE_NAME  = "io.log"
                        DATA_FILE_NAME = "saved_variables.h5"
                        LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
                        DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
                        Base.Filesystem.mkpath(RESULTS_DIR)
                        io = open(LOG_FILE_PATH, "w")
                        completed_matrix, type_tracker, tracker, imputed = complete(OneShotADMM(),
                                                                                    A               = input_matrix,
                                                                                    maxiter         = 200,
                                                                                    ρ               = 0.3,
                                                                                    use_autodiff    = false,
                                                                                    gd_iter         = 3,
                                                                                    io              = io,
                                                                                    debug_mode      = false,
                                                                                    project_rank    = nothing,
                                                                                    type_assignment = manual_type_matrix)
                        predicted_matrix = predict(MatrixCompletionModel(),
                                                   completed_matrix = completed_matrix,
                                                   type_tracker     = type_tracker)

                        
                        summary_object  = summary(MatrixCompletionModel(),
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
                        close(io)
                    end
                    let 
                        RESULTS_DIR    = GLOBAL_SIMULATION_RESULTS_DIR *
                            "bernoulli/small(" * string(ROW) * "x" * string(COL) * ")" * "(vary_missing_200_iter)/" *
                            "rank" * string(input_rank) * "/"  *
                            "sample" * string(input_sample) * "/chained/"
                        LOG_FILE_NAME  = "io.log"
                        DATA_FILE_NAME = "saved_variables.h5"
                        LOG_FILE_PATH  = RESULTS_DIR * LOG_FILE_NAME
                        DATA_FILE_PATH = RESULTS_DIR * DATA_FILE_NAME
                        Base.Filesystem.mkpath(RESULTS_DIR)
                        io = open(LOG_FILE_PATH, "w")

                        completed_matrix, type_tracker, tracker, imputed = complete(ChainedADMM(),
                                                                                    A               = deepcopy(input_matrix),
                                                                                    maxiter         = 200,
                                                                                    ρ               = 0.3,
                                                                                    use_autodiff    = false,
                                                                                    gd_iter         = 3,
                                                                                    imputation_round = 5,
                                                                                    io              = io,
                                                                                    debug_mode      = false,
                                                                                    project_rank    = nothing,
                                                                                    type_assignment = deepcopy(manual_type_matrix))
                        predicted_matrix = predict(MatrixCompletionModel(),
                                                   completed_matrix = completed_matrix,
                                                   type_tracker     = type_tracker)
                        
                        summary_object  = summary(MatrixCompletionModel(),
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
                        close(io)
                    end
                catch 
                    # nothing
                    @info("got exception not log")
                end
            end
        end
    end
end
