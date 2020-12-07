@info("Simulation: Chained Equations vs Chained ALM")

let
    Random.seed!(65536)
    ROW = 500
    COL = 500
    input_rank = 50
    input_sample = 80
    truth_matrix        = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1), rank = input_rank), ROW, COL)])
    sample_model        = provide(Sampler{BernoulliModel}(), rate = input_sample / 100)
    input_matrix        = sample_model.draw(truth_matrix)
    manual_type_matrix  = Array{Symbol}(undef, ROW, COL)
    manual_type_matrix .= :Gaussian
    complete(MICE(), A = input_matrix, type_assignment = manual_type_matrix)
end

