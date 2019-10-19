using MatrixCompletion
import Distributions

@testset "$(format("GD Optimizer: Negative Binomial Loss [Small][Forgiving][Native]"))" begin
  let
    input_size = 1600*400
    y = rand(Distributions.NegativeBinomial(10, 0.6), input_size) * 1.0
    mle_x = MatrixCompletion.Losses.negative_binomial_train(fx      = rand(input_size),
                                                            y       = y,
                                                            c       = zeros(input_size),
                                                            ρ       = 0,
                                                            γ       = 0.2,
                                                            iter    = 200,
                                                            verbose = true,
                                                            r_estimate = 10)
    @show(mle_x[1:20])
    prediction = predict(NegativeBinomial(),forward_map(NegativeBinomial(),mle_x,r_estimate=10))
    tc = provide(Diagnostics{Poisson()}(),
                 input_data=prediction, reference=y)
    display(tc)
  end
end
