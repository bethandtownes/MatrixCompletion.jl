using MatrixCompletion

import Distributions

@testset "$(format("SGD Optimizer: Bernoulli Loss [Small][Forgiving][Native]"))" begin
  let
    input_size = 500
    y = rand(Distributions.Bernoulli(0.6), input_size) * 1.0
    mle_x = MatrixCompletion.Losses.sgd_train(Loss{Bernoulli}(),
                                              fx         = rand(input_size),
                                              y          = y,
                                              c          = zeros(input_size),
                                              ρ          = 0,
                                              α          = 0.2,
                                              ρ₁         = 0.9,
                                              ρ₂         = 0.999,
                                              batch_size = 500, 
                                              epoch      = 10);
    prediction = predict(Bernoulli(),forward_map(Bernoulli(),mle_x))
    tc = provide(Diagnostics{Poisson()}(),
                 input_data=prediction, reference=y)
    display(tc)
  end
end
