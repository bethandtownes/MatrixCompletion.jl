


using Test,TimerOutputs,Printf
import Distributions





function unit_test_train_subloss(dist               = Poisson();
                                 gradient_eval      = Losses.provide(Loss{Poisson}()),
                                 input_distribution = Distributions.Poisson(5),
                                 input_size         = 500,
                                 ρ = 0,
                                 step_size = 0.1,
                                 max_iter = 100)
    y = rand(input_distribution, input_size) * 1.0
    mle_x = train(gradient_eval,
                  fx    = rand(input_size),
                  y     = y,
                  c     = zeros(input_size),
                  ρ     = ρ,
                  iter  = max_iter,
                  γ     = step_size);
    prediction = predict(dist,forward_map(dist,mle_x))
    return provide(Diagnostics{Poisson()}(),
                   input_data=prediction, reference=y)
    
end

