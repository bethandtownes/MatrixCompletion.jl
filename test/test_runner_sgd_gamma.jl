
import LinearAlgebra

function relative_error_l2(a, b)
  return LinearAlgebra.norm(a - b)^2 / LinearAlgebra.norm(b)^2
end


# @testset "$(format("SGD Optimizer: Bernoulli Loss [Small][Forgiving][Native]"))" begin
#   let
#     input_size = 1600* 400
#     y = rand(Distributions.Gamma(10, 2), input_size) * 1.0
#     mle_x = MatrixCompletion.Losses.sgd_train(Loss{Gamma}(),
#                                               fx         = y,
#                                               y          = y,
#                                               c          = zeros(input_size),
#                                               ρ          = 0.2,
#                                               α          = 0.2,
#                                               ρ₁         = 0.9,
#                                               ρ₂         = 0.999,
#                                               batch_size = 1600 * 400, 
#                                               epoch      = 20);
#     prediction = predict(Gamma(),forward_map(Gamma(),mle_x))
#     # @show(relative_error_l2(prediction, y))
#     tc = provide(Diagnostics{Poisson()}(),
#                  input_data=prediction, reference=y)
#     display(tc)
#   end
# end



@testset "$(format("GD Optimizer: Gamma Loss [Small][Forgiving][Native]"))" begin
  let
    input_size = 200 * 200
    y = rand(Distributions.Gamma(10, 2), input_size) * 1.0
    mle_x = MatrixCompletion.Losses.train(Loss{Gamma}(),
                                          fx      = rand(input_size),
                                          y       = y,
                                          c       = zeros(input_size),
                                          ρ       = 0,
                                          γ       = 0.2,
                                          iter    = 500,
                                          verbose = true)
    @show(mle_x[1:20])
    prediction = predict(Gamma(),forward_map(Gamma(),mle_x))
    tc = provide(Diagnostics{Poisson()}(),
                 input_data=prediction, reference=y)
    display(tc)
  end
end
