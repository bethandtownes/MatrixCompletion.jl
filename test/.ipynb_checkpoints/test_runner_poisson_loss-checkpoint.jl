
using Test,TimerOutputs,Printf
using MatrixCompletion.Losses
using MatrixCompletion.Concepts
import Random,Distributions




function forword_map(distribution::AbstractPoisson,
                         canonical_parameter::Array{Float64,1};
                         non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                         non_canonical_map = nothing)
    if !isnothing(non_canonical_parameter)
        ## TODO
    end
    return exp.(canonical_parameter)    
end


function predict(distribution::AbstractPoisson,input::Array{Float64,1};
                 non_canonical_prediction_map = nothing)
    if !isnothing(non_canonical_prediction_map)
        #TODO
    end
    return round.(input)
end


function forword_map(distribution::AbstractGamma,
                         canonical_parameter::Array{Float64,1};
                         non_canonical_parameter::Union{Array{Float64},Nothing} = nothing,
                         non_canonical_map = nothing)
    if !isnothing(non_canonical_parameter)
        ## TODO
    end
    return -1 ./ canonical_parameter
end


function predict(distribution::AbstractGamma,input::Array{Float64,1};
                 non_canonical_prediction_map = nothing)
    if !isnothing(non_canonical_prediction_map)
        #TODO
    end
    return input # 
end





function unit_test_train_subloss(dist       = AbstractPoisson();
                                 gradient_eval      = Losses.provide(Loss{AbstractPoisson}()),
                                 input_distribution = Distributions.Poisson(5),
                                 input_size         = 500,
                                 ρ = 0,
                                 step_size = 0.1,
                                 max_iter = 100)
    y = Random.rand(input_distribution, input_size) * 1.0
    mle_x = train(gradient_eval,
                  fx    = rand(input_size),
                  y     = y,
                  c     = zeros(input_size),
                  ρ     = ρ,
                  iter  = max_iter,
                  γ     = step_size);
    prediction = predict(dist,forword_map(dist,mle_x))
    errRate = sum(abs.(prediction .- y) .> 1) / input_size;
    return 1- errRate;
end



################################################################################
#                                TEST SETS                                     #
################################################################################



include("test_impl_poissonloss.jl")
include("test_impl_gammaloss.jl")



@label END_OF_TEST
