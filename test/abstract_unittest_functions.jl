using MatrixCompletion
using Test,TimerOutputs,Printf
using HDF5
using JSON
using DataFrames
import Distributions, Random
import Serialization

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


# struct MatrixCompletionModel end

# function predict(model::MatrixCompletionModel;
#                  completed_matrix, type_tracker)
#   predicted_matrix = similar(completed_matrix)
#   for dist in keys(type_tracker.indices)
#     idx = type_tracker[convert(Symbol,dist)]
#     predicted_matrix[idx] .= predict(dist, forward_map(dist,  completed_matrix[idx]))
#   end
#   return predicted_matrix
# end





function Base.truncate(::Type{Dict{String, Number}}, object::Dict{String, T}) where T<:Any
  return convert(Dict{String, Number},
                 filter(x -> typeof(x.second) <: Number, object))
end

function log_simulation_result(dist::ExponentialFamily, completed_matrix, truth_matrix, type_tracker, tracker; io = Base.stdout)
  predicted_matrix = similar(completed_matrix)
  predicted_matrix[type_tracker[convert(Symbol,dist)]] .= predict(dist,
                                                                  forward_map(dist,
                                                                              completed_matrix[type_tracker[convert(Symbol, dist)]]))
  
  summary_missing_only = provide(Diagnostics{Any}(),
                                 reference  = truth_matrix[tracker[convert(Symbol, dist)][:Missing]],
                                 input_data = predicted_matrix[tracker[convert(Symbol, dist)][:Missing]])
  @printf(io, "\nSummary on %s (Only Missing)\n%s\n", string(convert(Symbol, dist)), repeat("-", 80))
  show(io, MIME("text/plain"), summary_missing_only)
  print(io, "\n\n")
  summary_all  = provide(Diagnostics{Any}(),
                         reference  = truth_matrix[type_tracker[convert(Symbol, dist)]],
                         input_data = predicted_matrix[type_tracker[convert(Symbol, dist)]])
  @printf(io, "\nSummary on %s (Missing && Observed)\n%s\n", string(convert(Symbol, dist)), repeat("-", 80))
  show(io, MIME("text/plain"), summary_all)
  print(io, "\n\n")
  # show(io, MIME("text/plain"), timer)
  # close(io)
end





function Base.convert(::Type{Array}, object::Tuple{T, T}) where T
  return [object[1],object[2]]
end


function Base.convert(::Type{Array}, object::Array{Tuple{T, T}}) where T<:Real
  converted = Array{T, 2}(undef, length(object), 2)
  for col in 1:length(object)
    converted[col,:] .= convert(Array, object[col])
  end
  return converted
end
  
function Serialization.serialize(object::Array{T}) where T<:Real
  return object
end

function Serialization.serialize(object::Array{T}) where T<:CartesianIndex
  return convert(Array, convert.(Tuple, object))
end

function Serialization.serialize(object::Dict)
  return JSON.json(object)
end

function Serialization.serialize(object::Tuple)
  return object
end


function pickle(filepath::String, vars...)
  h5open(filepath, "w") do file
    for v in vars
      write(file, v.first, Serialization.serialize(v.second))
    end
  end
end

function read_pickled(filepath::String, var_name)
  c = h5open(filepath, "r") do file
    read(file, var_name);
  end;
  return c
end


const GLOBAL_SIMULATION_RESULTS_DIR = "/home/jasonsun0310/datavolume/matrix_completion_simulation_result/"
