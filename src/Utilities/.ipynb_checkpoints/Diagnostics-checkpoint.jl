module Diagn

export relative_error,
    total_error


using MatrixCompletion.Concepts
using LinearAlgebra 


struct ErrorMatrix{T<:ExponentialFamily} end


function relative_error(metric::L2Distance;input::Array{Float64} = nothing, reference::Array{Float64})
   return norm(input .- reference,2)/norm(reference,2)
end



function relative_error(metric::L0Distance;input::Array{Float64}=nothing,reference::Array{Float64}=nothing,stack::Float64=1e-5)
    return sum(Int.(input - reference .> slack)) / length(reference)
end






function total_error(metric::L2Distance;input::Array{Float64} = nothing, reference::Array{Float64} = nothing)
    return norm(input .- reference,2)
end




function total_error(metric::L0Distance;input::Array{Float64} =nothing,reference::Array{Float64}=nothing)
    return sum(Int.(input == reference))
end




function MatrixCompletion.Concepts.provide(object::Diagnostics{AbstractGamma};
                 input_data::Array{Float64} = nothing,
                 reference ::Array{Float64} = nothing)
    

    return Dict{"relative-error(L2 Distance)" => relative_error(L2Distance(),input_data,reference),
                "error-matrix" => (input_data - reference),
                "total-error(L2 Distance)"    => total_error(L2Distance,input_data, reference)}
end



function Concepts.provide(object::Diagnostics{AbstractGaussian};
                 input_data::Array{Float64} = nothing,
                 reference ::Array{Float64} = nothing)

    return Dict{"relative-error(L2 Distance)" => relative_error(L2Distance(),input_data,reference),
                "error-matrix" => (input_data - reference),
                "total-error(L2 Distance)"    => total_error(L2Distance,input_data, reference)}
end


function Concepts.provide(object::Diagnostics{AbstractBinomial};
                 input_data::Array{Float64} = nothing,
                 reference::Array{Float64}  = nothing)
    return Dict{"relative-error(L0 Distance)" => relative_error(L0Distance(),input_data,reference),
                "error-matrix" => (input_data-reference),
                "total_error(L0 Distance)" => total_error(L0Distance,input_data,reference)}
end



function Concepts.provide(object::Diagnostics{AbstractPoisson};
                 input_data::Array{Float64} = nothing,
                 reference::Array{Float64}  = nothing)
    return Dict{"relative-error(L0 Distance)" => relative_error(L0Distance(),input_data,reference),
                "relative-error(L0 Distance,slack=1)" => relative_error(L0Distance(),input_data,reference),
                "relative-error(L0 Distance,slack=2)" => relative_error(L0Distance(),input_data,reference),
                "error-matrix" => input_data-reference,
                "total_error(L0 Distance)" => total_error(L0Distance,input_data,reference)}
end


# function Concepts.provide(object::Diagnostics{AbstractNegativeBinomial};
#                  input_data::Array{Float64} = nothing,
#                  reference::Array{Float64}  = nothing)
#     return Dict{"relative-error(L0 Distance)" => relative_error(L0Distance(),input_data,reference),
#                 "error-matrix" => (input_data-reference),
#                 "total_error(L0 Distance)" => total_error(L0Distance,input_data,reference)}
# end


end
