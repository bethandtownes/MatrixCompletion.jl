using MatrixCompletion.Concepts
import LinearAlgebra 




struct RelativeError end
struct AbsoluteError end



function within_radius(x;of=1e-5,at=0)
  return sum(Int.(abs.(x .- at) .> of))
end


function Concepts.provide(object::RelativeError,x,y;
                          metric::Any = x -> LinearAlgebra.norm(x,2),
                          base_metric::Any=metric)
  return metric(x-y)/base_metric(y)
end




function Concepts.provide(object::AbsoluteError,x,y;
                          metric::Any= x-> LinearAlgebra.norm(x,2))
  return metric(x - y)
end



function Concepts.provide(object::Diagnostics{Int64})
  return "int64"
end




function Concepts.provide(object::Diagnostics{<:Any};
                          reference::Optional{VecOrMatOfReals}=nothing,
                          input_data::Optional{VecOrMatOfReals}=nothing) 
  if isnothing(input_data) && isnothing(reference)  
    throw(DomainError("[provide(Diagnostics)]: input_data and/or reference variable missing))"))
  end
  return Dict("relative-error[#within-radius(1e-5)]" => Concepts.provide(RelativeError(),    
                                                                         input_data,reference,
                                                                         metric      = x -> within_radius(x),
                                                                         base_metric = x -> LinearAlgebra.norm(abs.(x) .+ 1 ,0)),
              "absolute-error[#within-radius(1e-5)]" => Concepts.provide(AbsoluteError(),
                                                                         input_data, reference,
                                                                         metric      = x -> within_radius(x)),
              "relative-error[#within-radius(1)]" => Concepts.provide(RelativeError(),    
                                                                      input_data,reference,
                                                                      metric      = x -> within_radius(x,of=1),
                                                                      base_metric = x -> LinearAlgebra.norm(abs.(x) .+ 1 ,0)),
              "absolute-error[#within-radius(1)]" => Concepts.provide(AbsoluteError(),
                                                                      input_data, reference,
                                                                      metric      = x -> within_radius(x,of=1)),
              "relative-error[L1]" => Concepts.provide(RelativeError(),
                                                       input_data,reference,
                                                       metric = x -> LinearAlgebra.norm(x,1)),
              "absolute-error[L1]" => Concepts.provide(AbsoluteError(),
                                                       input_data, reference,
                                                       metric = x -> LinearAlgebra.norm(x,1)),
              "relative-error[L2]" => Concepts.provide(RelativeError(),
                                                       input_data,reference,
                                                       metric = x -> LinearAlgebra.norm(x,2)^2),
              "absolute-error[L2]" => Concepts.provide(AbsoluteError(),
                                                       input_data,reference,
                                                       metric = x -> LinearAlgebra.norm(x,2)^2)
              )
  
end





# function print_diagnostics(object::Diagnostics{<:Any};
#                            reference::Optional{VecOrMatOfReals}=nothing,
#                            input_data::Optional{VecOrMatOfReals}=nothing)
  
  
# end

