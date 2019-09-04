# import ..Concepts:AbstractSamplingModels,
#     BernoulliModel,
#     VecOrMatOfNumbers,
#     provide

using ..Concepts

import StatsBase
                 
                 


struct Sampler{T<:AbstractSamplingModels}
    model::T
    draw

    function Sampler{BernoulliModel}()
        # abstract type constructor
        return new{BernoulliModel}()
    end

    function Sampler{UniformModel}()
        # abstract type constructor
        return new{UniformModel}()
    end

    function Sampler{NonUniformModel}()
        # abstract type constructor
        return new{NonUniformModel}()
    end

    function Sampler{BernoulliModel}(model::BernoulliModel)
        draw = begin
            function(x::VecOrMatOf{Number})
                if isa(x,Vector)
                    n = length(x)
                    mask = [rand(Distributions.Bernoulli(model.rate)) == 1 ? 1 : missing for i in 1:n]
                    return mask .* x
                end
                n,m = size(x);
                mask = [rand(Distributions.Bernoulli(model.rate)) == 1 ? 1 : missing for i in 1:n,j in 1:m]
                return mask .* x
            end
        end
        return new{BernoulliModel}(model,draw)
    end

    function Sampler{UniformModel}(model::UniformModel)
        draw = begin
            function(x::VecOrMatOf{T}) where T<:Any
                sampled_object = nothing
                if isa(x,Vector)
                    n = length(x)
                    mask  = [CartesianIndex(i) for i in StatsBase.sample(1:n,Int.(n * model.rate))]
                    sampled_object = convert(MaybeMissing{T},Array{Missing}(undef,n))
                    for i in mask
                        sampled_object[i] = x[i]
                    end
                else  
                    row,col = size(x)
                    mask = [CartesianIndex(StatsBase.sample(1:row),StatsBase.sample(1:col)) for i in 1:Int.(row * col * model.rate)]
#                    display(mask)
                    sampled_object = convert(MaybeMissing{T},Array{Missing,2}(undef,row,col))
#                    display(sampled_object)
#                    display(x)
                    for i in mask
                        sampled_object[i] = x[i]
                    end
                end
                return sampled_object
            end
        end
        
        return new(model,draw)
    end
end


const Sampler(model::BernoulliModel) = Sampler{BernoulliModel}(model)
const Sampler(model::UniformModel)   = Sampler{UniformModel}(model)




@overload
function Concepts.provide(object::Sampler{Concepts.BernoulliModel};rate)
    return Sampler(BernoulliModel(rate))
end

@overload
function Concepts.provide(object::Sampler{Concepts.UniformModel};rate)
    return Sampler(UniformModel(rate))
end


