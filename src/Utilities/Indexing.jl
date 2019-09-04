using ..Concepts
using ..ModelFitting
import LinearAlgebra


struct IndexTracker{T<:Any} <: AbstractTracker
    indices::Optional{Dict}

    @abstract_instance
    function IndexTracker{T}() where T<:Any
        return new{T}(nothing)
    end

    function IndexTracker{Symbol}(data::Vararg{Array{Symbol}})
        ret = Dict()
        # this is not a proper FP way.. will make it proper FP when I have time!!
        has_same_dimension = length(unique(map(x -> size(x),data))) == 1
        if !has_same_dimension
            throw(DimensionMismatch())
        end
        list_of_unique_symbols = map(x -> unique(x),data)
        number_of_unique_symbols               = mapreduce(x -> length(x),+,list_of_unique_symbols)
        number_of_unique_symbols_after_flatten = length(unique(collect(Base.Iterators.flatten(list_of_unique_symbols))))
        if number_of_unique_symbols > number_of_unique_symbols_after_flatten
            throw(DomainError("There are duplicate symbols"))
        end
        for (available_symbols,sub_data) in zip(list_of_unique_symbols,data)
            for symbol in available_symbols
                ret[symbol] = findall(x -> x == symbol, sub_data)
            end
        end
        return new(ret)
    end
end

@overload
function Base.getindex(object::IndexTracker,i::Symbol)
    if isnothing(object.indices)
        @warn "indices are not constructed."
        return nothing
    end
    return object.indices[i]
end





mutable struct DistributionTracker <: AbstractTracker
#    view::Array{Symbol}
     indices::IndexTracker
    
    function DistributionTracker(data::Array{MaybeMissing{T},1}) where T<:Real
        view = Array{Symbol}(undef,size(data))
        # more to be added
        if check(:continuous,x) == true
            view .= choose(:Gaussian,:Gamma,x)
        end
        if check(:integral,x) == true
            if check(:binary,x) == true
                view .= :Bernoulli
            end
            view .= choose(:Poisson,:NegativeBinomial,x)
        end
        view .= :Unknown
        view[findall(x->ismissing,data)] = :Missing
        
        new_object.indices = IndexTracker(data)
    end

    # function DistributionTracker(data::Array{MaybeMissing{T},2};data_layout::Symbol = :bycol) where T<:Real
    #     view = Array{Symbol}(undef,size(data))
    #     if data_layout == :bycol
    #         for i = size(data)[2]
    #             local x = filter(x -> !ismissing(x),data[:,i])
    #             if check(:continuous,x) == true
    #                 view[:,i] .= choose(:Gaussian,:Gamma,x)
    #             end
    #             if check(:integral,x) == true
    #                 if check(:binary,x) == true
    #                    view[:,i] .= :Bernoulli 
    #                 end
    #                 view[:.i] .= choose(:Poisson,:NegativeBinomial,x)
    #             end
    #         end
    #         view[:,i] .= :Unknown
    #     end
    #     if data_layout == :byrow
    #         for i = size(data)[1]
    #             local x = filter(x -> !ismissing(x),data[i,:])
    #             if check(:continuous,x) == true
    #                 view[i,:] .= choose(:Gaussian,:Gamma,x)
    #             end
    #             if check(:integral,x) == true
    #                 if check(:binary,x)
    #                     view[i,:] .= :Bernoulli
    #                 end
    #                 view[i,:] .= choose(:Poisson,:NegativeBinomial,x)
    #             end
    #         end
    #         view[i,:] .= :Unknown
    #     end
    #     return new_object
    # end
end
