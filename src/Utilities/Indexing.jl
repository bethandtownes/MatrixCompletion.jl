using ..Concepts
using ..ModelFitting
import LinearAlgebra

export IndexTracker


function has_same_dimension(data)
  if length(data) == 0
    return false
  end
  first_sz = size(data[1])
  for _data in data
    if size(_data) != first_sz
      return false
    end
  end
  return true
end


mutable struct IndexTracker{T<:Any} <: AbstractTracker
  indices::Optional{Dict{T, Array{CartesianIndex{N}, 1}}} where N<:Any
  dimension::Optional{Tuple{Vararg{Int64}}}

  function IndexTracker{T}() where {T<:Any}
    return new(nothing, nothing)
  end
  
  function IndexTracker{T}(data::Vararg{Array{T, N}}) where {T<:Any, N<:Any}
    if !has_same_dimension(data)
      @warn("Data stream of different dimension. Won't construct.")
      throw(DimensionMismatch())
    end
    new_object = new(Dict{T,Array{CartesianIndex{N}, 1}}(), size(data[1]))
    for _data in data
      disjoint_join(new_object, _data)
    end
    return new_object
  end

end


@overload
function Base.getindex(object::IndexTracker{T}, i::T) where T<:Any
  if isnothing(object.indices)
    @warn "indices are not constructed."
    return nothing
  end
  return object.indices[i]
end


@overload
function Base.getproperty(object::IndexTracker{T}, sym::Symbol) where T<:Any
  if sym == :keys
    return collect(keys(object.indices))
  elseif sym == :dimension
    return getfield(object, :dimension)
  elseif sym == :indices
    return getfield(object, :indices)
  elseif sym == :size
    return object.dimension
  elseif sym == :dim
    return object.dimension
  end
end 


@overload
function Base.size(object::IndexTracker)
  return object.dimension
end


@overload
function Concepts.provide(object::IndexTracker{T}, data::Vararg{Array{T}}) where T<:Any
    return IndexTracker{T}(data);
end


@overload
function Concepts.groupby(obj::IndexTracker{T}, list::Array{T}) where T<:Any
    result = Dict{T, Dict{T, Array{<:CartesianIndex}}}()
    for a_key in collect(unique(obj.keys))
        result[a_key] = Dict{T, Array{<:CartesianIndex}}()
        for sym in list   
            result[a_key][sym] = intersect(obj[a_key], obj[sym])
        end
    end
    return result
end


@overload
function Base.convert(::Type{Array{<:CartesianIndex}}, x::Union{Array{Int64, 1}, Array{CartesianIndex}})
  if typeof(x) <: Array{Int64,1}
    return [CartesianIndex{1}(_x) for _x in x]
  end
  return x
end


@overload 
function Concepts.disjoint_join(a::IndexTracker{T}, b::Array{T, N}) where {T<:Any, N<:Any}
  if isnothing(a.dimension)
    a.dimension = size(b)
  elseif size(a) != size(b)
    @show(size(a))
    @show(size(b))
    throw(DimensionMismatch())
  end
  
  if isnothing(a.indices)
    a.indices = Dict{T, Array{CartesianIndex{N}, 1}}()
  end
  if length(intersect(unique(a.keys), unique(b))) > 0
    @warn("New View is not disjoint from the old")
    throw(MethodError())
  end
  for sym in collect(unique(b))
    a.indices[sym] = convert(Array{<:CartesianIndex} ,findall(x -> x == sym, b))
  end
end
