module BatchUtils

using ...Concepts

abstract type BatchingStrategy end
abstract type SequentialScan           <: BatchingStrategy end
abstract type SampleWithReplacement    <: BatchingStrategy end
abstract type SampleWithoutReplacement <: BatchingStrategy end


export BatchFactory,
  next,
  reset,
  has_next,
  initialize,
  BatchingStrategy,
  SequentialScan


mutable struct BatchFactory{T<:Any} 
  n::Optional{Int64}
  batch_size::Optional{Int64}
  cur_batch::Optional{Int64}


  function BatchFactory{S}(; size::Int64) where S<:BatchingStrategy
    return new{S}(nothing, size, nothing)
  end
    
  function BatchFactory{S}(n, batch_size) where S<:BatchingStrategy
    return new{S}(n, batch_size, 1)
  end
end

function initialize(obj::BatchFactory{T}, x::Array{S, 1}) where {T<:BatchingStrategy, S<:Any}
  obj.n = length(x)
  obj.cur_batch = 1
end

function next(obj::BatchFactory{SequentialScan})

  start_pos = obj.batch_size * (obj.cur_batch - 1)  + 1
  end_pos   = min(start_pos + obj.batch_size - 1, obj.n)
  obj.cur_batch = obj.cur_batch + 1
  return start_pos:end_pos
end

function Base.reset(obj::BatchFactory{SequentialScan})
  obj.cur_batch = 1
end

function has_next(obj::BatchFactory{SequentialScan})
  return obj.batch_size * (obj.cur_batch - 1)  + 1 <= obj.n
end


end # end of BatchUtils
