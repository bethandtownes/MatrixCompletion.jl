module MathLibSignatures
using ..Concepts

export MathematicalObject,
  Cone,
  Interval,
  SemidefiniteCone,
  ClosedInterval
  


abstract type MathematicalObject end
abstract type Cone     <: MathematicalObject  end
abstract type Interval <: MathematicalObject  end

struct SemidefiniteCone <: Cone
  rank::Optional{Int64}
  function SemidefiniteCone(; rank::Optional{Int64} = nothing)
    if rank == nothing
      return new(nothing)
    end
    if rank <= 0
      @warn("Rank should never be below zero. Check again.")
      throw(MethodError())
    end
    return new(rank)
  end
end

struct ClosedInterval{T} <: Interval where T<:Real
  ll::Optional{T}
  rr::Optional{T}

  function ClosedInterval{T}(ll::T, rr::T) where T<:Real
    if ll > rr
      @warn("Right end point of the interval shoud be bigger than left, return an empty interval instead.")
      return new{T}(nothing, nothing)
    end
    return new{T}(ll, rr)
  end
end


function project() end

function project!() end

function scope_test()
  @info("inside MathLib.Concepts")
end

end
