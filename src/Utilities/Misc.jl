using Printf

const bad_compare_msg(obj,exp,got) = @sprintf("Expected %s to be %s, instead got %s",string(obj),string(exp),string(got))


@overload
function Concepts.check(object::Type{Val{:rank}},of::Matrix{T},is::Optional{Integer}=nothing) where T<:Number
    if isnothing(is)
        return LinearAlgebra.rank(of)
    end
    got_rank = LinearAlgebra.rank(of)
    if got_rank == is
        return true
    end
    @warn bad_compare_msg(:rank,is,got_rank)
    return false
    
end


@overload
function Concepts.check(object::Type{Val{:dimension}},of::Array{T,2},is::Optional{Tuple}=nothing) where T<:Any
    if isnothing(is)
        return size(of)
    end
    got_size = size(of)
    if got_size == is
        return true
    end
    @warn bad_compare_msg(:dimension,is,got_size)
    return false
end



@overload
function Concepts.check(object::Union{Type{Val{:l2difference}},Type{Val{:l2diff}}},
                        a::Union{T1,Array{T1}},b::Union{T2,Array{T2}},
                        against::Optional{S}=nothing) where {T1<:Number,T2<:Number,S<:Real}
    if isnothing(against)
        return LinearAlgebra.norm(a-b,2)
    end
    got_diff = LinearAlgebra.norm(a-b,2)
    if abs(got_diff - against) < 1e-5
        return true
    end
    @warn bad_compare_msg(:l2difference,against,got_diff)
    return false
end



@overload
function Base.zeros(like::Array{T}) where T<:Any
    return zeros(size(like))
end


# This is necessary due to compiler bug??
@overload
const Concepts.check(object::Symbol,arg1) = Concepts.check(Val{object},arg1)

@overload
const Concepts.check(object::Symbol,arg1,arg2) = Concepts.check(Val{object},arg1,arg2)

@overload
const Concepts.check(object::Symbol,arg1,arg2,arg3) =  Concepts.check(Val{object},arg1,arg2,arg3)
