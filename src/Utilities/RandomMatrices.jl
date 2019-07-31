module RandomMatrices

import Distributions
import Random
import StatsBase


"""
Type-aliaes:
"""
const UnivariateDistributions = Distributions.Distribution{Distributions.Univariate,S} where S<:Distributions.ValueSupport





"""
Helper function to ensure the random matrix to be generated is actually feasible.
"""
function _ensure_feasible(row::T,col::T,rank::T) where T<:Integer
        if col <0 || row <0
            throw(DomainError("The dimension of the matrix should be positive integers."))
        end
        if rank <= 0 || rank > min(row,col)
            throw(DomainError("The rank of the matrix should be a positive integer less than min(row,col)"))
        end

end


"""
Helper function to ensure the input of mixed_dists is valid
"""
function _ensure_feasible(mixed_dists::Vector{Tuple{T,Pair{I,I},I}}) where {T<:UnivariateDistributions,I<:Integer}
    if length(mixed_dists) <= 0
        throw(DomainError("The dimension of the mixed distributions should be >= 1."))
    end
end





"""
Generate a random matrix of given rank with distribution of 'dist' of given size (rol,col) from user input.

Precondition(s):
1. The distribution has to be an element from the "Distributions.jl" and of subtype "Univariate".
"""
function rand(dist::T,row::I,col::I;target_rank::I) where {T<:UnivariateDistributions,I<:Integer}
    _ensure_feasible(row,col,target_rank);
    base_matrix = Random.rand(dist,row,target_rank)
    redundant_matrix = base_matrix[:,StatsBase.sample(1:target_rank,col-target_rank)];
    return hcat(base_matrix,redundant_matrix) * 1.0;
    # for remaining = 1:col-target_rank
    #     base_matrix = hcat(base_matrix,base_matrix[:,StatsBase.sample(1:target_rank)])
    # end
    # return base_matrix * 1.0;
end





"""
Generate a random matrix of consisting multiple types of distributions.

Precondition(s):
1. length(mixed_dists) has to be >= 1.
"""
function rand(mixed_dists::Vector{Tuple{T,Pair{I,I},I}}) where {T<:UnivariateDistributions,I<:Integer}
    _ensure_feasible(mixed_dists)
    gen_mat = mapreduce(x -> rand(x[1],x[2].first,x[2].second;target_rank = x[3]),(x,y)->hcat(x,y),mixed_dists);
    return gen_mat;
end


end
