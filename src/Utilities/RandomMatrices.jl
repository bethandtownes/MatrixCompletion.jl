using ..Concepts
import Random
import Distributions





struct FixedRankMatrix{T<:UnivariateDistributions} <: AbstractFixedRankMatrix
    dist::T
    rank::Optional{Integer}
    draw
    @abstract_instance
    function FixedRankMatrix{T}() where T<:UnivariateDistributions
        #@debug "abstract constructor of FixedRankMatrix"
        return new{T}()
    end

    
    function FixedRankMatrix{T}(dist::T;rank::Optional{Integer}=nothing) where T<:UnivariateDistributions
        return new{T}(dist,rank)
    end
end



const FixedRankMatrix(dist::T;rank::Optional{Integer}=nothing) where T<:UnivariateDistributions =
    begin
    isnothing(rank) ? FixedRankMatrix{T}(dist) : FixedRankMatrix{T}(dist,rank=rank)
    end


@overload
function Concepts.provide(object::FixedRankMatrix{T};row::Integer=10,col::Integer=10) where T<:UnivariateDistributions
    return rand(object,row,col)
    
end


function GaussianMatrix(row::Integer,col::Integer;
                     rank::Optional{Integer}=nothing,
                     μ::T=0, σ::T=1) where T<:Real
    if isnothing(rank)
        return rand(Distributions.Gaussian(μ,σ),row,col)
    end
    return rand(FixedRankMatrix(Distributions.Gaussian(μ,σ),rank=rank),row,col)
end


function PoissonMatrix(row::Integer,col::Integer;
                       rank::Optional{Integer}=nothing,
                       λ = 5) where T<:Real
    if isnothing(rank)
        return rand(Distributions.Poisson(λ),row,col)
    end
    return rand(FixedRankMatrix(Distributions.Poisson(λ),rank=rank),row,col)
end


function BernoulliMatrix(row::Integer,col::Integer;
                         rank::Optional{Integer}=nothing,
                         p = 0.5) where T<:Real
    if isnothing(rank)
        return rand(Distributions.Bernoulli(p),row,col)
    end
    return rand(FixedRankMatrix(Distributions.Bernoulli(p),rank=rank),row,col)
end



function GammaMatrix(row::Integer,col::Integer;
                     rank::Optional{Integer}=nothing,
                     α=5,θ=0.5) where T<:Real
    if isnothing(rank)
        return rand(Distributions.Gamma(α,θ),row,col)
    end
    return rand(FixedRankMatrix(Distributions.Gamma(α,θ),rank=rank),row,col)
end



@overload
function Random.rand(object::FixedRankMatrix{T},row::Integer,col::Integer) where T<:UnivariateDistributions
    used_rank = nothing
    if isnothing(object.rank)
        @warn "Rank is not specified. Using formula: rank= ⌊0.3 * (row ∧ col)⌋"
        used_rank = Int(0.3 * floor(min(row,col)))
    elseif object.rank > max(row,col)
        used_rank = min(row,col)
    else
        used_rank = object.rank
    end
    ensure_feasible(row,col,used_rank);
    base_matrix = Random.rand(object.dist,row,used_rank)
    redundant_matrix = base_matrix[:,StatsBase.sample(1:used_rank,col-used_rank)];
    return hcat(base_matrix,redundant_matrix) * 1.0;
end



@overload
function Random.rand(mixed_dists::Vector{Tuple{T,I,I}}) where {T<:FixedRankMatrix{<:UnivariateDistributions}, I<: Integer}
    #ensure_feasible(mixed_dists)
    # TODO: Make a ensure feasible function 
    gen_mat = mapreduce(x -> rand(x[1],x[2],x[3]),(x,y)->hcat(x,y),mixed_dists);
    return gen_mat;
end





function ensure_feasible(row::T,col::T,rank::T) where T<:Integer
        if col <0 || row <0
            throw(DomainError("The dimension of the matrix should be positive integers."))
        end
        if rank <= 0 || rank > min(row,col)
            throw(DomainError("The rank of the matrix should be a positive integer less than min(row,col)"))
        end
end































"""
Generate a random matrix of given rank with distribution of 'dist' of given size (rol,col) from user input.

Precondition(s):
1. The distribution has to be an element from the "Distributions.jl" and of subtype "Univariate".
"""

# function rand(dist::T,row::I,col::I;target_rank::I) where {T<:UnivariateDistributions,I<:Integer}
#     ensure_feasible(row,col,target_rank);
#     base_matrix = Random.rand(dist,row,target_rank)
#     redundant_matrix = base_matrix[:,StatsBase.sample(1:target_rank,col-target_rank)];
#     return hcat(base_matrix,redundant_matrix) * 1.0;
# end






"""
Generate a random matrix of consisting multiple types of distributions.

Precondition(s):
1. length(mixed_dists) has to be >= 1.
"""
# function rand(mixed_dists::Vector{Tuple{T,Pair{I,I},I}}) where {T<:UnivariateDistributions,I<:Integer}
#     _ensure_feasible(mixed_dists)
#     gen_mat = mapreduce(x -> rand(x[1],x[2].first,x[2].second;target_rank = x[3]),(x,y)->hcat(x,y),mixed_dists);
#     return gen_mat;
# end







# end
