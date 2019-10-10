
using .MathLibSignatures
using ..Utilities.FastEigen

using LinearAlgebra


function MathLibSignatures.project(to::SemidefiniteCone, mat::Array{Float64, 2};
                                   eigs_implementation = KrylovMethods())
  if isnothing(rank(to))
    # do a full projection
    Λ, X = eigen(mat)
    Λ₊_idx = findall(x -> x > 0, Λ)
    Λ₊ = Λ[Λ₊_idx]
    X₊ = X[Λ₊_idx] 
    return X₊ * diagm(0 => Λ₊) * X₊';
  end
  # we are computing the full projection
  Λ, X = eigs(eigs_implementation, mat, nev = to.rank)
  return X * diagm(0 => Λ) * X'

  # TODO: further extract positive eigen values
end

function MathLibSignatures.project(to::ClosedInterval, x::AutoboxedArray{Float64})
  return max.(to.ll, min.(x, to.rr))
end

function MathLibSignatures.project!(to::ClosedInterval, x::Array{Float64})
  @. x = max.(to.ll, min.(x, to.rr))
end





  
