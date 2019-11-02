using .MathLibSignatures
using ..Utilities.FastEigen

using LinearAlgebra





function MathLibSignatures.project(to::SemidefiniteCone, mat::Array{Float64, 2};
                                   eigs_implementation = KrylovMethods())
  if isnothing(rank(to))
    # do a full projection
    # @warn("Doing full eigen projection, could be costly!")
    eigDecomposition    = eigen(mat);
    posEigenValuesIndex = findall(x -> real(x) > 0,eigDecomposition.values);
    posEigenValues      = eigDecomposition.values[posEigenValuesIndex];
    posEigenVectors     = eigDecomposition.vectors[:,posEigenValuesIndex];
    projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
    return projectedMatrix;
  end
  # we are computing the full projection
  Λ, X = eigs(eigs_implementation, mat, nev = to.rank)
  # return X * diagm(0 => Λ) * X' 
  id = findall(x -> real(x) > 0, Λ)
  return X[:, id] * diagm(0 => Λ[id]) * (X[:,id])' 
end

function MathLibSignatures.project(to::ClosedInterval, x::AutoboxedArray{Float64})
  return max.(to.ll, min.(x, to.rr))
end

function MathLibSignatures.project!(to::ClosedInterval, x::Array{Float64})
  @. x = max.(to.ll, min.(x, to.rr))
end

