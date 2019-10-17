# early exit
    # if (max(primfeas, dualfeas) < stoptol) | (iter == maxiter) 
    #   breakyes = 1
    # end
    # if (max(primfeas, dualfeas) < sqrt(stoptol)) & (dualfeas > 1.5 * minimum(runhist.dualfeas[max(iter - 49, 1):iter])) & (iter > 150)
    #   breakyes = 2
    # end


# tune ρ
    # if (ρReset > 0) & (rem(iter, 10)==0)
    #   if (primfeas < 0.5 * dualfeas)
    #     ρ = 0.7 * ρ
    #   elseif (primfeas > 2 * dualfeas)
    #     ρ = 1.3 * ρ
    #   end
    # end
    # if (breakyes > 0)
    #   @printf("\n break = %1.0f\n", breakyes)
    #   break;
# end




# function sdpProjection(mat::Array{Float64, 2})
#   λ, X    = eigs(KrylovMethods(), mat, nev = 20)
#   # @show(λ)
#   # @show(X)
#   # @show(size(λ))
#   # @show(size(X))
#   return project(λ, X)
#   # posEigenValuesIndex = findall(x -> x > 0, λ);
#   # posEigenValues      = λ[posEigenValuesIndex];
#   # posEigenVectors     = X[:,posEigenValuesIndex];
#   # projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
#   # return projectedMatrix;
# end

# function sdpProjection0(data)
#   eigDecomposition    = eigen(data);
#   posEigenValuesIndex = findall(x -> x>0,eigDecomposition.values);
#   posEigenValues      = eigDecomposition.values[posEigenValuesIndex];
#   posEigenVectors     = eigDecomposition.vectors[:,posEigenValuesIndex];
#   projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
#   return projectedMatrix;
# end

# # function logisticLoss(x,y)
# #   f_x = Losses.σ.(x);
# #   return -sum(y .* log.(f_x) + (1 .- y) .* log.(1 .- f_x));
# # end


 
  # return X * diagm(0 => Λ) * X'

  # TODO: further extract positive eigen values
