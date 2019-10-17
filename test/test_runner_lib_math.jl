using MatrixCompletion.MathLib
import LinearAlgebra


function create_symmetric_matrix(n)
    a = rand(n,n)*5
    return a+a'
end

function correct_output_sparseeigen(input,k)
    eigen_dcp = LinearAlgebra.eigen(input);
    eigen_val = eigen_dcp.values;
    eigen_vec = eigen_dcp.vectors;
    first_k_idx = Base.sortperm(eigen_val,rev=true)[1:k];
    return eigen_val[first_k_idx],eigen_vec[:,first_k_idx];
end


@testset "$(format("Math Library: Projection[SemidefiniteCone]"))" begin
  let
    for i in 1:50
      data = create_symmetric_matrix(100)
      rk = rand(2:30)
      tc = project(SemidefiniteCone(rank = rk), data)
      λ0, X0 = correct_output_sparseeigen(data, rk)
      tc_comp = X0 * LinearAlgebra.Diagonal(λ0) * X0'
      @test LinearAlgebra.norm(tc_comp - tc)^2 / LinearAlgebra.norm(tc_comp) ^ 2 < 0.02
    end
  end
end

