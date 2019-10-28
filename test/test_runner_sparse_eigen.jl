# try include("sparse_eigen_test.jl") catch end 
# try include("./test/sparse_eigen_test.jl") catch end

# test_native_eigen()

import LinearAlgebra

function create_symmetric_matrix(n)
    a = rand(n,n) * 5
    return a+a'
end


function correct_output_sparseeigen(input,k)
    eigen_dcp = LinearAlgebra.eigen(input);
    eigen_val = eigen_dcp.values;
    eigen_vec = eigen_dcp.vectors;
    first_k_idx = Base.sortperm(eigen_val,rev=true)[1:k];
    return eigen_val[first_k_idx],eigen_vec[:,first_k_idx];
end


function project(v,e)
    return e * LinearAlgebra.Diagonal(v) * e';
end


function test_native_eigen(;n = 200, nev = 20)
  input = create_symmetric_matrix(n);
  @time λ, X = eigs(NativeEigen(), input ;nev = nev);
  λ₀, X₀ = correct_output_sparseeigen(input, nev);
  p = project(λ, X)
  p₀ = project(λ₀, X₀)
  @test LinearAlgebra.norm(p - p₀) < 0.01
end


function test_krylov(;n = 200, nev = 20)
  a = create_symmetric_matrix(n) * 5
  @info("doing krylov eigen")
  @time λ, X = eigs(KrylovMethods(), a, nev = nev)
  @info("doing full eigen")
  @time λ₀, X₀ = correct_output_sparseeigen(a, nev)
  p = project(λ, X)
  p₀ = project(λ₀, X₀)
  @test LinearAlgebra.norm(p - p₀)^2 / LinearAlgebra.norm(p₀)^2 < 0.01
end


function test_lobpcg(;dim= 2000,nev=20,repeat=5)
  input = create_symmetric_matrix(dim);
  for i = 1:repeat
    @time begin
      λ, X = eigs(NativeLOBPCG(), input; nev=nev)
      p1 = get_projection(λ, X)
    end 
    @time begin
      v0, e0 =  correct_output_sparseeigen(input, 20)
      p2 = get_projection(v0, e0)
    end
    @test LinearAlgebra.norm(p1 - p2) < 0.01
  end
end



@testset "$(format("Sparse Eigen: KrylovKit Wrapper"))" begin
  let
    for i in 1:10
      test_krylov(n = 2000, nev = 20)
    end
  end
end




# @testset "$(format("Sparse Eigen: NativeEigen Wrapper"))" begin
#   let
#     for i in 1:10
#       test_native_eigen(n = 2000, nev = 20)
#     end
#   end
# end
