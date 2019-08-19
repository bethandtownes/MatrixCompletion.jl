using MatrixCompletion.Utilities.FastEigen
using IterativeSolvers
using LinearAlgebra
using Test




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


function get_projection(v,e)
    return e * Diagonal(v) * e';
end


function test_native_eigen(;dim=500,nev=20,repeat=5)
    for i = 1:repeat   
        input = create_symmetric_matrix(dim);
        @time λ,X = eigs(NativeEigen(),input;nev=nev);
        λ0,X0 = correct_output_sparseeigen(input,nev);
        @test norm(get_projection(λ,X) - get_projection(λ0,X0),2)<1e-3;
    end
end



function test_lobpcg_wrapper(;dim=1000,nev=20,repeat=5)
    input = create_symmetric_matrix(dim);
    for i = 1:repeat
        @time λ,X = eigs(NativeLOBPCG(),input;nev=nev);
    end
end

function test_lobpcg_via_import(;dim=1000,nev=20,repeat=5)
    input = create_symmetric_matrix(dim);
    for i = 1:5
        @time lobpcg(input,true,nev);
    end
end

