using MatrixCompletion.Utilities.FastEigen
using IterativeSolvers
import LinearAlgebra



function create_symmetric_matrix(n)
    a = rand(n,n)*5
    return a+a'
end




function correct_output_sparseeigen(input,n)
    eigen_dcp = LinearAlgebra.eigen(input);
    
end


function get_projection(e,v)
    return e * diagm(v) * e
end

function test_native_eigen(;dim=500,nev=20,repeat=5)
    input = create_symmetric_matrix(dim);
    for i = 1:repeat
        @time λ,X = eigs(input;nev=nev,algorithm=NativeEigen());

    end
end


function test_lobpcg_wrapper(;dim=1000,nev=20,repeat=5)
    input = create_symmetric_matrix(dim);
    for i = 1:repeat
        @time λ,X = eigs(input;nev=nev,algorithm=NativeLOBPCG());
    end
end

function test_lobpcg_via_import(;dim=1000,nev=20,repeat=5)
    input = create_symmetric_matrix(dim);
    for i = 1:5
        @time lobpcg(input,true,nev);
    end
end

