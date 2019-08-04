module FastEigen

export FortranArnoldiMethod,
       NativeLanczos,
       NativeEigen,
       NativeLOBPCG,
       NativeArnoldiMethod,
       NativeEigen

export eigs

import KrylovKit,IterativeSolvers,ArnoldiMethod,Arpack,LinearAlgebra





mutable struct FortranArnoldiMethod 
    maxiter::Union{Float64,Nothing}
    tol::Union{Float64,Nothing}
    function FortranArnoldiMethod(;maxiter=nothing,tol=nothing)
        instance = new();
        instance.maxiter = maxiter;
        instance.tol = tol;
        return instance;
    end
end

mutable struct NativeLanczos 
    maxiter::Union{Float64,Nothing}
    tol::Union{Float64,Nothing}
    function NativeLanczos(;maxiter=nothing,tol=nothing)
        instance = new();
        instance.maxiter = maxiter;
        instance.tol = tol;
        return instance;
    end
end

mutable struct NativeEigen end 

mutable struct NativeArnoldiMethod 
    maxiter::Union{Float64,Nothing}
    tol::Union{Float64,Nothing} 
    function NativeArnoldiMethod(;maxiter=nothing,tol=nothing)
        instance = new();
        instance.maxiter = maxiter;
        instance.tol = tol;
        return instance;
    end
end


mutable struct NativeLOBPCG 
    maxiter::Union{Float64,Nothing}
    tol::Union{Float64,Nothing}
    function NativeLOBPCG(;maxiter=200,tol=nothing)
        instance = new();
        instance.maxiter = maxiter;
        instance.tol = tol;
        return instance;
    end
end


function eigs(x::Array{Float64,2};
              nev::Integer=6,eigen_vectors::Bool=true,order::Symbol=:LR,algorithm::NativeEigen)
    local eigen_val_id;
    eigen_decomp = LinearAlgebra.eigen(x);
    if order == :LR
        eigen_val_id = Base.partialsortperm(eigen_decomp.eigenvalues,nev,rev=true);
    elseif order == :SR
        eigen_val_id = Base.partialsortperm(eigen_decomp.eigenvalues,nev,rev=false)
    end
    if eigen_vectors == true
        return eigen_decomp.eigenvalues[eigen_val_id],eigen_decomp.eigen_vectorss[:,eigen_val_id] 
    end
    return eigen_decomp.eigenvalues[eigen_val_id];
end


 
function eigs(x::Array{Float64,2};
              nev::Integer=6,eigen_vectors::Bool=true,order::Symbol=:LR,algorithm::NativeLOBPCG) 
    local eigen_decomp;
    if order == :LR
        eigen_decomp = IterativeSolvers.lobpcg(x,true,6);
    end
    if order == :SR 
        eigen_decomp = IterativeSolvers.lobpcg(x,false,6);
    end
    if eigen_vectors == true
        return eigen_decomp.λ, eigen_decomp.X;
    end
    return eigen_decomp.λ
end





end