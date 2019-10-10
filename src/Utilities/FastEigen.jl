module FastEigen

export FortranArnoldiMethod,
  NativeLanczos,
  NativeEigen,
  NativeLOBPCG,
  NativeArnoldiMethod,
  NativeEigen,
  KrylovMethods


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



struct KrylovMethods end



function eigs(algorithm::KrylovMethods, x::Array{Float64, 2}; nev::Int64 = 10, order::Symbol = :LR, symmetric::Bool = true)
  λ, X = KrylovKit.eigsolve(x, nev, order;issymmetric = symmetric)
  X = hcat(X...)
  return λ[1:nev], X[:, 1:nev]
end



function eigs(algorithm::NativeEigen,x::Array{Float64,2};
              nev::Integer=6,eigen_vectors::Bool=true,order::Symbol=:LR)
  local eigen_val_id;
  eigen_decomp = LinearAlgebra.eigen(x);
  if order == :LR
    eigen_val_id = Base.partialsortperm(eigen_decomp.values,1:nev,rev=true);
  elseif order == :SR
    eigen_val_id = Base.partialsortperm(eigen_decomp.values,1:nev,rev=false)
  end
  if eigen_vectors == true
    return eigen_decomp.values[eigen_val_id],eigen_decomp.vectors[:,eigen_val_id] 
  end
  return eigen_decomp.values[eigen_val_id];
end



function eigs(algorithm::NativeLOBPCG,x::Array{Float64,2};
              nev::Integer=6,eigen_vectors::Bool=true,order::Symbol=:LR) 
  local eigen_decomp;
  if order == :LR
    eigen_decomp = IterativeSolvers.lobpcg(x,true, nev);
  end
  if order == :SR 
    eigen_decomp = IterativeSolvers.lobpcg(x,false, nev);
  end
  if eigen_vectors == true
    return eigen_decomp.λ, eigen_decomp.X;
  end
  return eigen_decomp.λ
end



end
