module Losses

using Printf
import Random, AutoGrad, Distributions

using ..Concepts

struct SGD end



####################### HELPER METHODS ###################
# vectorized sigmoid function
σ(z) = 1.0 ./ (1.0 .+ exp.(-z))
##########################################################



# function Poisson()
#     L(x,y,c,ρ) = sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
#     return L;
# end


# function Logistic()
#     L(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);



#     return L;
# end


# abstract type ExponentialFamily                             end
# abstract type AbstractBinomial         <: ExponentialFamily end
# abstract type AbstractGaussian         <: ExponentialFamily end
# abstract type AbstractPoisson          <: ExponentialFamily end
# abstract type AbstractGamma            <: ExponentialFamily end
# abstract type AbstractExponential      <: AbstractGamma     end
# abstract type AbstractNegativeBinomial <: ExponentialFamily end
# abstract type AbstractGeometric        <: AbstractNegativeBinomial  end



#struct Loss{T} end 



#export Loss,ExponentialFamily,
 #   AbstractBinomial,AbstractGaussian,AbstractPoisson,AbstractGamma,AbstractNegativeBinomial,AbstractGeometric
export evaluate,grad,train


loss_logistic(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);



################################################################################
#                             Gaussian Loss                                    # 
################################################################################

function Concepts.provide(loss::Loss{AbstractGaussian})
    #TODO
end




function evaluate(loss::Loss{AbstractGaussian},
                  x,y,c,ρ)
    #TODO
end


function grad(loss::Loss{AbstractGaussian} ,
              x,y,c,ρ)
    #TODO
end


################################################################################
#                            Binomial (Logistic) Loss                          # 
################################################################################



function Concepts.provide(loss::Loss{AbstractBinomial})
     L(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);
    return L
end



function evaluate(loss::Loss{AbstractBinomial},
                  x,y,c,ρ)
    return  -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x)))
          .+ sum(ρ .* (x .- c).^2);
end


function grad(loss::Loss{AbstractBinomial},
              x,y,c,ρ)
    ex = exp.(x)
    inv_ex1 = 1 ./(ex .+ 1)
    return inv_ex1 .* (-y + (1 .-y) .* ex) .+ (2*ρ) .* (x .- c)
end



################################################################################
#                             Poisson Loss                                     #
################################################################################



function Concepts.provide(loss::Loss{AbstractPoisson})
    L(x,y,c,ρ) = sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
    return L;
end



function evaluate(loss::Loss{AbstractPoisson},
                  x,y,c,ρ)
    return sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
end



function grad(loss::Loss{AbstractPoisson},
              x,y,c,ρ)
    #    return # sum(exp.(x) .- y)  .+ (2*ρ) .* (x .- c)
    return exp.(x) .- y .+ (2*ρ) .* (x .- c)
end


################################################################################
#                              Gamma Loss                                      #
################################################################################


function Concepts.provide(loss::Loss{AbstractGamma})
    L(x,y,c,ρ) = sum(-x .* y .- log.(-x))+ sum(ρ .* (x .- c).^2)
   return L
end


function evaluate(loss::Loss{AbstractGamma},x,y,c,ρ)
    return sum(x .* y .- log.(x))+ sum(ρ .* (x .- c).^2)
end




## Use the reciprocal link instead of the negative reciprocal link
function grad(loss::Loss{AbstractGamma},x,y,c,ρ)
    return y .- (1 ./ x) .+ (2*ρ) .* (x .- c)
end
   



function grad_logistic(x,y,c,ρ)
    ex = exp.(x)
    inv_ex1 = 1 ./(ex .+ 1);
    return inv_ex1 .* (-y + (1 .-y) .* ex) .+ (2*ρ) .* (x .- c);
    # return (-y .* inv_ex1 + (1 .- y) .* (ex .* inv_ex1)) .+ (2*ρ) .* (x.-c);
end



function train_logistic(fx,y,c,ρ;γ=0.02,iter=20)
    # ∇ = AutoGrad.grad(loss);
    curFx = fx;
    for i = 1:iter
        curFx = curFx .- γ .* grad_logistic(curFx,y,c,ρ);
        if i == iter
        @printf("iter:%d, loss[bernoulli]=%f\n",i,loss_logistic(curFx,y,c,ρ));
        end
    end
    # @printf("loss:%f\n",loss_logistic(curFx,y,c,ρ));
    return curFx;
end




function train(loss;fx,y,c,ρ,γ=0.02,iter=20,verbose=false)
    ∇ = AutoGrad.grad(loss);
    curFx = fx;
    for i = 1:iter
        curFx = curFx .- γ .* ∇(curFx,y,c,ρ);
    end
    return curFx;
end


function train(native_loss::Loss{T};
               fx,y,c,ρ,γ=0.02,iter=20,verbose=false) where T<:ExponentialFamily
    curFx = fx;
    for i = 1:iter 
        curFx = curFx .- γ .* grad(native_loss,curFx,y,c,ρ);
        if verbose == true
            @printf("loss:%f\n",evaluate(native_loss,curFx,y,c,ρ ))
        end
    end
    return curFx;
end



# function train(algo::SGD,loss,fx,y,c,ρ;γ=0.02,epoch=10,num_of_batch=20)
#     ∇ = AutoGrad.grad(loss);
#     n = length(fx);
#     curFx = fx;
#     batch_size = Int(n/num_of_batch);
#     for i = 1:epoch
#         for sample in 1:batch_size:n
#             batch = sample:min(sample+batch_size-1,n);
#             curFx[batch] = curFx[batch] .- γ .* ∇(curFx[batch],y[batch],c[batch],ρ);
#             @printf("loss:%f\n",loss(curFx,y,c,ρ));
#         end
#         @printf("iter:%d, loss:%f\n",i,loss(curFx,y,c,ρ));
#     end
#     return curFx;
# end



end
