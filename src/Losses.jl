module Losses
using Printf
import Random
import AutoGrad

struct SGD end



####################### HELPER METHODS ###################
# vectorized sigmoid function
σ(z) = 1.0 ./ (1.0 .+ exp.(-z))
##########################################################



function Poisson()
    L(x,y,c,ρ) = sum(exp.(x) .- y .* x) + sum(ρ .* (x .- c).^2)
    return L;
end


function Logistic()
    L(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);
    return L;
end



loss_logistic(x,y,c,ρ) = -sum(y .* log.(σ.(x)) .+ (1 .- y) .* log.(1 .- σ.(x))) .+  sum(ρ .* (x .- c).^2);


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


function train(loss,fx,y,c,ρ;γ=0.02,iter=20)
    ∇ = AutoGrad.grad(loss);
    curFx = fx;
    for i = 1:iter
        curFx = curFx .- γ .* ∇(curFx,y,c,ρ);
        # @printf("iter:%d, loss[bernoulli]=%f\n",i,∇(curFx,y,c,ρ));
    end
    return curFx;
end



function train(algo::SGD,loss,fx,y,c,ρ;γ=0.02,epoch=10,num_of_batch=20)
    ∇ = AutoGrad.grad(loss);
    n = length(fx);
    curFx = fx;
    batch_size = Int(n/num_of_batch);
    for i = 1:epoch
        for sample in 1:batch_size:n
            batch = sample:min(sample+batch_size-1,n);
            curFx[batch] = curFx[batch] .- γ .* ∇(curFx[batch],y[batch],c[batch],ρ);
            @printf("loss:%f\n",loss(curFx,y,c,ρ));
        end
        @printf("iter:%d, loss:%f\n",i,loss(curFx,y,c,ρ));
    end
    return curFx;
end



end
