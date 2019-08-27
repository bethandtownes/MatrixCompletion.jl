module ADMM

using Printf
using LinearAlgebra
using SparseArrays


# import MatrixCompletion.Losses:train,train_logistic
# import MatrixCompletion.Losses
# import MatrixCompletion.Utilities.Indexing:DIST_FLAGS,Bernoulli,Gaussian,Poisson,Gamma,NegativeBinomial
# import MatrixCompletion.Utilities.Indexing:construct_type_matrix,construct_index_trakcer




#import ...Losses:train,train_logistic
#import ...Losses


using ...Losses
using ...Utilities.Indexing




const DefaultMatrix = Array{Union{T,Missing}} where T<:Real
const TypeMatrix = Union{Array{Union{Missing,DIST_FLAGS},2},Array{DIST_FLAGS,2}}



mutable struct RunHistory
    primfeas::Array{Float64,1}
    dualfeas::Array{Float64,1}

    function RunHistory()
        new(Array{Float64,1}(),Array{Float64,1}())
    end
end


function sdpProjection(data)
    eigDecomposition    = eigen(data);
    posEigenValuesIndex = findall(x -> x>0,eigDecomposition.values);
    posEigenValues      = eigDecomposition.values[posEigenValuesIndex];
    posEigenVectors     = eigDecomposition.vectors[:,posEigenValuesIndex];
    projectedMatrix     = posEigenVectors * diagm(0 => posEigenValues) *posEigenVectors';
    return projectedMatrix;
end


function logisticLoss(x,y)
    f_x = Losses.σ.(x);
    return -sum(y .* log.(f_x) + (1 .- y) .* log.(1 .- f_x));
end


function l1BallProjection(v,b)
    if (norm(v,1) <= b);
        return v
    end
    n = length(v);
    nvec = hcat(1:n...)'[:];
    vv = sort(abs.(v),lt = (x,y) -> !isless(x,y));
    idxsort = sortperm(abs.(v),lt = (x,y) -> !isless(x,y));
    vsum = cumsum(vv);
    tmp  = vv .-(vsum .- b)./nvec;
    idx  = findall(x->x>0,tmp);
    if !isempty(idx);
        k = maximum(idx);
    else
        println("something is wrong")
    end
    lam = (vsum[k] .- b) ./ k;
    xx = zeros(length(idxsort));
    xx[idxsort,1] = max.(vv .- lam,0);
    x = sign.(v).*xx;
    return x;
end






function _column_is_all_missing(col)
    return mapreduce(x->ismissing(x),(x,y)->x&&y,col);
end


function _has_totally_corrupted_columns(x)
    return reduce((x,y)->x&&y,mapslices(_column_is_all_missing,x,dims=1));
end



function _get_column_distribution_type(x)
    return x[findfirst(x->!ismissing(x),x)];
end




function _get_complete_type_matrix(type_matrix::TypeMatrix)
    if _has_totally_corrupted_columns(type_matrix)
        throw(BoundsError("The input type matrix has totally corrupted columns"))
    end
    n,m = size(type_matrix);
    complete_type_matrix = Array{DIST_FLAGS,2}(undef,n,m);
    for col = 1:m
        # column_type = Ref(_get_column_distribution_type(type_matrix[:,col]));
        complete_type_matrix[:,col] .= Ref(_get_column_distribution_type(type_matrix[:,col]))
        # fill(complete_type_matrix[:,col],_get_column_distribution_type(complete_type_matrix[:,col]));
    end
    return complete_type_matrix;
end




function _predict_gaussian(x::Array{T,2}) where T<:Union{Real,Missing}
    prediction = deepcopy(x);
    # already_observed_indices = findall(x->!ismissing(x),obs);
    # prediction[already_observed_indices] = obs[already_observed_indices];
    return prediction;
end

function _predict_bernoulli(x)
    prediction = (Int.(sign.(x)) .+ 1)/2
    # already_observed_indices = findall(x->!ismissing(x),obs);
    # prediction[already_observed_indices] = obs[already_observed_indices];
    return prediction
end





function _predict_poisson(x::Array{T,2}) where T<:Union{Real,Missing}
end

function _predict_gamma(;x::Array{T,2}) where T<:Union{Real,Missing}
end

function _predict_negative_binomial(;x::Array{T,2}) where T<:Union{Real,Missing}
end


function predict(;x::Array{T,2},obs::TypeMatrix) where T<:Union{Real,Missing}
    prediction = deepcopy(x);
    column_types = _get_complete_type_matrix(obs);
    columns_bernoulli = findall(x->x==Ref(Bernoulli),column_types);
    columns_gaussian = findall(x->x==Ref(Gaussian),column_types);
    columns_poisson = findall(x->x==Ref(Poisson),column_types);
    columns_gamma = findall(x->x==Ref(Gamma),column_types);
    columns_negative_binomial = findall(x->x==Ref(NegativeBinomial),column_types);
    if !isempty(columns_gaussian)
        prediction[columns_gaussian] = _predict_gaussian(prediction[columns_gaussian]);
    end
    if !isempty(columns_bernoulli)
        prediction[columns_bernoulli] = _predict_bernoulli(prediction[columns_bernoulli]);
    end

    if !isempty(columns_poisson)
        prediction[columns_poisson] = _predict_poisson(prediction[columns_poisson]);
    end
    if !isempty(columns_gamma)
        prediction[columns_gamma] = _predict_gamma(prediction[columns_gamma]);
    end
    if !isempty(columns_negative_binomial)
        prediction[columns_negative_binomial] = _predict_negative_binomial(prediction[columns_negative_binomial]);
    end
end



mutable struct MatrixCompletionModel
    observed
    completed
    prediction
    type_matrix
    function MatrixCompletionModel(;observed=nothing,completed=nothing,type_matrix=nothing)
        model = new();
        model.observed = deepcopy(observed);
        model.completed = deepcopy(completed);
        model.type_matrix = deepcopy(type_matrix);
        model.prediction = predict(x=model.completed,obs=model.type_matrix)
    end
end




function complete(;A::DefaultMatrix   = nothing,
                   α::Float64         = maximum(A[findall(x -> !ismissing(x),A)]),
                   λ::Float64         = 5e-1,
                   μ::Float64         = 5e-4,
                   σ::Float64         = 0.3,
                   τ::Float64         = 1.618,
                   maxiter::Int64     = 200,
                   stoptol::Float64   = 1e-5,
                   use_autodiff::Bool = false,
                   gd_iter::Int64     = 50,
                   debug_mode::Bool   = false,
                   interactive_plot   = false)
    if isnothing(A)
        error("please provide data matrix");
    end
    Fnorm         = x -> norm(x,2);
    d1,d2         = size(A);
    type_matrix   = construct_type_matrix(A);
    index_tracker = construct_index_tracker(input_type_matrix = type_matrix);
    Aobs        = A[index_tracker.Observed];
    AobsBinary  = A[index_tracker.Bernoulli];
    AobsCont    = A[index_tracker.Gaussian];
    Y  = zeros(d1+d2,d1+d2);
    X  = zeros(d1+d2,d1+d2);
    W  = zeros(d1+d2,d1+d2);
    C  = zeros(d1+d2,d1+d2);
    R  = zeros(d1+d2,d1+d2);
    II = sparse(1.0I, d1+d2, d1+d2)
    σReset    = 1
    breakyes  = 0;
    runhist = RunHistory();
    binaryCnt = length(index_tracker.Bernoulli);
    warm_up_bernoulli = rand(binaryCnt,1);
    for iter = 1:maxiter
        debug_mode && @printf("current admm iter:%d\n",iter)
        # if debug_mode
        #     @printf("current admm iter:%d\n",iter);
        # end
        σInv     = 1/σ
        Xinput   = Y + σInv * W;
        X        = sdpProjection(Xinput-(σInv*μ)*II);
        debug_mode && @printf("sdp projection done\n")
        Xdual    = -σ*(X-Xinput);
        # Step 2
        C     = X - σInv * W;
        diagC = diag(C);
        Y12   = C[1:d1,d1+1:d1+d2];
        if !isempty(index_tracker.Gaussian)
            Y12[index_tracker.Gaussian] = (1/(1+σ))*(A[index_tracker.Gaussian]+σ*Y12[index_tracker.Gaussian]);
            debug_mode && @printf("gaussian update done\n")
        end
        if !isempty(index_tracker.Bernoulli)
            C12ObsBinary           = Y12[index_tracker.Bernoulli];
            binaryCnt              = length(index_tracker.Bernoulli);
            if use_autodiff == true
                # binaryPartUpdate   = train(Losses.Logistic(),rand(binaryCnt,1),A[index_tracker.Bernoulli],C12ObsBinary,σ;iter = gd_iter,γ=0.2);
                binaryPartUpdate   = train(provide(Loss{Binomial}()),
                                                  fx   = warm_up_bernoulli,
                                                  y    = A[index_tracker.Bernoulli],
                                                  c    = C12ObsBinary,
                                                  ρ    = σ,
                                                  iter = gd_iter,
                                                  γ    = 0.2);
            elsepp
                # binaryPartUpdate   = train_logistic(rand(binaryCnt,1),A[index_tracker.Bernoulli],C12ObsBinary,σ;iter=gd_iter,γ=0.2);
                # binaryPartUpdate   = train_logistic(warm_up_bernoulli,A[index_tracker.Bernoulli],C12ObsBinary,σ;iter=gd_iter,γ=0.2);
                binaryPartUpdate   = train(Loss{Binomial}(),
                                           fx   = warm_up_bernoulli,
                                           y    = A[index_tracker.Bernoulli],
                                           c    = C12ObsBinary,
                                           ρ    = σ,
                                           iter = gd_iter,
                                           γ    = 0.2); 
            end
            
            warm_up_bernoulli = binaryPartUpdate;
            Y12[index_tracker.Bernoulli]  = binaryPartUpdate;
            debug_mode && @printf("bernoulli update done\n")
        end
        if !isempty(index_tracker.Poisson)
        end
        if !isempty(index_tracker.NegativeBinomial)
        end
        if !isempty(index_tracker.Gamma)
        end
        Y12      = max.(-α,min.(Y12,α));
        debug_mode && @printf("part1 done\n")
        ϵ        = λ*σInv;
        debug_mode && @printf("part2 done\n")
        diagYtmp = diagC - ϵ*l1BallProjection(diagC/ϵ,1);
        debug_mode && @printf("part3 done\n")
        Y11      = C[1:d1,1:d1];
        debug_mode && @printf("part4 done\n")
        Y22      = C[d1+1:d1+d2,d1+1:d1+d2];
        debug_mode && @printf("part5 done\n")
        Y        = [Y11  Y12; Y12' Y22];
        debug_mode && @printf("part6 done\n")
        Y        = Y+spdiagm(0 => diagYtmp - diag(Y));
        debug_mode && @printf("part7 done\n")
        Ydual    = σ*(Y-C);
        debug_mode && @printf("part8 done\n")
        # step 3
        W = W + τ*σ*(Y-X);
        debug_mode && @printf("part9 done\n")
        normX    = 1+Fnorm(X);
        debug_mode && @printf("part10 done\n")
       primfeas = Fnorm(X-Y)/normX;
        debug_mode && @printf("part11 done\n")
        err1     = σInv*Fnorm(W-Xdual);
        debug_mode && @printf("part12 done\n")
        err2     = σInv*Fnorm(W-Ydual);
        debug_mode && @printf("part13 done\n")
        dualfeas = maximum([err1,err2])/normX;
        debug_mode && @printf("part14 done\n")
        push!(runhist.primfeas, primfeas);
        push!(runhist.dualfeas, dualfeas);
        debug_mode && @printf("part15 done\n")
        if (max(primfeas,dualfeas) < stoptol) | (iter==maxiter)
           breakyes=1;
        end
        if (max(primfeas,dualfeas) < sqrt(stoptol)) & (dualfeas > 1.5*minimum(runhist.dualfeas[max(iter-49,1):iter]))& (iter > 150)
           breakyes=2;
        end
        if (rem(iter,20)==1) | (breakyes > 0)
           R = abs.(maximum(diag(Y)));
           obj1 = norm(Y12[index_tracker.Gaussian]-AobsCont)^2 + logisticLoss(Y12[index_tracker.Bernoulli],AobsBinary);
           # obj1 = norm(Y12[obsContDataIdx]-AobsCont)^2 + logisticLoss(Y12[obsBinaryDataIdx],AobsBinary);
           obj2 = λ*R;
           obj3 = μ * tr(II*X);
           maxY12 = maximum(abs.(Y12));
           @printf("\n %3.0f %3.2e %3.2e| %3.2e %5.2f %5.2f %3.2e| %3.2e|",iter,primfeas,dualfeas,λ,R,maxY12,μ,σ);
           @printf("| obj1: %3.2e obj2: %3.2e obj3: %3.2e|",obj1,obj2,obj3);
        end
        if (σReset > 0) & (rem(iter,10)==0)
           if (primfeas < 0.5*dualfeas);
              σ = 0.7*σ;
           elseif (primfeas > 2*dualfeas)
              σ = 1.3*σ;
           end
        end
        if (breakyes > 0 )
           @printf("\n break = %1.0f\n",breakyes);
           break;
        end
    end
    completedMatrix = C[1:d1,d1+1:d1+d2];
    return completedMatrix
end


end
