using Test,Printf,LinearAlgebra
import Random,Distributions
import MatrixCompletion.Losses
import MatrixCompletion.Losses:train,SGD,train_logistic
import MatrixCompletion.Convex.ADMM:complete
import MatrixCompletion.Utilities.RandomMatrices:rand
import MatrixCompletion.Utilities.Sampling:sample,BernoulliModel
import MatrixCompletion.Utilities.Indexing:construct_index_trakcer,construct_type_matrix
import MatrixCompletion.Utilities.Indexing:Bernoulli,Gaussian,Poisson,Gamma,NegativeBinomial,Missing,DIST_FLAGS
import MatrixCompletion.Convex.ADMM:MatrixCompletionModel,
       predict,
       _get_column_distribution_type,
       _get_complete_type_matrix,
       _predict_bernoulli,
       _predict_poisson,
       _predict_gamma,
       _predict_negative_binomial,
       _predict_gaussian

function test_train_logistic(;size=1000,ρ=0.1,γ=0.2,maxIter=20)
   y = Int.(Random.bitrand(size));
   mle_x = train(Losses.Logistic(),Random.rand(size),y,zeros(size),ρ ,iter = maxIter, γ = γ);
   @test sum((Int.(sign.(mle_x)).+1)/2 .== y)/size >0.99
end

function test_train_logistic_optimized(;size=1000,ρ=0.1,γ=0.2,maxIter=20)
   y = Int.(Random.bitrand(size));
   mle_x = train_logistic(Random.rand(size),y,zeros(size),ρ ,iter = maxIter, γ = γ);
   @test sum((Int.(sign.(mle_x)).+1)/2 .== y)/size >0.99
end


function test_train_logistic_sgd(size=3000*9000,ρ=0.1,γ=0.2,ep=10)
    y = Int.(Random.bitrand(size));
    mle_x = train(SGD(),Losses.Logistic(),Random.rand(size),y,zeros(size),ρ ,epoch = ep, γ = γ,num_of_batch=20);
    print(sum((Int.(sign.(mle_x)).+1)/2 .== y)/size);
    @test sum((Int.(sign.(mle_x)).+1)/2 .== y)/size >0.99
end




function test_train_poisson(;size=500,ρ=0.05,γ=0.1,maxIter=100)
   y = rand(Distributions.Poisson(10),size)*1.0
   mle_x = train(Poisson(),rand(size),y,rand(size),ρ ,iter = maxIter, γ = γ);
   recoveredX = round.(exp.(mle_x));
   errRate = sum(abs.(recoveredX .- y) .>0) / size;
   # @printf("recovery rate: %f\n",1-errRate);
   @test 1-errRate >= 0.99
end



function test_train_gamma(;size=500,ρ=0.05,γ=0.1,maxIter=100)
end


function test_train_negative_binomial(;size=500,ρ=0.05,γ=0.1,maxIter=100)
end



function test_complete_type_matrix()
    test1_input = Array{Union{Missing,DIST_FLAGS},2}(undef,10,10)
    test1_input[:,1:2] .= Ref(Gaussian);
    test1_input[:,3:4] .= Ref(Bernoulli);
    test1_input[:,5:6] .= Ref(Gamma);
    test1_input[:,7:8] .= Ref(Poisson);
    test1_input[:,9:10] .= Ref(NegativeBinomial);
    test1_expect = deepcopy(test1_input);
    test1_input[diagind(test1_input)] .= missing;
    test1_output = _get_complete_type_matrix(test1_input);
    @test test1_output == test1_expect;
    test2_input = Array{DIST_FLAGS,2}(undef,10,10)
    test2_input[:,1:2] .= Ref(Gaussian);
    test2_input[:,3:4] .= Ref(Bernoulli);
    test2_input[:,5:6] .= Ref(Gamma);
    test2_input[:,7:8] .= Ref(Poisson);
    test2_input[:,9:10] .= Ref(NegativeBinomial);
    test2_expect = deepcopy(test2_input);
    test2_output = _get_complete_type_matrix(test2_input);
    @test test2_output == test2_expect;
end



function test_predict_bernoulli()
    size=500;ρ=0.1;γ=0.1;maxIter=500;
    y = Int.(Random.bitrand(size));
    mle_x = train(Losses.Logistic(),Random.rand(size),y,zeros(size),ρ ,iter = maxIter, γ = γ);
    @test sum(_predict_bernoulli(mle_x) .== y)/size > 0.99
end



# test_predict_bernoulli()


function test_predict()

end







function accuracyImputedBinaryPart(;truth::Array{Float64,2}=nothing, completedMatrix::Array{Float64,2}=nothing)
    typeM = construct_type_matrix(truth);
#     binaryColumns = find(x->x==BINARY,typeM[1,:]);
    binaryColumns = findall(x->x==Bernoulli,typeM[1,:]);
    imputedBinaryPart  = (sign.(completedMatrix[:,binaryColumns]) .+ 1)/2
    return sum(Int.(truth[:,binaryColumns] .== imputedBinaryPart)) / (length(binaryColumns) * size(truth)[1]);
end


function accuracyImputedContinuousPart(;truth::Array{Float64,2}=nothing,completedMatrix::Array{Float64,2}=nothing)
    typeM = construct_type_matrix(truth);
    continuousColumns = findall(x->x==Gaussian,typeM[1,:])
    imputedContinuousPart = completedMatrix[:,continuousColumns];
    imputedContinuousPart - truth[:,continuousColumns]
    return norm(imputedContinuousPart - truth[:,continuousColumns])^2/norm(truth[:,continuousColumns])^2
end




function test_admm_with_autodiff_smallinput(;gd_iter=3,dbg=false)
    admm_test_matrix1 = rand([(Distributions.Bernoulli(0.7),100=>50,3),(Distributions.Gaussian(3,1),100=>50,3)])
    admm_test_matrix_missing1 = sample(BernoulliModel(),x = admm_test_matrix1,rate = 0.8)
    @time admm_test_matrix_output_1 = complete(A = admm_test_matrix_missing1,maxiter=200,use_autodiff=true,gd_iter=gd_iter,debug_mode=dbg);
    gaussian_acc = accuracyImputedContinuousPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
    bernoulli_acc = accuracyImputedBinaryPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
    @printf("gaussian acc: %f\n", gaussian_acc)
    @printf("bernoulli acc: %f\n",bernoulli_acc)
end


function test_admm_without_autodiff_smallinput(;gd_iter=3,dbg=false)
    admm_test_matrix1 = rand([(Distributions.Bernoulli(0.7),100=>50,3),(Distributions.Gaussian(3,1),100=>50,3)])
    admm_test_matrix_missing1 = sample(BernoulliModel(),x = admm_test_matrix1,rate = 0.8)
    @time admm_test_matrix_output_1 = complete(A = admm_test_matrix_missing1,maxiter=200,use_autodiff=false,gd_iter=gd_iter,debug_mode=dbg)
    gaussian_acc = accuracyImputedContinuousPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
    bernoulli_acc = accuracyImputedBinaryPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
    @printf("gaussian acc: %f\n", gaussian_acc)
    @printf("bernoulli acc: %f\n",bernoulli_acc)
end


function test_admm_without_autodiff_largeinput(;gd_iter=3,dbg=false)
    admm_test_matrix1 = rand([(Distributions.Bernoulli(0.7),6000=>3000,10),(Distributions.Gaussian(3,1),6000=>3000,10)])
    admm_test_matrix_missing1 = sample(BernoulliModel(),x = admm_test_matrix1,rate = 0.8)
    @time admm_test_matrix_output_1 = complete(A = admm_test_matrix_missing1,maxiter=200,use_autodiff=false,gd_iter=gd_iter,debug_mode=dbg)
    gaussian_acc = accuracyImputedContinuousPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
    bernoulli_acc = accuracyImputedBinaryPart(truth=admm_test_matrix1,completedMatrix = admm_test_matrix_output_1)
end





#
#
#
# admm_test_matrix2 = rand([(Distributions.Bernoulli(0.7),5000=>2500,100),(Distributions.Gaussian(3,1),5000=>2500,100)])
# admm_test_matrix_missing2 = sample(BernoulliModel(),x = admm_test_matrix2,rate = 0.8)
#
# a = construct_type_matrix(admm_test_matrix2)
# construct_index_trakcer(input_type_matrix=a)
# admm_test_matrix_output_2 = complete(A = admm_test_matrix_missing2)
# accuracyImputedContinuousPart(truth=admm_test_matrix2,completedMatrix = admm_test_matrix_output_2)
# accuracyImputedBinaryPart(truth=admm_test_matrix2,completedMatrix = admm_test_matrix_output_2)
#
#
#
# #
# # _get_column_distribution_type(type_mat[:,1])
# # _get_column_distribution_type(type_mat[:,51])
# # type_mat[:,90] .= Ref(_get_column_distribution_type(type_mat[:,51]))
#
#
# complete_type_mat = _get_complete_type_matrix(type_mat)
# complete_type_mat[:,90]
# predict(x=ot,obs=type_mat)
#
#
# MatrixCompletionModel(observed=admm_test_matrix_missing1,completed=ot,type_matrix=type)
#
#
# test_train_logistic()
# test_train_poisson()
# test_complete_type_matrix()
