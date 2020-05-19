# MatrixCompletion.jl

MatrixCompletion.jl is Julia package for completion low rank matrices with missing entries. The problem of matrix completion has a wide range of applications, such as collaborative filtering, system identification, data imputation, and Internet of things localization. 

MatrixCompletion.jl by default uses algorithm proposed in Robust Matrix Completion with Mix Data Types, see [paper](), which is a convex algorithm that minimizes the joint likelihood reguluaized by both nuclear norm and max norm. It can complete low rank matrices with hetergenous data (columnwise or row wise) data types, which are in the exponential family such as:
* Bernoulli (binary data)
* Poisson (count data)
* Gaussian (continuous data)
* Negative Binomial (count)
* Gamma (skewed continuous data)

## Installation 
To install, simply type 
```julia
Pkg.add("MatrixCompletion)
```
in the Julia REPL. 

## Minimal Working Example 
For example, the following simulation snippet demonstrates how to complete a manually generated mixed-type low rank matrix.
```
using MatrixCompletion
import Distributions, Random
input_rank = 20
truth_matrix       = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1),           rank = input_rank), 500, 100),
                           (FixedRankMatrix(Distributions.Bernoulli(0.5),           rank = input_rank), 500, 100),
                           (FixedRankMatrix(Distributions.Gamma(10, 0.5),           rank = input_rank), 500, 100),
                           (FixedRankMatrix(Distributions.Poisson(5),               rank = input_rank), 500, 100),
                           (FixedRankMatrix(Distributions.NegativeBinomial(6, 0.8), rank = input_rank), 500, 100)])
sample_model       = provide(Sampler{BernoulliModel}(), rate = 80 / 100)
input_matrix       = sample_model.draw(truth_matrix)
manual_type_matrix = Array{Symbol}(undef, 500, 500)
manual_type_matrix[:, 1:100]   .= :Gaussian
manual_type_matrix[:, 101:200] .= :Bernoulli
manual_type_matrix[:, 201:300] .= :Gamma
manual_type_matrix[:, 301:400] .= :Poisson
manual_type_matrix[:, 401:500] .= :NegativeBinomial
user_input_estimators = Dict(:NegativeBinomial=> Dict(:r=>6, :p=>0.8))

completed_matrix, type_tracker, tracker = complete(A                     = input_matrix,
                                                    maxiter               = 200,
                                                    ρ                     = 0.3,
                                                    use_autodiff          = false,
                                                    gd_iter               = 3,
                                                    debug_mode            = false,
                                                    user_input_estimators = user_input_estimators,
                                                    project_rank          = input_rank * 10,
                                                    io                    = io,
                                                    type_assignment       = manual_type_matrix)


predicted_matrix = predict(MatrixCompletionModel(),
                            completed_matrix = completed_matrix,
                            type_tracker     = type_tracker,
                            estimators       = user_input_estimators)

summary_object   = summary(MatrixCompletionModel(),
                            predicted_matrix = predicted_matrix,
                            truth_matrix     = truth_matrix,
                            type_tracker     = type_tracker,
                            tracker          = tracker)
```

## Automatic data type detection
We note that in the MWE, the data type of the observed matrix was entered manually. Although doing so in most cases guarantees maximum recovery rate, in reality it is often unknown that what are the exact distributions of the underlying data. To address this issue, we provided an API that allows the algorithm to automatically detect the best fitting distributed within the supported range and after doing so, also acquire the MLE of the corresponding parameters. Traditional goodness-and-fit often has less power when the input data size are large. To address this problem, we adopted a different approach combining a simple trivial decision tree and comparing the empirical distribution to its exponential family candidates in the frequency domain, i.e. MGF. In order to use automatic data type detection, one simply need to not provide the `user_input_estimators` parameter in the `complete` function. 

**Note.** While useful, according to preliminary simulations results, using automatic data type detection often results in recovery rate loss when the underlying data is continuous and strongly skewed. 


## Simulation Utilities

### Different Fast Eigen Solvers
Due to potential unforseen numerical instabilitis, multiple fast eigen libraries are shipped with MatrixCompletion.jl. Arpack is tested to be the fastest when using the MKL patched Julia 1.2. By default, this is not enabled due to licensing issues; and a Julia native Lancozs library is used by default.

### Sampling Schemes
MatrixCompletion.jl supports two sampling schemes, Uniform Sampling and Bernoulli Sampling. Researchers can use `x = provide(Sampler{BernoulliModel}(), rate = 80 / 100)` to get an instance of the corresponding sampler and bind it to `x`, and use `x.draw(M)` to draw a partially observed matrix from full input matrix `M`. 

### Random Fixed Rank Matrix Generator
MatrixCompletion.jl also provides a random fixed rank matrix generator that supports generating low rank matrix of target rank and specified (combination of) distributions. For example, 
```julia
input_rank = 10
M = rand([(FixedRankMatrix(Distributions.Gaussian(0, 1),           rank = input_rank), 500, 100),
          (FixedRankMatrix(Distributions.Bernoulli(0.5),           rank = input_rank), 500, 100),
          (FixedRankMatrix(Distributions.Gamma(10, 0.5),           rank = input_rank), 500, 100),
          (FixedRankMatrix(Distributions.Poisson(5),               rank = input_rank), 500, 100),
          (FixedRankMatrix(Distributions.NegativeBinomial(6, 0.8), rank = input_rank), 500, 100)])
```
generates a 500*500 matrix `M` with target rank 50, with the five listed distribution divided according to the dimension parameters.


# On going work

## GUI front end
A Gui front end is to be created using Genie to 
* provide interactive plotting of convergence path and regularization path. 
* provide a simple and easy to use interface for user to try to apply the algorithm to pictures of users' choice.

## More Algorithms

MatrixCompletion.jl aims to be the most comprehensive matrix completion algorithm library in Julia consisting of both popular non-convex and convex algorithms proposed in the past decade, including (on going work):

* LMaFit: Low-Rank Matrix Fitting [(Wen et al. 2012)](http://link.springer.com/article/10.1007%2Fs12532-012-0044-1) [website](http://lmafit.blogs.rice.edu/)

* OptSpace: Matrix Completion from Noisy Entries  [(Keshavan et al. 2009)](http://arxiv.org/pdf/0906.2027v1.pdf) [website](http://web.engr.illinois.edu/~swoh/software/optspace/code.html)

* OR1MP: Orthogonal rank-one matrix pursuit for low rank matrix completion [(Wang et al. 2015)](https://arxiv.org/abs/1404.1377)

* SVT: A singular value thresholding algorithm for matrix completion [(Cai et al. 2008)](http://arxiv.org/pdf/0810.3286.pdf) [website](http://svt.stanford.edu/)

