module Tst

using Revise
#doodle =true

#include("MatrixCompletion.jl")f

using MatrixCompletion
using Test
import Distributions
using LinearAlgebra


FLAG_TEST_SAMPLING       = false
FLAG_TEST_CONCEPTS       = false
FLAG_TEST_RANDOM_OBJECTS = true


FLAG_TEST_CONCEPTS ?
    include("../test/test_impl_concepts.jl") : println("concepts test skipped")


FLAG_TEST_SAMPLING ?
    include("../test/test_impl_sampling.jl") : println("sampling test skipped")


#tc1 = rand(FixedRankMatrix{Gaussian}(Gaussian(),10),50,50)



@testset "$(format("Misc: check[:rank]"))" begin
    # object = :Rank
    let
        tc = ones(5,5)
        @test check(Val{:rank},tc,1)
        @test check(:rank,tc,1)
        @test check(:rank,tc) == 1
        @test check(:rank,tc,2) == false
    end
    # object = :dimension
    let
        tc = rand(5,5)
        @test check(:dimension,tc) == (5,5)
        @test check(:dimension,tc,(5,5)) == true
        @test check(:dimension,tc,(4,5)) == false
    end
end




@testset "$(format("Random Structure: FixedRankMatrix[Constructor]"))" begin
    @test_logs (:info,"abstract constructor of FixedRankMatrix") FixedRankMatrix{Distributions.Poisson}()
    # test data type
    @test isa(FixedRankMatrix{Distributions.Poisson},DataType) == true
    @test isa(FixedRankMatrix{Distributions.Poisson}(),DataType) == false
    # test default constructor
    @test FixedRankMatrix{Distributions.Poisson}(Distributions.Poisson();rank=4).rank==4
    # test shorthanded constructor
    @test FixedRankMatrix(Distributions.Poisson(5),rank=2).rank == 2
    @test typeof(FixedRankMatrix(Distributions.Poisson(5),rank=2).dist) == Distributions.Poisson{Float64}
    # test mixed distribution
    
end

@testset "$(format("Random Structure: FixedRankMatrix[overload: Base.rand]"))" begin
    let 
        tc1 = rand(FixedRankMatrix(Distributions.Gaussian(0,1),rank =3),
                   10,10)
        @test_logs (:warn,"Rank is not specified. Using formula: rank= ⌊0.3 * (row ∧ col)⌋") rand(FixedRankMatrix(Distributions.Gaussian(0,1)), 100,30)
        tc2 = rand(FixedRankMatrix(Distributions.Gaussian(0,1)),
                   100,30)
        
        @test rank(tc1) == 3
        @test rank(tc2) == 9
        tc3 = rand([(FixedRankMatrix(Distributions.Gaussian(0,1),  rank=2),10,3),
                    (FixedRankMatrix(Distributions.Poisson(5),     rank=2),10,3),
                    (FixedRankMatrix(Distributions.Bernoulli(0.5), rank=2),10,4)])
        @test LinearAlgebra.rank(tc3) == 6
    end
    # test provide interface
    let
        tc = provide(FixedRankMatrix(Distributions.Gaussian(0,1),rank=5),
                     row = 10,col=10)
        @test check(:dimension,tc,(10,10))
        @test check(:rank,tc,5) 
        
    end
    # test gaussian matrix function
    let
        tc = GaussianMatrix(20,20,rank=10,μ=0.0,σ=1.0)
        @test check(:dimension,tc,(20,20))
        @test check(:rank,tc,10)
        # more power test
        tc1 = GaussianMatrix(20,20;μ=0,σ=1)
        @test check(:dimension,tc1,(20,20))
        @test check(:rank,tc1,20)
        
    end
    # test poisson matrix function
    let
        
    end
end 





#@show FixedRankMatrix{Distributions.Poisson}()




# sample_bernoulli_model0 = Sampler(BernoulliModel(0.8))
# tc1 = sample_bernoulli_model0.draw(ones(5,5))
# display(tc1)
# @test isa(tc1, Array{MaybeMissing{Float64},2}) || Array{Float64,2}

# tc2 = sample_bernoulli_model0.draw([1,2,3,4,5])
# display(tc2)
# @test isa(tc2, Array{MaybeMissing{Int64},1}) || isa(tc2,Array{Int64})

# #@show typeof(Sampler{BernoulliModel}())


# sample_bernoulli_model1= provide(Sampler{BernoulliModel}(),rate = 0.8)
# tc3 = sample_bernoulli_model1.draw(ones(5,5))
# display(tc3)
# @test isa(tc3, Array{MaybeMissing{Float64},2}) || Array{Float64,2}

# tc4 = sample_bernoulli_model1.draw([1,2,3,4,5])
# display(tc4)
# @test isa(tc4, Array{MaybeMissing{Int64},1}) || isa(tc4,Array{Int64})

end
