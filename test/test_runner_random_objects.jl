import Distributions
import LinearAlgebra



const VISUAL_RANDOM_STRUCTURE = true


@testset "$(format("Random Structure: FixedRankMatrix[Constructor]"))" begin
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
        @test check(:rank,tc1,3)
        @test check(:rank,tc2,9)

        tc3 = rand([(FixedRankMatrix(Distributions.Gaussian(0,1),  rank=2),10,3),
                    (FixedRankMatrix(Distributions.Poisson(5),     rank=2),10,3),
                    (FixedRankMatrix(Distributions.Bernoulli(0.5), rank=2),10,4)])
        @test LinearAlgebra.rank(tc3) == 6
    end
end 



@testset "$(format("Random Structure: FixedRankMatrix[overload: Concepts.provide]"))" begin
    # test provide interface
    let
        tc = provide(FixedRankMatrix(Distributions.Gaussian(0,1),rank=5),
                     row = 10,col=10)
        @test check(:dimension,tc,(10,10))
        @test check(:rank,tc,5) 
    end
end

@testset "$(format("Random Structure: GaussianMatrix"))" begin
    let
        tc = GaussianMatrix(20,20,rank=10,μ=0.0,σ=1.0)
        @test check(:dimension,tc,(20,20))
        @test check(:rank,tc,10)
        # test optional rank parameter
        tc1 = GaussianMatrix(20,20;μ=0,σ=1)
        @test check(:dimension,tc1,(20,20))
        @test check(:rank,tc1,20)        
    end
end


@testset "$(format("Random Structure: PoissonMatrix"))" begin
    let
        tc = PoissonMatrix(20,20,rank=10,λ=3)
        @test check(:dimension,tc,(20,20))
        @test check(:rank,tc,10)
        # test optional rank parameter
        tc1 = PoissonMatrix(20,20;λ=3)
        @test check(:dimension,tc1,(20,20))
        @test check(:rank,tc1,20)
    end
end


@testset "$(format("Random Structure: BernoulliMatrix"))" begin
    let
        tc = BernoulliMatrix(20,20,rank=10,p=0.5)
        @test check(:dimension,tc,(20,20))
        @test check(:rank,tc,10)
        # test optional rank parameter
        tc1 = BernoulliMatrix(20,20;p=0.5)
        @test check(:dimension,tc1,(20,20))
        @test check(:rank,tc1,20)
    end
end



