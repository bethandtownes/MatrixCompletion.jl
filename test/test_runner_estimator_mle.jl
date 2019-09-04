import Distributions
using MatrixCompletion

@testset "$(format("Estimator: MLE[construction]"))"  begin
    @test typeof(MLE{Gaussian}()) == MLE{Gaussian}
    @test typeof(MLE(Gaussian())) == MLE{Gaussian}
    @test typeof(MLE(:Gaussian))  == MLE{Gaussian}
end


@testset "$(format("Estimator: MLE[Gaussian]"))" begin
    for i = 1:10
        input_σ = rand() * 10
        input_μ = rand() * 10
        tc = rand(Distributions.Gaussian(input_μ,input_σ),1000)
        out_1 = Distributions.fit_mle(Distributions.Gaussian,tc)
        out_2 = estimator(MLE{Gaussian}(),tc)
        @test out_1.μ == out_2[:μ] && out_1.σ == out_2[:σ]
    end
end

@testset "$(format("Estimator: MLE[Gamma]"))" begin
    for i = 1:10
        input_α = rand() * 10
        input_θ = rand() * 10
        tc = rand(Distributions.Gamma(input_α,input_θ),10000)
        out_1 = Distributions.fit_mle(Distributions.Gamma,tc)
        out_2 = estimator(MLE{Gamma}(),tc)
        @test out_1.α == out_2[:α] && out_1.θ == out_2[:θ]
    end
end


@testset "$(format("Estimator: MLE[Poisson]"))" begin
    for i = 1:10
        input_λ = rand() * 10
        tc = rand(Distributions.Poisson(input_λ),10000)
        out_1 = sum(tc) / length(tc)
        out_2 = estimator(MLE{Poisson}(),tc)
        @test check(:l2diff,out_1,out_2[:λ]) < 0.5
    end
end


@testset "$(format("Estimator: MLE[Bernoulli]"))" begin
    for i = 1:10
        input_p = rand()
        tc = rand(Distributions.Bernoulli(input_p),10000)
        out_1 = sum(tc) / length(tc)
        out_2 = estimator(MLE{Bernoulli}(),tc)
        @test check(:l2diff,out_1,out_2[:p]) < 0.5
    end
end
