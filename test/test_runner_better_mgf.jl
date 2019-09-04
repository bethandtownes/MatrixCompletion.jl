import Distributions
@testset "$(format("MGF: construction"))" begin
    let
        @test typeof(MGF{Poisson}(Poisson()))                 == MGF{Poisson}
        @test typeof(MGF{Poisson}(Poisson();logscale = true)) == MGF{Poisson}
        @test typeof(MGF(Poisson()))                          == MGF{Poisson}
        @test typeof(MGF(Poisson(),logscale=false))           == MGF{Poisson}
        @test typeof(MGF(:Poisson))                           == MGF{Poisson}
        #@test_throws UnrecognizedSymbolException MGF(:POisson)
        # test argument with constructor
        let
            tc = MGF{Poisson}(Poisson(),logscale=true)
            @test typeof(tc) == MGF{Poisson}
            @test tc.OPTION_LOG_SCALE == true
        end
        let
            tc = MGF(Poisson(),logscale=true)
            @test typeof(tc) == MGF{Poisson}
            @test tc.OPTION_LOG_SCALE == true
        end
        let
            tc = MGF(:Poisson,logscale=true)
            @test typeof(tc) == MGF{Poisson}
            @test tc.OPTION_LOG_SCALE == true
        end
    end
end


@testset "$(format("SampleMGF: construction"))" begin
    let
        @test typeof(SampleMGF()) == SampleMGF
    end
    # test constuctor with argument
    let
        tc = SampleMGF(logscale = false)
        @test typeof(tc) == SampleMGF
        @test tc.OPTION_LOG_SCALE == false
    end
end

@testset "$(format("MGF: evaluation[Poisson]"))" begin
    let
        for i = 1:5
            t = collect(1:0.05:1.5)
            tc_λ = rand() * 3
            tc = evaluate(MGF(:Poisson),t,λ=tc_λ)
            compare = Distributions.mgf.(Distributions.Poisson(tc_λ),t)
            @test check(:l2diff,tc,compare) < 1e-2
            tc_log = evaluate(MGF(:Poisson,logscale=true),t,λ=tc_λ)
            compare_log = log.(Distributions.mgf.(Distributions.Poisson(tc_λ),t))
            @test check(:l2diff,tc_log,compare_log) < 1e-1
        end
    end
end


@testset "$(format("MGF: evaluation[Gaussian]"))" begin
    let
        for i = 1:5
            t = collect(1:0.05:1.5)
            tc_μ = rand() * 5
            tc_σ = rand() * 5
            tc = evaluate(MGF(:Gaussian),t;μ = tc_μ, σ = tc_σ)
            compare = Distributions.mgf.(Distributions.Gaussian(tc_μ,tc_σ),t)
            @test check(:l2diff,tc,compare) < 1e-2
            tc_log = evaluate(MGF(:Gaussian,logscale=true),t;μ = tc_μ,σ = tc_σ)
            compare_log = log.(Distributions.mgf.(Distributions.Gaussian(tc_μ,tc_σ),t))
            @test check(:l2diff,tc_log,compare_log) < 1e-2
        end
    end
end


@testset "$(format("MGF: evaluation[Gamma]"))" begin
    let
        for i = 1:5
            tc_α = rand() * 5
            tc_θ = rand() * 5
            # ensure support
            t = collect(0.01:0.05: (1/tc_θ))
            tc = evaluate(MGF(:Gamma),t;α = tc_α, θ = tc_θ)
            compare = Distributions.mgf.(Distributions.Gamma(tc_α,tc_θ),t)
            @test check(:l2diff,tc,compare) < 1e-2
            tc_log = evaluate(MGF(:Gamma,logscale=true),t;α = tc_α,θ = tc_θ)
            compare_log = log.(Distributions.mgf.(Distributions.Gamma(tc_α,tc_θ),t))
            @test check(:l2diff,tc_log,compare_log) < 1e-2
        end
    end
end


@testset "$(format("MGF: evaluation[Bernoulli]"))" begin
    let
        for i = 1:5
            tc_p = rand()
            # ensure support
            t = collect(0.01:0.05: 0.5)
            tc = evaluate(MGF(:Bernoulli),t;p = tc_p)
            compare = Distributions.mgf.(Distributions.Bernoulli(tc_p),t)
            @test check(:l2diff,tc,compare) < 1e-2
            tc_log = evaluate(MGF(:Bernoulli,logscale=true),t;p = tc_p)
            compare_log = log.(Distributions.mgf.(Distributions.Bernoulli(tc_p),t))
            @test check(:l2diff,tc_log,compare_log) < 1e-2
        end
    end
end


@testset "$(format("MGF: evaluation[NegativeBinomial]"))" begin
    let
        for i = 1:5
            tc_p = rand()
            tc_r = rand() * 5
            t = collect(0.01:0.05: -log(tc_p))
            tc = evaluate(MGF(:NegativeBinomial),t;
                          p = tc_p, r = tc_r)
            compare = Distributions.mgf.(Distributions.NegativeBinomial(tc_r,tc_p),t)
            @test check(:l2diff,tc,compare) < 1e-2
            tc_log = evaluate(MGF(:NegativeBinomial,logscale=true),t;p = tc_p,r=tc_r)
            compare_log = log.(Distributions.mgf.(Distributions.NegativeBinomial(tc_r, tc_p),t))
            @test check(:l2diff,tc_log,compare_log) < 1e-2
        end
    end
end


@testset "$(format("SampleMGF: evaluation"))" begin
    # there is not way to make sure its correctness. we try to a few distributions
    # and show that as order increases the approximation gets more accurate.
    let
        total_passed = 0 
        for i = 1:1000
            t            = collect(0:0.0001:0.001)
            sample       = rand(Distributions.Gamma(rand(1:100),rand(1:100)),5000)
            sample_mgf   = evaluate(SampleMGF(),t,data = sample,order = 15)
            real_mgf     = Distributions.mgf.(Distributions.fit_mle(Distributions.Gamma,sample) ,t)
            not_real_mgf = Distributions.mgf.(Distributions.fit_mle(Distributions.Normal,sample),t)
            if check(:l2diff,sample_mgf,real_mgf) < check(:l2diff,sample_mgf,not_real_mgf)
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.80
    end

    let
        total_passed = 0 
        for i = 1:1000
            t            = collect(0:0.01:0.1)
            sample       = rand(Distributions.Normal(rand(10:100),rand(1:10)),5000)
            sample_mgf   = evaluate(SampleMGF(),t,data = sample,order = 15)
            real_mgf     = Distributions.mgf.(Distributions.fit_mle(Distributions.Normal,sample) ,t)
            not_real_mgf = nothing
            try
                not_real_mgf = Distributions.mgf.(Distributions.fit_mle(Distributions.Gamma,sample),t)
            catch
                continue
            end
            if check(:l2diff,sample_mgf,real_mgf) < check(:l2diff,sample_mgf,not_real_mgf)
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.80
    end
end
