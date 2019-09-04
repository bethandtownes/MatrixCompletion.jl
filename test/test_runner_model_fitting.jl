@testset "$(format("Model Fitting: choose[Gaussian v.s. Poisson]"))" begin
    let
        total_passed = 0 
        for i = 1:1000
            sample = rand(Distributions.Gamma(rand(1:100),rand(1:100)),5000)
            if choose(Gaussian(),Gamma(),data=sample) == :Gamma
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000.0 > 0.9
        @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
    end

    let
        total_passed = 0
        for i = 1:1000
            sample = rand(Distributions.Gamma(rand(1:10),rand(1:10)),5000)
            if choose(Gaussian(),Gamma(),data=sample,
                      comp=Comparator{MGF}(MGF(),eval_at = collect(0.01:0.001:0.02))) == :Gamma
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.9
        @info @sprintf("[Success rate][Gamma[α∈(1,10),θ∈(1,10)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
    end

    let
        total_passed = 0
        for i = 1:1000
            sample = rand(Distributions.Gamma(rand(1:10),rand(1:10)),5000)
            if choose(Gaussian(),Gamma(),data=sample,
                      comp=Comparator{MGF}(MGF())) == :Gamma
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.9
        @info @sprintf("[Success rate][Gamma[α∈(1,10),θ∈(1,10)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
    end

    let
        total_passed = 0
        for i = 1:1000
            sample = rand(Distributions.Gamma(rand(1:100),rand(1:100)),5000)
            if choose(Gaussian(),Gamma(),data=sample,
                      comp=Comparator{MGF}(MGF())) == :Gamma
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.9
        @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
    end

    
    let
        total_passed = 0
        for i = 1:1000
            sample = rand(Distributions.Gaussian(rand(50:100),rand(1:5)),5000)
            if choose(Gaussian(),Gamma(),data=sample,
                      comp=Comparator{MGF}(MGF())) == :Gaussian
                total_passed = total_passed + 1
            end
        end
        @test total_passed/1000 > 0.9
        @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)] v.s. Gaussian(true)]: %f\n",total_passed/1000.0)
    end
end




