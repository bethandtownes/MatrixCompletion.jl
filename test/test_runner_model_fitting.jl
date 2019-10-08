import Distributions 
# @testset "$(format("Model Fitting: choose[Gaussian v.s. Poisson]"))" begin
#     let
#         total_passed = 0 
#         for i = 1:1000
#             sample = rand(Distributions.Gamma(rand(1:100),rand(1:100)),5000)
#             if choose(Gaussian(),Gamma(),data=sample) == :Gamma
#                 total_passed = total_passed + 1
#             end
#         end
#         @test total_passed/1000.0 > 0.9
#         @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
#     end

#     let
#         total_passed = 0
#         for i = 1:1000
#             sample = rand(Distributions.Gamma(rand(1:10),rand(1:10)),5000)
#             if choose(Gaussian(),Gamma(),data=sample,
#                       comp=Comparator{MGF}(MGF(),eval_at = collect(0.01:0.001:0.02))) == :Gamma
#                 total_passed = total_passed + 1
#             end
#         end
#         @test total_passed/1000 > 0.9
#         @info @sprintf("[Success rate][Gamma[α∈(1,10),θ∈(1,10)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
#     end

#     let
#         total_passed = 0
#         for i = 1:1000
#             sample = rand(Distributions.Gamma(rand(1:10),rand(1:10)),5000)
#             if choose(Gaussian(),Gamma(),data=sample,
#                       comp=Comparator{MGF}(MGF())) == :Gamma
#                 total_passed = total_passed + 1
#             end
#         end
#         @test total_passed/1000 > 0.9
#         @info @sprintf("[Success rate][Gamma[α∈(1,10),θ∈(1,10)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
#     end

#     let
#         total_passed = 0
#         for i = 1:1000
#             sample = rand(Distributions.Gamma(rand(1:100),rand(1:100)),5000)
#             if choose(Gaussian(),Gamma(),data=sample,
#                       comp=Comparator{MGF}(MGF())) == :Gamma
#                 total_passed = total_passed + 1
#             end
#         end
#         @test total_passed/1000 > 0.9
#         @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)](true) v.s. Gaussian]: %f\n",total_passed/1000.0)
#     end
#     let
#         total_passed = 0
#         for i = 1:1000
#             sample = rand(Distributions.Gaussian(rand(50:100),rand(1:5)),5000)
#             if choose(Gaussian(),Gamma(),data=sample,
#                       comp=Comparator{MGF}(MGF())) == :Gaussian
#                 total_passed = total_passed + 1
#             end
#         end
#         @test total_passed/1000 > 0.9
#         @info @sprintf("[Success rate][Gamma[α∈(1,100),θ∈(1,100)] v.s. Gaussian(true)]: %f\n",total_passed/1000.0)
#     end
# end

@testset "$(format("Model Fitting: check[continuous/integral]"))" begin
    # test the continuous case
    
    [let
     tc1 = rand(100) * 100 
     @test check(:continuous,tc1) == true
     @test check(:integral,tc1)   == false
     tc2 = rand(100,100) * 100
     @test check(:continuous,tc2) == true
     @test check(:integral,tc2)   == false
     end for i = 1:5] 
    # test the integral case
    [let
     tc1 = rand(1:10,100)
     @test check(:continuous,tc1) == false
     @test check(:integral,tc1)   == true
     tc2 = rand(1:10,100,100)
     @test check(:continuous,tc2) == false
     @test check(:integral,tc2)   == true
     end for i = 1:5]
end

@testset "$(format("Model Fitting: check[Bernoulli]"))"  begin
    [let
     tc1 = rand(0:1,100)
     @test check(:Bernoulli,tc1) ==  true
     @test check(:Bernoulli,tc1 .* 1.0) == true
     @test check(:Bernoulli,tc1 .* 1.5) == false
     tc2 = rand(0:1,100,100)
     @test check(:Bernoulli,tc2) == true
     @test check(:Bernoulli,tc2 .* 1.0) == true
     @test check(:Bernoulli,tc2 .* 1.5) == false
    end for i = 1:5]
end


@testset "$(format("Model Fitting: check[Poisson/NB]"))" begin
    let
        tc1 = rand(Distributions.Poisson(10),100)
        @test choose(:Poisson,:NegativeBinomial;data = tc1) == :Poisson
        @test choose(Poisson,NegativeBinomial;data=tc1)     == :Poisson
        @test choose(Poisson(),NegativeBinomial();data = tc1)  == :Poisson
    end
end


@testset "$(format("Model Fitting: check[support]"))" begin
    let
        tc = collect(1:10)
        @test check(:support,tc,layout=:flatten) == (1,10)
        @test check(:support,collect(-5:5),layout=:flatten) == (-5,5)
    end
end


@testset "$(format("Model Fitting: check[Gamma/Gaussian]"))" begin
    [let
        tc_vec = rand(Distributions.Gaussian(rand()*100,rand()*10),10000)
        @test choose(Gaussian,Gamma,data= tc_vec) == :Gaussian
        @test choose(Gaussian(),Gamma(),data= tc_vec) == :Gaussian
        @test choose(:Gaussian,:Gamma,data= tc_vec) == :Gaussian
        tc_mat = rand(Distributions.Gaussian(rand()*100,rand()*10),1000,1000)
        @test choose(Gaussian,Gamma,data= tc_mat) == :Gaussian
        @test choose(Gaussian(),Gamma(),data= tc_mat) == :Gaussian
        @test choose(:Gaussian,:Gamma,data= tc_mat) == :Gaussian
    end for i =1:5]
    [let
        tc_vec = rand(Distributions.Gamma(rand()*10,rand()*10),10000)
        @test choose(Gaussian,Gamma,data= tc_vec)     == :Gamma
        @test choose(Gaussian(),Gamma(),data= tc_vec) == :Gamma
        @test choose(:Gaussian,:Gamma,data= tc_vec)   == :Gamma
        tc_mat = rand(Distributions.Gamma(rand()*10,rand()*10),1000,1000)
        @test choose(Gaussian,Gamma,data= tc_mat)     == :Gamma
        @test choose(Gaussian(),Gamma(),data= tc_mat) == :Gamma
        @test choose(:Gaussian,:Gamma,data= tc_mat)   == :Gamma
        tc_mat = rand(Distributions.Gamma(rand()*100,rand()*100),1000,1000)
        @test choose(Gaussian,Gamma,data= tc_mat)     == :Gamma
        @test choose(Gaussian(),Gamma(),data= tc_mat) == :Gamma
        @test choose(:Gaussian,:Gamma,data= tc_mat)   == :Gamma
    end for i=1:5]
end
