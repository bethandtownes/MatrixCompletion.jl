@testset "$(format("Sampling: UniformModel[VecOrMat]"))" begin
    #==================== Vector Case ====================#
    let
        tc1 = rand(10000)
        output = Sampler(UniformModel(0.5)).draw(tc1)
        @test count(x->ismissing(x),output) > 1000
        @test count(x->!ismissing(x),output) > 1000
        # stronger
        tc2 = rand(10000)
        output = Sampler(UniformModel(0.1)).draw(tc2)
        @test count(x->ismissing(x),output) >1000
        @test count(x->!ismissing(x),output) <1000
    end

    #==================== Matrix Case ====================#
    let
        @test 4 <= count(x -> !ismissing(x),Sampler(UniformModel(0.1)).draw(ones(10,10))) <= 10
        @test 10 < count(x -> ismissing(x), Sampler(UniformModel(0.1)).draw(ones(10,10)))
    end
    #==================== Factory Mode ====================#
    let
        sampler = provide(Sampler{UniformModel}(),rate = 0.5)
        tc1 = rand(10000)
        output = sampler.draw(tc1)
        @test count(x->ismissing(x),output) > 1000
        @test count(x->!ismissing(x),output) > 1000
        # stronger
        tc2 = rand(10000)
        sampler2 = provide(Sampler{UniformModel}(),rate = 0.1)
        output = sampler2.draw(tc2)
        @test count(x->ismissing(x),output) >1000
        @test count(x->!ismissing(x),output) <=1000
        @test 4 <= count(x -> !ismissing(x),sampler2.draw(ones(10,10))) <= 10
        @test 10 < count(x -> ismissing(x),sampler2.draw(ones(10,10)))
    end
end
