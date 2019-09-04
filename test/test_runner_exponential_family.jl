using MatrixCompletion


@testset "$(format("Exponential Family: forward_map[poisson]"))" begin
    let
        tc1 = rand(100)
        @test check(:l2diff, forward_map(Poisson(),tc1),exp.(tc1),0)
        @test check(:l2diff, forward_map(:Poisson,tc1),exp.(tc1),0)
        tc2 = rand(100,100)
        @test check(:l2diff, forward_map(Poisson(),tc2),exp.(tc2),0)
        @test check(:l2diff, forward_map(:Poisson,tc2),exp.(tc2),0)
    end
end


@testset "$(format("Exponential Family: forward_map[gamma]"))" begin
    let
        tc1 = rand(100)
        @test check(:l2diff, forward_map(Gamma(),tc1),1 ./ tc1,0)
        @test check(:l2diff, forward_map(:Gamma,tc1),1 ./ tc1,0)
        tc2 = rand(100,100)
        @test check(:l2diff, forward_map(Gamma(),tc2),1 ./ tc2,0)
        @test check(:l2diff, forward_map(:Gamma,tc2),1 ./ tc2,0)
    end
end


@testset "$(format("Exponential Family: forward_map[gaussian]"))" begin
    let
        tc1 = rand(100)
        @test check(:l2diff, forward_map(Gaussian(),tc1),tc1,0)
        @test check(:l2diff, forward_map(:Gaussian,tc1), tc1,0)
        tc2 = rand(100,100)
        @test check(:l2diff, forward_map(Gaussian(),tc2),tc2,0)
        @test check(:l2diff, forward_map(:Gaussian,tc2),tc2,0)
    end
end


@testset "$(format("Exponential Family: forward_map[bernoulli]"))" begin
    let
        logit = (x) -> log.(x./(1 .- x))
        tc1 = rand(100)
        tc1_logit = logit(tc1)
        @test check(:l2diff, forward_map(Bernoulli(),tc1_logit),tc1,0)
        @test check(:l2diff, forward_map(:Bernoulli,tc1_logit), tc1,0)
        tc2 = rand(100,100)
        tc2_logit = logit(tc2)
        @test check(:l2diff, forward_map(Bernoulli(),tc2_logit),tc2,0)
        @test check(:l2diff, forward_map(:Bernoulli,tc2_logit), tc2,0)
    end
end


@testset "$(format("Exponential Family: predict[poisson]"))" begin
    let
        tc1 = rand(2:20,100)
        log_tc1 = log.(tc1)
        @test check(:l2diff, predict(Poisson(),forward_map(Poisson(),log_tc1)),tc1,0.0)
        @test check(:l2diff, predict(:Poisson,forward_map(:Poisson,log_tc1)),tc1,0.0)
        @test predict(:Poisson,[1,1];custom_prediction_function=1) == -1 # 
        tc2 = rand(2:20,100,100)
        log_tc2 = log.(tc2)
        @test check(:l2diff, predict(Poisson(),forward_map(Poisson(),log_tc2)),tc2,0.0)
        @test check(:l2diff, predict(:Poisson,forward_map(:Poisson,log_tc2)),tc2,0.0)
        @test predict(:Poisson,[1 1;1 1];custom_prediction_function=1) == -1 # 
    end
end



@testset "$(format("Exponential Family: predict[bernoulli]"))" begin
    let
        logit = (x) -> log.(x./(1 .- x))
        tc1 = rand(100)
        tc1_int = Int.(tc1 .> 0.5)
        logit_tc1 = logit(tc1)
        @test check(:l2diff, predict(Bernoulli(),forward_map(Bernoulli(),logit_tc1)),tc1_int,0.0)
        @test check(:l2diff, predict(:Bernoulli,forward_map(:Bernoulli,logit_tc1)),tc1_int,0.0)
        @test predict(:Bernoulli,[1,1];custom_prediction_function=1) == -1 # 
        tc2 = rand(100,100)
        tc2_int = Int.(tc2 .> 0.5)
        logit_tc2 = logit(tc2)
        @test check(:l2diff, predict(Bernoulli(),forward_map(Bernoulli(),logit_tc2)),tc2_int,0.0)
        @test check(:l2diff, predict(:Bernoulli,forward_map(:Bernoulli,logit_tc2)),tc2_int,0.0)
        @test predict(:Bernoulli,[1 1;1 1];custom_prediction_function=1) == -1 # 
    end
end



@testset "$(format("Exponential Family: predict[gaussian]"))" begin
    let
        tc1 = rand(100)
        @test check(:l2diff, predict(Gaussian(),forward_map(Gaussian(),tc1)),tc1,0.0)
        @test check(:l2diff, predict(:Gaussian,forward_map(:Gaussian,tc1)),tc1,0.0)
        @test predict(:Gaussian,[1,1];custom_prediction_function=1) == -1 # 
        tc2 = rand(100,100)
        @test check(:l2diff, predict(Gaussian(),forward_map(Gaussian(),tc2)),tc2,0.0)
        @test check(:l2diff, predict(:Gaussian,forward_map(:Gaussian,tc2)),tc2,0.0)
        @test predict(:Poisson,[1 1;1 1];custom_prediction_function=1) == -1 # 
    end
end


@testset "$(format("Exponential Family: predict[gaussian]"))" begin
    let
        tc1 = rand(100)
        inv_tc1 = 1 ./ tc1
        @test check(:l2diff, predict(Gamma(),forward_map(Gamma(),1 ./ tc1)),tc1,0.0)
        @test check(:l2diff, predict(:Gamma,forward_map(:Gamma,1 ./ tc1)),tc1,0.0)
        @test predict(:Gamma,[1,1];custom_prediction_function=1) == -1 # 
        tc2 = rand(100,100)
        inv_tc2 = 1 ./ tc2 
        @test check(:l2diff, predict(Gamma(),forward_map(Gamma(),inv_tc2)),tc2,0.0)
        @test check(:l2diff, predict(:Gamma,forward_map(:Gamma,inv_tc2)),tc2,0.0)
        @test predict(:Poisson,[1 1;1 1];custom_prediction_function=1) == -1 # 
    end
end
