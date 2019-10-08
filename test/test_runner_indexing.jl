using MatrixCompletion



@testset "$(format("Tracker: IndexTracker[convert]"))" begin
  let
    data = [1, 2, 3, 4]
    tc = convert(Array{<:CartesianIndex}, data)
    @test typeof(tc) == Array{CartesianIndex{1}, 1}
  end
  let
    data = rand(5, 5)
    tc = convert(Array{<:CartesianIndex}, findall(x -> x < 0.5 , data))
    @test typeof(tc) == Array{CartesianIndex{2}, 1}
  end
end


@testset "$(format("Tracker: IndexTracker[data stream initialization]"))" begin
  let
    data1 = [:Gaussian :Bernoulli;
             :Gaussian :Bernoulli]
    data2 = [:Observed :Observed;
             :Missing  :Missing]
    tc = IndexTracker{Symbol}(data1, data2)
  end
  let
    data1 = [:Gaussian :Bernoulli;
             :Gaussian :Bernoulli]
    data2 = [:Observed :Observed :Observed;
             :Missing  :Missing :Missing]
    @test_throws DimensionMismatch  tc = IndexTracker{Symbol}(data1, data2)
  end
end


@testset "$(format("Tracker: IndexTracker[disjoint join]"))" begin
  let
    tc = IndexTracker{Symbol}([:a, :b])
    @test_throws DimensionMismatch disjoint_join(tc, [:a])
    @test_throws MethodError disjoint_join(tc, [:a, :c])
  end
  let
    tc = IndexTracker{Symbol}([:a :b; :c :d])
  end
end


@testset "$(format("Tracker: IndexTracker[getindex]"))" begin
  let
    data = [:a,:b]
    tc = IndexTracker{Symbol}(data)
    @test data[tc[:a]] == data[findall(x -> x == :a, data)]
  end
  let
    data = [:a :b;
            :c :d]
    tc = IndexTracker{Symbol}(data)
    @test data[tc[:a]] == data[findall(x -> x == :a, data)]
    @test data[tc[:b]] == data[findall(x -> x == :b, data)]
    @test data[tc[:c]] == data[findall(x -> x == :c, data)]
    @test data[tc[:d]] == data[findall(x -> x == :d, data)]
  end
end


@testset "$(format("Tracker: IndexTracker[symbol]"))" begin
  let
    tc1 = IndexTracker{Symbol}([:a, :b])
    @test typeof(tc1) == IndexTracker{Symbol}
    @test size(tc1) == (2, )
    @test tc1.dimension == (2, )
  end
end



@testset "$(format("Tracker: IndexTracker[groupby]"))" begin
  let
    data = Array{Symbol}(undef, 20, 20)
    data[:, 1:10]  .= :Gaussian
    data[:, 11:20] .= :Binomial
    data2 = Array{Symbol}(undef, 20, 20)
    data2[:, 1:5]   .= :Observed
    data2[:, 10:15] .= :Observed
    data2[:, 6:10]  .= :Missing
    data2[:, 16:20] .= :Missing
    tc = IndexTracker{Symbol}(data)
    # @show(size(Iterators.flatten(data[:, 1:10])))
    @test tc[:Gaussian] == findall(x -> x == :Gaussian, data)
    @test tc[:Binomial] == findall(x -> x == :Binomial, data)
    disjoint_join(tc, data2)
    @test tc[:Observed] == findall(x -> x == :Observed, data2)
    @test tc[:Missing] == findall(x -> x == :Missing, data2)
    tc2 = groupby(tc, [:Observed, :Missing])
    @test tc2[:Gaussian][:Observed] == intersect(findall(x -> x == :Gaussian, data),
                                                 findall(x -> x == :Observed, data2))
    @test tc2[:Gaussian][:Missing] == intersect(findall(x -> x == :Gaussian, data),
                                                findall(x -> x == :Missing, data2))
    @test tc2[:Binomial][:Observed] == intersect(findall(x -> x == :Binomial, data),
                                                 findall(x -> x == :Observed, data2))
    @test tc2[:Binomial][:Missing] == intersect(findall(x -> x == :Binomial, data),
                                                findall(x -> x == :Missing, data2))
  end
end

