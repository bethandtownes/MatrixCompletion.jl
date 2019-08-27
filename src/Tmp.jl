
include("MatrixCompletion.jl")

using .MatrixCompletion
using Test

FLAG_TEST_SAMPLING = false
FLAG_TEST_CONCEPTS = false




FLAG_TEST_CONCEPTS ?
    include("../test/test_impl_concepts.jl") : println("concepts test skipped")


FLAG_TEST_SAMPLING ?
    include("../test/test_impl_sampling.jl") : println("sampling test skipped")





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
