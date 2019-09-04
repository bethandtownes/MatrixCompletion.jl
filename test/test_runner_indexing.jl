@testset "$(format("Tracker: IndexTracker[symbol]"))" begin
    let
        tc1 = IndexTracker{Any}()
        @test typeof(tc1) == IndexTracker{Any}
        @test_throws DomainError tc2 = IndexTracker{Symbol}([:a,:b],[:a,:b])
    end
end



# @testset "$(format("Tracker: check[integral/continuous]"))" begin
#     #integral for vector and matrices
#     let
#         [@test check(:integral,rand(-100:100,100))       == true for i in 1:5]
#         [@test check(:integral,rand(-100:100,100,100))   == true for i in 1:5]
#         [@test check(:integral,rand(100))                == false for i in 1:5]
#         [@test check(:integral,rand(100,100))            == false for i in 1:5]
#     end
#     # continuous test for vector and matrices
#     let
#         [@test check(:continuous,rand(100))              == true for i in 1:5]
#         [@test check(:continuous,rand(100,100))          == true for i in 1:5]
#         [@test check(:continuous,rand(-100:100, 100))    == false for i in 1:5]
#         [@test check(:continuous,rand(-100:100,100,100)) == false for i in 1:5] 
#     end
# end



# @testset "$(format("Tracker: check[gaussian]"))" begin
#     let
        
#     end
# end


# @testset "$(format("Tracker: check[poisson]"))" begin
#     let
        
#     end
# end


# @testset "$(format("Tracker: check[negative-binomial]"))" begin
#     let
#         @warn "[check] for negative binomial yet to be implemented."
#     end
# end


# @testset "$(format("Tracker: check[bernoulli]"))" begin
#     let
        
#     end
# end



# @testset "$(format("Tracker: check[gamma]"))" begin
#     let
        
#     end
# end






