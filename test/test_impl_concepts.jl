
@testset "$(format("Concept: MaybeMissing[Conversion][VecOrMat]"))" begin
    let
        tc1     = [1,2,3,4]
        output1 = convert(MaybeMissing{Number},tc1)
        @test isa(output1,Array{MaybeMissing{Number}}) == true
        @test output1 == tc1
        #see(output1)
        output2 = convert(MaybeMissing{Float64},tc1)
        @test isa(output2, Array{MaybeMissing{Float64}}) == true
        #display(output2)
        # test array of missing
        tc2 = [missing, missing, missing]
        output3 = convert(MaybeMissing{Float64},tc2)
        @test isa(output3,Array{MaybeMissing{Float64}}) == true
        output3[1] = 1.0
        @test output3[1] == 1.0 && ismissing(output3[2]) && ismissing(output3[3])
        @test isa(convert(MaybeMissing{Float64},[missing missing;missing missing]),
            Array{MaybeMissing{Float64}}) == true
    end
end