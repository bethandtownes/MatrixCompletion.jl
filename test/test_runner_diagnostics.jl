using MatrixCompletion



function unit_test_error_metric(metric;input,reference)
    return provide(metric)(input-reference)
end


function unit_test_relative_error(metric;input,reference)
    return  relative_error(metric,input,reference)
end


function unit_test_total_error(metric;input,reference)
    
end



function unit_test_diagnostics(;input =nothing, reference = nothing)
    
end





@testset "Diagnostic Test: Error Metric" begin
    # set 1
    tc1_input = rand(20)
    tc1_reference = rand(20)
    slack = 1e-5
    unit_test_error_metric(LpSpace(1),tc1_input(),tc2_reference, mapreduce(x-> x>slack ? 1 : 0 ,+,abs.(tc1_input .- tc1_reference)))
 end
                          
