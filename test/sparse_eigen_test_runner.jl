try include("sparse_eigen_test.jl") catch end 
try include("./test/sparse_eigen_test.jl") catch end

test_native_eigen()

