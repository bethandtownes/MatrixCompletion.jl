include("./admm_test.jl")

using Distributed


@time test_train_logistic()
@time test_train_logistic_optimized()
@time test_train_logistic(size=1000*1000)
@time @everywhere test_train_logistic_optimized(size=1000*1000)

test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
test_admm_without_autodiff_smallinput(gd_iter=3)
test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)
