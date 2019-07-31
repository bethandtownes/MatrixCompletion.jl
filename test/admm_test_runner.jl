include("./admm_test.jl")


#
@time test_train_logistic()
@time test_train_logistic_optimized()
@time test_train_logistic(size=9000*3000)
@time test_train_logistic_optimized(size=9000*3000)

test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
test_admm_without_autodiff_smallinput(gd_iter=3)
test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)

a = Random.rand(6000,6000).-0.5
b = Random.rand(6000,6000)
c = Random.rand(6000,6000)


@time a * b * c
@time (a * b) * c

eigen(a)



asym = a' + a

m = Symmetric(asym)

@time eigen(m)



@time eigen(m,0,100)
