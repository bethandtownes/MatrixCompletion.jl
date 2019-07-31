include("./admm_test.jl")


#
@time test_train_logistic()
@time test_train_logistic_optimized()
@time test_train_logistic(size=9000*3000)
@time test_train_logistic_optimized(size=9000*3000)

test_admm_with_autodiff_smallinput(gd_iter=3,dbg=true)
test_admm_without_autodiff_smallinput(gd_iter=3)
test_admm_without_autodiff_largeinput(gd_iter=3,dbg=true)
#
# a = Random.rand(3000,3000).-0.5
# b = Random.rand(6000,6000)
# c = Random.rand(6000,6000)
#
#
# @time a * b * c
# @time (a * b) * c

eigen(a)
a = Random.rand(10000,10000)
asym = a' + a

t  = asym * asym
# m = Symmetric(asym)
#
# @time eigen(m)
#
# @time eigen(m,0,100)
#
# @time eigvals(m)
# @time eigvecs(m)
# import LowRankApprox
#
#
# a = Random.rand(2000,2000)
# m = Symmetric(a+a')
# @time LowRankApprox.pheigfact(m)
#
# @time eigen(m)
#



a = Random.rand(5000,5000)

ta = a'+a
@time eigs(ta, nev = 20, which=:BE)

@time eigen(ta)









using Arpack
using LinearAlgebra
import Random


# eig_dec = eigen(test_matrix);
F = svd(test_matrix)
truncated = deepcopy(F.S)
truncated[51:end] .= 0
truncated_mat = F.U * Diagonal(truncated) * F.Vt
rank(truncated_mat)
a_truncated = truncated_mat[101:200,1:100]
rank(a_truncated)

a = Random.rand(2500,2500)
b = Random.rand(2500,2500)
c = Random.rand(2500,2500)
a = a+a'
b = b+b'
c = c+c'
test_matrix = [b a;a' c]
@time F = svd(test_matrix)
@time F_cheap = svds(test_matrix,nsv=100)
