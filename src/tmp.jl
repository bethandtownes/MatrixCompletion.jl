


function test_1()
  cur = zeros(100000)
  for i = 1:100
    cur = cur - 0.55 * rand(100000)
  end
  return cur
end




function test_2()
  cur = zeros(100000)
  for i = 1:100
    cur .-= 0.55 * rand(100000)
  end
  return cur
end



@time test_1();

@time test_2();




import Arpack
import Random
import KrylovKit




a = rand(4000, 4000) * 5
a = a + a'


@time Arpack.eigs(a; nev= 1000, which=:LR, maxiter=200)

@time KrylovKit.eigsolve(a, 1000, :LR;issymmetric = true, krylovdim = 1000, maxiter = 200)
