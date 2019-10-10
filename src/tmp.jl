


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


