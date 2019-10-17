using MatrixCompletion.Utilities.BatchUtils


@testset "$(format("BatchUtils: Tables Header"))" begin
  let
    tc = collect(1:100)
    batch = BatchFactory{SequentialScan}(size = 10)
    initialize(batch, tc)

    expected = [1:10, 11:20, 21:30, 31:40, 41:50, 51:60, 61:70, 71:80, 81:90, 91:100]
    ptr = 1
    while has_next(batch)
      @test tc[next(batch)] == collect(expected[ptr])
      ptr += 1
    end
  end

  let
    tc = collect(1:100)
    batch = BatchFactory{SequentialScan}(size = 64)
    initialize(batch, tc)
    expected = [1:64, 65:100]
    ptr = 1
    while has_next(batch)
      @test tc[next(batch)] == collect(expected[ptr])
      ptr += 1
    end
  end
end
