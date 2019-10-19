using MatrixCompletion
import Distributions


@testset "$(format("Estimator: MOM[NegativeBinomial]]"))"  begin
  let
    for i in 1:10
      p_test = rand()
      r_test = rand() * 20
      data = rand(Distributions.NegativeBinomial(r_test, p_test), 1000 * 200)
      tc = estimator(MOM{NegativeBinomial}(), data)
      @test abs(tc[:p] - p_test) / p_test < 0.05
      @test abs(tc[:r] - r_test) / r_test < 0.05
    end
  end
end
