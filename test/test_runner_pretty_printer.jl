using MatrixCompletion.Utilities.PrettyPrinter
using Printf

@testset "$(format("PrettyPrinter: Tables Header"))" begin
  let
    header_list = ["Iter", "R(primal)", " R(dual)",  "ℒ(Gaussian)", "ℒ(Bernoulli)", "ℒ(Poisson)", "ℒ(Gamma)", "λ‖diag(Z)‖ᵢ", " μ⟨I, X⟩", " ‖Z₁₂‖ᵢ "]
    row = map(x -> @sprintf("%3.2e", x), rand(10))
    row[1] = "100"
    table_header(header_list)
    add_row(header_list, data = row) 
  end
end
