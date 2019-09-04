@info "Visualization: [PoissonMatrix (5x5) (λ=5)]"
display(PoissonMatrix(5,5,rank=3,λ=5))
@info "Visualization: [GaussianMatrix (5x5) (μ=0,σ=1)]"
display(GaussianMatrix(5,5,rank=3,σ=1,μ=0))
@info "Visualization: [BinomialMatrix (5x5) (p=0.5)]"
display(BernoulliMatrix(5,5,rank=3,p=0.5))
@info "Visualization: [GammaMatrix (5,5) (α=5,θ=0.5)]"
display(GammaMatrix(5,5,rank=3,α=5,θ=0.5))

