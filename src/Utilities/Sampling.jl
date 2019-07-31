module Sampling

struct BernoulliModel end
struct UniformModel end
struct NonUniformModel end

import Distributions





function sample(model::BernoulliModel;x::Array{T,2},rate::T) where T<:AbstractFloat
    n,m = size(x);
    mask = [rand(Distributions.Bernoulli(rate)) == 1 ? 1 : missing for i in 1:n,j in 1:m]
    return mask .* x
end


end
