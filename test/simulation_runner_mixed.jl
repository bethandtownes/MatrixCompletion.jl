const FLAG_SIMULATION_MIXED_VARY_RANK_SMALL                 = false
const FLAG_SIMULATION_MIXED_VARY_RANK_MEDIUM                = true
const FLAG_SIMULATION_MIXED_VARY_RANK_LARGE                 = false

const FLAG_SIMULATION_MIXED_VARY_MISSING_PERCENTAGE_SMALL   = false
const FLAG_SIMULATION_MIXED_VARY_MISSING_PERCENTAGE_MEDIUM  = false
const FLAG_SIMULATION_MIXED_VARY_MISSING_PERCENTAGE_LARGE   = false

FLAG_SIMULATION_MIXED_VARY_RANK_SMALL ?
  include("simulation_mixed_vary_rank_small.jl")  : nothing

FLAG_SIMULATION_MIXED_VARY_RANK_MEDIUM ?
  include("simulation_mixed_vary_rank_medium.jl")  : nothing



